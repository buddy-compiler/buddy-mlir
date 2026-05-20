# ===- buddy_model.cmake - Config-driven model build ------------------------===
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===------------------------------------------------------------------------===
#
# Thin CMake layer that delegates heavy lifting to the buddy-codegen Python
# scripts.  A model's CMakeLists.txt simply calls:
#
#   include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
#
#   buddy_add_model(
#     NAME        deepseek_r1
#     SPEC        ${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json
#     RUNNER_SRC  DeepSeekR1Runner.cpp
#     RUNNER_HDR  include/buddy/runtime/models/DeepSeekR1Runner.h
#   )
#
# This creates:
#   - Target:  buddy_models_<NAME>        (static library)
#   - Target:  <NAME>_model_so            (shared library with MLIR .o)
#   - Target:  <NAME>_rax                 (.rax manifest + vocab.txt)
#
# ===----------------------------------------------------------------------===

set(BUDDY_CODEGEN_DIR "${CMAKE_SOURCE_DIR}/tools/buddy-codegen"
    CACHE PATH "Path to buddy-codegen scripts")

option(BUDDY_RAX_EMBED_PAYLOAD
  "Embed model .so / weights / vocab into .rax payload segment"
  ON)

option(IS_RVV_CROSSCOMPILE
  "Enable RVV cross-compilation for model.so (riscv64 target)"
  OFF)
option(BUDDY_MODEL_LAYER_PARTITION
  "Build supported models with validated layer-partitioned prefill compilation"
  ON)
option(BUDDY_MODEL_LAYER_PARTITION_DEBUG_WRAPPERS
  "Emit per-partition forward_* debug wrapper MLIR files for layer partitioning"
  OFF)
option(BUDDY_MODEL_REUSE_WEIGHTS
  "Reuse existing model weight data when a matching weight manifest is present"
  ON)
set(RISCV_GNU_TOOLCHAIN "" CACHE PATH
  "Path to RISCV GNU toolchain root (expects <root>/sysroot)")
set(RISCV_OMP_SHARED "" CACHE FILEPATH
  "Path to target OpenMP shared library for RVV link (e.g. libomp.so)")
set(RISCV_MLIR_C_RUNNER_UTILS "" CACHE FILEPATH
  "Path to target mlir_c_runner_utils shared library for RVV link")
if(NOT DEFINED BUDDY_MLIR_BUILD_DIR)
  if(DEFINED BUDDY_BUILD_DIR)
    set(_BUDDY_MLIR_BUILD_DIR_DEFAULT "${BUDDY_BUILD_DIR}")
  else()
    set(_BUDDY_MLIR_BUILD_DIR_DEFAULT "${CMAKE_BINARY_DIR}")
  endif()
  set(BUDDY_MLIR_BUILD_DIR "${_BUDDY_MLIR_BUILD_DIR_DEFAULT}" CACHE PATH
    "buddy-mlir build dir used to derive ../llvm/build/bin/clang(++)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# buddy_add_model(
#   NAME          <model_family>            e.g. deepseek_r1
#   SPEC          <variant_spec.json>       full path to variant spec
#   RUNNER_SRC    <file.cpp>                model-specific runner source
#   [RUNNER_PLUGIN_SRC <file.cpp>]          C ABI plugin wrapper source
#   [HF_CONFIG    <config.json>]            optional HuggingFace config path
#   [LOCAL_MODEL  <dir>]                    optional: HF snapshot dir for import
#                                           (sets DEEPSEEKR1_MODEL_PATH)
#   [BUILD_DIR    <dir>]                    mode A: use pre-built .o files
#   [MLIR_DIR     <dir>]                    mode B: use pre-generated MLIR files
#   [NUM_THREADS  <N>]                      OpenMP threads (default from spec)
#   [LLC_ATTRS    <string>]                 LLC target attributes
#   [COMPILE_JOBS <N>]                      parallel MLIR compilation jobs
#   [TIERED_KV_CACHE ON|OFF]                build multiple cache-sized entrypoints
#   [TIERED_CACHE_SIZES <list>]             e.g. "32;64;128;256;512;1024"
# )
# ──────────────────────────────────────────────────────────────────────────────
function(buddy_add_model)
  cmake_parse_arguments(
    MDL                                      # prefix
    ""                                       # flags
    "NAME;SPEC;RUNNER_SRC;RUNNER_PLUGIN_SRC;HF_CONFIG;LOCAL_MODEL;BUILD_DIR;MLIR_DIR;NUM_THREADS;LLC_ATTRS;COMPILE_JOBS;TIERED_KV_CACHE"
    "TIERED_CACHE_SIZES"                     # multi-value
    ${ARGN}
  )

  if(NOT MDL_NAME OR NOT MDL_SPEC OR NOT MDL_RUNNER_SRC)
    message(FATAL_ERROR "buddy_add_model: NAME, SPEC and RUNNER_SRC are required")
  endif()

  # Models are configured before tools/frontend Python; without this,
  # Python3_EXECUTABLE is empty and ninja invokes *.py directly → Permission denied
  # when the script is not chmod +x.
  if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()

  set(BIN      "${CMAKE_CURRENT_BINARY_DIR}")
  set(GEN_DIR  "${BIN}/generated")

  # Platform detection for LLC
  if(IS_RVV_CROSSCOMPILE)
    set(MDL_LLC_ATTRS "-march=riscv64 -mattr=+m,+d,+v -mtriple=riscv64-unknown-linux-gnu")
  elseif(NOT MDL_LLC_ATTRS)
    if(HAVE_LOCAL_RVV)
      set(MDL_LLC_ATTRS "-mcpu=native -mattr=+m,+d,+v")
    else()
      set(MDL_LLC_ATTRS "-mcpu=native")
    endif()
  endif()

  if(NOT MDL_COMPILE_JOBS)
    set(MDL_COMPILE_JOBS 1)
  endif()
  if(NOT MDL_RUNNER_PLUGIN_SRC)
    set(MDL_RUNNER_PLUGIN_SRC "${MDL_RUNNER_SRC}")
  endif()

  if(MDL_TIERED_KV_CACHE)
    set(MDL_TIERED_KV_CACHE ON)
  else()
    set(MDL_TIERED_KV_CACHE OFF)
  endif()
  if(MDL_TIERED_KV_CACHE AND NOT MDL_TIERED_CACHE_SIZES)
    set(MDL_TIERED_CACHE_SIZES 32 64 128 256 512 1024)
  endif()

  set(MDL_LAYER_PARTITION OFF)
  if(BUDDY_MODEL_LAYER_PARTITION AND NOT MDL_BUILD_DIR)
    if(MDL_TIERED_KV_CACHE)
      message(STATUS
        "[${MDL_NAME}] Layer partitioning is disabled for tiered KV cache builds.")
    elseif(IS_RVV_CROSSCOMPILE)
      message(STATUS
        "[${MDL_NAME}] Layer partitioning is disabled for RVV cross-compilation.")
    elseif(MDL_MLIR_DIR AND NOT EXISTS "${MDL_MLIR_DIR}/layer_partitioned/partition_manifest.json")
      message(STATUS
        "[${MDL_NAME}] Layer partitioning is disabled because MLIR_DIR has no layer_partitioned manifest.")
    else()
      set(MDL_LAYER_PARTITION ON)
    endif()
  endif()

  set(MDL_GEN_MANIFEST_ARGS)
  set(MDL_EXTRA_STAGE4_DEPS)

  if(IS_RVV_CROSSCOMPILE)
    if(NOT RISCV_GNU_TOOLCHAIN)
      message(FATAL_ERROR
        "IS_RVV_CROSSCOMPILE=ON requires RISCV_GNU_TOOLCHAIN (toolchain root with sysroot).")
    endif()
    if(NOT RISCV_OMP_SHARED)
      message(FATAL_ERROR
        "IS_RVV_CROSSCOMPILE=ON requires RISCV_OMP_SHARED (target OpenMP shared library path).")
    endif()
    if(NOT RISCV_MLIR_C_RUNNER_UTILS)
      message(FATAL_ERROR
        "IS_RVV_CROSSCOMPILE=ON requires RISCV_MLIR_C_RUNNER_UTILS (target mlir_c_runner_utils path).")
    endif()

    get_filename_component(RISCV_OMP_BASENAME "${RISCV_OMP_SHARED}" NAME)
    get_filename_component(RISCV_MLIR_RUNNER_BASENAME "${RISCV_MLIR_C_RUNNER_UTILS}" NAME)
    if(RISCV_OMP_BASENAME STREQUAL "")
      message(FATAL_ERROR "RISCV_OMP_SHARED has no basename: ${RISCV_OMP_SHARED}")
    endif()
    if(RISCV_MLIR_RUNNER_BASENAME STREQUAL "")
      message(FATAL_ERROR
        "RISCV_MLIR_C_RUNNER_UTILS has no basename: ${RISCV_MLIR_C_RUNNER_UTILS}")
    endif()

    set(RISCV_OMP_LOCAL "${BIN}/${RISCV_OMP_BASENAME}")
    set(RISCV_MLIR_RUNNER_LOCAL "${BIN}/${RISCV_MLIR_RUNNER_BASENAME}")

    add_custom_command(
      OUTPUT "${RISCV_OMP_LOCAL}" "${RISCV_MLIR_RUNNER_LOCAL}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RISCV_OMP_SHARED}" "${RISCV_OMP_LOCAL}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RISCV_MLIR_C_RUNNER_UTILS}" "${RISCV_MLIR_RUNNER_LOCAL}"
      DEPENDS "${RISCV_OMP_SHARED}" "${RISCV_MLIR_C_RUNNER_UTILS}"
      COMMENT "[${MDL_NAME}] Copying RVV runtime deps (omp/mlir_c_runner_utils)"
      VERBATIM
    )

    list(APPEND MDL_GEN_MANIFEST_ARGS
      --dep-shared-lib "file:${RISCV_OMP_BASENAME}"
      --dep-shared-lib "file:${RISCV_MLIR_RUNNER_BASENAME}")

    list(APPEND MDL_EXTRA_STAGE4_DEPS
      "${RISCV_OMP_LOCAL}"
      "${RISCV_MLIR_RUNNER_LOCAL}")
  endif()

  # ════════════════════════════════════════════════════════════════════════════
  # Part 0: Code generation (variant spec → config → C++ / MLIR manifest)
  # ════════════════════════════════════════════════════════════════════════════

  set(GEN_CONFIG  "${GEN_DIR}/config.json")
  # Header under buddy/runtime/models/ so #include "buddy/runtime/models/ModelSession.h"
  # resolves with -I ${GEN_DIR} only (no checked-in copy under models/<name>/include).
  set(GEN_SESS_H  "${GEN_DIR}/buddy/runtime/models/ModelSession.h")
  set(GEN_SESS_CC "${GEN_DIR}/ModelSession.cpp")
  set(GEN_RHAL    "${GEN_DIR}/${MDL_NAME}.mlir")
  set(RUNNER_PLUGIN_NAME "${MDL_NAME}_runner.so")

  # ── gen_config.py ─────────────────────────────────────────────────────────
  set(GEN_CONFIG_CMD
    "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_config.py"
    --spec "${MDL_SPEC}" -o "${GEN_CONFIG}"
  )
  if(MDL_HF_CONFIG)
    list(APPEND GEN_CONFIG_CMD --hf-config "${MDL_HF_CONFIG}")
  endif()

  add_custom_command(
    OUTPUT  "${GEN_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_DIR}"
    COMMAND ${GEN_CONFIG_CMD}
    DEPENDS "${MDL_SPEC}" "${BUDDY_CODEGEN_DIR}/gen_config.py"
    COMMENT "[${MDL_NAME}] Generating config.json from ${MDL_SPEC}"
    VERBATIM
  )

  # ── gen_session.py ────────────────────────────────────────────────────────
  add_custom_command(
    OUTPUT  "${GEN_SESS_H}" "${GEN_SESS_CC}"
    COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_session.py"
            --config "${GEN_CONFIG}" --output-dir "${GEN_DIR}"
    DEPENDS "${GEN_CONFIG}" "${BUDDY_CODEGEN_DIR}/gen_session.py"
    COMMENT "[${MDL_NAME}] Generating ModelSession.{h,cpp}"
    VERBATIM
  )

  # ── gen_manifest.py ───────────────────────────────────────────────────────
  add_custom_command(
    OUTPUT  "${GEN_RHAL}"
    COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/gen_manifest.py"
            --config "${GEN_CONFIG}" -o "${GEN_RHAL}"
            --runner-library "${RUNNER_PLUGIN_NAME}"
            ${MDL_GEN_MANIFEST_ARGS}
    DEPENDS "${GEN_CONFIG}" "${BUDDY_CODEGEN_DIR}/gen_manifest.py"
    COMMENT "[${MDL_NAME}] Generating ${MDL_NAME}.mlir (RHAL manifest)"
    VERBATIM
  )

  # ════════════════════════════════════════════════════════════════════════════
  # Part 1: Runtime static library
  # ════════════════════════════════════════════════════════════════════════════

  set(LIB_TARGET "buddy_models_${MDL_NAME}")

  add_library(${LIB_TARGET} STATIC
    "${GEN_SESS_CC}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${MDL_RUNNER_SRC}"
  )

  target_include_directories(${LIB_TARGET} PUBLIC
    "${GEN_DIR}"
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BUDDY_SOURCE_DIR}/frontend/Interfaces>
    $<BUILD_INTERFACE:${BUDDY_BINARY_DIR}/frontend/Interfaces>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/buddy-mlir>
  )
  target_compile_features(${LIB_TARGET} PUBLIC cxx_std_17)
  target_link_libraries(${LIB_TARGET} PUBLIC
    buddy_runtime_core
    buddy_runtime_llm
    ${CMAKE_DL_LIBS}
    LLVMSupport
  )
  install(FILES ${MDL_RUNNER_SRC}
    DESTINATION include/buddy-mlir/buddy/runtime/models/
    COMPONENT buddy_runtime
  )
  install(TARGETS ${LIB_TARGET}
    EXPORT BuddyMLIRTargets
    COMPONENT buddy_runtime
  )

  set(RUNNER_PLUGIN_TARGET "buddy_models_${MDL_NAME}_runner")
  add_library(${RUNNER_PLUGIN_TARGET} SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/${MDL_RUNNER_PLUGIN_SRC}"
  )
  set_target_properties(${RUNNER_PLUGIN_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BIN}"
    RUNTIME_OUTPUT_DIRECTORY "${BIN}"
    OUTPUT_NAME "${MDL_NAME}_runner"
    PREFIX ""
  )
  target_link_libraries(${RUNNER_PLUGIN_TARGET} PRIVATE ${LIB_TARGET})
  target_compile_features(${RUNNER_PLUGIN_TARGET} PRIVATE cxx_std_17)

  # ════════════════════════════════════════════════════════════════════════════
  # Part 2: Model compilation pipeline (MLIR → .o → .so)
  #
  # Rough pipe (same as legacy dsr1_* macros): buddy-opt → mlir-opt (TOSA) →
  # buddy-opt (bufferize, vectorize, lower) → mlir-translate → llvm-as → llc → .o
  # Subgraph / decode file naming and extra flags are handled in compile_pipeline.py.
  # ════════════════════════════════════════════════════════════════════════════

  set(MODEL_SO "${BIN}/${MDL_NAME}_model.so")

  set(OBJ_FILES)
  set(MLIR_COMPILE_DEPS)
  if(MDL_TIERED_KV_CACHE)
    foreach(CACHE_SIZE ${MDL_TIERED_CACHE_SIZES})
      list(APPEND OBJ_FILES
        "${BIN}/forward_prefill_${CACHE_SIZE}.o"
        "${BIN}/subgraph_prefill_${CACHE_SIZE}.o"
        "${BIN}/forward_decode_${CACHE_SIZE}.o"
        "${BIN}/subgraph_decode_${CACHE_SIZE}.o")
    endforeach()
  else()
    list(APPEND OBJ_FILES
      "${BIN}/forward_prefill.o"
      "${BIN}/subgraph_prefill.o"
      "${BIN}/forward_decode.o"
      "${BIN}/subgraph_decode.o")
  endif()

  if(MDL_BUILD_DIR)
    # ── Mode A: pre-built .o ───────────────────────────────────────────────
    set(OBJ_FILES)
    if(MDL_TIERED_KV_CACHE)
      foreach(CACHE_SIZE ${MDL_TIERED_CACHE_SIZES})
        list(APPEND OBJ_FILES
          "${MDL_BUILD_DIR}/forward_prefill_${CACHE_SIZE}.o"
          "${MDL_BUILD_DIR}/subgraph_prefill_${CACHE_SIZE}.o"
          "${MDL_BUILD_DIR}/forward_decode_${CACHE_SIZE}.o"
          "${MDL_BUILD_DIR}/subgraph_decode_${CACHE_SIZE}.o")
      endforeach()
    else()
      list(APPEND OBJ_FILES
        "${MDL_BUILD_DIR}/forward_prefill.o"
        "${MDL_BUILD_DIR}/subgraph_prefill.o"
        "${MDL_BUILD_DIR}/forward_decode.o"
        "${MDL_BUILD_DIR}/subgraph_decode.o")
    endif()

  else()
    # Determine MLIR source directory
    if(MDL_MLIR_DIR)
      # Mode B: pre-generated MLIR
      set(MLIR_SRC "${MDL_MLIR_DIR}")
    else()
      # Mode C: full pipeline (import → compile)
      if(NOT BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
        message(FATAL_ERROR
          "buddy_add_model (${MDL_NAME}): PyTorch→MLIR import needs the Buddy Python package under "
          "build/python_packages. Re-configure with:\n"
          "  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON\n"
          "tools/buddy-codegen/build_model.py passes this by default.")
      endif()

      set(MLIR_SRC "${BIN}")

      # Stamp output: import_model may write *-w8a16.mlir etc.; a single stamp keeps
      # CMake deps correct without duplicating variant-specific filenames here.
      set(IMPORT_STAMP "${BIN}/.buddy_import_done")
      # Synced by frontend/Python → build/python_packages/buddy/compiler (target
      # python-package-buddy). import_model needs PYTHONPATH to that tree.
      set(BUDDY_PY_PKG_ROOT "${CMAKE_BINARY_DIR}/python_packages")
      set(IMPORT_DEPS "${GEN_CONFIG}" "${BUDDY_CODEGEN_DIR}/import_model.py")
      # BUDDY_MLIR_ENABLE_PYTHON_PACKAGES is required above; target is always defined.
      if(TARGET python-package-buddy)
        list(APPEND IMPORT_DEPS python-package-buddy)
      endif()
      if(MDL_LOCAL_MODEL)
        set(_IMPORT_ENV ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${BUDDY_PY_PKG_ROOT}"
          "DEEPSEEKR1_MODEL_PATH=${MDL_LOCAL_MODEL}")
      else()
        set(_IMPORT_ENV ${CMAKE_COMMAND} -E env "PYTHONPATH=${BUDDY_PY_PKG_ROOT}")
      endif()
      set(_IMPORT_MODEL_EXTRA_ARGS)
      if(MDL_LAYER_PARTITION)
        list(APPEND _IMPORT_MODEL_EXTRA_ARGS
          --experimental-layer-partitioned
          --skip-full-mlir)
        if(BUDDY_MODEL_LAYER_PARTITION_DEBUG_WRAPPERS)
          list(APPEND _IMPORT_MODEL_EXTRA_ARGS
            --layer-partition-debug-wrappers)
        endif()
      endif()
      if(BUDDY_MODEL_REUSE_WEIGHTS)
        list(APPEND _IMPORT_MODEL_EXTRA_ARGS --reuse-existing-weights)
      endif()
      add_custom_command(
        OUTPUT "${IMPORT_STAMP}"
        COMMAND ${_IMPORT_ENV}
                "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/import_model.py"
                --config "${GEN_CONFIG}" --output-dir "${BIN}"
                ${_IMPORT_MODEL_EXTRA_ARGS}
        COMMAND "${CMAKE_COMMAND}" -E touch "${IMPORT_STAMP}"
        DEPENDS ${IMPORT_DEPS}
        COMMENT "[${MDL_NAME}] Stage 1: importing model → MLIR + weights"
        VERBATIM
      )
      set(MLIR_COMPILE_DEPS "${IMPORT_STAMP}")
    endif()

    if(MDL_MLIR_DIR)
      if(MDL_LAYER_PARTITION)
        set(MLIR_COMPILE_DEPS
          "${MLIR_SRC}/layer_partitioned/partition_manifest.json"
          "${MLIR_SRC}/layer_partitioned/forward_prefill.mlir"
          "${MLIR_SRC}/layer_partitioned/forward_decode.mlir")
      elseif(MDL_TIERED_KV_CACHE)
        set(MLIR_COMPILE_DEPS)
        foreach(CACHE_SIZE ${MDL_TIERED_CACHE_SIZES})
          list(APPEND MLIR_COMPILE_DEPS
            "${MLIR_SRC}/forward_prefill_${CACHE_SIZE}.mlir"
            "${MLIR_SRC}/subgraph0_prefill_${CACHE_SIZE}.mlir"
            "${MLIR_SRC}/forward_decode_${CACHE_SIZE}.mlir"
            "${MLIR_SRC}/subgraph0_decode_${CACHE_SIZE}.mlir")
        endforeach()
      else()
        set(MLIR_COMPILE_DEPS
          "${MLIR_SRC}/forward_prefill.mlir"
          "${MLIR_SRC}/subgraph0_prefill.mlir"
          "${MLIR_SRC}/forward_decode.mlir"
          "${MLIR_SRC}/subgraph0_decode.mlir"
        )
      endif()
    endif()

    if(MDL_LAYER_PARTITION)
      # ── Stage 2/3: partitioned MLIR → .o → .so via compile_pipeline.py ───
      set(PARTITIONED_MLIR_SRC "${MLIR_SRC}/layer_partitioned")
      set(PARTITIONED_OBJ_DIR "${BIN}/obj_partitioned")
      add_custom_command(
        OUTPUT "${MODEL_SO}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PARTITIONED_OBJ_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
                --config "${GEN_CONFIG}"
                --compile-partitioned
                --link
                --mlir-dir "${PARTITIONED_MLIR_SRC}"
                --output-dir "${PARTITIONED_OBJ_DIR}"
                --output-so "${MODEL_SO}"
                --buddy-opt "${BUDDY_BINARY_DIR}/buddy-opt"
                --llvm-tools-dir "${LLVM_TOOLS_BINARY_DIR}"
                "--llc-attrs=${MDL_LLC_ATTRS}"
                --cxx "${CMAKE_CXX_COMPILER}"
                --llvm-lib-dir "${LLVM_LIBRARY_DIR}"
                -j "${MDL_COMPILE_JOBS}"
        DEPENDS
          buddy-opt
          "${GEN_CONFIG}"
          "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
          ${MLIR_COMPILE_DEPS}
        COMMENT "[${MDL_NAME}] Stage 2/3: partitioned MLIR → ${MDL_NAME}_model.so"
        VERBATIM
      )
    else()
      # ── Stage 2: MLIR → .o via compile_pipeline.py ───────────────────────
      add_custom_command(
        OUTPUT ${OBJ_FILES}
        COMMAND "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
                --config "${GEN_CONFIG}"
                --compile-all
                --mlir-dir "${MLIR_SRC}"
                --output-dir "${BIN}"
                --buddy-opt "${BUDDY_BINARY_DIR}/buddy-opt"
                --llvm-tools-dir "${LLVM_TOOLS_BINARY_DIR}"
                "--llc-attrs=${MDL_LLC_ATTRS}"
                -j "${MDL_COMPILE_JOBS}"
        DEPENDS
          buddy-opt
          "${GEN_CONFIG}"
          "${BUDDY_CODEGEN_DIR}/compile_pipeline.py"
          ${MLIR_COMPILE_DEPS}
        COMMENT "[${MDL_NAME}] Stage 2: MLIR → .o (compile_pipeline.py)"
        VERBATIM
      )
    endif()
  endif()

  # ── Stage 3: link .o → .so ─────────────────────────────────────────────
  set(MDL_STAGE3_LINKER "${CMAKE_CXX_COMPILER}")
  set(MDL_STAGE3_LINK_OPTS)
  set(MDL_STAGE3_LIBS -lomp -lmlir_c_runner_utils -lm)

  if(IS_RVV_CROSSCOMPILE)
    set(CMAKE_C_COMPILER "${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang")
    set(CMAKE_CXX_COMPILER
        "${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang++")
    set(MDL_STAGE3_LINKER "${CMAKE_CXX_COMPILER}")

    set(RISCV_LINK_OPTS
      --target=riscv64-unknown-linux-gnu
      "--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot"
      "--gcc-toolchain=${RISCV_GNU_TOOLCHAIN}")
    set(MDL_STAGE3_LINK_OPTS
      ${RISCV_LINK_OPTS}
      "-Wl,-rpath,\$ORIGIN")
    set(MDL_STAGE3_LIBS
      "${RISCV_OMP_SHARED}"
      "${RISCV_MLIR_C_RUNNER_UTILS}"
      -lm)
  endif()

  if(APPLE)
    set(_BUDDY_MODEL_LINK_FLAGS
      "-Wl,-install_name,@rpath/${MDL_NAME}_model.so"
    )
  else()
    set(_BUDDY_MODEL_LINK_FLAGS
      "-Wl,-soname,${MDL_NAME}_model.so"
      "-Wl,--allow-multiple-definition"
    )
  endif()

  if(NOT MDL_LAYER_PARTITION)
    add_custom_command(
      OUTPUT "${MODEL_SO}"
      COMMAND ${MDL_STAGE3_LINKER}
                ${MDL_STAGE3_LINK_OPTS}
                -shared -fPIC
                ${_BUDDY_MODEL_LINK_FLAGS}
                -o "${MODEL_SO}"
                ${OBJ_FILES}
                "-L${LLVM_LIBRARY_DIR}"
                "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
                ${MDL_STAGE3_LIBS}
      DEPENDS ${OBJ_FILES}
      COMMENT "[${MDL_NAME}] Stage 3: linking ${MDL_NAME}_model.so"
      VERBATIM
    )
  endif()

  add_custom_target(${MDL_NAME}_model_so
    DEPENDS "${MODEL_SO}"
    COMMENT "${MDL_NAME}_model.so → ${MODEL_SO}"
  )

  # ════════════════════════════════════════════════════════════════════════════
  # Part 3: rax-pack → .rax
  # ════════════════════════════════════════════════════════════════════════════

  # Copy vocab.txt alongside the .rax (and make it visible to rax-pack payload
  # embedding via file:vocab.txt URI).
  set(VOCAB_SRC "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt")
  set(VOCAB_DST "${BIN}/vocab.txt")

  add_custom_command(
    OUTPUT  "${VOCAB_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VOCAB_SRC}" "${VOCAB_DST}"
    DEPENDS "${VOCAB_SRC}"
    COMMENT "[${MDL_NAME}] Copying vocab.txt"
    VERBATIM
  )

  set(MODEL_RAX "${BIN}/${MDL_NAME}.rax")
  set(RAX_PACK_ARGS)
  if(BUDDY_RAX_EMBED_PAYLOAD)
    list(APPEND RAX_PACK_ARGS --embed-payload)
  endif()

  set(MDL_STAGE4_DEPS
    rax-pack
    "${GEN_RHAL}"
    "${MODEL_SO}"
    ${RUNNER_PLUGIN_TARGET}
    "${VOCAB_DST}")
  list(APPEND MDL_STAGE4_DEPS ${MDL_EXTRA_STAGE4_DEPS})

  add_custom_command(
    OUTPUT "${MODEL_RAX}"
    COMMAND "${CMAKE_BINARY_DIR}/bin/rax-pack"
            "${GEN_RHAL}" -o "${MODEL_RAX}" ${RAX_PACK_ARGS}
    DEPENDS ${MDL_STAGE4_DEPS}
    COMMENT "[${MDL_NAME}] Stage 4: packing ${MDL_NAME}.rax"
    VERBATIM
  )

  add_custom_target(${MDL_NAME}_rax
    DEPENDS "${MODEL_RAX}" "${VOCAB_DST}"
    COMMENT "${MDL_NAME}.rax + vocab.txt → ${BIN}"
  )

endfunction()
