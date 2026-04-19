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
#   [HF_CONFIG    <config.json>]            optional HuggingFace config path
#   [LOCAL_MODEL  <dir>]                    optional: HF snapshot dir for import
#                                           (sets DEEPSEEKR1_MODEL_PATH)
#   [BUILD_DIR    <dir>]                    mode A: use pre-built .o files
#   [MLIR_DIR     <dir>]                    mode B: use pre-generated MLIR files
#   [NUM_THREADS  <N>]                      OpenMP threads (default from spec)
#   [LLC_ATTRS    <string>]                 LLC target attributes
#   [COMPILE_JOBS <N>]                      parallel MLIR compilation jobs
# )
# ──────────────────────────────────────────────────────────────────────────────
function(buddy_add_model)
  cmake_parse_arguments(
    MDL                                      # prefix
    ""                                       # flags
    "NAME;SPEC;RUNNER_SRC;HF_CONFIG;LOCAL_MODEL;BUILD_DIR;MLIR_DIR;NUM_THREADS;LLC_ATTRS;COMPILE_JOBS"
    ""                                       # multi-value
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
            --config "${GEN_CONFIG}" -o "${GEN_RHAL}" ${MDL_GEN_MANIFEST_ARGS}
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

  # ════════════════════════════════════════════════════════════════════════════
  # Part 2: Model compilation pipeline (MLIR → .o → .so)
  #
  # Rough pipe (same as legacy dsr1_* macros): buddy-opt → mlir-opt (TOSA) →
  # buddy-opt (bufferize, vectorize, lower) → mlir-translate → llvm-as → llc → .o
  # Subgraph / decode file naming and extra flags are handled in compile_pipeline.py.
  # ════════════════════════════════════════════════════════════════════════════

  set(MODEL_SO "${BIN}/${MDL_NAME}_model.so")

  set(OBJ_FP "${BIN}/forward_prefill.o")
  set(OBJ_SP "${BIN}/subgraph_prefill.o")
  set(OBJ_FD "${BIN}/forward_decode.o")
  set(OBJ_SD "${BIN}/subgraph_decode.o")

  if(MDL_BUILD_DIR)
    # ── Mode A: pre-built .o ───────────────────────────────────────────────
    set(OBJ_FP "${MDL_BUILD_DIR}/forward_prefill.o")
    set(OBJ_SP "${MDL_BUILD_DIR}/subgraph_prefill.o")
    set(OBJ_FD "${MDL_BUILD_DIR}/forward_decode.o")
    set(OBJ_SD "${MDL_BUILD_DIR}/subgraph_decode.o")

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
      add_custom_command(
        OUTPUT "${IMPORT_STAMP}"
        COMMAND ${_IMPORT_ENV}
                "${Python3_EXECUTABLE}" "${BUDDY_CODEGEN_DIR}/import_model.py"
                --config "${GEN_CONFIG}" --output-dir "${BIN}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${IMPORT_STAMP}"
        DEPENDS ${IMPORT_DEPS}
        COMMENT "[${MDL_NAME}] Stage 1: importing model → MLIR + weights"
        VERBATIM
      )
      set(MLIR_COMPILE_DEPS "${IMPORT_STAMP}")
    endif()

    if(MDL_MLIR_DIR)
      set(MLIR_COMPILE_DEPS
        "${MLIR_SRC}/forward_prefill.mlir"
        "${MLIR_SRC}/subgraph0_prefill.mlir"
        "${MLIR_SRC}/forward_decode.mlir"
        "${MLIR_SRC}/subgraph0_decode.mlir"
      )
    endif()

    # ── Stage 2: MLIR → .o via compile_pipeline.py ─────────────────────────
    add_custom_command(
      OUTPUT "${OBJ_FP}" "${OBJ_SP}" "${OBJ_FD}" "${OBJ_SD}"
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

  # ── Stage 3: link .o → .so ─────────────────────────────────────────────
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

  add_custom_command(
    OUTPUT "${MODEL_SO}"
    COMMAND ${MDL_STAGE3_LINKER}
              ${MDL_STAGE3_LINK_OPTS}
              -shared -fPIC
              ${_BUDDY_MODEL_LINK_FLAGS}
              -o "${MODEL_SO}"
              "${OBJ_FP}" "${OBJ_SP}" "${OBJ_FD}" "${OBJ_SD}"
              "-L${LLVM_LIBRARY_DIR}"
              "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
              ${MDL_STAGE3_LIBS}
    DEPENDS "${OBJ_FP}" "${OBJ_SP}" "${OBJ_FD}" "${OBJ_SD}"
    COMMENT "[${MDL_NAME}] Stage 3: linking ${MDL_NAME}_model.so"
    VERBATIM
  )

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
