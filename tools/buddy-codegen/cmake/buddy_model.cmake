# ===- buddy_model.cmake - Config-driven model build ──────────────────────===
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
# ===----------------------------------------------------------------------===
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

# ──────────────────────────────────────────────────────────────────────────────
# buddy_add_model(
#   NAME          <model_family>            e.g. deepseek_r1
#   SPEC          <variant_spec.json>       full path to variant spec
#   RUNNER_SRC    <file.cpp>                model-specific runner source
#   [HF_CONFIG    <config.json>]            optional HuggingFace config path
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
    "NAME;SPEC;RUNNER_SRC;HF_CONFIG;BUILD_DIR;MLIR_DIR;NUM_THREADS;LLC_ATTRS;COMPILE_JOBS"
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
  if(NOT MDL_LLC_ATTRS)
    if(HAVE_LOCAL_RVV)
      set(MDL_LLC_ATTRS "-mcpu=native -mattr=+m,+d,+v")
    else()
      set(MDL_LLC_ATTRS "-mcpu=native")
    endif()
  endif()

  if(NOT MDL_COMPILE_JOBS)
    set(MDL_COMPILE_JOBS 1)
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
            --config "${GEN_CONFIG}" -o "${GEN_RHAL}"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${BUDDY_SOURCE_DIR}/frontend/Interfaces"
    "${BUDDY_BINARY_DIR}/frontend/Interfaces"
  )
  target_compile_features(${LIB_TARGET} PUBLIC cxx_std_17)
  target_link_libraries(${LIB_TARGET} PUBLIC
    buddy_runtime_core
    buddy_runtime_llm
    ${CMAKE_DL_LIBS}
    LLVMSupport
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
    message(STATUS "[${MDL_NAME}] Mode A: pre-built .o from ${MDL_BUILD_DIR}")
    set(OBJ_FP "${MDL_BUILD_DIR}/forward_prefill.o")
    set(OBJ_SP "${MDL_BUILD_DIR}/subgraph_prefill.o")
    set(OBJ_FD "${MDL_BUILD_DIR}/forward_decode.o")
    set(OBJ_SD "${MDL_BUILD_DIR}/subgraph_decode.o")

  else()
    # Determine MLIR source directory
    if(MDL_MLIR_DIR)
      # Mode B: pre-generated MLIR
      message(STATUS "[${MDL_NAME}] Mode B: MLIR from ${MDL_MLIR_DIR}")
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

      message(STATUS "[${MDL_NAME}] Mode C: full pipeline (import + compile)")
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
      add_custom_command(
        OUTPUT "${IMPORT_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${BUDDY_PY_PKG_ROOT}"
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
  add_custom_command(
    OUTPUT "${MODEL_SO}"
    COMMAND ${CMAKE_CXX_COMPILER}
              -shared -fPIC
              "-Wl,-soname,${MDL_NAME}_model.so"
              -Wl,--allow-multiple-definition
              -o "${MODEL_SO}"
              "${OBJ_FP}" "${OBJ_SP}" "${OBJ_FD}" "${OBJ_SD}"
              "-L${LLVM_LIBRARY_DIR}"
              "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
              -lomp -lmlir_c_runner_utils -lm
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

  set(MODEL_RAX "${BIN}/${MDL_NAME}.rax")

  add_custom_command(
    OUTPUT "${MODEL_RAX}"
    COMMAND "${CMAKE_BINARY_DIR}/bin/rax-pack"
            "${GEN_RHAL}" -o "${MODEL_RAX}"
    DEPENDS rax-pack "${GEN_RHAL}" "${MODEL_SO}"
    COMMENT "[${MDL_NAME}] Stage 4: packing ${MDL_NAME}.rax"
    VERBATIM
  )

  # Copy vocab.txt alongside the .rax
  set(VOCAB_SRC "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt")
  set(VOCAB_DST "${BIN}/vocab.txt")

  add_custom_command(
    OUTPUT  "${VOCAB_DST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VOCAB_SRC}" "${VOCAB_DST}"
    DEPENDS "${VOCAB_SRC}"
    COMMENT "[${MDL_NAME}] Copying vocab.txt"
    VERBATIM
  )

  add_custom_target(${MDL_NAME}_rax
    DEPENDS "${MODEL_RAX}" "${VOCAB_DST}"
    COMMENT "${MDL_NAME}.rax + vocab.txt → ${BIN}"
  )

  # ════════════════════════════════════════════════════════════════════════════
  # Summary
  # ════════════════════════════════════════════════════════════════════════════

  message(STATUS "[${MDL_NAME}] Targets:")
  message(STATUS "  ${LIB_TARGET}       → static runtime lib")
  message(STATUS "  ${MDL_NAME}_model_so → ${MODEL_SO}")
  message(STATUS "  ${MDL_NAME}_rax      → ${MODEL_RAX}")
  message(STATUS "[${MDL_NAME}] After build, run inference with:")
  message(STATUS "  buddy-cli --model ${MODEL_RAX} --prompt '...'")

endfunction()
