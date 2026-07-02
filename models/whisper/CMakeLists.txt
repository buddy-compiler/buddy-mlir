# ══════════════════════════════════════════════════════════════════════════════
# Whisper (encoder-decoder ASR) — buddy-cli / .rax integration
#
# Self-contained build (does NOT use tools/buddy-codegen/buddy_add_model, which is
# specialized for LLM prefill/decode + KV cache).  Produces the same runtime
# contract as models/deepseek_r1:
#
#   whisper_model.so   compiled MLIR kernels (exports _mlir_ciface_forward)
#   whisper_runner.so  InferenceRunner plugin (dlopen'd by buddy-cli)
#   whisper.rax        RHAL manifest (weights + model.so embedded as payload)
#   vocab.txt          tokenizer vocab (resolved file: relative to the .rax)
#   audio.wav          sample input (convenience; pass via --audio)
#
# Run:  buddy-cli --model whisper.rax --audio audio.wav
# ══════════════════════════════════════════════════════════════════════════════

set(WHISPER_DIR        "${CMAKE_CURRENT_SOURCE_DIR}")
set(WHISPER_CODEGEN    "${WHISPER_DIR}/codegen")
set(WHISPER_EXAMPLE    "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper")
set(BIN                "${CMAKE_CURRENT_BINARY_DIR}")

set(BUDDY_WHISPER_SPEC
  "${WHISPER_DIR}/specs/base.json"
  CACHE FILEPATH "Whisper variant spec JSON, e.g. specs/base.json")
set(WHISPER_SPEC "${BUDDY_WHISPER_SPEC}")

# OpenMP runtime (libomp.so) lives under the LLVM runtimes build tree, not in
# ${LLVM_LIBRARY_DIR}. Mirror examples/CMakeLists.txt so -lomp resolves.
get_filename_component(WHISPER_LLVM_BUILD_DIR "${LLVM_LIBRARY_DIR}" DIRECTORY)
set(WHISPER_OMP_DIR
  "${WHISPER_LLVM_BUILD_DIR}/runtimes/runtimes-bins/openmp/runtime/src")

# Optional local HF snapshot; when set it is exported as WHISPER_MODEL_PATH so the
# importer loads weights from disk instead of downloading openai/whisper-base.
set(BUDDY_WHISPER_MODEL_PATH "" CACHE PATH
  "Local HuggingFace whisper snapshot dir (sets WHISPER_MODEL_PATH for import)")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# ── Stage 1: import PyTorch model → forward.mlir / subgraph0.mlir / arg0.data ──
if(NOT BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  message(FATAL_ERROR
    "models/whisper: PyTorch→MLIR import needs the Buddy Python package under "
    "build/python_packages. Re-configure with -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON.")
endif()

set(WHISPER_PY_PKG_ROOT "${CMAKE_BINARY_DIR}/python_packages")
if(BUDDY_WHISPER_MODEL_PATH)
  set(WHISPER_IMPORT_ENV ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${WHISPER_PY_PKG_ROOT}"
      "WHISPER_MODEL_PATH=${BUDDY_WHISPER_MODEL_PATH}")
else()
  set(WHISPER_IMPORT_ENV ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${WHISPER_PY_PKG_ROOT}")
endif()

set(WHISPER_IMPORT_DEPS
  "${WHISPER_CODEGEN}/import-whisper.py" "${WHISPER_SPEC}")
if(TARGET python-package-buddy)
  list(APPEND WHISPER_IMPORT_DEPS python-package-buddy)
endif()

add_custom_command(
  OUTPUT  "${BIN}/forward.mlir" "${BIN}/subgraph0.mlir" "${BIN}/arg0.data"
  COMMAND ${WHISPER_IMPORT_ENV}
          "${Python3_EXECUTABLE}" "${WHISPER_CODEGEN}/import-whisper.py"
          --spec "${WHISPER_SPEC}" --output-dir "${BIN}"
  DEPENDS ${WHISPER_IMPORT_DEPS}
  COMMENT "[whisper] Stage 1: importing model → forward/subgraph0.mlir + arg0.data"
  VERBATIM)

# ── Stage 2: forward.mlir → forward.o ─────────────────────────────────────────
# (pass pipeline mirrors examples/BuddyWhisper/CMakeLists.txt exactly)
add_custom_command(
  OUTPUT "${BIN}/forward.o"
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt "${BIN}/forward.mlir"
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops),matmul-parallel-vectorization-optimize, batchmatmul-optimize, eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-scf-to-openmp, convert-openmp-to-llvm, convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O0 -o "${BIN}/forward.o"
  DEPENDS "${BIN}/forward.mlir" buddy-opt
  COMMENT "[whisper] Stage 2: forward.mlir → forward.o"
  VERBATIM)

# ── Stage 2: subgraph0.mlir → subgraph0.o ─────────────────────────────────────
add_custom_command(
  OUTPUT "${BIN}/subgraph0.o"
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt "${BIN}/subgraph0.mlir"
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
            -test-linalg-transform-patterns=test-decompose-pad-tensor |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize=bufferize-function-boundaries
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -matmul-parallel-vectorization-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -convert-linalg-to-loops
            -convert-vector-to-scf
            -expand-strided-metadata
            -lower-affine
            -cse
            -convert-vector-to-llvm
            -memref-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3 -o "${BIN}/subgraph0.o"
  DEPENDS "${BIN}/subgraph0.mlir" buddy-opt
  COMMENT "[whisper] Stage 2: subgraph0.mlir → subgraph0.o"
  VERBATIM)

# ── Stage 3: link forward.o + subgraph0.o → whisper_model.so ──────────────────
set(WHISPER_MODEL_SO "${BIN}/whisper_model.so")
add_custom_command(
  OUTPUT "${WHISPER_MODEL_SO}"
  COMMAND ${CMAKE_CXX_COMPILER}
            -shared -fPIC
            -Wl,-soname,whisper_model.so
            -Wl,--allow-multiple-definition
            -o "${WHISPER_MODEL_SO}"
            "${BIN}/forward.o" "${BIN}/subgraph0.o"
            "-L${LLVM_LIBRARY_DIR}"
            "-L${WHISPER_OMP_DIR}"
            "-Wl,-rpath,${LLVM_LIBRARY_DIR}"
            "-Wl,-rpath,${WHISPER_OMP_DIR}"
            -lomp -lmlir_c_runner_utils -lm
  DEPENDS "${BIN}/forward.o" "${BIN}/subgraph0.o"
  COMMENT "[whisper] Stage 3: linking whisper_model.so"
  VERBATIM)

add_custom_target(whisper_model_so DEPENDS "${WHISPER_MODEL_SO}")

# ── Runtime: runner static library + plugin shared library ────────────────────
add_library(buddy_models_whisper STATIC WhisperRunner.cpp)
target_include_directories(buddy_models_whisper PUBLIC
  $<BUILD_INTERFACE:${WHISPER_DIR}/include>
  $<BUILD_INTERFACE:${BUDDY_SOURCE_DIR}/frontend/Interfaces>
  $<BUILD_INTERFACE:${BUDDY_BINARY_DIR}/frontend/Interfaces>)
target_compile_features(buddy_models_whisper PUBLIC cxx_std_17)
# The runner does not execute OpenMP itself (DAP preprocessing has no omp); the
# model .so it dlopens carries its own libomp via rpath. So omp is not linked here.
target_link_directories(buddy_models_whisper PUBLIC
  "${LLVM_LIBRARY_DIR}" "${WHISPER_OMP_DIR}")
target_link_libraries(buddy_models_whisper PUBLIC
  buddy_runtime_core
  BuddyLibDAP
  mlir_c_runner_utils
  LLVMSupport
  ${CMAKE_DL_LIBS})

add_library(buddy_models_whisper_runner SHARED WhisperRunnerPlugin.cpp)
set_target_properties(buddy_models_whisper_runner PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${BIN}"
  RUNTIME_OUTPUT_DIRECTORY "${BIN}"
  OUTPUT_NAME "whisper_runner"
  PREFIX "")
target_link_libraries(buddy_models_whisper_runner PRIVATE buddy_models_whisper)
target_compile_features(buddy_models_whisper_runner PRIVATE cxx_std_17)

# ── Stage 4: RHAL manifest → whisper.rax ──────────────────────────────────────
set(WHISPER_MANIFEST "${BIN}/whisper.mlir")
add_custom_command(
  OUTPUT  "${WHISPER_MANIFEST}"
  COMMAND "${Python3_EXECUTABLE}" "${WHISPER_CODEGEN}/gen_whisper_manifest.py"
          --spec "${WHISPER_SPEC}"
          --runner-library "whisper_runner.so"
          -o "${WHISPER_MANIFEST}"
  DEPENDS "${WHISPER_CODEGEN}/gen_whisper_manifest.py" "${WHISPER_SPEC}"
  COMMENT "[whisper] Stage 4: generating whisper.mlir (RHAL manifest)"
  VERBATIM)

# vocab.txt + audio.wav copied next to the .rax (vocab resolved file: at runtime).
set(WHISPER_VOCAB_DST "${BIN}/vocab.txt")
set(WHISPER_AUDIO_DST "${BIN}/audio.wav")
add_custom_command(
  OUTPUT  "${WHISPER_VOCAB_DST}" "${WHISPER_AUDIO_DST}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${WHISPER_EXAMPLE}/vocab.txt" "${WHISPER_VOCAB_DST}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${WHISPER_EXAMPLE}/audio.wav" "${WHISPER_AUDIO_DST}"
  DEPENDS "${WHISPER_EXAMPLE}/vocab.txt" "${WHISPER_EXAMPLE}/audio.wav"
  COMMENT "[whisper] Copying vocab.txt + audio.wav"
  VERBATIM)

set(WHISPER_RAX "${BIN}/whisper.rax")
add_custom_command(
  OUTPUT  "${WHISPER_RAX}"
  COMMAND "${CMAKE_BINARY_DIR}/bin/rax-pack"
          "${WHISPER_MANIFEST}" -o "${WHISPER_RAX}" --embed-payload
  DEPENDS rax-pack
          "${WHISPER_MANIFEST}"
          "${WHISPER_MODEL_SO}"
          "${BIN}/arg0.data"
          buddy_models_whisper_runner
          "${WHISPER_VOCAB_DST}"
  COMMENT "[whisper] Stage 4: packing whisper.rax"
  VERBATIM)

add_custom_target(whisper_rax
  DEPENDS "${WHISPER_RAX}" "${WHISPER_AUDIO_DST}"
  COMMENT "whisper.rax + vocab.txt + audio.wav → ${BIN}")
