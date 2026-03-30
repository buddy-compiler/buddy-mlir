# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Runtime library
# ══════════════════════════════════════════════════════════════════════════════

# Model shape constants — must match the compiled .so.
set(BUDDY_DSR1_HEAD_NUM       2          CACHE STRING "DeepSeekR1 KV head count")
set(BUDDY_DSR1_MAX_TOKEN_LEN  1024       CACHE STRING "DeepSeekR1 max token length")
set(BUDDY_DSR1_HIDDEN_SIZE    128        CACHE STRING "DeepSeekR1 KV hidden size per head")
set(BUDDY_DSR1_VOCAB_SIZE     151936     CACHE STRING "DeepSeekR1 vocabulary size (Qwen tokenizer: 151936 entries, EOS=151643)")
set(BUDDY_DSR1_KV_LAYERS      56         CACHE STRING "DeepSeekR1 KV cache layers")
set(BUDDY_DSR1_PARAMS_SIZE    1777088064 CACHE STRING "DeepSeekR1 parameter count (floats)")
# Default --max-tokens / RunConfig::maxNewTokens (total context incl. prompt). Keep in sync with
# buddy::runtime::RunConfig in InferenceRunner.h and buddy-cli default.
set(BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS 1024 CACHE STRING
    "DeepSeekR1 default max total tokens (prompt + generation) for buddy-cli / RunConfig")

add_library(buddy_models_deepseekr1 STATIC
  ModelSession.cpp
  DeepSeekR1Runner.cpp
)

target_include_directories(buddy_models_deepseekr1
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BUDDY_SOURCE_DIR}/frontend/Interfaces
    ${BUDDY_BINARY_DIR}/frontend/Interfaces
)

target_compile_features(buddy_models_deepseekr1 PUBLIC cxx_std_17)

target_compile_definitions(buddy_models_deepseekr1 PUBLIC
  BUDDY_DSR1_HEAD_NUM=${BUDDY_DSR1_HEAD_NUM}
  BUDDY_DSR1_MAX_TOKEN_LEN=${BUDDY_DSR1_MAX_TOKEN_LEN}
  BUDDY_DSR1_HIDDEN_SIZE=${BUDDY_DSR1_HIDDEN_SIZE}
  BUDDY_DSR1_VOCAB_SIZE=${BUDDY_DSR1_VOCAB_SIZE}
  BUDDY_DSR1_KV_LAYERS=${BUDDY_DSR1_KV_LAYERS}
  BUDDY_DSR1_PARAMS_SIZE=${BUDDY_DSR1_PARAMS_SIZE}LL
  BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS=${BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS}
)

# buddy_runtime_core → InferenceRunner, ModelManifest, BufferPool + buddy_rax (transitive)
# CMAKE_DL_LIBS      → -ldl for dlopen/dlsym/dlclose
target_link_libraries(buddy_models_deepseekr1 PUBLIC
  buddy_runtime_core
  ${CMAKE_DL_LIBS}
  LLVMSupport
)

# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Model compilation pipeline
# ══════════════════════════════════════════════════════════════════════════════

# ── User-configurable options ───────────────────────────────────────────────

set(DEEPSEEKR1_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(DEEPSEEKR1_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")
set(DEEPSEEKR1_NUM_THREADS "48" CACHE STRING
  "OpenMP thread count for MLIR compilation (--convert-scf-to-openmp=num-threads=N)")

# ── Platform detection ───────────────────────────────────────────────────────

if(HAVE_LOCAL_RVV)
  set(DSR1_LLC_ATTRS   "-mcpu=native -mattr=+m,+d,+v")
else()
  set(DSR1_LLC_ATTRS   "-mcpu=native")
endif()

# ── Common pass options ──────────────────────────────────────────────────────

set(DSR1_BUFFERIZE_OPTS "bufferize-function-boundaries")

set(DSR1_TOSA_PIPELINE
  "builtin.module(func.func(tosa-to-linalg-named),\
func.func(tosa-to-linalg),\
func.func(tosa-to-tensor),\
func.func(tosa-to-arith))")

set(DSR1_LOWER_TO_LLVM
  -memref-expand
  -arith-expand
  -convert-vector-to-llvm
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
  -reconcile-unrealized-casts
)

# Pipe: buddy-opt → mlir-opt (TOSA) → buddy-opt (vectorize + lower) →
#       mlir-translate → llvm-as → llc → .o
macro(dsr1_mlir_to_obj TARGET_NAME INPUT_MLIR OUTPUT_OBJ EXTRA_OPT_FLAGS)
  separate_arguments(DSR1_EXTRA_FLAGS UNIX_COMMAND "${EXTRA_OPT_FLAGS}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${INPUT_MLIR}
              -simplify-tosa-reshape
              ${DSR1_EXTRA_FLAGS}
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline=${DSR1_TOSA_PIPELINE}
              |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize="${DSR1_BUFFERIZE_OPTS}"
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -canonicalize
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -cse
              -canonicalize
              -optimize-allocation-liveness
              -matmul-vectorization-blis
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-parallelize
              -convert-vector-to-scf
              -lower-affine
              "-convert-scf-to-openmp=num-threads=${DEEPSEEKR1_NUM_THREADS}"
              ${DSR1_LOWER_TO_LLVM}
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
              |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as
              |
            ${LLVM_TOOLS_BINARY_DIR}/llc ${DSR1_LLC_ATTRS}
              -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    DEPENDS buddy-opt ${INPUT_MLIR}
    COMMENT "[DSR1] Compiling ${OUTPUT_OBJ}"
    VERBATIM
  )
endmacro()

# subgraph variant: needs extra -convert-elementwise-to-linalg and subgraph opts
macro(dsr1_subgraph_to_obj TARGET_NAME INPUT_MLIR OUTPUT_OBJ EXTRA_SUBGRAPH_FLAGS)
  separate_arguments(DSR1_SG_EXTRA UNIX_COMMAND "${EXTRA_SUBGRAPH_FLAGS}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${INPUT_MLIR}
              -simplify-tosa-reshape
              ${DSR1_SG_EXTRA}
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline=${DSR1_TOSA_PIPELINE}
              |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -one-shot-bufferize="${DSR1_BUFFERIZE_OPTS}"
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -canonicalize
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -cse
              -canonicalize
              -optimize-allocation-liveness
              -matmul-vectorization-blis
              -batchmatmul-optimize
              -batchmatmul-transpose-b-vectorization
              -convert-linalg-to-affine-loops
              -affine-parallelize
              -convert-vector-to-scf
              -lower-affine
              "-convert-scf-to-openmp=num-threads=${DEEPSEEKR1_NUM_THREADS}"
              -cse
              ${DSR1_LOWER_TO_LLVM}
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
              |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as
              |
            ${LLVM_TOOLS_BINARY_DIR}/llc ${DSR1_LLC_ATTRS}
              -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    DEPENDS buddy-opt ${INPUT_MLIR}
    COMMENT "[DSR1] Compiling ${OUTPUT_OBJ}"
    VERBATIM
  )
endmacro()

# subgraph_decode needs an extra decode-specific vectorization pass
macro(dsr1_subgraph_decode_to_obj INPUT_MLIR OUTPUT_OBJ)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${INPUT_MLIR}
              -simplify-tosa-reshape
              -simplify-tosa-matmul-scalar
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -pass-pipeline=${DSR1_TOSA_PIPELINE}
              |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -convert-elementwise-to-linalg
              -one-shot-bufferize="${DSR1_BUFFERIZE_OPTS}"
              -expand-strided-metadata
              -ownership-based-buffer-deallocation
              -canonicalize
              -buffer-deallocation-simplification
              -bufferization-lower-deallocations
              -cse
              -canonicalize
              -optimize-allocation-liveness
              -eliminate-memref-copy
              -assume-tight-memref-layout
              -staticize-memref-layout
              "-matmul-vectorization-decode=vector-size=32"
              "-batch-matmul-vectorization-decode=vector-size=128"
              "-batchmatmul-transpose-b-vectorization=vector-size=16"
              -convert-linalg-to-affine-loops
              -convert-vector-to-scf
              -lower-affine
              "-convert-scf-to-openmp=num-threads=${DEEPSEEKR1_NUM_THREADS}"
              -cse
              ${DSR1_LOWER_TO_LLVM}
              |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir
              |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as
              |
            ${LLVM_TOOLS_BINARY_DIR}/llc ${DSR1_LLC_ATTRS}
              -filetype=obj -relocation-model=pic -O3
              -o ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_OBJ}
    DEPENDS buddy-opt ${INPUT_MLIR}
    COMMENT "[DSR1] Compiling ${OUTPUT_OBJ} (decode vectorization)"
    VERBATIM
  )
endmacro()

# ── Determine where .o files come from ──────────────────────────────────────

set(DSR1_BIN ${CMAKE_CURRENT_BINARY_DIR})
set(DSR1_MODEL_SO ${DSR1_BIN}/deepseek_r1_model.so)

set(DSR1_OBJ_PREFILL   ${DSR1_BIN}/forward_prefill.o)
set(DSR1_OBJ_SUBPREFIL ${DSR1_BIN}/subgraph_prefill.o)
set(DSR1_OBJ_DECODE    ${DSR1_BIN}/forward_decode.o)
set(DSR1_OBJ_SUBDECODE ${DSR1_BIN}/subgraph_decode.o)

if(DEEPSEEKR1_BUILD_DIR)
  # ── Mode A: pre-built .o from an external build ────────────────────────────
  message(STATUS "[DSR1] Mode A: using pre-built .o from ${DEEPSEEKR1_BUILD_DIR}")

  set(DSR1_OBJ_PREFILL   ${DEEPSEEKR1_BUILD_DIR}/forward_prefill.o)
  set(DSR1_OBJ_SUBPREFIL ${DEEPSEEKR1_BUILD_DIR}/subgraph_prefill.o)
  set(DSR1_OBJ_DECODE    ${DEEPSEEKR1_BUILD_DIR}/forward_decode.o)
  set(DSR1_OBJ_SUBDECODE ${DEEPSEEKR1_BUILD_DIR}/subgraph_decode.o)

else()
  # ── Mode B / C: compile .o from MLIR ──────────────────────────────────────
  if(DEEPSEEKR1_MLIR_DIR)
    # Mode B: pre-generated MLIR files
    message(STATUS "[DSR1] Mode B: compiling from MLIR in ${DEEPSEEKR1_MLIR_DIR}")
    set(DSR1_MLIR_DIR ${DEEPSEEKR1_MLIR_DIR})
  else()
    # Mode C: full pipeline — run Python import first
    message(STATUS "[DSR1] Mode C: full pipeline (Python import + compile)")
    message(STATUS "  Set DEEPSEEKR1_MLIR_DIR to skip the import step.")
    set(DSR1_MLIR_DIR ${DSR1_BIN})

    # ── Stage 1: Python import ───────────────────────────────────────────────
    set(DSR1_IMPORT_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/import_deepseek_r1.py)

    add_custom_command(
      OUTPUT
        ${DSR1_BIN}/forward_prefill.mlir
        ${DSR1_BIN}/subgraph0_prefill.mlir
        ${DSR1_BIN}/forward_decode.mlir
        ${DSR1_BIN}/subgraph0_decode.mlir
        ${DSR1_BIN}/arg0.data
      COMMAND ${Python3_EXECUTABLE} ${DSR1_IMPORT_SCRIPT}
              --output-dir ${DSR1_BIN}
      DEPENDS ${DSR1_IMPORT_SCRIPT}
      COMMENT "[DSR1] Stage 1: importing model from Hugging Face → MLIR"
      VERBATIM
    )
  endif()

  # ── Stage 2: MLIR → .o ──────────────────────────────────────────────────────

  dsr1_mlir_to_obj(forward_prefill
    ${DSR1_MLIR_DIR}/forward_prefill.mlir
    forward_prefill.o
    ""
  )

  dsr1_subgraph_to_obj(subgraph_prefill
    ${DSR1_MLIR_DIR}/subgraph0_prefill.mlir
    subgraph_prefill.o
    ""
  )

  dsr1_mlir_to_obj(forward_decode
    ${DSR1_MLIR_DIR}/forward_decode.mlir
    forward_decode.o
    ""
  )

  dsr1_subgraph_decode_to_obj(
    ${DSR1_MLIR_DIR}/subgraph0_decode.mlir
    subgraph_decode.o
  )

endif()  # DEEPSEEKR1_BUILD_DIR

# ── Stage 3: Link .o → deepseek_r1_model.so ─────────────────────────────────

add_custom_command(
  OUTPUT ${DSR1_MODEL_SO}
  COMMAND ${CMAKE_CXX_COMPILER}
            -shared -fPIC
            -Wl,-soname,deepseek_r1_model.so
            -Wl,--allow-multiple-definition
            -o ${DSR1_MODEL_SO}
            ${DSR1_OBJ_PREFILL}
            ${DSR1_OBJ_SUBPREFIL}
            ${DSR1_OBJ_DECODE}
            ${DSR1_OBJ_SUBDECODE}
            -L${LLVM_LIBRARY_DIR}
            -Wl,-rpath,${LLVM_LIBRARY_DIR}
            -lomp
            -lmlir_c_runner_utils
            -lm
  DEPENDS
    ${DSR1_OBJ_PREFILL}
    ${DSR1_OBJ_SUBPREFIL}
    ${DSR1_OBJ_DECODE}
    ${DSR1_OBJ_SUBDECODE}
  COMMENT "[DSR1] Stage 3: linking deepseek_r1_model.so"
  VERBATIM
)

add_custom_target(deepseek_r1_model_so
  DEPENDS ${DSR1_MODEL_SO}
  COMMENT "deepseek_r1_model.so → ${DSR1_MODEL_SO}"
)

# ── Stage 4: rax-pack → deepseek_r1.rax ─────────────────────────────────────
#
# Output .rax goes to the binary dir alongside the .so so that
# the relative URIs in deepseek_r1.mlir ("file:deepseek_r1_model.so" etc.)
# resolve correctly at runtime.

set(DSR1_RAX ${DSR1_BIN}/deepseek_r1.rax)

add_custom_command(
  OUTPUT ${DSR1_RAX}
  COMMAND ${CMAKE_BINARY_DIR}/bin/rax-pack
            ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_r1.mlir
            -o ${DSR1_RAX}
  DEPENDS
    rax-pack
    ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_r1.mlir
    ${DSR1_MODEL_SO}
  COMMENT "[DSR1] Stage 4: packing deepseek_r1.rax"
  VERBATIM
)

# vocab.txt lives in the source tree (examples/BuddyDeepSeekR1/vocab.txt) and
# is copied into the build directory alongside the .rax so that the relative
# URI "file:vocab.txt" in the manifest resolves at runtime without any manual
# file-copying by the user.
set(DSR1_VOCAB_SRC ${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt)
set(DSR1_VOCAB     ${DSR1_BIN}/vocab.txt)

add_custom_command(
  OUTPUT ${DSR1_VOCAB}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DSR1_VOCAB_SRC} ${DSR1_VOCAB}
  DEPENDS ${DSR1_VOCAB_SRC}
  COMMENT "[DSR1] Copying vocab.txt to build directory"
  VERBATIM
)

add_custom_target(deepseek_r1_rax
  DEPENDS ${DSR1_RAX} ${DSR1_VOCAB}
  COMMENT "deepseek_r1.rax + vocab.txt → ${DSR1_BIN}"
)

# ── Summary ──────────────────────────────────────────────────────────────────

message(STATUS "[DSR1] Targets:")
message(STATUS "  deepseek_r1_model_so → ${DSR1_MODEL_SO}")
message(STATUS "  deepseek_r1_rax      → ${DSR1_RAX}  (+ vocab.txt)")
message(STATUS "[DSR1] After build, run inference with:")
message(STATUS "  buddy-cli --model ${DSR1_RAX} --prompt '...'")
