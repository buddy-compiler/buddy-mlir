add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/subgraph0.mlir ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/arg0.data
  COMMAND ${Python3_EXECUTABLE} ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/import-whisper.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and arg0.data..."
)
set(PATTERN_ARG "test-generalize-pad-tensor")
add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -pass-pipeline "builtin.module( func.func(buffer-deallocation-simplification, convert-linalg-to-loops),matmul-paralell-vectorization-optimize, batchmatmul-optimize, eliminate-empty-tensors,func-bufferize-dynamic-offset, func.func(llvm-request-c-wrappers),convert-scf-to-openmp, convert-openmp-to-llvm, convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
          ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyWhisper/forward.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
    OUTPUT subgraph0.o
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/subgraph0.mlir 
              -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              -test-linalg-transform-patterns=${PATTERN_ARG} |
            ${BUDDY_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -convert-elementwise-to-linalg
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -matmul-paralell-vectorization-optimize
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
              -convert-scf-to-openmp
              -func-bufferize-dynamic-offset
              -tensor-bufferize
              -convert-linalg-to-loops
              -finalizing-bufferize
              -convert-vector-to-scf
              -expand-strided-metadata
              -cse
              -convert-vector-to-llvm
              -memref-expand
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-openmp-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm 
              -convert-func-to-llvm
              -reconcile-unrealized-casts |
            ${LLVM_TOOLS_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
            ${LLVM_TOOLS_BINARY_DIR}/llvm-as |
            ${LLVM_TOOLS_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O3 -o ${BUDDY_BINARY_DIR}/../examples/BuddyWhisper/subgraph0.o
    DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyWhisper/subgraph0.mlir
    COMMENT "Building subgraph0.o "
    VERBATIM)

add_library(WHISPER STATIC forward.o subgraph0.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  WHISPER
  PROPERTIES
  LINKER_LANGUAGE C)

set(BUDDY_WHISPER_FILES
  whisper-main.h
  whisper-main.cpp
)

add_executable(buddy-whisper-run ${BUDDY_WHISPER_FILES})
target_link_directories(buddy-whisper-run PRIVATE ${LLVM_LIBRARY_DIR})

set(BUDDY_WHISPER_LIBS
  WHISPER
  BuddyLibDAP
  mlir_c_runner_utils
  omp
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_WHISPER_LIBS mimalloc)
endif()

target_link_libraries(buddy-whisper-run ${BUDDY_WHISPER_LIBS})
