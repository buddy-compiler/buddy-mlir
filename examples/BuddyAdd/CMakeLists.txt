add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyAdd/forward.mlir ${BUDDY_EXAMPLES_DIR}/BuddyAdd/subgraph0.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyAdd/import-model.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyAdd/forward.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -matmul-paralell-vectorization-optimize
            -batchmatmul-optimize
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -affine-parallelize
            -lower-affine
            -convert-scf-to-openmp
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
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
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llvm-as |
        ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj -relocation-model=pic -O3
          -o ${BUDDY_BINARY_DIR}/../../examples/BuddyAdd/forward.o
  DEPENDS buddy-opt ${BUDDY_EXAMPLES_DIR}/BuddyAdd/forward.mlir
  COMMENT "Building forward.o "
  VERBATIM)

add_library(ADD STATIC forward.o)

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  ADD
  PROPERTIES
  LINKER_LANGUAGE C)

add_executable(buddy-add-run Add-main.cpp)
target_link_directories(buddy-add-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_ADD_LIBS
  ADD
  mlir_c_runner_utils
  omp
  gcc_s
)
if(BUDDY_MLIR_USE_MIMALLOC)
  list(APPEND BUDDY_ADD_LIBS mimalloc)
endif()

target_link_libraries(buddy-add-run ${BUDDY_ADD_LIBS})
