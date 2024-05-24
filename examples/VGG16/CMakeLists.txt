add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/VGG16/arg0.data ${BUDDY_EXAMPLES_DIR}/VGG16/forward.mlir ${BUDDY_EXAMPLES_DIR}/VGG16/subgraph0.mlir
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/VGG16/import-vgg16.py
  COMMENT "Generating forward.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/VGG16/forward.mlir 
  -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -pass-pipeline "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), eliminate-empty-tensors, func.func(llvm-request-c-wrappers),convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf,  convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 
          -o ${BUDDY_BINARY_DIR}/../examples/VGG16/forward.o
            DEPENDS ${BUDDY_EXAMPLES_DIR}/VGG16/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/VGG16/subgraph0.mlir
  -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-tensor-to-linalg 
            -linalg-bufferize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize-dynamic-offset
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llvm-as |
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0
          -o ${BUDDY_BINARY_DIR}/../examples/VGG16/subgraph0.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/VGG16/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)
add_library(VGG16 STATIC forward.o subgraph0.o ) #forward.o subgraph0.o)

SET_TARGET_PROPERTIES(VGG16 PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-vgg16-run buddy-vgg16-main.cpp)

target_link_directories(buddy-vgg16-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_VGG16_LIBS VGG16 mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(buddy-vgg16-run ${BUDDY_VGG16_LIBS})



