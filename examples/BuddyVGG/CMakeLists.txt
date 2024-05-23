add_custom_command(
  OUTPUT ${BUDDY_EXAMPLES_DIR}/BuddyVGG/vgg.mlir ${BUDDY_EXAMPLES_DIR}/BuddyVGG/arg0.data
  COMMAND python3 ${BUDDY_EXAMPLES_DIR}/BuddyVGG/buddy-vgg-import.py
  COMMENT "Generating vgg.mlir and parameter files"
)

add_custom_command(
  OUTPUT vgg.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/BuddyVGG/vgg.mlir 
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
            -eliminate-empty-tensors
            -convert-tensor-to-linalg 
            -linalg-bufferize
            -convert-linalg-to-affine-loops
            -lower-affine
            -func-bufferize
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
          ${LLVM_MLIR_BINARY_DIR}/llc -filetype=obj  -relocation-model=pic -O0 -o ${BUDDY_BINARY_DIR}/../examples/BuddyVGG/vgg.o
  DEPENDS ${BUDDY_EXAMPLES_DIR}/BuddyVGG/vgg.mlir
  COMMENT "Building vgg.o"
  VERBATIM)

add_library(VGG STATIC vgg.o)

SET_TARGET_PROPERTIES(VGG PROPERTIES LINKER_LANGUAGE C)

add_executable(buddy-vgg-run buddy-vgg-main.cpp)
target_link_directories(buddy-vgg-run PRIVATE ${LLVM_MLIR_LIBRARY_DIR})

set(BUDDY_VGG_LIBS VGG mlir_c_runner_utils ${OpenCV_LIBS})
target_link_libraries(buddy-vgg-run ${BUDDY_VGG_LIBS})
