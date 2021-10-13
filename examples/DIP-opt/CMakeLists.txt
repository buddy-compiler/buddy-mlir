add_custom_command(OUTPUT corr2d.o
COMMAND ${BUDDY_BINARY_DIR}/DIP-opt ${BUDDY_EXAMPLES_DIR}/DIP-opt/corr2d.mlir -lower-DIP -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DIP-opt/corr2d.o
DEPENDS DIP-opt)

add_library(Corr2D STATIC corr2d.o)

message(${BUDDY_EXAMPLES_DIR})

SET_TARGET_PROPERTIES(
Corr2D
PROPERTIES
LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(correlation2D correlation2D.cpp)
add_dependencies(correlation2D DIP-opt)
target_link_libraries(correlation2D ${OpenCV_LIBS} Corr2D)

