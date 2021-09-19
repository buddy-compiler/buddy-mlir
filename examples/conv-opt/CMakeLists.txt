add_custom_command(OUTPUT conv2d.o
  COMMAND ${BUDDY_BINARY_DIR}/conv-opt ${BUDDY_EXAMPLES_DIR}/conv-opt/conv2d.mlir -conv-vectorization="strip-mining=${BUDDY_CONV_OPT_STRIP_MINING}" -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/conv-opt/conv2d.o
  DEPENDS conv-opt)

# add_custom_command(OUTPUT conv2d.o
#   COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/conv-opt/conv2d.mlir -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' | 
#           ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
#           ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} --filetype=obj -o ${BUDDY_SOURCE_DIR}/build/examples/conv-opt/conv2d.o
#   DEPENDS conv-opt)

add_library(Conv2D STATIC conv2d.o)

message(${BUDDY_EXAMPLES_DIR})

SET_SOURCE_FILES_PROPERTIES(
  template.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true)

SET_TARGET_PROPERTIES(
  Conv2D
  PROPERTIES
  LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(edge-detection edge-detection.cpp)
add_dependencies(edge-detection conv-opt)
target_link_libraries(edge-detection ${OpenCV_LIBS} Conv2D)

add_subdirectory(comparison)
