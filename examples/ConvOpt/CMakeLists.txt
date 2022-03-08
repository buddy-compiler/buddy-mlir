if (${BUDDY_CONV_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_CONV_OPT_STRIP_MINING})
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256)
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128)
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64)
endif()


message(STATUS "Spliting size: ${SPLITING_SIZE}")


add_custom_command(OUTPUT conv2d.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/ConvOpt/conv2d.mlir -conv-vectorization="strip-mining=${SPLITING_SIZE}" -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/ConvOpt/conv2d.o
  DEPENDS buddy-opt)

# add_custom_command(OUTPUT conv2d.o
#   COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_EXAMPLES_DIR}/ConvOpt/conv2d.mlir -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf --convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
#           ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
#           ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/ConvOpt/conv2d.o
#   DEPENDS buddy-opt)

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
add_dependencies(edge-detection buddy-opt)
target_link_libraries(edge-detection ${OpenCV_LIBS} Conv2D)

add_subdirectory(comparison)
