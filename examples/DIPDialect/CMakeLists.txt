if (${BUDDY_DIP_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_DIP_OPT_STRIP_MINING})
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256)
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128)
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64)
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64)
endif()

message(STATUS "Spliting size: ${SPLITING_SIZE}")

add_custom_command(OUTPUT dip.o
COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/DIPDialect/dip.mlir -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" -arith-expand -lower-affine -convert-scf-to-cf -convert-math-to-llvm -convert-vector-to-llvm -convert-memref-to-llvm -llvm-request-c-wrappers -convert-func-to-llvm -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DIPDialect/dip.o
DEPENDS buddy-opt)

add_library(DIP STATIC dip.o)

SET_TARGET_PROPERTIES(
DIP
PROPERTIES
LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(correlation2D correlation2D.cpp)
add_dependencies(correlation2D buddy-opt)
target_link_libraries(correlation2D ${OpenCV_LIBS} DIP)

add_executable(rotation2D rotation2D.cpp)
add_dependencies(rotation2D buddy-opt)
target_link_libraries(rotation2D ${OpenCV_LIBS} DIP)

add_executable(resize2D resize2D.cpp)
add_dependencies(resize2D buddy-opt)
target_link_libraries(resize2D ${OpenCV_LIBS} DIP)
