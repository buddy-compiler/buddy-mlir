add_custom_command(OUTPUT corr2d.o
COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/DIPDialect/corr2d.mlir -lower-dip="DIP-strip-mining=${BUDDY_DIP_OPT_STRIP_MINING}" -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/DIPDialect/corr2d.o
DEPENDS buddy-opt)

add_library(Corr2D STATIC corr2d.o)

SET_TARGET_PROPERTIES(
Corr2D
PROPERTIES
LINKER_LANGUAGE C)

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(correlation2D correlation2D.cpp)
add_dependencies(correlation2D buddy-opt)
target_link_libraries(correlation2D ${OpenCV_LIBS} Corr2D)
