if (${BUDDY_DIP_OPT_STRIP_MINING})
    set(SPLITING_SIZE ${BUDDY_DIP_OPT_STRIP_MINING})
elseif (HAVE_AVX512)
    set(SPLITING_SIZE 64)
elseif (HAVE_AVX2)
    set(SPLITING_SIZE 16)
elseif (HAVE_SSE)
    set(SPLITING_SIZE 16)
elseif (HAVE_NEON)
    set(SPLITING_SIZE 16)
endif ()

add_custom_command(OUTPUT resize4d.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/buddy-opt ${BUDDY_EXAMPLES_DIR}/Resize4D/resize4d_test1.mlir
        # -lower-dip="DIP-strip-mining=${SPLITING_SIZE}"
        -arith-expand
        -lower-affine
        -convert-scf-to-cf
        -convert-math-to-llvm
        -convert-vector-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc
        -mtriple=${BUDDY_TARGET_TRIPLE}
        -mattr=${BUDDY_OPT_ATTR}
        --filetype=obj
        -o ${CMAKE_CURRENT_BINARY_DIR}/resize4d.o
        DEPENDS buddy-opt
        )

add_library(Resize4D STATIC resize4d.o)

SET_TARGET_PROPERTIES(Resize4D PROPERTIES
  LINKER_LANGUAGE C
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )
if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_executable(resize4d resize4d.cpp)
add_dependencies(resize4d 
                buddy-opt
                buddy-translate
                )
target_link_libraries(resize4d ${OpenCV_LIBS} Resize4D)
