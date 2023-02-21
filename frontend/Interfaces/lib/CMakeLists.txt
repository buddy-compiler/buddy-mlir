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

add_custom_command(OUTPUT DIP.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/DIP.mlir 
              -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
              -arith-expand
              -lower-affine
              -convert-scf-to-cf
              -convert-math-to-llvm
              -convert-vector-to-llvm
              -convert-memref-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc 
              -mtriple=${BUDDY_TARGET_TRIPLE}
              -mattr=${BUDDY_OPT_ATTR}
              --filetype=obj
              -o ${CMAKE_CURRENT_BINARY_DIR}/DIP.o
  DEPENDS buddy-opt
  )

add_library(BuddyLibDIP STATIC DIP.o)

SET_TARGET_PROPERTIES(BuddyLibDIP PROPERTIES
  LINKER_LANGUAGE C
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )
