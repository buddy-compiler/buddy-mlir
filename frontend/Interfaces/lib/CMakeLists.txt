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

add_custom_command(OUTPUT DIP.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/DIP.mlir
        -lower-dip="DIP-strip-mining=${SPLITING_SIZE}"
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
        -o ${CMAKE_CURRENT_BINARY_DIR}/DIP.o
        DEPENDS buddy-opt
        )

add_library(BuddyLibDIP STATIC DIP.o)

SET_TARGET_PROPERTIES(BuddyLibDIP PROPERTIES
  LINKER_LANGUAGE C
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )

add_custom_command(OUTPUT DAP.o
  COMMAND ${CMAKE_BINARY_DIR}/bin/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/DAP.mlir 
              -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}"
              --convert-linalg-to-affine-loops
              -arith-expand
              -lower-affine
              -convert-scf-to-cf
              -convert-math-to-llvm
              -convert-vector-to-llvm
              -finalize-memref-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc 
              -mtriple=${BUDDY_TARGET_TRIPLE}
              -mattr=${BUDDY_OPT_ATTR}
              --filetype=obj
              -o ${CMAKE_CURRENT_BINARY_DIR}/DAP.o
  DEPENDS buddy-opt
  )

add_library(BuddyLibDAP STATIC DAP.o)

SET_TARGET_PROPERTIES(BuddyLibDAP PROPERTIES
  LINKER_LANGUAGE CXX
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )

  add_custom_command(OUTPUT DAPVectorization.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/DAP.mlir |
          sed 's/buddy_fir/buddy_fir_vectorization/' |
          sed 's/buddy_iir/buddy_iir_vectorization/' |
          sed 's/buddy_biquad/buddy_biquad_vectorization/' |
          ${CMAKE_BINARY_DIR}/bin/buddy-opt
              -vectorize-dap
              -convert-linalg-to-affine-loops
              -arith-expand
              -lower-affine
              -convert-scf-to-cf
              -convert-math-to-llvm
              -convert-vector-to-llvm
              -finalize-memref-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc 
              -mtriple=${BUDDY_TARGET_TRIPLE}
              -mattr=${BUDDY_OPT_ATTR}
              -filetype=obj
              -o ${CMAKE_CURRENT_BINARY_DIR}/DAPVectorization.o
  DEPENDS buddy-opt
  )

add_library(BuddyLibDAPVectorization STATIC DAPVectorization.o)

SET_TARGET_PROPERTIES(BuddyLibDAPVectorization PROPERTIES
  LINKER_LANGUAGE CXX
  ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
  )
