if (${BUDDY_DAP_OPT_VECTOR_SPLITTING})
  set(SPLITING_SIZE ${BUDDY_DAP_OPT_VECTOR_SPLITTING})
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

#-------------------------------------------------------------------------------
# Buddy DAP Dialect FIR operation
#-------------------------------------------------------------------------------

add_executable(buddy-fir-scalar FIRLowpass.cpp)
add_dependencies(buddy-fir-scalar buddy-opt)
target_link_libraries(buddy-fir-scalar
  BuddyLibDAP
  mlir_c_runner_utils
)

add_executable(buddy-fir-vectorization FIRVectorization.cpp)
add_dependencies(buddy-fir-vectorization buddy-opt)
target_link_libraries(buddy-fir-vectorization
  BuddyLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect Biquad Operation
#-------------------------------------------------------------------------------

add_executable(buddy-biquad biquad.cpp)
add_dependencies(buddy-biquad buddy-opt)
target_link_libraries(buddy-biquad
  BuddyLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect IIR Operation
#-------------------------------------------------------------------------------

add_executable(buddy-iir-scalar IIRLowpass.cpp)
add_dependencies(buddy-iir-scalar buddy-opt)
target_link_libraries(buddy-iir-scalar 
  BuddyLibDAP
  mlir_c_runner_utils
)

add_executable(buddy-iir-vectorization IIRVectorization.cpp)
add_dependencies(buddy-iir-vectorization buddy-opt)
target_link_libraries(buddy-iir-vectorization
  BuddyLibDAP
  mlir_c_runner_utils
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect WhisperPreprocess Operation
#-------------------------------------------------------------------------------

add_executable(buddy-whisper-preprocess WhisperPreprocess.cpp)
add_dependencies(buddy-whisper-preprocess buddy-opt)
target_link_libraries(buddy-whisper-preprocess
  BuddyLibDAP
  mlir_c_runner_utils
)

add_executable(buddy-rfft RFFT.cpp)
add_dependencies(buddy-rfft buddy-opt)
target_link_libraries(buddy-rfft
  BuddyLibDAP
  mlir_c_runner_utils
)
