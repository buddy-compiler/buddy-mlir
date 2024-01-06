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

add_executable(buddy-fir FIRLowpass.cpp)
add_dependencies(buddy-fir buddy-opt)
target_link_libraries(buddy-fir
  BuddyLibDAP
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect Biquad Operation
#-------------------------------------------------------------------------------

add_executable(buddy-biquad biquad.cpp)
add_dependencies(buddy-biquad buddy-opt)
target_link_libraries(buddy-biquad
  BuddyLibDAP
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect IIR Operation
#-------------------------------------------------------------------------------

add_executable(buddy-iir-scalar IIRLowpass.cpp)
add_dependencies(buddy-iir-scalar buddy-opt)
target_link_libraries(buddy-iir-scalar 
  BuddyLibDAP
)

add_executable(buddy-iir-vectorization IIRVectorization.cpp)
add_dependencies(buddy-iir-vectorization buddy-opt)
target_link_libraries(buddy-iir-vectorization
  BuddyLibDAPVectorization
)
