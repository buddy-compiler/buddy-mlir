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

add_executable(firLowpass firLowpass.cpp)
add_dependencies(firLowpass buddy-opt)
target_link_libraries(firLowpass
  BuddyLibDAP
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect Biquad Operation
#-------------------------------------------------------------------------------

add_executable(biquad biquad.cpp)
add_dependencies(biquad buddy-opt)
target_link_libraries(biquad
  BuddyLibDAP
)

#-------------------------------------------------------------------------------
# Buddy DAP Dialect IIR Operation
#-------------------------------------------------------------------------------

add_executable(iirLowpass iirLowpass.cpp)
add_dependencies(iirLowpass buddy-opt)
target_link_libraries(iirLowpass 
  BuddyLibDAP
)

add_executable(iirVectorization iirVectorization.cpp)
add_dependencies(iirVectorization buddy-opt)
target_link_libraries(iirVectorization
  BuddyLibDAPVectorization
)
