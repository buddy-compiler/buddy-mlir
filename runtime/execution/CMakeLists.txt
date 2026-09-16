add_library(buddy_runtime_execution STATIC
  RaxExecutionSession.cpp
  RaxExecutor.cpp
)
target_link_libraries(buddy_runtime_execution PUBLIC
  buddy_runtime_core
  ${CMAKE_DL_LIBS}
)
target_compile_features(buddy_runtime_execution PUBLIC cxx_std_17)

install(TARGETS buddy_runtime_execution
        EXPORT BuddyMLIRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT buddy_runtime)

if(BUDDY_RUNTIME_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(buddy_runtime_mpi STATIC
    MpiCommunicator.cpp
  )
  target_link_libraries(buddy_runtime_mpi PUBLIC
    buddy_runtime_core
    MPI::MPI_CXX
  )
  target_compile_features(buddy_runtime_mpi PUBLIC cxx_std_17)

  install(TARGETS buddy_runtime_mpi
          EXPORT BuddyMLIRTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT buddy_runtime)
endif()
