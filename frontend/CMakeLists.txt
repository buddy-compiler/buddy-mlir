if(FeGen)
  add_subdirectory(FrontendGen)
endif()

add_subdirectory(Interfaces)

if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  add_subdirectory(Python)
endif()
