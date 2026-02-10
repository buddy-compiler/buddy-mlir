add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  # Set Python bindings domain to avoid conflicts with other MLIR projects
  if(NOT MLIR_BINDINGS_PYTHON_NB_DOMAIN)
    set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "buddy_mlir" CACHE STRING "" FORCE)
  endif()
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
