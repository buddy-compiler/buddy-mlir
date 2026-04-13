add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
