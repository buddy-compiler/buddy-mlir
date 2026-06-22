add_subdirectory(include)
add_subdirectory(lib)

if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
  include(MLIRDetectPythonEnv)

  mlir_configure_python_dev_packages()
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
