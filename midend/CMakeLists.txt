add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_detect_pybind11_install()
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
  COMPONENTS Interpreter Development NumPy REQUIRED)
  find_package(pybind11 2.6 CONFIG REQUIRED)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()