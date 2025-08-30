add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
  COMPONENTS Interpreter Development NumPy REQUIRED)
  find_package(Python ${LLVM_MINIMUM_PYTHON_VERSION}
  COMPONENTS Interpreter Development NumPy REQUIRED)
  mlir_detect_pybind11_install()
  find_package(pybind11 2.10 CONFIG REQUIRED)
  mlir_detect_nanobind_install()
  find_package(nanobind 2.4 CONFIG REQUIRED)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
