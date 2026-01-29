add_subdirectory(include)
add_subdirectory(lib)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)

  # We support multiple Python versions
  # Development.Module asks CMake to locate the build artifacts needed to
  # compile extension modules (headers + linkable module libs). See:
  # https://cmake.org/cmake/help/latest/module/FindPython3.html
  find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
  COMPONENTS Interpreter Development.Module NumPy REQUIRED)
  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  find_package(Python
  COMPONENTS Interpreter Development.Module NumPy REQUIRED)
  mlir_detect_pybind11_install()
  find_package(pybind11 2.10 CONFIG REQUIRED)
  mlir_detect_nanobind_install()
  find_package(nanobind 2.4 CONFIG REQUIRED)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
