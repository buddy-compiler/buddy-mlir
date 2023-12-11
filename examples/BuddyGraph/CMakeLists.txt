cmake_minimum_required(VERSION 3.22)
project(examples LANGUAGES C CXX)

function(mlir_detect_pybind11_install)
  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(STATUS "not found (install via 'pip install pybind11' or set pybind11_DIR)")
      return()
    endif()
    message(STATUS "found (${PACKAGE_DIR})")
    set(pybind11_DIR "${PACKAGE_DIR}" PARENT_SCOPE)
  endif()
endfunction()
find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION}
    COMPONENTS Interpreter ${_python_development_component} NumPy REQUIRED)
mlir_detect_pybind11_install()
find_package(pybind11 2.9 CONFIG REQUIRED)
pybind11_add_module(example main.cc)
add_library(test STATIC test.o)

SET_TARGET_PROPERTIES(
  test
  PROPERTIES
  LINKER_LANGUAGE C)

target_link_directories(example PRIVATE /root/buddy-mlir/llvm/build/lib)

set(BUDDY_LLAMA_LIBS
  test
  mlir_c_runner_utils
)

target_link_libraries(example PRIVATE ${BUDDY_LLAMA_LIBS})