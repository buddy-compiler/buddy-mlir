##===- CMakeLists.txt - buddy-mlir cmake root -----------------*- cmake -*-===//
##
## Configure the buddy-mlir build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(buddy-mlir LANGUAGES CXX C)
  
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)
set(LLVM_MLIR_SOURCE_DIR ${MLIR_DIR}/../../../../mlir)
set(LLVM_PROJECT_SOURCE_DIR ${MLIR_DIR}/../../../../)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# BUDDY configuration
#-------------------------------------------------------------------------------

# BUDDY project.
set(BUDDY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(BUDDY_EXAMPLES_DIR ${BUDDY_SOURCE_DIR}/examples)
set(BUDDY_MIDEND_INCLUDE_DIR ${BUDDY_SOURCE_DIR}/midend/include/)
set(BUDDY_THIRDPARTY_INCLUDE_DIR ${BUDDY_SOURCE_DIR}/thirdparty/include/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUDDY_BINARY_DIR})

set(BUDDY_EXAMPLES OFF CACHE BOOL "Build examples")
set(BUDDY_ENABLE_OPENCV OFF CACHE BOOL "Enable OpenCV support.")

# Generate libraries into `lib` of build directory.
set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Add BUDDY files to the include path
include_directories(${BUDDY_MAIN_INCLUDE_DIR})
include_directories(${BUDDY_MIDEND_INCLUDE_DIR})
include_directories(${BUDDY_MIDEND_INCLUDE_DIR}/Interface)
include_directories(${BUDDY_MIDEND_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/midend/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/backend/include)
include_directories(${BUDDY_SOURCE_DIR}/lib)
include_directories(${BUDDY_THIRDPARTY_INCLUDE_DIR})
include_directories(${BUDDY_SOURCE_DIR}/frontend/Interfaces)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Hardware detection
#-------------------------------------------------------------------------------

include(${BUDDY_SOURCE_DIR}/cmake/check_simd.cmake)
include(${BUDDY_SOURCE_DIR}/cmake/check_toolchain.cmake)
check_simd()
check_toolchain()

#-------------------------------------------------------------------------------
# Antlr Configuration
#-------------------------------------------------------------------------------
# NB: currently, ANTLR is used in dsl examples only,
# however, there is a plan to use in the frontend,
# so it is kept in the top-level cmake
if(BUDDY_DSL_EXAMPLES)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Antlr)

    # required if linking to static library
    add_definitions(-DANTLR4CPP_STATIC)

    # add external build for antlrcpp
    include(ExternalAntlr4Cpp)
    # add antrl4cpp artifacts to project environment
    include_directories(${ANTLR4_INCLUDE_DIRS})

    # set variable pointing to the antlr tool that supports C++
    # this is not required if the jar file can be found under PATH environment
    set(ANTLR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/antlr/antlr-4.10.1-complete.jar)
    # add macros to generate ANTLR Cpp code from grammar
    find_package(ANTLR REQUIRED)
endif()

#-------------------------------------------------------------------------------
# OpenMP setup
#-------------------------------------------------------------------------------

find_library(OPENMP_LIB omp PATHS ${LLVM_MLIR_BINARY_DIR}/../lib)
if(OPENMP_LIB)
    message("-- OpenMP FOUND")
    set(CMAKE_CXX_COMPILER ${LLVM_MLIR_BINARY_DIR}/clang++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
    link_libraries(${OPENMP_LIB})
else()
    message("-- OpenMP NOT FOUND")
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(cmake)
add_subdirectory(frontend)
add_subdirectory(midend)
add_subdirectory(backend)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(tests)
