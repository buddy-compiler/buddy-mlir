##===- CMakeLists.txt - buddy-mlir cmake root -----------------*- cmake -*-===//
##
## Configure the buddy-mlir build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(buddy-mlir LANGUAGES CXX C)
  
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#-------------------------------------------------------------------------------
# BUDDY configuration
#-------------------------------------------------------------------------------

# BUDDY project.
set(BUDDY_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set(BUDDY_MAIN_INCLUDE_DIR ${BUDDY_MAIN_SRC_DIR}/include)

set(BUDDY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BUDDY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(BUDDY_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add BUDDY files to the include path
include_directories(${BUDDY_MAIN_INCLUDE_DIR})
include_directories(${BUDDY_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(tools)
