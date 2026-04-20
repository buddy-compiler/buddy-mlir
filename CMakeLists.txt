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
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()
#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(buddy-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
include(ExternalProject)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)
message(STATUS "buddy-mlir install-tree build.")

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
foreach(_buddy_lit_tool IN ITEMS FileCheck count not)
  if(NOT TARGET ${_buddy_lit_tool})
    add_custom_target(${_buddy_lit_tool})
  endif()
endforeach()

if(NOT LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT lit)
endif()

list(GET LLVM_LIBRARY_DIRS 0 BUDDY_LLVM_LIBRARY_DIR)
set(BUDDY_OPENMP_RUNTIME_LIBRARY
  "${BUDDY_LLVM_LIBRARY_DIR}/${LLVM_HOST_TRIPLE}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}")

#-------------------------------------------------------------------------------
# BUDDY configuration
#-------------------------------------------------------------------------------

# BUDDY project.
set(BUDDY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_BUILD_DIR ${CMAKE_BINARY_DIR})
set(BUDDY_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)

set(BUDDY_EXAMPLES_DIR ${BUDDY_SOURCE_DIR}/examples)
set(BUDDY_MLIR_INTERFACE_DIR ${BUDDY_SOURCE_DIR}/frontend/Interfaces)
set(BUDDY_MIDEND_INCLUDE_DIR ${BUDDY_SOURCE_DIR}/midend/include)
set(BUDDY_THIRDPARTY_INCLUDE_DIR ${BUDDY_SOURCE_DIR}/thirdparty/include)
set(BUDDY_MLIR_PYTHON_PACKAGES_DIR ${BUDDY_BUILD_DIR}/python_packages)

set(BUDDY_EXAMPLES OFF CACHE BOOL "Build examples")
set(BUDDY_TORQ_EXAMPLES OFF CACHE BOOL
    "Build BuddyTorq example (PyTorch->MLIR + TORQ-Tile RVV GEMM); requires RISC-V RVV toolchain")
set(BUDDY_RUNTIME_MATMUL_TORQ OFF CACHE BOOL
    "Build buddy_matmul_torq (runtime buddy_matmul_f32 via TORQ-Tile RVV); requires RISC-V RVV and find_package(TorqTile)")
set(BUDDY_TRANSFORMER_EXAMPLES OFF CACHE BOOL "Build transformer examples")
set(BUDDY_ENABLE_OPENCV OFF CACHE BOOL "Enable OpenCV support.")

# DeepSeek R1 (buddy-codegen): heavy codegen, MLIR import, .so / .rax. Off by default
# so plain `ninja` does not build model artifacts; enable when using
# tools/buddy-codegen/build_model.py or -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON.
option(BUDDY_BUILD_DEEPSEEK_R1_MODEL
  "Build models/deepseek_r1 (buddy_models_deepseek_r1, .so, .rax) and link into buddy-cli"
  OFF)

set(_buddy_require_jpeg OFF)
set(_buddy_require_png OFF)

if(BUDDY_ENABLE_OPENCV)
  add_definitions(-DBUDDY_ENABLE_OPENCV)
  set(_buddy_require_jpeg ON)
  set(_buddy_require_png ON)
endif()

if(BUDDY_MLIR_ENABLE_DIP_LIB)
  add_definitions(-DBUDDY_MLIR_ENABLE_DIP_LIB)
  set(_buddy_require_jpeg ON)
  set(_buddy_require_png ON)
endif()

if(BUDDY_ENABLE_PNG)
  add_definitions(-DBUDDY_ENABLE_PNG)
  set(_buddy_require_png ON)
endif()

if(_buddy_require_jpeg)
  find_package(JPEG REQUIRED)
endif()

if(_buddy_require_png)
  find_package(PNG REQUIRED)
endif()

if(BUDDY_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Add BUDDY files to the include path
include_directories(${BUDDY_MIDEND_INCLUDE_DIR})
include_directories(${BUDDY_MIDEND_INCLUDE_DIR}/Interface)
include_directories(${BUDDY_MIDEND_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/midend/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BUDDY_SOURCE_DIR}/lib)
include_directories(${BUDDY_THIRDPARTY_INCLUDE_DIR})
include_directories(${BUDDY_SOURCE_DIR}/frontend/Interfaces)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if((DEFINED LLVM_ENABLE_ASSERTIONS AND NOT LLVM_ENABLE_ASSERTIONS) OR
   (DEFINED ENABLE_ASSERTIONS AND NOT ENABLE_ASSERTIONS))
  add_compile_definitions(NDEBUG)
endif()

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
if(BUDDY_DSL_EXAMPLES)
  message(WARNING
    "BUDDY_DSL_EXAMPLES depends on MLIR Toy example sources and is no longer "
    "maintained by buddy-mlir. ToyDSL examples will be disabled.")
endif()
set(BUDDY_DSL_EXAMPLES OFF CACHE BOOL "Build DSL examples" FORCE)

#-------------------------------------------------------------------------------
# The `mimalloc` Configuration
#-------------------------------------------------------------------------------

if(BUDDY_MLIR_USE_MIMALLOC)
  if(MIMALLOC_BUILD_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${MIMALLOC_BUILD_DIR})
  endif()
  find_package(mimalloc REQUIRED)
endif()

#-------------------------------------------------------------------------------
# flatc — FlatBuffers schema compiler
# Required for generating buddy/runtime/rax/RAX.h from runtime/rax/rax.fbs.
#-------------------------------------------------------------------------------

find_program(FLATC_EXECUTABLE flatc
  HINTS /usr/bin /usr/local/bin
  DOC "FlatBuffers schema compiler (flatc)"
)
if(NOT FLATC_EXECUTABLE)
  message(FATAL_ERROR
    "flatc not found. Install it with:\n"
    "  sudo apt install flatbuffers-compiler\n"
    "or set -DFLATC_EXECUTABLE=/path/to/flatc")
endif()
message(STATUS "Found flatc: ${FLATC_EXECUTABLE}")

#-------------------------------------------------------------------------------
# The RISC-V toolchain
#-------------------------------------------------------------------------------

if(BUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN)
  set(RISCV_GNU_TOOLCHAIN_DIR "${BUDDY_SOURCE_DIR}/thirdparty/riscv-gnu-toolchain")
  set(RISCV_GNU_TOOLCHAIN_INSTALL_DIR "${CMAKE_BINARY_DIR}/thirdparty/riscv-gnu-toolchain")
  ExternalProject_Add(
    riscv-gnu-toolchain
    SOURCE_DIR        ${RISCV_GNU_TOOLCHAIN_DIR}
    PREFIX            ${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    CONFIGURE_COMMAND ${RISCV_GNU_TOOLCHAIN_DIR}/configure --prefix=${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    BUILD_COMMAND     make clean && make linux build-qemu -j
    BUILD_IN_SOURCE   TRUE
    INSTALL_COMMAND   ""
  )
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(cmake)
add_subdirectory(runtime)
# Before models/: codegen (buddy_add_model) needs target python-package-buddy and
# ${CMAKE_BINARY_DIR}/python_packages for import_model.py (buddy.compiler.*).
add_subdirectory(frontend)
add_subdirectory(models)
add_subdirectory(midend)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(tests)
#-------------------------------------------------------------------------------
# Target check-buddy
#-------------------------------------------------------------------------------

add_custom_target(check-buddy
  DEPENDS check-examples check-tests
)

#-------------------------------------------------------------------------------
# Target install
#-------------------------------------------------------------------------------

set(BUDDY_MLIR_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/BuddyMLIR")

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuddyMLIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BuddyMLIRConfig.cmake
  INSTALL_DESTINATION ${BUDDY_MLIR_INSTALL_CMAKE_DIR}
)

option(BUDDY_PACKAGE_VERSION "Release VERSION" "0.0.0")
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/BuddyMLIRConfigVersion.cmake
  VERSION ${BUDDY_PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/BuddyMLIRConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/BuddyMLIRConfigVersion.cmake
  DESTINATION ${BUDDY_MLIR_INSTALL_CMAKE_DIR}
)
