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
option(BUDDY_MLIR_OUT_OF_TREE_BUILD "Specifies an out of tree build" OFF)
option(BUDDY_ENABLE_TESTS "Enable buddy-mlir test targets." ON)

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
  find_program(LLVM_EXTERNAL_LIT llvm-lit
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "${LLVM_TOOLS_DIR}"
    NO_DEFAULT_PATH)
endif()
if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT llvm-lit lit)
endif()

list(GET LLVM_LIBRARY_DIRS 0 BUDDY_LLVM_LIBRARY_DIR)
get_filename_component(BUDDY_LLVM_BUILD_DIR "${BUDDY_LLVM_LIBRARY_DIR}" DIRECTORY)
set(BUDDY_OPENMP_RUNTIME_CANDIDATE_DIRS
  "${BUDDY_LLVM_LIBRARY_DIR}"
  "${BUDDY_LLVM_LIBRARY_DIR}/${LLVM_HOST_TRIPLE}"
  "${LLVM_BINARY_DIR}/runtimes/runtimes-bins/openmp/runtime/src"
  "${BUDDY_LLVM_BUILD_DIR}/runtimes/runtimes-bins/openmp/runtime/src"
)
set(BUDDY_OPENMP_RUNTIME_DIR "")
set(BUDDY_OPENMP_RUNTIME_LIBRARY "")
foreach(_buddy_openmp_runtime_dir IN LISTS BUDDY_OPENMP_RUNTIME_CANDIDATE_DIRS)
  set(_buddy_openmp_runtime_library
    "${_buddy_openmp_runtime_dir}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}")
  if(EXISTS "${_buddy_openmp_runtime_library}")
    set(BUDDY_OPENMP_RUNTIME_DIR "${_buddy_openmp_runtime_dir}")
    set(BUDDY_OPENMP_RUNTIME_LIBRARY "${_buddy_openmp_runtime_library}")
    break()
  endif()
endforeach()
if(BUDDY_OPENMP_RUNTIME_LIBRARY)
  message(STATUS "Using OpenMP runtime: ${BUDDY_OPENMP_RUNTIME_LIBRARY}")
else()
  message(WARNING "OpenMP runtime libomp${CMAKE_SHARED_LIBRARY_SUFFIX} was not found in the LLVM build tree.")
endif()

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

option(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES
  "Build buddy.compiler Python package files used by model lowering scripts"
  ON)
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
  "Build models/deepseek_r1 (buddy_models_deepseek_r1, .so, .rax)"
  OFF)
option(BUDDY_BUILD_LLAMA31_TT_MODEL
  "Build models/llama31_tt and link the Tenstorrent Llama 3.1 runner into buddy-cli"
  OFF)
option(BUDDY_BUILD_LLAMA32_TT_MODEL
  "Build the Llama 3.2 TT package target and link the shared Tenstorrent Llama runner into buddy-cli"
  OFF)
option(BUDDY_ENABLE_TENSTORRENT
  "Enable optional Tenstorrent/tt-mlir configuration checks for TTIR->TTNN flows"
  OFF)
set(BUDDY_TT_MLIR_SOURCE_DIR "${BUDDY_SOURCE_DIR}/thirdparty/tt-mlir"
  CACHE PATH "Optional tt-mlir source tree for Tenstorrent flows")
set(BUDDY_TT_MLIR_BUILD_DIR ""
  CACHE PATH "Optional tt-mlir build directory containing ttmlir tools")
set(BUDDY_TT_METAL_HOME ""
  CACHE PATH "Optional tt-metal source/runtime directory used by ttrt")

if(BUDDY_ENABLE_TENSTORRENT)
  if(NOT EXISTS "${BUDDY_TT_MLIR_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "BUDDY_ENABLE_TENSTORRENT=ON requires tt-mlir sources. Run "
      "`git submodule update --init thirdparty/tt-mlir` or set "
      "BUDDY_TT_MLIR_SOURCE_DIR.")
  endif()

  find_program(BUDDY_TTMLIR_OPT ttmlir-opt
    HINTS "${BUDDY_TT_MLIR_BUILD_DIR}/bin")
  find_program(BUDDY_TTMLIR_TRANSLATE ttmlir-translate
    HINTS "${BUDDY_TT_MLIR_BUILD_DIR}/bin")
  if(NOT BUDDY_TTMLIR_OPT OR NOT BUDDY_TTMLIR_TRANSLATE)
    message(FATAL_ERROR
      "BUDDY_ENABLE_TENSTORRENT=ON requires ttmlir-opt and "
      "ttmlir-translate in PATH or BUDDY_TT_MLIR_BUILD_DIR/bin.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import ttrt, ttrt.runtime"
    RESULT_VARIABLE BUDDY_TTRT_IMPORT_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)
  if(NOT BUDDY_TTRT_IMPORT_RESULT EQUAL 0)
    message(WARNING
      "Python package ttrt was not found in "
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}. Native buddy-cli inference "
      "does not import ttrt, but Tenstorrent package-generation targets still "
      "require the tt-mlir Python environment.")
  endif()

  message(STATUS "Tenstorrent support enabled")
  message(STATUS "  tt-mlir source: ${BUDDY_TT_MLIR_SOURCE_DIR}")
  message(STATUS "  ttmlir-opt: ${BUDDY_TTMLIR_OPT}")
  message(STATUS "  ttmlir-translate: ${BUDDY_TTMLIR_TRANSLATE}")
  if(BUDDY_TT_METAL_HOME)
    message(STATUS "  tt-metal home: ${BUDDY_TT_METAL_HOME}")
  endif()
endif()

# Whisper (buddy-codegen, encoder-decoder ASR): import → .so / .rax + runner
# plugin loaded by buddy-cli. Off by default; enable with
# -DBUDDY_BUILD_WHISPER_MODEL=ON (requires BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON).
option(BUDDY_BUILD_WHISPER_MODEL
  "Build models/whisper (whisper_model.so, whisper.rax, whisper_runner.so)"
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
#
# When Buddy is configured against a tt-mlir toolchain, prefer that toolchain's
# flatc and FlatBuffers headers. Older system flatc versions can generate C++
# that is incompatible with the newer FlatBuffers headers pulled in through
# MLIR/tt-mlir include paths.
#-------------------------------------------------------------------------------

set(BUDDY_FLATBUFFERS_BIN_HINTS)
set(BUDDY_FLATBUFFERS_INCLUDE_HINTS)
if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  list(APPEND BUDDY_FLATBUFFERS_BIN_HINTS "$ENV{TTMLIR_TOOLCHAIN_DIR}/bin")
  list(APPEND BUDDY_FLATBUFFERS_INCLUDE_HINTS "$ENV{TTMLIR_TOOLCHAIN_DIR}/include")
endif()

find_program(FLATC_EXECUTABLE flatc
  HINTS ${BUDDY_FLATBUFFERS_BIN_HINTS} /usr/bin /usr/local/bin
  DOC "FlatBuffers schema compiler (flatc)"
)
if(NOT FLATC_EXECUTABLE)
  message(FATAL_ERROR
    "flatc not found. Install it with:\n"
    "  sudo apt install flatbuffers-compiler\n"
    "or set -DFLATC_EXECUTABLE=/path/to/flatc")
endif()
message(STATUS "Found flatc: ${FLATC_EXECUTABLE}")

if(FLATC_EXECUTABLE)
  get_filename_component(BUDDY_FLATC_BIN_DIR "${FLATC_EXECUTABLE}" DIRECTORY)
  get_filename_component(BUDDY_FLATC_PREFIX_DIR "${BUDDY_FLATC_BIN_DIR}" DIRECTORY)
  list(APPEND BUDDY_FLATBUFFERS_INCLUDE_HINTS
    "${BUDDY_FLATC_PREFIX_DIR}/include")
endif()

#-------------------------------------------------------------------------------
# The RISC-V toolchain
#-------------------------------------------------------------------------------

if(BUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN)
  find_package(Git REQUIRED)
  set(RISCV_GNU_TOOLCHAIN_DIR "${BUDDY_SOURCE_DIR}/thirdparty/riscv-gnu-toolchain")
  set(RISCV_GNU_TOOLCHAIN_INSTALL_DIR "${CMAKE_BINARY_DIR}/thirdparty/riscv-gnu-toolchain")
  ExternalProject_Add(
    riscv-gnu-toolchain
    SOURCE_DIR        ${RISCV_GNU_TOOLCHAIN_DIR}
    PREFIX            ${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    DOWNLOAD_COMMAND  ${GIT_EXECUTABLE} -C ${BUDDY_SOURCE_DIR}
                      submodule update --init -- thirdparty/riscv-gnu-toolchain
    CONFIGURE_COMMAND ${RISCV_GNU_TOOLCHAIN_DIR}/configure --prefix=${RISCV_GNU_TOOLCHAIN_INSTALL_DIR}
    BUILD_COMMAND     make clean && make linux build-qemu -j
    BUILD_IN_SOURCE   TRUE
    INSTALL_COMMAND   ""
  )
endif()

#-------------------------------------------------------------------------------
# External Dialects Configuration
#-------------------------------------------------------------------------------

set(BUDDY_EXTERNAL_DIALECTS_DIR "" CACHE PATH "Path to external dialects directory")
set(BUDDY_ENABLE_EXTERNAL_DIALECTS OFF)

if(BUDDY_EXTERNAL_DIALECTS_DIR)
  if(NOT EXISTS "${BUDDY_EXTERNAL_DIALECTS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "BUDDY_EXTERNAL_DIALECTS_DIR is set but "
      "${BUDDY_EXTERNAL_DIALECTS_DIR}/CMakeLists.txt does not exist")
  endif()

  set(BUDDY_ENABLE_EXTERNAL_DIALECTS ON)
  message(STATUS "Including external dialects from: ${BUDDY_EXTERNAL_DIALECTS_DIR}")
  include_directories(${BUDDY_EXTERNAL_DIALECTS_DIR})
  include_directories(${CMAKE_BINARY_DIR}/external_dialects)
  add_compile_definitions(BUDDY_EXTERNAL_DIALECTS)
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

# Add external dialects before midend so they can be referenced
if(BUDDY_ENABLE_EXTERNAL_DIALECTS)
  add_subdirectory(${BUDDY_EXTERNAL_DIALECTS_DIR} ${CMAKE_BINARY_DIR}/external_dialects)
endif()

add_subdirectory(midend)
add_subdirectory(tools)
get_property(_buddy_llama_tt_rax_targets_set GLOBAL PROPERTY
             BUDDY_LLAMA_TT_RAX_TARGETS SET)
if(_buddy_llama_tt_rax_targets_set)
  get_property(_buddy_llama_tt_rax_targets GLOBAL PROPERTY
               BUDDY_LLAMA_TT_RAX_TARGETS)
  list(REMOVE_DUPLICATES _buddy_llama_tt_rax_targets)
  foreach(_buddy_llama_tt_rax IN LISTS _buddy_llama_tt_rax_targets)
    if(TARGET ${_buddy_llama_tt_rax})
      add_dependencies(${_buddy_llama_tt_rax} buddy-cli rax-pack)
      if(TARGET python-package-buddy)
        add_dependencies(${_buddy_llama_tt_rax} python-package-buddy)
      endif()
      if(TARGET python-package-buddy-mlir)
        add_dependencies(${_buddy_llama_tt_rax} python-package-buddy-mlir)
      endif()
    endif()
  endforeach()
endif()
add_subdirectory(examples)
if(BUDDY_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
#-------------------------------------------------------------------------------
# Target check-buddy
#-------------------------------------------------------------------------------

if(BUDDY_ENABLE_TESTS)
  add_custom_target(check-buddy
    DEPENDS check-examples check-tests
  )
endif()

#-------------------------------------------------------------------------------
# Target install
#-------------------------------------------------------------------------------

set(BUDDY_MLIR_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/BuddyMLIR")

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuddyMLIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/BuddyMLIRConfig.cmake
  INSTALL_DESTINATION ${BUDDY_MLIR_INSTALL_CMAKE_DIR}
)

set(BUDDY_PACKAGE_VERSION "0.0.0" CACHE STRING "Release VERSION")
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
