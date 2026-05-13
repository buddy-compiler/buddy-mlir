# ===- CMakeLists.txt - Llama 3.1 TT runner ------------------*- cmake -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# Llama 3.1 Tenstorrent runner.
#
# This target links the tt-mlir C++ runtime directly. Python is still used by
# the package target to capture/lower the model and prepare artifacts, but
# buddy-cli inference itself stays in-process in native C++.

add_library(buddy_models_llama31_tt
  Llama31TTRunner.cpp
)

target_compile_features(buddy_models_llama31_tt PRIVATE cxx_std_20)

target_include_directories(buddy_models_llama31_tt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(buddy_models_llama31_tt
  PUBLIC
    buddy_runtime_core
    buddy_runtime_llm
  PRIVATE
    LLVMSupport
)

if(NOT BUDDY_ENABLE_TENSTORRENT)
  message(FATAL_ERROR
    "BUDDY_BUILD_LLAMA31_TT_MODEL=ON requires BUDDY_ENABLE_TENSTORRENT=ON")
endif()

find_library(BUDDY_TTMLIR_RUNTIME_LIB
  NAMES TTMLIRRuntime libTTMLIRRuntime.so
  HINTS
    "${BUDDY_TT_MLIR_BUILD_DIR}/runtime/lib"
    "${BUDDY_TT_MLIR_BUILD_DIR}/python_packages/ttrt/runtime"
  NO_DEFAULT_PATH)
if(NOT BUDDY_TTMLIR_RUNTIME_LIB)
  message(FATAL_ERROR
    "Could not find libTTMLIRRuntime.so. Build tt-mlir with "
    "-DTTMLIR_ENABLE_RUNTIME=ON and set BUDDY_TT_MLIR_BUILD_DIR.")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"
  OUTPUT_VARIABLE _BUDDY_LLAMA31_PYTHON_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY') or '')"
  OUTPUT_VARIABLE _BUDDY_LLAMA31_PYTHON_LDLIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_library(BUDDY_LLAMA31_PYTHON_LIB
  NAMES "${_BUDDY_LLAMA31_PYTHON_LDLIBRARY}" python3.12 python3
  HINTS "${_BUDDY_LLAMA31_PYTHON_LIBDIR}"
  NO_DEFAULT_PATH)
if(NOT BUDDY_LLAMA31_PYTHON_LIB)
  message(FATAL_ERROR
    "Could not find the Python shared library required by libTTMLIRRuntime. "
    "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
endif()

target_include_directories(buddy_models_llama31_tt SYSTEM
  PRIVATE
    "${BUDDY_TT_MLIR_SOURCE_DIR}/runtime/include"
    "${BUDDY_TT_MLIR_SOURCE_DIR}/include"
    "${BUDDY_TT_MLIR_BUILD_DIR}/runtime/include"
    "${BUDDY_TT_MLIR_BUILD_DIR}/include"
)

target_link_libraries(buddy_models_llama31_tt
  PRIVATE
    "${BUDDY_TTMLIR_RUNTIME_LIB}"
    "${BUDDY_LLAMA31_PYTHON_LIB}"
)

get_filename_component(_BUDDY_TTMLIR_RUNTIME_LIB_DIR
  "${BUDDY_TTMLIR_RUNTIME_LIB}" DIRECTORY)
get_filename_component(_BUDDY_LLAMA31_PYTHON_LIB_DIR
  "${BUDDY_LLAMA31_PYTHON_LIB}" DIRECTORY)
set(_BUDDY_TT_METAL_BUILD_LIB_DIR
  "${BUDDY_TT_MLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build_Release/lib")
set_property(TARGET buddy_models_llama31_tt APPEND PROPERTY BUILD_RPATH
  "${_BUDDY_TTMLIR_RUNTIME_LIB_DIR}"
  "${_BUDDY_TT_METAL_BUILD_LIB_DIR}"
  "${_BUDDY_LLAMA31_PYTHON_LIB_DIR}"
  "${BUDDY_TT_MLIR_BUILD_DIR}/python_packages/ttrt/runtime"
)
target_link_options(buddy_models_llama31_tt INTERFACE
  "LINKER:-rpath,${_BUDDY_TTMLIR_RUNTIME_LIB_DIR}"
  "LINKER:-rpath,${_BUDDY_TT_METAL_BUILD_LIB_DIR}"
  "LINKER:-rpath,${_BUDDY_LLAMA31_PYTHON_LIB_DIR}"
  "LINKER:-rpath,${BUDDY_TT_MLIR_BUILD_DIR}/python_packages/ttrt/runtime"
)

set(BUDDY_LLAMA31_MAX_CACHE_LEN "1024" CACHE STRING
  "Static KV cache length used when packaging Llama 3.1 TT artifacts")
set(BUDDY_LLAMA31_MODEL_PATH "" CACHE STRING
  "Optional local Hugging Face Llama 3.1 model path for packaging")
set(BUDDY_LLAMA31_EMBED_RAX_PAYLOAD ON CACHE BOOL
  "Embed Llama TTNN flatbuffers and chat artifacts into llama31_tt.rax")
set(BUDDY_LLAMA31_OFFLINE ON CACHE BOOL
  "Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE while packaging Llama 3.1 TT")

if(DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
  set(_BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DEFAULT "$ENV{TTMLIR_TOOLCHAIN_DIR}")
else()
  set(_BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DEFAULT
    "${CMAKE_SOURCE_DIR}/build-ttmlir-toolchain")
endif()
set(BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DIR
  "${_BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DEFAULT}" CACHE PATH
  "tt-mlir toolchain directory used by the Llama 3.1 TT package target")

set(_BUDDY_LLAMA31_EMBED_RAX_PAYLOAD 0)
if(BUDDY_LLAMA31_EMBED_RAX_PAYLOAD)
  set(_BUDDY_LLAMA31_EMBED_RAX_PAYLOAD 1)
endif()

set(_BUDDY_LLAMA31_OFFLINE 0)
if(BUDDY_LLAMA31_OFFLINE)
  set(_BUDDY_LLAMA31_OFFLINE 1)
endif()

set(_BUDDY_LLAMA31_PACKAGE_ENV
  "BUDDY_BUILD=${CMAKE_BINARY_DIR}"
  "BUDDY_TT_SKIP_ACTIVATE=1"
  "BUDDY_LLAMA31_ARTIFACT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  "TTIR_OUT=${CMAKE_CURRENT_BINARY_DIR}/ttir_out_static"
  "BASE_CHAT_ART=${CMAKE_CURRENT_BINARY_DIR}/chat_artifacts"
  "CHAT_ART=${CMAKE_CURRENT_BINARY_DIR}/chat_artifacts"
  "RAX_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}"
  "RAX_STEM=llama31_tt"
  "TTRT_SYS_DIR=${CMAKE_CURRENT_BINARY_DIR}/ttrt_sys"
  "TTMLIR_SOURCE=${BUDDY_TT_MLIR_SOURCE_DIR}"
  "TTMLIR_BUILD=${BUDDY_TT_MLIR_BUILD_DIR}"
  "TTMLIR_TOOLCHAIN_DIR=${BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DIR}"
  "MAX_CACHE_LEN=${BUDDY_LLAMA31_MAX_CACHE_LEN}"
  "PACKAGE_ONLY=1"
  "RUN_WITH_BUDDY_CLI=0"
  "MAX_NEW_TOKENS=0"
  "SKIP_LOWER=0"
  "SKIP_PREPARE=0"
  "SKIP_PACKAGE=0"
  "DEVICE_ARGMAX=0"
  "FULL_ARG_ATTRS=1"
  "EMBED_RAX_PAYLOAD=${_BUDDY_LLAMA31_EMBED_RAX_PAYLOAD}"
  "HF_HUB_OFFLINE=${_BUDDY_LLAMA31_OFFLINE}"
  "TRANSFORMERS_OFFLINE=${_BUDDY_LLAMA31_OFFLINE}")

if(BUDDY_LLAMA31_MODEL_PATH)
  list(APPEND _BUDDY_LLAMA31_PACKAGE_ENV
    "LLAMA31_MODEL_PATH=${BUDDY_LLAMA31_MODEL_PATH}")
endif()

set(BUDDY_LLAMA31_RAX "${CMAKE_CURRENT_BINARY_DIR}/llama31_tt.rax")
set(BUDDY_LLAMA31_RHAL "${CMAKE_CURRENT_BINARY_DIR}/llama31_tt.rhal.mlir")

add_custom_command(
  OUTPUT "${BUDDY_LLAMA31_RAX}"
  BYPRODUCTS "${BUDDY_LLAMA31_RHAL}"
  COMMAND ${CMAKE_COMMAND} -E env ${_BUDDY_LLAMA31_PACKAGE_ENV}
          bash "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_p150_chat.sh"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/_env.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy-llama31-lower-ttir.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama31_chat_prepare.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_p150_chat.sh"
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
  COMMENT "Generating self-contained llama31_tt.rax"
  USES_TERMINAL
  VERBATIM
)

add_custom_target(llama31_tt_rax
  DEPENDS "${BUDDY_LLAMA31_RAX}"
  COMMENT "llama31_tt.rax -> ${BUDDY_LLAMA31_RAX}"
)
if(TARGET python-package-buddy)
  add_dependencies(llama31_tt_rax python-package-buddy)
endif()

add_custom_target(llama31_tt_package
  COMMENT "Alias for llama31_tt_rax"
)
add_dependencies(llama31_tt_package llama31_tt_rax)
