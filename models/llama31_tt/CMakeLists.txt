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
    "Building Llama TT package targets requires BUDDY_ENABLE_TENSTORRENT=ON")
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
set(_BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH
  "${_BUDDY_LLAMA31_PYTHON_LIB_DIR}")
if(DEFINED ENV{LD_LIBRARY_PATH})
  string(APPEND _BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH ":$ENV{LD_LIBRARY_PATH}")
endif()
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
set(BUDDY_LLAMA31_DECODE_RMSNORM_FUSION ON CACHE BOOL
  "Use TTIR RMSNorm fusion for Llama 3.1 fixed-batch decode packages")
set(BUDDY_LLAMA31_DECODE_FOLD_IDENTITY_MUL ON CACHE BOOL
  "Fold generated Llama 3.1 fixed-batch decode multiply-by-one ops")
set(BUDDY_LLAMA31_DECODE_PACK_QKV ON CACHE BOOL
  "Pre-pack Llama 3.1 fixed-batch decode Q/K/V static weight slots")
set(BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD ON CACHE BOOL
  "Split the Llama 3.1 fixed-batch decode LM-head vocab projection into fixed vocab-row shards")
set(BUDDY_LLAMA31_DECODE_LM_HEAD_SPLITS "8" CACHE STRING
  "Number of vocab-row shards used when BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD is enabled")
set(BUDDY_LLAMA31_DECODE_SPLIT_EMBEDDING_WEIGHT ON CACHE BOOL
  "Duplicate the tied Llama 3.1 fixed-batch decode embedding/LM-head weight as separate TTNN layout slots")
set(BUDDY_LLAMA31_DECODE_KEEP_STATIC_WEIGHT_INPUTS ON CACHE BOOL
  "Remove generated Llama 3.1 fixed-batch decode dealloc ops for static weight inputs")

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

set(_BUDDY_LLAMA31_DECODE_RMSNORM_FUSION 0)
if(BUDDY_LLAMA31_DECODE_RMSNORM_FUSION)
  set(_BUDDY_LLAMA31_DECODE_RMSNORM_FUSION 1)
endif()

set(_BUDDY_LLAMA31_DECODE_FOLD_IDENTITY_MUL 0)
if(BUDDY_LLAMA31_DECODE_FOLD_IDENTITY_MUL)
  set(_BUDDY_LLAMA31_DECODE_FOLD_IDENTITY_MUL 1)
endif()

set(_BUDDY_LLAMA31_DECODE_PACK_QKV 0)
if(BUDDY_LLAMA31_DECODE_PACK_QKV)
  set(_BUDDY_LLAMA31_DECODE_PACK_QKV 1)
endif()

set(_BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD 0)
if(BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD)
  set(_BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD 1)
endif()

set(_BUDDY_LLAMA31_DECODE_SPLIT_EMBEDDING_WEIGHT 0)
if(BUDDY_LLAMA31_DECODE_SPLIT_EMBEDDING_WEIGHT)
  set(_BUDDY_LLAMA31_DECODE_SPLIT_EMBEDDING_WEIGHT 1)
endif()

set(_BUDDY_LLAMA31_DECODE_KEEP_STATIC_WEIGHT_INPUTS 0)
if(BUDDY_LLAMA31_DECODE_KEEP_STATIC_WEIGHT_INPUTS)
  set(_BUDDY_LLAMA31_DECODE_KEEP_STATIC_WEIGHT_INPUTS 1)
endif()

set(_BUDDY_LLAMA31_PACKAGE_ENV
  "BUDDY_BUILD=${CMAKE_BINARY_DIR}"
  "BUDDY_TT_SKIP_ACTIVATE=1"
  "BUDDY_LLAMA31_ARTIFACT_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  "LD_LIBRARY_PATH=${_BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH}"
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
          bash "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/_env.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy-llama31-lower-ttir.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama31_chat_prepare.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
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

function(_buddy_register_llama_tt_rax_target target_name)
  get_property(_buddy_llama_tt_rax_targets_set GLOBAL PROPERTY
               BUDDY_LLAMA_TT_RAX_TARGETS SET)
  if(_buddy_llama_tt_rax_targets_set)
    get_property(_buddy_llama_tt_rax_targets GLOBAL PROPERTY
                 BUDDY_LLAMA_TT_RAX_TARGETS)
  else()
    set(_buddy_llama_tt_rax_targets "")
  endif()
  list(APPEND _buddy_llama_tt_rax_targets "${target_name}")
  list(REMOVE_DUPLICATES _buddy_llama_tt_rax_targets)
  set_property(GLOBAL PROPERTY BUDDY_LLAMA_TT_RAX_TARGETS
               "${_buddy_llama_tt_rax_targets}")
endfunction()

_buddy_register_llama_tt_rax_target(llama31_tt_rax)

set(BUDDY_LLAMA31_FIXED_BATCH_SIZES "32" CACHE STRING
  "Semicolon-separated fixed batch sizes to package for Llama 3.1 TT (batch1 is always emitted as llama31_tt_rax)")

function(_buddy_llama31_add_fixed_batch_package batch_size)
  if(NOT "${batch_size}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "BUDDY_LLAMA31_FIXED_BATCH_SIZES must contain integers; got '${batch_size}'")
  endif()
  if(batch_size LESS 2)
    message(FATAL_ERROR
      "Llama 3.1 batch packages only generate fixed batch sizes >= 2; "
      "batch1 is already provided by llama31_tt_rax")
  endif()

  set(_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT
    "${CMAKE_BINARY_DIR}/models/llama31_tt_b${batch_size}")
  set(_BUDDY_LLAMA31_BATCH_STEM "llama31_tt_b${batch_size}")
  set(_BUDDY_LLAMA31_BATCH_TTIR_STEM "llama31_b${batch_size}")
  set(_BUDDY_LLAMA31_BATCH_DISPLAY_NAME
    "Llama-3.1-8B-Instruct batch${batch_size}")
  set(_BUDDY_LLAMA31_BATCH_EMBED_RAX_PAYLOAD 0)
  if(BUDDY_LLAMA31_EMBED_RAX_PAYLOAD)
    set(_BUDDY_LLAMA31_BATCH_EMBED_RAX_PAYLOAD 1)
  endif()
  set(_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX 0)
  set(_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX_PREFILL 0)
  set(_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX_DECODE 1)
  # Keep attention_mask in the traced graph for every fixed-batch package.
  # This matches the official batch-oriented demo structure more closely and
  # keeps left-padded multi-prompt batches semantically correct.
  set(_BUDDY_LLAMA31_BATCH_RUNTIME_ATTENTION_MASK 1)
  set(_BUDDY_LLAMA31_BATCH_PACKAGE_ENV
    "BUDDY_BUILD=${CMAKE_BINARY_DIR}"
    "BUDDY_TT_SKIP_ACTIVATE=1"
    "BUDDY_LLAMA31_ARTIFACT_ROOT=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}"
    "LD_LIBRARY_PATH=${_BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH}"
    "TTIR_OUT=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/ttir_out_static"
    "BASE_CHAT_ART=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/chat_artifacts"
    "CHAT_ART=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/chat_artifacts"
    "RAX_PACKAGE_DIR=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}"
    "RAX_STEM=${_BUDDY_LLAMA31_BATCH_STEM}"
    "RAX_MODEL_NAME=${_BUDDY_LLAMA31_BATCH_STEM}"
    "TTIR_STEM=${_BUDDY_LLAMA31_BATCH_TTIR_STEM}"
    "PROMPT_FORMAT=chat"
    "PREFILL_KV_OUTPUT_ORDER=key_value"
    "LLAMA_DEFAULT_MODEL=meta-llama/Llama-3.1-8B-Instruct"
    "LLAMA_DISPLAY_NAME=${_BUDDY_LLAMA31_BATCH_DISPLAY_NAME}"
    "TTRT_SYS_DIR=${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/ttrt_sys"
    "TTMLIR_SOURCE=${BUDDY_TT_MLIR_SOURCE_DIR}"
    "TTMLIR_BUILD=${BUDDY_TT_MLIR_BUILD_DIR}"
    "TTMLIR_TOOLCHAIN_DIR=${BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DIR}"
    "MAX_CACHE_LEN=${BUDDY_LLAMA31_MAX_CACHE_LEN}"
    "BATCH_SIZE=${batch_size}"
    "BUDDY_TT_DECODE_RMSNORM_FUSION=${_BUDDY_LLAMA31_DECODE_RMSNORM_FUSION}"
    "BUDDY_TT_DECODE_FOLD_IDENTITY_MUL=${_BUDDY_LLAMA31_DECODE_FOLD_IDENTITY_MUL}"
    "BUDDY_TT_DECODE_PACK_QKV=${_BUDDY_LLAMA31_DECODE_PACK_QKV}"
    "BUDDY_TT_DECODE_SPLIT_LM_HEAD=${_BUDDY_LLAMA31_DECODE_SPLIT_LM_HEAD}"
    "BUDDY_TT_DECODE_LM_HEAD_SPLITS=${BUDDY_LLAMA31_DECODE_LM_HEAD_SPLITS}"
    "BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT=${_BUDDY_LLAMA31_DECODE_SPLIT_EMBEDDING_WEIGHT}"
    "BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS=${_BUDDY_LLAMA31_DECODE_KEEP_STATIC_WEIGHT_INPUTS}"
    "PACKAGE_ONLY=1"
    "RUN_WITH_BUDDY_CLI=0"
    "MAX_NEW_TOKENS=0"
    "DEVICE_ARGMAX=${_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX}"
    "DEVICE_ARGMAX_PREFILL=${_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX_PREFILL}"
    "DEVICE_ARGMAX_DECODE=${_BUDDY_LLAMA31_BATCH_DEVICE_ARGMAX_DECODE}"
    "RUNTIME_ATTENTION_MASK=${_BUDDY_LLAMA31_BATCH_RUNTIME_ATTENTION_MASK}"
    "SKIP_LOWER=0"
    "SKIP_PREPARE=0"
    "SKIP_PACKAGE=0"
    "FULL_ARG_ATTRS=1"
    "EMBED_RAX_PAYLOAD=${_BUDDY_LLAMA31_BATCH_EMBED_RAX_PAYLOAD}"
    "HF_HUB_OFFLINE=${_BUDDY_LLAMA31_OFFLINE}"
    "TRANSFORMERS_OFFLINE=${_BUDDY_LLAMA31_OFFLINE}")

  if(BUDDY_LLAMA31_MODEL_PATH)
    list(APPEND _BUDDY_LLAMA31_BATCH_PACKAGE_ENV
      "LLAMA_MODEL_PATH=${BUDDY_LLAMA31_MODEL_PATH}"
      "LLAMA31_MODEL_PATH=${BUDDY_LLAMA31_MODEL_PATH}")
  endif()

  set(_BUDDY_LLAMA31_BATCH_RAX
    "${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/${_BUDDY_LLAMA31_BATCH_STEM}.rax")
  set(_BUDDY_LLAMA31_BATCH_RHAL
    "${_BUDDY_LLAMA31_BATCH_ARTIFACT_ROOT}/${_BUDDY_LLAMA31_BATCH_STEM}.rhal.mlir")

  add_custom_command(
    OUTPUT "${_BUDDY_LLAMA31_BATCH_RAX}"
    BYPRODUCTS "${_BUDDY_LLAMA31_BATCH_RHAL}"
    COMMAND ${CMAKE_COMMAND} -E env ${_BUDDY_LLAMA31_BATCH_PACKAGE_ENV}
            bash "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/_env.sh"
      "${CMAKE_CURRENT_SOURCE_DIR}/buddy-llama31-lower-ttir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/llama31_chat_prepare.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/pack_decode_qkv_ttnn.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
      "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
    COMMENT "Generating self-contained ${_BUDDY_LLAMA31_BATCH_STEM}.rax"
    USES_TERMINAL
    VERBATIM
  )

  add_custom_target(${_BUDDY_LLAMA31_BATCH_STEM}_rax
    DEPENDS "${_BUDDY_LLAMA31_BATCH_RAX}"
    COMMENT "${_BUDDY_LLAMA31_BATCH_STEM}.rax -> ${_BUDDY_LLAMA31_BATCH_RAX}"
  )
  if(TARGET python-package-buddy)
    add_dependencies(${_BUDDY_LLAMA31_BATCH_STEM}_rax python-package-buddy)
  endif()

  add_custom_target(${_BUDDY_LLAMA31_BATCH_STEM}_package
    COMMENT "Alias for ${_BUDDY_LLAMA31_BATCH_STEM}_rax"
  )
  add_dependencies(${_BUDDY_LLAMA31_BATCH_STEM}_package
    ${_BUDDY_LLAMA31_BATCH_STEM}_rax)
  _buddy_register_llama_tt_rax_target(${_BUDDY_LLAMA31_BATCH_STEM}_rax)
endfunction()

if(BUDDY_LLAMA31_FIXED_BATCH_SIZES)
  list(REMOVE_DUPLICATES BUDDY_LLAMA31_FIXED_BATCH_SIZES)
  foreach(_buddy_llama31_batch_size IN LISTS BUDDY_LLAMA31_FIXED_BATCH_SIZES)
    _buddy_llama31_add_fixed_batch_package("${_buddy_llama31_batch_size}")
  endforeach()
endif()

set(BUDDY_LLAMA32_MAX_CACHE_LEN "128" CACHE STRING
  "Static KV cache length used when packaging Llama 3.2 TT artifacts")
set(BUDDY_LLAMA32_MODEL_PATH "" CACHE STRING
  "Optional local Hugging Face Llama 3.2 model path for packaging")
set(BUDDY_LLAMA32_EMBED_RAX_PAYLOAD ON CACHE BOOL
  "Embed Llama 3.2 TTNN flatbuffers and chat artifacts into llama32_tt.rax")
set(BUDDY_LLAMA32_OFFLINE ON CACHE BOOL
  "Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE while packaging Llama 3.2 TT")
set(BUDDY_LLAMA32_FIXED_BATCH_SIZES "32" CACHE STRING
  "Semicolon-separated additional fixed batch sizes to package for Llama 3.2 TT (batch1 is always emitted as llama32_tt_rax)")
set(BUDDY_LLAMA32_DECODE_RMSNORM_FUSION ON CACHE BOOL
  "Experimental: fuse Llama 3.2 decode RMSNorm into a TTNN op")
set(BUDDY_LLAMA32_DECODE_ARGMAX_TILE OFF CACHE BOOL
  "Experimental: let Llama 3.2 decode argmax consume tiled LM-head logits directly")
set(BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL OFF CACHE BOOL
  "Experimental: fold generated Llama 3.2 decode multiply-by-one ops")
set(BUDDY_LLAMA32_DECODE_PACK_QKV ON CACHE BOOL
  "Experimental: pre-pack Llama 3.2 decode Q/K/V static weight slots")
set(BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP OFF CACHE BOOL
  "Experimental: pre-pack Llama 3.2 decode MLP gate/up static weight slots")
set(BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE ON CACHE BOOL
  "Experimental: replace generated Llama 3.2 decode RoPE trig ops with runtime inputs")
set(BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2 OFF CACHE BOOL
  "Experimental: use HiFi2/no-fp32 accumulation only for the Llama 3.2 decode LM-head matmul")
set(BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD OFF CACHE BOOL
  "Experimental: split the Llama 3.2 decode LM-head vocab projection into fixed vocab-row shards")
set(BUDDY_LLAMA32_DECODE_LM_HEAD_SPLITS "8" CACHE STRING
  "Number of vocab-row shards used when BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD is enabled")
set(BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC OFF CACHE BOOL
  "Experimental: attach official-style DRAM-sharded matmul program config to Llama 3.2 decode LM-head projection matmul(s)")
set(BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC OFF CACHE BOOL
  "Experimental: attach 1D multicast matmul program config to Llama 3.2 decode LM-head projection matmul(s)")
set(BUDDY_LLAMA32_DECODE_LM_HEAD_PROGRAM_CORES "48" CACHE STRING
  "Number of cores used by LM-head program-config experiments")
set(BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT ON CACHE BOOL
  "Experimental: duplicate the tied Llama 3.2 decode embedding/LM-head weight as separate TTNN layout slots")
set(BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS OFF CACHE BOOL
  "Experimental: remove generated Llama 3.2 decode dealloc ops for static weight inputs")
set(BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO OFF CACHE BOOL
  "Experimental: use row-major u32 token tensors for Llama 3.2 decode input/output")
set(BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS OFF CACHE BOOL
  "Experimental: use ttnn.nlp_create_qkv_heads_decode for Llama 3.2 decode QKV head creation")
set(BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS OFF CACHE BOOL
  "Experimental: use ttnn.nlp_concat_heads_decode for Llama 3.2 decode attention output")
set(BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT OFF CACHE BOOL
  "Experimental: ask Llama 3.2 decode SDPA to directly produce the sharded layout consumed by ttnn.nlp_concat_heads_decode")

set(_BUDDY_LLAMA32_EMBED_RAX_PAYLOAD 0)
if(BUDDY_LLAMA32_EMBED_RAX_PAYLOAD)
  set(_BUDDY_LLAMA32_EMBED_RAX_PAYLOAD 1)
endif()

set(_BUDDY_LLAMA32_OFFLINE 0)
if(BUDDY_LLAMA32_OFFLINE)
  set(_BUDDY_LLAMA32_OFFLINE 1)
endif()

set(_BUDDY_LLAMA32_DECODE_ARGMAX_TILE 0)
if(BUDDY_LLAMA32_DECODE_ARGMAX_TILE)
  set(_BUDDY_LLAMA32_DECODE_ARGMAX_TILE 1)
endif()

set(_BUDDY_LLAMA32_DECODE_RMSNORM_FUSION 0)
if(BUDDY_LLAMA32_DECODE_RMSNORM_FUSION)
  set(_BUDDY_LLAMA32_DECODE_RMSNORM_FUSION 1)
endif()

set(_BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL 0)
if(BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL)
  set(_BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL 1)
endif()

set(_BUDDY_LLAMA32_DECODE_PACK_QKV 0)
if(BUDDY_LLAMA32_DECODE_PACK_QKV)
  set(_BUDDY_LLAMA32_DECODE_PACK_QKV 1)
endif()

set(_BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP 0)
if(BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP)
  set(_BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP 1)
endif()

set(_BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE 0)
if(BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE)
  set(_BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE 1)
endif()

set(_BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2 0)
if(BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2)
  set(_BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2 1)
endif()

set(_BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD 0)
if(BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD)
  set(_BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD 1)
endif()

set(_BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC 0)
if(BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC)
  set(_BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC 1)
endif()

set(_BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC 0)
if(BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC)
  set(_BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC 1)
endif()

set(_BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT 0)
if(BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT)
  set(_BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT 1)
endif()

set(_BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS 0)
if(BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS)
  set(_BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS 1)
endif()

set(_BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO 0)
if(BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO)
  set(_BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO 1)
endif()

set(_BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS 0)
if(BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS)
  set(_BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS 1)
endif()

set(_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS 0)
if(BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS)
  set(_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS 1)
endif()

set(_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT 0)
if(BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT)
  set(_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT 1)
endif()

set(_BUDDY_LLAMA32_ARTIFACT_ROOT "${CMAKE_BINARY_DIR}/models/llama32_tt")
set(_BUDDY_LLAMA32_PACKAGE_ENV
  "BUDDY_BUILD=${CMAKE_BINARY_DIR}"
  "BUDDY_TT_SKIP_ACTIVATE=1"
  "BUDDY_LLAMA31_ARTIFACT_ROOT=${_BUDDY_LLAMA32_ARTIFACT_ROOT}"
  "LD_LIBRARY_PATH=${_BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH}"
  "TTIR_OUT=${_BUDDY_LLAMA32_ARTIFACT_ROOT}/ttir_out_static"
  "BASE_CHAT_ART=${_BUDDY_LLAMA32_ARTIFACT_ROOT}/chat_artifacts"
  "CHAT_ART=${_BUDDY_LLAMA32_ARTIFACT_ROOT}/chat_artifacts"
  "RAX_PACKAGE_DIR=${_BUDDY_LLAMA32_ARTIFACT_ROOT}"
  "RAX_STEM=llama32_tt"
  "RAX_MODEL_NAME=llama32_tt"
  "TTIR_STEM=llama32"
  "PROMPT_FORMAT=completion"
  "PREFILL_KV_OUTPUT_ORDER=key_value"
  "LLAMA_DEFAULT_MODEL=meta-llama/Llama-3.2-3B"
  "LLAMA_DISPLAY_NAME=Llama-3.2-3B"
  "LLAMA_USE_LLAMA32_ENV=1"
  "TTRT_SYS_DIR=${_BUDDY_LLAMA32_ARTIFACT_ROOT}/ttrt_sys"
  "TTMLIR_SOURCE=${BUDDY_TT_MLIR_SOURCE_DIR}"
  "TTMLIR_BUILD=${BUDDY_TT_MLIR_BUILD_DIR}"
  "TTMLIR_TOOLCHAIN_DIR=${BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DIR}"
  "MAX_CACHE_LEN=${BUDDY_LLAMA32_MAX_CACHE_LEN}"
  "BUDDY_TT_DECODE_RMSNORM_FUSION=${_BUDDY_LLAMA32_DECODE_RMSNORM_FUSION}"
  "BUDDY_TT_DECODE_FOLD_IDENTITY_MUL=${_BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL}"
  "BUDDY_TT_DECODE_PACK_QKV=${_BUDDY_LLAMA32_DECODE_PACK_QKV}"
  "BUDDY_TT_DECODE_PACK_MLP_GATE_UP=${_BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP}"
  "BUDDY_TT_DECODE_PRECOMPUTE_ROPE=${_BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE}"
  "BUDDY_TT_DECODE_ARGMAX_TILE=${_BUDDY_LLAMA32_DECODE_ARGMAX_TILE}"
  "BUDDY_TT_DECODE_LMHEAD_HIFI2=${_BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2}"
  "BUDDY_TT_DECODE_SPLIT_LM_HEAD=${_BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD}"
  "BUDDY_TT_DECODE_LM_HEAD_SPLITS=${BUDDY_LLAMA32_DECODE_LM_HEAD_SPLITS}"
  "BUDDY_TT_DECODE_LM_HEAD_DRAM_PC=${_BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC}"
  "BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC=${_BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC}"
  "BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES=${BUDDY_LLAMA32_DECODE_LM_HEAD_PROGRAM_CORES}"
  "BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT=${_BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT}"
  "BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS=${_BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS}"
  "BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO=${_BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO}"
  "BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS=${_BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS}"
  "BUDDY_TT_DECODE_FUSE_CONCAT_HEADS=${_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS}"
  "BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT=${_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT}"
  "PACKAGE_ONLY=1"
  "RUN_WITH_BUDDY_CLI=0"
  "MAX_NEW_TOKENS=0"
  "SKIP_LOWER=0"
  "SKIP_PREPARE=0"
  "SKIP_PACKAGE=0"
  "FULL_ARG_ATTRS=1"
  "EMBED_RAX_PAYLOAD=${_BUDDY_LLAMA32_EMBED_RAX_PAYLOAD}"
  "HF_HUB_OFFLINE=${_BUDDY_LLAMA32_OFFLINE}"
  "TRANSFORMERS_OFFLINE=${_BUDDY_LLAMA32_OFFLINE}")

if(BUDDY_LLAMA32_MODEL_PATH)
  list(APPEND _BUDDY_LLAMA32_PACKAGE_ENV
    "LLAMA_MODEL_PATH=${BUDDY_LLAMA32_MODEL_PATH}"
    "LLAMA32_MODEL_PATH=${BUDDY_LLAMA32_MODEL_PATH}")
endif()

set(BUDDY_LLAMA32_RAX "${_BUDDY_LLAMA32_ARTIFACT_ROOT}/llama32_tt.rax")
set(BUDDY_LLAMA32_RHAL "${_BUDDY_LLAMA32_ARTIFACT_ROOT}/llama32_tt.rhal.mlir")

add_custom_command(
  OUTPUT "${BUDDY_LLAMA32_RAX}"
  BYPRODUCTS "${BUDDY_LLAMA32_RHAL}"
  COMMAND ${CMAKE_COMMAND} -E env ${_BUDDY_LLAMA32_PACKAGE_ENV}
          bash "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/_env.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/buddy-llama31-lower-ttir.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama31_chat_prepare.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/pack_decode_qkv_ttnn.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
  COMMENT "Generating self-contained llama32_tt.rax"
  USES_TERMINAL
  VERBATIM
)

add_custom_target(llama32_tt_rax
  DEPENDS "${BUDDY_LLAMA32_RAX}"
  COMMENT "llama32_tt.rax -> ${BUDDY_LLAMA32_RAX}"
)
if(TARGET python-package-buddy)
  add_dependencies(llama32_tt_rax python-package-buddy)
endif()

add_custom_target(llama32_tt_package
  COMMENT "Alias for llama32_tt_rax"
)
add_dependencies(llama32_tt_package llama32_tt_rax)

_buddy_register_llama_tt_rax_target(llama32_tt_rax)

function(_buddy_llama32_add_fixed_batch_package batch_size)
  if(NOT "${batch_size}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "BUDDY_LLAMA32_FIXED_BATCH_SIZES must contain integers; got '${batch_size}'")
  endif()
  if(batch_size LESS 2)
    message(FATAL_ERROR
      "Llama 3.2 batch packages only generate fixed batch sizes >= 2; "
      "batch1 is already provided by llama32_tt_rax")
  endif()

  set(_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT
    "${CMAKE_BINARY_DIR}/models/llama32_tt_b${batch_size}")
  set(_BUDDY_LLAMA32_BATCH_STEM "llama32_tt_b${batch_size}")
  set(_BUDDY_LLAMA32_BATCH_TTIR_STEM "llama32_b${batch_size}")
  set(_BUDDY_LLAMA32_BATCH_DISPLAY_NAME
    "Llama-3.2-3B batch${batch_size}")
  set(_BUDDY_LLAMA32_BATCH_EMBED_RAX_PAYLOAD 0)
  if(BUDDY_LLAMA32_EMBED_RAX_PAYLOAD)
    set(_BUDDY_LLAMA32_BATCH_EMBED_RAX_PAYLOAD 1)
  endif()

  set(_BUDDY_LLAMA32_BATCH_PACKAGE_ENV
    "BUDDY_BUILD=${CMAKE_BINARY_DIR}"
    "BUDDY_TT_SKIP_ACTIVATE=1"
    "BUDDY_LLAMA31_ARTIFACT_ROOT=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}"
    "LD_LIBRARY_PATH=${_BUDDY_LLAMA_PACKAGE_LD_LIBRARY_PATH}"
    "TTIR_OUT=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/ttir_out_static"
    "BASE_CHAT_ART=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/chat_artifacts"
    "CHAT_ART=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/chat_artifacts"
    "RAX_PACKAGE_DIR=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}"
    "RAX_STEM=${_BUDDY_LLAMA32_BATCH_STEM}"
    "RAX_MODEL_NAME=${_BUDDY_LLAMA32_BATCH_STEM}"
    "TTIR_STEM=${_BUDDY_LLAMA32_BATCH_TTIR_STEM}"
    "PROMPT_FORMAT=completion"
    "PREFILL_KV_OUTPUT_ORDER=key_value"
    "LLAMA_DEFAULT_MODEL=meta-llama/Llama-3.2-3B"
    "LLAMA_DISPLAY_NAME=${_BUDDY_LLAMA32_BATCH_DISPLAY_NAME}"
    "LLAMA_USE_LLAMA32_ENV=1"
    "TTRT_SYS_DIR=${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/ttrt_sys"
    "TTMLIR_SOURCE=${BUDDY_TT_MLIR_SOURCE_DIR}"
    "TTMLIR_BUILD=${BUDDY_TT_MLIR_BUILD_DIR}"
    "TTMLIR_TOOLCHAIN_DIR=${BUDDY_LLAMA31_TTMLIR_TOOLCHAIN_DIR}"
    "MAX_CACHE_LEN=${BUDDY_LLAMA32_MAX_CACHE_LEN}"
    "BATCH_SIZE=${batch_size}"
    "BUDDY_TT_DECODE_RMSNORM_FUSION=${_BUDDY_LLAMA32_DECODE_RMSNORM_FUSION}"
    "BUDDY_TT_DECODE_FOLD_IDENTITY_MUL=${_BUDDY_LLAMA32_DECODE_FOLD_IDENTITY_MUL}"
    "BUDDY_TT_DECODE_PACK_QKV=${_BUDDY_LLAMA32_DECODE_PACK_QKV}"
    "BUDDY_TT_DECODE_PACK_MLP_GATE_UP=${_BUDDY_LLAMA32_DECODE_PACK_MLP_GATE_UP}"
    "BUDDY_TT_DECODE_PRECOMPUTE_ROPE=${_BUDDY_LLAMA32_DECODE_PRECOMPUTE_ROPE}"
    "BUDDY_TT_DECODE_ARGMAX_TILE=${_BUDDY_LLAMA32_DECODE_ARGMAX_TILE}"
    "BUDDY_TT_DECODE_LMHEAD_HIFI2=${_BUDDY_LLAMA32_DECODE_LMHEAD_HIFI2}"
    "BUDDY_TT_DECODE_SPLIT_LM_HEAD=${_BUDDY_LLAMA32_DECODE_SPLIT_LM_HEAD}"
    "BUDDY_TT_DECODE_LM_HEAD_SPLITS=${BUDDY_LLAMA32_DECODE_LM_HEAD_SPLITS}"
    "BUDDY_TT_DECODE_LM_HEAD_DRAM_PC=${_BUDDY_LLAMA32_DECODE_LM_HEAD_DRAM_PC}"
    "BUDDY_TT_DECODE_LM_HEAD_MCAST1D_PC=${_BUDDY_LLAMA32_DECODE_LM_HEAD_MCAST1D_PC}"
    "BUDDY_TT_DECODE_LM_HEAD_PROGRAM_CORES=${BUDDY_LLAMA32_DECODE_LM_HEAD_PROGRAM_CORES}"
    "BUDDY_TT_DECODE_SPLIT_EMBEDDING_WEIGHT=${_BUDDY_LLAMA32_DECODE_SPLIT_EMBEDDING_WEIGHT}"
    "BUDDY_TT_DECODE_KEEP_STATIC_WEIGHT_INPUTS=${_BUDDY_LLAMA32_DECODE_KEEP_STATIC_WEIGHT_INPUTS}"
    "BUDDY_TT_DECODE_NATIVE_U32_TOKEN_IO=${_BUDDY_LLAMA32_DECODE_NATIVE_U32_TOKEN_IO}"
    "BUDDY_TT_DECODE_FUSE_CREATE_QKV_HEADS=${_BUDDY_LLAMA32_DECODE_FUSE_CREATE_QKV_HEADS}"
    "BUDDY_TT_DECODE_FUSE_CONCAT_HEADS=${_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS}"
    "BUDDY_TT_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT=${_BUDDY_LLAMA32_DECODE_FUSE_CONCAT_HEADS_SDPA_OUTPUT}"
    "PACKAGE_ONLY=1"
    "RUN_WITH_BUDDY_CLI=0"
    "MAX_NEW_TOKENS=0"
    "DEVICE_ARGMAX=0"
    "DEVICE_ARGMAX_PREFILL=0"
    "DEVICE_ARGMAX_DECODE=1"
    "RUNTIME_ATTENTION_MASK=1"
    "SKIP_LOWER=0"
    "SKIP_PREPARE=0"
    "SKIP_PACKAGE=0"
    "FULL_ARG_ATTRS=1"
    "EMBED_RAX_PAYLOAD=${_BUDDY_LLAMA32_BATCH_EMBED_RAX_PAYLOAD}"
    "HF_HUB_OFFLINE=${_BUDDY_LLAMA32_OFFLINE}"
    "TRANSFORMERS_OFFLINE=${_BUDDY_LLAMA32_OFFLINE}")

  if(BUDDY_LLAMA32_MODEL_PATH)
    list(APPEND _BUDDY_LLAMA32_BATCH_PACKAGE_ENV
      "LLAMA_MODEL_PATH=${BUDDY_LLAMA32_MODEL_PATH}"
      "LLAMA32_MODEL_PATH=${BUDDY_LLAMA32_MODEL_PATH}")
  endif()

  set(_BUDDY_LLAMA32_BATCH_RAX
    "${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/${_BUDDY_LLAMA32_BATCH_STEM}.rax")
  set(_BUDDY_LLAMA32_BATCH_RHAL
    "${_BUDDY_LLAMA32_BATCH_ARTIFACT_ROOT}/${_BUDDY_LLAMA32_BATCH_STEM}.rhal.mlir")

  add_custom_command(
    OUTPUT "${_BUDDY_LLAMA32_BATCH_RAX}"
    BYPRODUCTS "${_BUDDY_LLAMA32_BATCH_RHAL}"
    COMMAND ${CMAKE_COMMAND} -E env ${_BUDDY_LLAMA32_BATCH_PACKAGE_ENV}
            bash "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/_env.sh"
      "${CMAKE_CURRENT_SOURCE_DIR}/buddy-llama31-lower-ttir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/llama31_chat_prepare.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/pack_decode_qkv_ttnn.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/run_llama31_tt_chat.sh"
      "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_tenstorrent_manifest.py"
    COMMENT "Generating self-contained ${_BUDDY_LLAMA32_BATCH_STEM}.rax"
    USES_TERMINAL
    VERBATIM
  )

  add_custom_target(${_BUDDY_LLAMA32_BATCH_STEM}_rax
    DEPENDS "${_BUDDY_LLAMA32_BATCH_RAX}"
    COMMENT "${_BUDDY_LLAMA32_BATCH_STEM}.rax -> ${_BUDDY_LLAMA32_BATCH_RAX}"
  )
  if(TARGET python-package-buddy)
    add_dependencies(${_BUDDY_LLAMA32_BATCH_STEM}_rax python-package-buddy)
  endif()

  add_custom_target(${_BUDDY_LLAMA32_BATCH_STEM}_package
    COMMENT "Alias for ${_BUDDY_LLAMA32_BATCH_STEM}_rax"
  )
  add_dependencies(${_BUDDY_LLAMA32_BATCH_STEM}_package
    ${_BUDDY_LLAMA32_BATCH_STEM}_rax)
  _buddy_register_llama_tt_rax_target(${_BUDDY_LLAMA32_BATCH_STEM}_rax)
endfunction()

if(BUDDY_LLAMA32_FIXED_BATCH_SIZES)
  list(REMOVE_DUPLICATES BUDDY_LLAMA32_FIXED_BATCH_SIZES)
  foreach(_buddy_llama32_batch_size IN LISTS BUDDY_LLAMA32_FIXED_BATCH_SIZES)
    _buddy_llama32_add_fixed_batch_package("${_buddy_llama32_batch_size}")
  endforeach()
endif()
