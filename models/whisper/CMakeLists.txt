# Whisper encoder-decoder ASR integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# Whisper keeps custom import and manifest generators because the generic
# DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_WHISPER_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/base.json"
  CACHE FILEPATH "Whisper variant spec JSON, e.g. specs/base.json")
set(BUDDY_WHISPER_MODEL_PATH "" CACHE PATH
  "Local HuggingFace whisper snapshot dir")
option(BUDDY_WHISPER_K3_IME
  "Use shared A100 FP16 IME GEMM and batch-matmul passes for Whisper"
  OFF)
file(READ "${BUDDY_WHISPER_SPEC}" _whisper_spec_json)
string(JSON _whisper_encoder_cache ERROR_VARIABLE _whisper_cache_error
       GET "${_whisper_spec_json}" encoder_cache)
if(_whisper_cache_error)
  set(_whisper_encoder_cache OFF)
endif()
string(JSON _whisper_kv_cache ERROR_VARIABLE _whisper_kv_error
       GET "${_whisper_spec_json}" kv_cache)
if(_whisper_kv_error)
  set(_whisper_kv_cache OFF)
endif()
set(WHISPER_SKIP_AFFINE_FUSION OFF)
if(_whisper_kv_cache AND BUDDY_WHISPER_K3_IME)
  # Fusing affine loops after all encoder GEMMs enter IME changes values in
  # the first Whisper encoder block on K3. The KV graph uses all IME GEMMs
  # and skips only that pass for its encoder.
  set(WHISPER_SKIP_AFFINE_FUSION ON)
endif()

if(BUDDY_WHISPER_K3_IME)
  if(NOT IS_RVV_CROSSCOMPILE OR NOT BUDDY_RISCV_VLEN STREQUAL "1024")
    message(FATAL_ERROR
      "BUDDY_WHISPER_K3_IME requires RISC-V cross compilation and VLEN=1024")
  endif()
  if(NOT BUDDY_RISCV_ENABLE_ZFH_ZVFH)
    message(FATAL_ERROR
      "BUDDY_WHISPER_K3_IME requires BUDDY_RISCV_ENABLE_ZFH_ZVFH=ON")
  endif()
  string(JSON _whisper_precision GET "${_whisper_spec_json}" precision)
  if(NOT _whisper_precision STREQUAL "f16")
    message(FATAL_ERROR "BUDDY_WHISPER_K3_IME requires precision=f16")
  endif()
endif()

set(WHISPER_EXAMPLE "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper")
if(IS_RVV_CROSSCOMPILE)
  # BuddyLibDAP's MLIR objects are compiled for BUDDY_TARGET_TRIPLE. Link
  # them only into the RISC-V plugins; the host plugin targets are not used.
  set(WHISPER_RUNTIME_LINK_LIBS)
  set(WHISPER_RISCV_RUNTIME_LINK_LIBS BuddyLibDAP)
else()
  set(WHISPER_RUNTIME_LINK_LIBS BuddyLibDAP mlir_c_runner_utils)
  set(WHISPER_RISCV_RUNTIME_LINK_LIBS)
endif()

buddy_add_model(
  NAME whisper
  TEMPLATE_PARTITION_CAPABLE ON
  MODEL_KIND single_forward
  SINGLE_FORWARD_IME "${BUDDY_WHISPER_K3_IME}"
  SINGLE_FORWARD_SKIP_AFFINE_FUSION "${WHISPER_SKIP_AFFINE_FUSION}"
  SINGLE_FORWARD_SPLIT "${_whisper_encoder_cache}"
  SINGLE_FORWARD_KV "${_whisper_kv_cache}"
  SPEC "${BUDDY_WHISPER_SPEC}"
  RUNNER_SRC WhisperRunner.cpp
  RUNNER_PLUGIN_SRC WhisperRunnerPlugin.cpp
  MODEL_PLUGIN_SRC WhisperAudioTranscriptionModelPlugin.cpp
  EXTRA_SRCS
    WhisperRuntime.cpp
    WhisperAudioTranscriptionModel.cpp
  LOCAL_MODEL "${BUDDY_WHISPER_MODEL_PATH}"
  LOCAL_MODEL_ENV WHISPER_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import_whisper.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_whisper_manifest.py"
  MODEL_SO_NAME whisper_model.so
  ASSET_FILES
    "${WHISPER_EXAMPLE}/vocab.txt"
    "${WHISPER_EXAMPLE}/audio.wav"
  RUNTIME_LINK_LIBS ${WHISPER_RUNTIME_LINK_LIBS}
  RISCV_RUNTIME_LINK_LIBS ${WHISPER_RISCV_RUNTIME_LINK_LIBS}
)

if(BUDDY_ENABLE_TESTS)
  include(${CMAKE_CURRENT_SOURCE_DIR}/WhisperTests.cmake)
endif()
