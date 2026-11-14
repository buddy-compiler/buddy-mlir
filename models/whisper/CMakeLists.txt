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
if(_whisper_encoder_cache AND BUDDY_WHISPER_K3_IME)
  message(FATAL_ERROR
    "Cached Whisper encoder currently requires BUDDY_WHISPER_K3_IME=OFF: "
    "the split encoder IME path fails transcription correctness on K3")
endif()

set(WHISPER_LLC_ATTRS "")
if(BUDDY_WHISPER_K3_IME)
  if(NOT IS_RVV_CROSSCOMPILE OR NOT BUDDY_RISCV_VLEN STREQUAL "1024")
    message(FATAL_ERROR
      "BUDDY_WHISPER_K3_IME requires RISC-V cross compilation and VLEN=1024")
  endif()
  file(READ "${BUDDY_WHISPER_SPEC}" _whisper_spec_json)
  string(JSON _whisper_precision GET "${_whisper_spec_json}" precision)
  if(NOT _whisper_precision STREQUAL "f16")
    message(FATAL_ERROR "BUDDY_WHISPER_K3_IME requires precision=f16")
  endif()
  # VLEN options (+zvl1024b, -riscv-v-vector-bits-max=1024) are appended by
  # buddy_rvv_vlen_llc_attrs() in buddy_model.cmake from BUDDY_RISCV_VLEN.
  set(WHISPER_LLC_ATTRS
    "-march=riscv64 -mcpu=spacemit-a100 -mattr=+m,+d,+v,+zfh,+zvfh,+xsmtvdotii -mtriple=riscv64-unknown-linux-gnu")
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
  SINGLE_FORWARD_SPLIT "${_whisper_encoder_cache}"
  SPEC "${BUDDY_WHISPER_SPEC}"
  LLC_ATTRS "${WHISPER_LLC_ATTRS}"
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
