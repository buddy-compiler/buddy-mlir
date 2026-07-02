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

set(WHISPER_EXAMPLE "${CMAKE_SOURCE_DIR}/examples/BuddyWhisper")

buddy_add_model(
  NAME whisper
  MODEL_KIND single_forward
  SPEC "${BUDDY_WHISPER_SPEC}"
  RUNNER_SRC WhisperRunner.cpp
  RUNNER_PLUGIN_SRC WhisperRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_WHISPER_MODEL_PATH}"
  LOCAL_MODEL_ENV WHISPER_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-whisper.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_whisper_manifest.py"
  MODEL_SO_NAME whisper_model.so
  ASSET_FILES
    "${WHISPER_EXAMPLE}/vocab.txt"
    "${WHISPER_EXAMPLE}/audio.wav"
  RUNTIME_LINK_LIBS
    BuddyLibDAP
    mlir_c_runner_utils
)
