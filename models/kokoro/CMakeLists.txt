# Kokoro-82M text-to-speech model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# Kokoro keeps a custom import and manifest generator (like ColBERTv2 / BGE-M3)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_KOKORO_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Kokoro-82M variant spec JSON, e.g. specs/f32.json")
set(BUDDY_KOKORO_MODEL_PATH "" CACHE PATH
  "Local HuggingFace hexgrad/Kokoro-82M snapshot dir")

buddy_add_model(
  NAME kokoro
  MODEL_KIND single_forward
  SPEC "${BUDDY_KOKORO_SPEC}"
  RUNNER_SRC KokoroRunner.cpp
  RUNNER_PLUGIN_SRC KokoroRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_KOKORO_MODEL_PATH}"
  LOCAL_MODEL_ENV KOKORO_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-kokoro.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_kokoro_manifest.py"
  MODEL_SO_NAME kokoro_model.so
  ASSET_FILES
    "${BUDDY_KOKORO_MODEL_PATH}/config.json"
)
