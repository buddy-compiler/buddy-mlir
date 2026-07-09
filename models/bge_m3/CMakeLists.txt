# BGE-M3 dense embedding model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# BGE-M3 keeps a custom import and manifest generator (like Whisper) because the
# generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_BGE_M3_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/base.json"
  CACHE FILEPATH "BGE-M3 variant spec JSON")
set(BUDDY_BGE_M3_MODEL_PATH ""
  CACHE PATH "Local HuggingFace BGE-M3 model directory")

buddy_add_model(
  NAME bge_m3
  MODEL_KIND single_forward
  SPEC "${BUDDY_BGE_M3_SPEC}"
  RUNNER_SRC BgeM3Runner.cpp
  RUNNER_PLUGIN_SRC BgeM3RunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_BGE_M3_MODEL_PATH}"
  LOCAL_MODEL_ENV BUDDY_LOCAL_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-bge-m3.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_bge_m3_manifest.py"
  MODEL_SO_NAME bge_m3_model.so
  ASSET_FILES
    "${BUDDY_BGE_M3_MODEL_PATH}/tokenizer.json"
)
