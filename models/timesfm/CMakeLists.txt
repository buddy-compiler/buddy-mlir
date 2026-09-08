# TimesFM 2.5 time-series foundation model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# TimesFM keeps a custom import and manifest generator (like ColBERTv2/BGE-M3)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_TIMESFM_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "TimesFM variant spec JSON, e.g. specs/f32.json")
set(BUDDY_TIMESFM_MODEL_PATH "" CACHE PATH
  "Local HuggingFace google/timesfm-2.5-200m-pytorch snapshot dir")

buddy_add_model(
  NAME timesfm
  MODEL_KIND single_forward
  SPEC "${BUDDY_TIMESFM_SPEC}"
  RUNNER_SRC TimesfmRunner.cpp
  RUNNER_PLUGIN_SRC TimesfmRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_TIMESFM_MODEL_PATH}"
  LOCAL_MODEL_ENV TIMESFM_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-timesfm.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_timesfm_manifest.py"
  MODEL_SO_NAME timesfm_model.so
  ASSET_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
)
