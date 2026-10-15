# SAM2-hiera-tiny vision-encoder model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# SAM2 keeps a custom import and manifest generator (like ColBERTv2) because the
# generic DeepSeek path is LLM prefill/decode + KV-cache specific. This model is
# a fixed-shape image encoder: import-sam2.py traces the Sam2VisionModel over a
# 1 x 3 x 256 x 256 image, and gen_sam2_manifest.py emits the RHAL manifest for
# the vision-encoder forward ABI.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_SAM2_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "SAM2 variant spec JSON, e.g. specs/f32.json")
set(BUDDY_SAM2_MODEL_PATH "" CACHE PATH
  "Local HuggingFace facebook/sam2-hiera-tiny snapshot dir")

buddy_add_model(
  NAME sam2
  MODEL_KIND single_forward
  SPEC "${BUDDY_SAM2_SPEC}"
  RUNNER_SRC Sam2Runner.cpp
  RUNNER_PLUGIN_SRC Sam2RunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_SAM2_MODEL_PATH}"
  LOCAL_MODEL_ENV SAM2_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-sam2.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_sam2_manifest.py"
  MODEL_SO_NAME sam2_model.so
  ASSET_FILES
    "${BUDDY_SAM2_MODEL_PATH}/config.json"
)
