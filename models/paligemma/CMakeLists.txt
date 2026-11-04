# PaliGemma-3B-224 vision-language model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# PaliGemma keeps a custom import and manifest generator (like ColBERTv2)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_PALIGEMMA_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "PaliGemma variant spec JSON, e.g. specs/f32.json")
set(BUDDY_PALIGEMMA_MODEL_PATH "" CACHE PATH
  "Local HuggingFace google/paligemma-3b-mix-224 snapshot dir")

buddy_add_model(
  NAME paligemma
  MODEL_KIND single_forward
  SPEC "${BUDDY_PALIGEMMA_SPEC}"
  RUNNER_SRC PaligemmaRunner.cpp
  RUNNER_PLUGIN_SRC PaligemmaRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/PaligemmaRunner.h
  LOCAL_MODEL "${BUDDY_PALIGEMMA_MODEL_PATH}"
  LOCAL_MODEL_ENV PALIGEMMA_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-paligemma.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_paligemma_manifest.py"
  MODEL_SO_NAME paligemma_model.so
  ASSET_FILES
    "${BUDDY_PALIGEMMA_MODEL_PATH}/tokenizer.json"
    "${BUDDY_PALIGEMMA_MODEL_PATH}/tokenizer_config.json"
    "${BUDDY_PALIGEMMA_MODEL_PATH}/special_tokens_map.json"
)
