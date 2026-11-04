# E5-Mistral-7B-Instruct sentence-embedding model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# E5-Mistral keeps a custom import and manifest generator (like ColBERTv2)
# because it is a Mistral-architecture encoder-only model, not an LLM
# prefill/decode + KV-cache model.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_E5_MISTRAL_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "E5-Mistral variant spec JSON, e.g. specs/f32.json")
set(BUDDY_E5_MISTRAL_MODEL_PATH "" CACHE PATH
  "Local HuggingFace intfloat/e5-mistral-7b-instruct snapshot dir")

buddy_add_model(
  NAME e5_mistral
  MODEL_KIND single_forward
  SPEC "${BUDDY_E5_MISTRAL_SPEC}"
  RUNNER_SRC E5MistralRunner.cpp
  RUNNER_PLUGIN_SRC E5MistralRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_E5_MISTRAL_MODEL_PATH}"
  LOCAL_MODEL_ENV E5_MISTRAL_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-e5-mistral.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_e5_mistral_manifest.py"
  MODEL_SO_NAME e5_mistral_model.so
  ASSET_FILES
    "${BUDDY_E5_MISTRAL_MODEL_PATH}/tokenizer.json"
)
