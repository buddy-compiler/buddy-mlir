# SmolVLM2-500M text-LM integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# SmolVLM2 keeps a custom import and manifest generator (like ColBERTv2/BGE-M3)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific,
# and the full VLM forward does not trace as a single graph. The AOT importer
# traces SmolVLMModel.text_model (the Llama3 text decoder) and the runner owns
# GPT2 byte-level BPE tokenization against the staged tokenizer.json.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_SMOLVLM2_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "SmolVLM2 variant spec JSON, e.g. specs/f32.json")
set(BUDDY_SMOLVLM2_MODEL_PATH "" CACHE PATH
  "Local HuggingFace HuggingFaceTB/SmolVLM2-500M-Instruct snapshot dir")

buddy_add_model(
  NAME smolvlm2
  MODEL_KIND single_forward
  SPEC "${BUDDY_SMOLVLM2_SPEC}"
  RUNNER_SRC Smolvlm2Runner.cpp
  RUNNER_PLUGIN_SRC Smolvlm2RunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_SMOLVLM2_MODEL_PATH}"
  LOCAL_MODEL_ENV SMOLVLM2_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-smolvlm2.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_smolvlm2_manifest.py"
  MODEL_SO_NAME smolvlm2_model.so
  ASSET_FILES
    "${BUDDY_SMOLVLM2_MODEL_PATH}/tokenizer.json"
)
