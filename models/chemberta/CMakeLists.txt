# ChemBERTa masked-LM encoder model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward.
# ChemBERTa keeps a custom import and manifest generator (like BGE-M3/ColBERTv2)
# because the generic DeepSeek path is LLM prefill/decode + KV-cache specific.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_CHEMBERTA_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "ChemBERTa variant spec JSON, e.g. specs/f32.json")
set(BUDDY_CHEMBERTA_MODEL_PATH "" CACHE PATH
  "Local HuggingFace DeepChem/ChemBERTa-77M-MLM snapshot dir")

buddy_add_model(
  NAME chemberta
  MODEL_KIND single_forward
  SPEC "${BUDDY_CHEMBERTA_SPEC}"
  RUNNER_SRC ChembertaRunner.cpp
  RUNNER_PLUGIN_SRC ChembertaRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_CHEMBERTA_MODEL_PATH}"
  LOCAL_MODEL_ENV CHEMBERTA_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-chemberta.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_chemberta_manifest.py"
  MODEL_SO_NAME chemberta_model.so
  ASSET_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/vocab.txt"
)
