# EmbeddingGemma (google/embeddinggemma-300m) sentence embedding integration.
#
# single_forward model. The compiled kernel is the SentenceTransformer pipeline
# (Gemma3TextModel -> mean pooling -> Dense(768->3072) -> Dense(3072->768) ->
# L2 normalize) traced as a single AOT graph (307,581,953 f32 weights).

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_EMBEDDINGGEMMA_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "EmbeddingGemma variant spec JSON, e.g. specs/f32.json")
set(BUDDY_EMBEDDINGGEMMA_MODEL_PATH "" CACHE PATH
  "Local HuggingFace google/embeddinggemma-300m snapshot dir")

buddy_add_model(
  NAME embeddinggemma
  MODEL_KIND single_forward
  SPEC "${BUDDY_EMBEDDINGGEMMA_SPEC}"
  RUNNER_SRC EmbeddinggemmaRunner.cpp
  RUNNER_PLUGIN_SRC EmbeddinggemmaRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}"
  LOCAL_MODEL_ENV EMBEDDINGGEMMA_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-embeddinggemma.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_embeddinggemma_manifest.py"
  MODEL_SO_NAME embeddinggemma_model.so
  ASSET_FILES
    "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}/tokenizer.json"
)
