# BGE-Reranker-v2-M3 cross-encoder reranker model integration.
#
# Uses the shared buddy-codegen model entry point with MODEL_KIND=single_forward
# and pluggable IMPORT_SCRIPT/MANIFEST_SCRIPT (like ColBERTv2 and BGE-M3): the
# importer AOT-traces the XLMRobertaForSequenceClassification forward into
# subgraph0.mlir/forward.mlir/arg0.data, and the manifest generator emits a RHAL
# manifest packing the kernels, weights, tokenizer, and the runner plugin into
# a single bge_reranker.rax.

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_BGE_RERANKER_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "BGE-Reranker variant spec JSON, e.g. specs/f32.json")
set(BUDDY_BGE_RERANKER_MODEL_PATH "" CACHE PATH
  "Local HuggingFace BAAI/bge-reranker-v2-m3 snapshot dir")

buddy_add_model(
  NAME bge_reranker
  MODEL_KIND single_forward
  SPEC "${BUDDY_BGE_RERANKER_SPEC}"
  RUNNER_SRC BgeRerankerRunner.cpp
  RUNNER_PLUGIN_SRC BgeRerankerRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_BGE_RERANKER_MODEL_PATH}"
  LOCAL_MODEL_ENV BGE_RERANKER_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-bge-reranker.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_bge_reranker_manifest.py"
  MODEL_SO_NAME bge_reranker_model.so
  ASSET_FILES
    "${BUDDY_BGE_RERANKER_MODEL_PATH}/tokenizer.json"
)
