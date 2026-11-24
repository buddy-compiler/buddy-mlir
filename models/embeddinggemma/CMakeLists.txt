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

file(READ "${BUDDY_EMBEDDINGGEMMA_SPEC}" _embeddinggemma_spec)
string(JSON _embeddinggemma_variant GET "${_embeddinggemma_spec}" variant)
string(JSON _embeddinggemma_num_threads ERROR_VARIABLE _embeddinggemma_threads_error
  GET "${_embeddinggemma_spec}" num_threads)
if(_embeddinggemma_threads_error OR NOT _embeddinggemma_num_threads)
  set(_embeddinggemma_num_threads 1)
endif()

set(_embeddinggemma_import_extra_deps)
set(_embeddinggemma_lower_extra_deps
  "${CMAKE_CURRENT_SOURCE_DIR}/codegen/tanh-approx.mlir")
set(_embeddinggemma_linker)
if(_embeddinggemma_variant STREQUAL "fp16_comp3")
  if(NOT IS_RVV_CROSSCOMPILE AND NOT HAVE_LOCAL_RVV)
    message(FATAL_ERROR "EmbeddingGemma fp16_comp3 requires a RISC-V build")
  endif()
  if(NOT "${BUDDY_RISCV_VLEN}" STREQUAL "1024")
    message(FATAL_ERROR "EmbeddingGemma fp16_comp3 requires BUDDY_RISCV_VLEN=1024")
  endif()
  # Link separately compiled F32 and A100 objects without retuning the F32
  # graph. buddy_add_model passes this linker to LOWER_SCRIPT.
  find_program(_embeddinggemma_linker NAMES ld.lld riscv64-unknown-linux-gnu-ld
    HINTS "${RISCV_GNU_TOOLCHAIN}/bin" "${LLVM_TOOLS_BINARY_DIR}"
    REQUIRED)
  list(APPEND _embeddinggemma_import_extra_deps
    "${CMAKE_CURRENT_SOURCE_DIR}/codegen/embeddinggemma_ime.py")
endif()

buddy_add_model(
  NAME embeddinggemma
  MODEL_KIND single_forward
  SPEC "${BUDDY_EMBEDDINGGEMMA_SPEC}"
  NUM_THREADS "${_embeddinggemma_num_threads}"
  RUNNER_SRC EmbeddinggemmaRunner.cpp
  RUNNER_PLUGIN_SRC EmbeddinggemmaRunnerPlugin.cpp
  RUNNER_HDR include/buddy/runtime/models/EmbeddinggemmaRunner.h
  LOCAL_MODEL "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}"
  LOCAL_MODEL_ENV EMBEDDINGGEMMA_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-embeddinggemma.py"
  IMPORT_EXTRA_DEPS ${_embeddinggemma_import_extra_deps}
  LOWER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh"
  LOWER_EXTRA_DEPS ${_embeddinggemma_lower_extra_deps}
  LOWER_LINKER "${_embeddinggemma_linker}"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_embeddinggemma_manifest.py"
  MODEL_SO_NAME embeddinggemma_model.so
  ASSET_FILES
    "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}/tokenizer.json"
)
