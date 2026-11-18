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

set(_embeddinggemma_lower "${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh")
file(READ "${BUDDY_EMBEDDINGGEMMA_SPEC}" _embeddinggemma_spec)
string(JSON _embeddinggemma_variant GET "${_embeddinggemma_spec}" variant)
if(_embeddinggemma_variant STREQUAL "fp16_comp3")
  if(NOT IS_RVV_CROSSCOMPILE AND NOT HAVE_LOCAL_RVV)
    message(FATAL_ERROR "EmbeddingGemma fp16_comp3 requires a RISC-V build")
  endif()
  if(NOT "${BUDDY_RISCV_VLEN}" STREQUAL "1024")
    message(FATAL_ERROR "EmbeddingGemma fp16_comp3 requires BUDDY_RISCV_VLEN=1024")
  endif()
  # Link separately compiled F32 and A100 objects without retuning the F32
  # graph. A build-local wrapper passes the target linker to LOWER_SCRIPT.
  find_program(_embeddinggemma_linker NAMES ld.lld riscv64-unknown-linux-gnu-ld
    HINTS "${RISCV_GNU_TOOLCHAIN}/bin" "${LLVM_TOOLS_BINARY_DIR}"
    REQUIRED)
  set(_embeddinggemma_lower "${CMAKE_CURRENT_BINARY_DIR}/lower_embeddinggemma.sh")
  file(GENERATE OUTPUT "${_embeddinggemma_lower}" CONTENT
    "#!/bin/sh\nexec \"${CMAKE_COMMAND}\" -E env \"EMBEDDINGGEMMA_OBJECT_LINKER=${_embeddinggemma_linker}\" bash \"${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh\" \"$@\"\n")
endif()

buddy_add_model(
  NAME embeddinggemma
  MODEL_KIND single_forward
  SPEC "${BUDDY_EMBEDDINGGEMMA_SPEC}"
  RUNNER_SRC EmbeddinggemmaRunner.cpp
  RUNNER_PLUGIN_SRC EmbeddinggemmaRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}"
  LOCAL_MODEL_ENV EMBEDDINGGEMMA_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-embeddinggemma.py"
  LOWER_SCRIPT "${_embeddinggemma_lower}"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_embeddinggemma_manifest.py"
  MODEL_SO_NAME embeddinggemma_model.so
  ASSET_FILES
    "${BUDDY_EMBEDDINGGEMMA_MODEL_PATH}/tokenizer.json"
)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/subgraph0.o"
  APPEND
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/codegen/tanh-approx.mlir"
          "${CMAKE_CURRENT_SOURCE_DIR}/codegen/lower_to_obj.sh"
)

if(_embeddinggemma_variant STREQUAL "fp16_comp3")
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/.buddy_import_done"
    APPEND
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/codegen/embeddinggemma_ime.py"
  )
endif()
