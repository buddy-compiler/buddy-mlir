# MoLFormer (ibm/MoLFormer-XL-both-10pct) molecular embedding integration.
#
# single_forward model. The compiled kernel is the *full* 12-layer linear-
# attention encoder traced as one AOT graph (44.7M params including the
# deterministic random-Fourier-feature projections). See README.md for the
# traceability notes (deterministic random features + patched attention).

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)

set(BUDDY_MOLFORMER_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "MoLFormer variant spec JSON, e.g. specs/f32.json")
set(BUDDY_MOLFORMER_MODEL_PATH "" CACHE PATH
  "Local HuggingFace ibm/MoLFormer-XL-both-10pct snapshot dir")

buddy_add_model(
  NAME molformer
  MODEL_KIND single_forward
  SPEC "${BUDDY_MOLFORMER_SPEC}"
  RUNNER_SRC MoLFormerRunner.cpp
  RUNNER_PLUGIN_SRC MoLFormerRunnerPlugin.cpp
  LOCAL_MODEL "${BUDDY_MOLFORMER_MODEL_PATH}"
  LOCAL_MODEL_ENV MOLFORMER_MODEL_PATH
  IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/import-molformer.py"
  MANIFEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_molformer_manifest.py"
  MODEL_SO_NAME molformer_model.so
  ASSET_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/vocab.txt"
)
