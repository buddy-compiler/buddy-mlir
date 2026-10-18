# ══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL-MoE (Qwen3-VL-30B-A3B), text decoder — buddy-codegen
#
# The text decoder of the vision-language model, imported from PyTorch as a
# Qwen3-MoE causal LM (gen_config.text_decoder_config,
# import_model.load_qwen3_moe): each MoE block's experts are traced as one
# node (frontend/Python/moe.py) and become an int4 kernel with variant
# w4g32 (graph/transform/k3_w4.py). Same pipeline as models/deepseek_r1:
# variant JSON → gen_config + ModelSession + RHAL manifest + import_model +
# compile_pipeline + .so + .rax.
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_QWEN3_VL_MOE_DEFAULT_COMPILE_JOBS)
if(_BUDDY_QWEN3_VL_MOE_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_QWEN3_VL_MOE_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_QWEN3_VL_MOE_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/k3_w4g32.json"
  CACHE FILEPATH "Qwen3-VL-MoE variant spec JSON")
set(BUDDY_QWEN3_VL_MOE_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json of the Qwen3-VL-MoE model")
set(BUDDY_QWEN3_VL_MOE_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF Qwen3-VL-MoE snapshot directory")
set(BUDDY_QWEN3_VL_MOE_COMPILE_JOBS
  "${_BUDDY_QWEN3_VL_MOE_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")

buddy_add_model(
  NAME qwen3_vl_moe
  SPEC "${BUDDY_QWEN3_VL_MOE_SPEC}"
  RUNNER_SRC Qwen3VLMoeRunner.cpp
  RUNNER_PLUGIN_SRC Qwen3VLMoeRunnerPlugin.cpp
  HF_CONFIG "${BUDDY_QWEN3_VL_MOE_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_QWEN3_VL_MOE_LOCAL_MODEL}"
  COMPILE_JOBS "${BUDDY_QWEN3_VL_MOE_COMPILE_JOBS}"
  ASSET_FILES "${CMAKE_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt"
)
