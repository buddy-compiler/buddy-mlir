_add_test_executable(buddy-sampler-test
  SamplerTest.cpp
  LINK_LIBS
    buddy_runtime_core
)

_add_test_executable(buddy-perplexity-test
  PerplexityTest.cpp
  LINK_LIBS
    buddy_runtime_llm
)

_add_test_executable(buddy-text-generation-test
  TextGenerationTest.cpp
  LINK_LIBS
    buddy_runtime_llm
)

_add_test_executable(buddy-legacy-generation-test
  LegacyGenerationTest.cpp
  LINK_LIBS
    buddy_runtime_llm
)

# Compile the production resident model with a scripted ModelSession so these
# tests do not require generated model code or downloaded weights.
_add_test_executable(buddy-deepseek-resident-model-test
  DeepSeekR1ResidentModelTest.cpp
  ${BUDDY_SOURCE_DIR}/models/deepseek_r1/DeepSeekR1ResidentModel.cpp
  LINK_LIBS
    buddy_runtime_llm
    LLVMSupport
)
target_include_directories(buddy-deepseek-resident-model-test BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/deepseek-resident
  ${BUDDY_SOURCE_DIR}/models/deepseek_r1/include
)

_add_test_executable(buddy-serving-generation-test
  ServingGenerationTest.cpp
  LINK_LIBS
    buddy_runtime_llm
)

target_include_directories(buddy-serving-generation-test PRIVATE
  ${BUDDY_SOURCE_DIR}/runtime/llm
)
