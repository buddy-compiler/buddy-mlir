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
