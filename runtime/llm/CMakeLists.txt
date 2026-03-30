# Model-agnostic LLM runtime utilities: text generation loop, interactive REPL.
add_library(buddy_runtime_llm STATIC
  TextGeneration.cpp
  InteractiveSession.cpp
)

target_include_directories(buddy_runtime_llm PUBLIC
  ${BUDDY_SOURCE_DIR}/runtime/include
  ${BUDDY_SOURCE_DIR}/frontend/Interfaces
  ${BUDDY_BINARY_DIR}/frontend/Interfaces
)

target_link_libraries(buddy_runtime_llm PUBLIC
  buddy_runtime_core
)

target_compile_features(buddy_runtime_llm PUBLIC cxx_std_17)
