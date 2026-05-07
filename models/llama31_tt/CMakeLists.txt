# Llama 3.1 Tenstorrent runner.
#
# This target intentionally has no compile-time dependency on tt-metal/ttrt.
# The first serving integration step launches the Python ttrt runner described
# by the .rax manifest, so buddy-cli can dispatch Tenstorrent artifacts without
# forcing every Buddy build to link Tenstorrent runtime libraries.

add_library(buddy_models_llama31_tt
  Llama31TTRunner.cpp
)

target_compile_features(buddy_models_llama31_tt PRIVATE cxx_std_17)

target_include_directories(buddy_models_llama31_tt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(buddy_models_llama31_tt
  PUBLIC
    buddy_runtime_core
    buddy_runtime_llm
)
