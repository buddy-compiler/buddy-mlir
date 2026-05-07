# ===- CMakeLists.txt - Llama 3.1 TT runner ------------------*- cmake -*-===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
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
