# ===- deps.cmake - Header-only third-party dependencies --------------------===//
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
# Fetches the header-only third-party libraries used by buddy-mlir. Enabled by
# default; set BUDDY_DOWNLOAD_<NAME> to OFF for offline builds and provide the
# headers on the include path instead.
#
# ===----------------------------------------------------------------------===//

include(FetchContent)

# ── jsoncons: JSON parsing for the BGE-M3 tokenizer ──────────────────────────
# Used instead of llvm::json so cross-compiled model plugins do not need to
# link LLVM.
option(BUDDY_DOWNLOAD_JSONCONS "Download the jsoncons JSON library" ON)
if(BUDDY_DOWNLOAD_JSONCONS)
  set(JSONCONS_BUILD_TESTS OFF)
  FetchContent_Declare(
    jsoncons
    URL https://github.com/danielaparker/jsoncons/archive/refs/tags/v1.8.1.tar.gz
    URL_HASH SHA3_256=c1f7957049ce756005ce67917ce8b6f09c0cf56e630664edeb272365229baada
  )
  FetchContent_MakeAvailable(jsoncons)
  set(BUDDY_JSONCONS_INCLUDE_DIR "${jsoncons_SOURCE_DIR}/include")
  include_directories(${BUDDY_JSONCONS_INCLUDE_DIR})
else()
  message(STATUS "jsoncons download disabled; expecting its headers on the include path")
endif()

# ── CLI11: command line parsing for the buddy tools ──────────────────────────
option(BUDDY_DOWNLOAD_CLI11 "Download the CLI11 command line parser" ON)
if(BUDDY_DOWNLOAD_CLI11)
  FetchContent_Declare(
    cli11_hpp
    URL https://github.com/CLIUtils/CLI11/releases/download/v2.6.2/CLI11.hpp
    DOWNLOAD_NO_EXTRACT YES
    URL_HASH SHA256=227a16fe5f9f8ada80c3c409492475536f597e7bd83a6c26eacc3c8c149a9295
  )
  FetchContent_MakeAvailable(cli11_hpp)
  add_library(CLI11 INTERFACE)
  target_include_directories(CLI11 SYSTEM INTERFACE "${cli11_hpp_SOURCE_DIR}")
else()
  add_library(CLI11 INTERFACE)
  message(STATUS "CLI11 download disabled; expecting CLI11.hpp on the include path")
endif()
