//===- ModelPluginDiscoveryTest.cpp - Discovery tests ---------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "ModelPluginDiscovery.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expectFailure(const std::string &path, const std::string &needle) {
  try {
    (void)buddy::server::discoverModelPlugin(path);
    assert(false && "expected plugin discovery failure");
  } catch (const std::exception &error) {
    assert(std::string(error.what()).find(needle) != std::string::npos);
  }
}

} // namespace

int main() {
  using buddy::runtime::ModelServiceKind;
  expectFailure("", "model_library path is empty");

  const auto resident =
      buddy::server::discoverModelPlugin(BUDDY_DISCOVERY_RESIDENT_PLUGIN);
  assert(resident.serviceKind == ModelServiceKind::Resident);
  const auto embedding =
      buddy::server::discoverModelPlugin(BUDDY_DISCOVERY_EMBEDDING_PLUGIN);
  assert(embedding.serviceKind == ModelServiceKind::Embedding);
  const auto maskedLM =
      buddy::server::discoverModelPlugin(BUDDY_DISCOVERY_MASKED_LM_PLUGIN);
  assert(maskedLM.serviceKind == ModelServiceKind::MaskedLM);
  const auto transcription =
      buddy::server::discoverModelPlugin(BUDDY_DISCOVERY_TRANSCRIPTION_PLUGIN);
  assert(transcription.serviceKind == ModelServiceKind::Transcription);

  expectFailure(BUDDY_DISCOVERY_MISSING_PLUGIN, "missing");
  expectFailure(BUDDY_DISCOVERY_NULL_PLUGIN, "null descriptor");
  expectFailure(BUDDY_DISCOVERY_VERSION_PLUGIN, "ABI version");
  expectFailure(BUDDY_DISCOVERY_SMALL_PLUGIN, "structSize");
  expectFailure(BUDDY_DISCOVERY_RESERVED_PLUGIN, "reserved field");
  expectFailure(BUDDY_DISCOVERY_UNKNOWN_PLUGIN, "unknown service kind");
  std::cout << "ModelPluginDiscovery tests passed\n";
}
