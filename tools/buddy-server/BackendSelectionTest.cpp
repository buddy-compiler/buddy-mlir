//===- BackendSelectionTest.cpp - Backend selection tests ----------------===//
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

#include "BackendSelection.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace {

template <typename Function> void expectFailure(Function &&function) {
  bool failed = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  assert(failed);
}

} // namespace

int main() {
  using namespace buddy::server;
  BackendPluginPaths manifest;
  manifest.transcription = "/tmp/whisper_transcription.so";
  auto selected = selectBackend({}, manifest);
  assert(selected.kind == BackendKind::Transcription);
  assert(selected.pluginPath == manifest.transcription);

  BackendPluginPaths explicitPaths;
  explicitPaths.embedding = "/tmp/override.so";
  selected = selectBackend(explicitPaths, manifest);
  assert(selected.kind == BackendKind::Embedding);

  expectFailure([] { (void)selectBackend({}, {}); });
  BackendPluginPaths ambiguousManifest;
  ambiguousManifest.resident = "a.so";
  ambiguousManifest.transcription = "b.so";
  expectFailure([&] { (void)selectBackend({}, ambiguousManifest); });
  BackendPluginPaths ambiguousExplicit;
  ambiguousExplicit.maskedLM = "a.so";
  ambiguousExplicit.transcription = "b.so";
  expectFailure([&] { (void)selectBackend(ambiguousExplicit, {}); });

  buddy::runtime::ModelManifest parsed;
  parsed.transcriptionLibraryPath = "payload-extracted.so";
  assert(pluginPathsFromManifest(parsed).transcription ==
         "payload-extracted.so");
  std::cout << "BackendSelection tests passed\n";
  return 0;
}
