//===- AudioTranscriptionModelPluginHandleTest.cpp - Loader tests --------===//
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

#include "AudioTranscriptionModelPluginHandle.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef BUDDY_TEST_TRANSCRIPTION_PLUGIN
#error "BUDDY_TEST_TRANSCRIPTION_PLUGIN is required"
#endif
#ifndef BUDDY_TEST_TRANSCRIPTION_MISSING_PLUGIN
#error "BUDDY_TEST_TRANSCRIPTION_MISSING_PLUGIN is required"
#endif
#ifndef BUDDY_TEST_TRANSCRIPTION_NULL_PLUGIN
#error "BUDDY_TEST_TRANSCRIPTION_NULL_PLUGIN is required"
#endif

namespace {

template <typename Function> void expectFailure(Function &&function) {
  bool failed = false;
  try {
    function();
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
}

} // namespace

int main() {
  expectFailure([] { buddy::server::AudioTranscriptionModelPluginHandle(""); });
  expectFailure([] {
    buddy::server::AudioTranscriptionModelPluginHandle(
        "/definitely/missing/transcription_plugin.so");
  });
  expectFailure([] {
    buddy::server::AudioTranscriptionModelPluginHandle(
        BUDDY_TEST_TRANSCRIPTION_MISSING_PLUGIN);
  });
  expectFailure([] {
    buddy::server::AudioTranscriptionModelPluginHandle handle(
        BUDDY_TEST_TRANSCRIPTION_NULL_PLUGIN);
    (void)handle.createModel();
  });

  const std::filesystem::path marker =
      std::filesystem::temp_directory_path() /
      "buddy-transcription-plugin-destroy.marker";
  std::filesystem::remove(marker);
  setenv("BUDDY_TRANSCRIPTION_DESTROY_MARKER", marker.c_str(), 1);

  buddy::server::AudioTranscriptionModelPluginHandle::ModelPtr model(
      nullptr, [](buddy::runtime::AudioTranscriptionModel *) {});
  {
    buddy::server::AudioTranscriptionModelPluginHandle handle(
        BUDDY_TEST_TRANSCRIPTION_PLUGIN);
    assert(handle.modelType() == "fake_transcription");
    model = handle.createModel();
    assert(model);
  }
  assert(!std::filesystem::exists(marker));
  model.reset();
  assert(std::filesystem::exists(marker));
  std::filesystem::remove(marker);

  std::cout << "AudioTranscriptionModelPluginHandle tests passed\n";
  return 0;
}
