//===- ModelManifestTranscriptionTest.cpp - Transcription URI tests ------===//
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

#include "buddy/runtime/core/ModelManifest.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  assert(argc == 3);
  const auto fileManifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
  assert(fileManifest.modelName == "fake_transcription");
  assert(
      std::filesystem::is_regular_file(fileManifest.transcriptionLibraryPath));
  assert(
      std::filesystem::path(fileManifest.transcriptionLibraryPath).filename() ==
      "transcription.so");
  assert(std::filesystem::is_regular_file(fileManifest.runnerLibraryPath));

  const auto payloadManifest =
      buddy::runtime::ModelManifest::loadFromRax(argv[2]);
  assert(std::filesystem::is_regular_file(
      payloadManifest.transcriptionLibraryPath));
  assert(std::filesystem::path(payloadManifest.transcriptionLibraryPath)
             .filename() == "transcription.so");
  assert(payloadManifest.transcriptionLibraryPath !=
         fileManifest.transcriptionLibraryPath);
  assert(payloadManifest.moduleAttrs.at("transcription_library") ==
         "payload:transcription.so");
  assert(payloadManifest.resolvedModuleAttrs.at("transcription_library") ==
         payloadManifest.transcriptionLibraryPath);
  std::cout << "ModelManifest transcription tests passed\n";
  return 0;
}
