//===- ModelManifestDuplicateAttributeTest.cpp - Duplicate attr test -----===//
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
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  assert(argc == 2 && "expected an output directory");
  namespace fs = std::filesystem;
  using namespace rhal::rax;

  const fs::path outputDir = argv[1];
  fs::create_directories(outputDir);
  const fs::path raxPath = outputDir / "duplicate-model-library.rax";

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<KV>> attrs = {
      CreateKVDirect(builder, "model_library", "file:first.so"),
      CreateKVDirect(builder, "model_library", "file:second.so")};
  std::vector<flatbuffers::Offset<CodeObject>> codeObjects = {
      CreateCodeObjectDirect(builder, 1, "model", CodeObjectKind_HostSharedLib,
                             nullptr, "file:model.so", nullptr, "cpu")};
  const auto module = CreateModule(
      builder, builder.CreateString("RAX"), CreateVersion(builder, 0, 1, 0),
      Endianness_Little, builder.CreateVector(attrs), 0, 0, 0,
      builder.CreateVector(codeObjects), 0);
  builder.Finish(module, "RAX0");

  std::ofstream output(raxPath, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
               static_cast<std::streamsize>(builder.GetSize()));
  output.close();
  assert(output && "failed to write duplicate attribute fixture");

  try {
    (void)buddy::runtime::ModelManifest::loadFromRax(raxPath.string());
    assert(false && "duplicate model_library must be rejected");
  } catch (const std::runtime_error &error) {
    assert(std::string(error.what()).find("duplicate model_library") !=
           std::string::npos);
  }

  std::cout << "ModelManifest duplicate attribute test passed\n";
  return 0;
}
