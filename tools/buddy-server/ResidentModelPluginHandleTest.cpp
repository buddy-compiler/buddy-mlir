//===- ResidentModelPluginHandleTest.cpp - Resident ABI loader tests ------===//
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

#include "ResidentModelPluginHandle.h"
#include "ModelPluginDiscovery.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> std::string expectFailure(Function &&function) {
  try {
    function();
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  throw std::runtime_error("expected the resident plugin operation to fail");
}

class CallbackMarker {
public:
  CallbackMarker() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "buddy-resident-abi-XXXXXX")
            .string();
    char *directory = mkdtemp(pattern.data());
    require(directory, "could not create resident ABI test directory");
    marker = std::filesystem::path(directory) / "callbacks";
    if (const char *previous = std::getenv("BUDDY_RESIDENT_ABI_TEST_MARKER")) {
      hadPrevious = true;
      previousValue = previous;
    }
    require(setenv("BUDDY_RESIDENT_ABI_TEST_MARKER", marker.c_str(), 1) == 0,
            "could not configure resident ABI test marker");
  }

  ~CallbackMarker() {
    if (hadPrevious)
      setenv("BUDDY_RESIDENT_ABI_TEST_MARKER", previousValue.c_str(), 1);
    else
      unsetenv("BUDDY_RESIDENT_ABI_TEST_MARKER");
    std::error_code error;
    std::filesystem::remove_all(marker.parent_path(), error);
  }

  std::string calls() const {
    std::ifstream input(marker);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

private:
  std::filesystem::path marker;
  bool hadPrevious = false;
  std::string previousValue;
};

void checkDiscovery(const char *path) {
  auto discovery = buddy::server::discoverModelPlugin(path);
  require(discovery.serviceKind == buddy::runtime::ModelServiceKind::Resident,
          "resident discovery must retain its v1 descriptor");
}

void checkRejectedPlugin(const char *path, const char *missingSymbol) {
  CallbackMarker marker;
  checkDiscovery(path);
  require(marker.calls().empty(), "discovery called a resident entry point");
  std::string message = expectFailure(
      [&] { buddy::server::ResidentModelPluginHandle handle(path); });
  require(message.find(missingSymbol) != std::string::npos,
          "missing-symbol error must name the required v2 export: " + message);
  require(message.find("Rebuild") != std::string::npos &&
              message.find("repackage") != std::string::npos &&
              message.find(".rax") != std::string::npos,
          "incompatible plugins need rebuild and repackage advice: " + message);
  require(marker.calls().empty(),
          "loader called a resident entry point before validating all exports");
}

void checkValidPlugin() {
  CallbackMarker marker;
  checkDiscovery(BUDDY_TEST_RESIDENT_V2_PLUGIN);
  buddy::server::ResidentModelPluginHandle handle(
      BUDDY_TEST_RESIDENT_V2_PLUGIN);
  require(handle.modelType() == "fake_resident_v2", "wrong v2 model type");
  require(marker.calls() == "type_v2\n", "constructor called the factory");
  auto model = handle.createModel();
  require(model->status().state == buddy::runtime::ModelLoadState::Unloaded,
          "unexpected initial model state");
  const auto capabilities = model->capabilities();
  require(!capabilities.rawTokenize && !capabilities.rawDetokenize &&
              !capabilities.tokenizeAddSpecial &&
              !capabilities.tokenizeParseSpecial &&
              !capabilities.tokenizeWithPieces,
          "default v2 codec capabilities must be conservative");
  require(capabilities.chatRoles.empty() && !capabilities.chatMultiTurn &&
              capabilities.maxImages == 0 && !capabilities.imagePaths &&
              !capabilities.imageBytes && !capabilities.streaming &&
              !capabilities.maxNewTokens && !capabilities.stop &&
              !capabilities.stopTokenIds,
          "default v2 serving capabilities must be conservative");
  const auto sampling = capabilities.sampling;
  require(!sampling.temperature && !sampling.topK && !sampling.topP &&
              !sampling.minP && !sampling.repeatPenalty &&
              !sampling.repeatLastN && !sampling.seed,
          "default v2 sampling capabilities must be conservative");
  buddy::runtime::ResidentModelConfig config;
  config.modelName = "resident-abi-test";
  model->load(config);
  require(model->isLoaded() && model->status().modelName == config.modelName &&
              model->status().backend == "abi-test",
          "v2 model status dispatch failed");
  const auto loadedCapabilities = model->capabilities();
  require(loadedCapabilities.rawDetokenize &&
              loadedCapabilities.chatRoles ==
                  std::vector<std::string>({"user"}),
          "v2 capability values did not cross the plugin boundary");
  buddy::runtime::DetokenizeRequest request;
  request.tokens = {7, 9};
  require(model->detokenize(request).content == "7,9",
          "v2 detokenize dispatch failed");
  require(model->detokenize({}).content.empty(),
          "v2 detokenize lost the empty request");
  require(model->tokenize({}).tokens == std::vector<int>({7, 9}),
          "existing tokenizer dispatch changed");
  require(marker.calls() == "type_v2\ncreate_v2\n",
          "model was destroyed before releasing ownership");
  model.reset();
  require(marker.calls() == "type_v2\ncreate_v2\ndestroy_v2\n",
          "model did not use the matching v2 destroy export");
}

void checkNullFactory() {
  CallbackMarker marker;
  buddy::server::ResidentModelPluginHandle handle(
      BUDDY_TEST_RESIDENT_NULL_PLUGIN);
  std::string message = expectFailure([&] { (void)handle.createModel(); });
  require(message.find("null model") != std::string::npos,
          "null factory error lost its diagnostic: " + message);
  require(marker.calls() == "type_v2\ncreate_v2\n",
          "null factory result must not be passed to destroy");
}

void checkInvalidType(const char *path) {
  CallbackMarker marker;
  checkDiscovery(path);
  std::string message = expectFailure(
      [&] { buddy::server::ResidentModelPluginHandle handle(path); });
  require(message.find("resident model type is empty") != std::string::npos,
          "empty type error lost its diagnostic: " + message);
  require(marker.calls() == "type_v2\n",
          "invalid type must be rejected before calling the factory");
}

} // namespace

int main() {
  try {
    (void)expectFailure(
        [] { buddy::server::ResidentModelPluginHandle handle(""); });
    (void)expectFailure([] {
      buddy::server::ResidentModelPluginHandle handle(
          "/definitely/missing/resident_plugin.so");
    });
    checkValidPlugin();
    checkRejectedPlugin(BUDDY_TEST_RESIDENT_V1_PLUGIN,
                        "buddy_create_resident_model_v2");
    checkRejectedPlugin(BUDDY_TEST_RESIDENT_MISSING_CREATE_PLUGIN,
                        "buddy_create_resident_model_v2");
    checkRejectedPlugin(BUDDY_TEST_RESIDENT_MISSING_DESTROY_PLUGIN,
                        "buddy_destroy_resident_model_v2");
    checkRejectedPlugin(BUDDY_TEST_RESIDENT_MISSING_TYPE_PLUGIN,
                        "buddy_resident_model_type_v2");
    checkNullFactory();
    checkInvalidType(BUDDY_TEST_RESIDENT_NULL_TYPE_PLUGIN);
    checkInvalidType(BUDDY_TEST_RESIDENT_EMPTY_TYPE_PLUGIN);
    std::cout << "ResidentModelPluginHandle tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ResidentModelPluginHandle test failed: " << error.what()
              << '\n';
    return 1;
  }
}
