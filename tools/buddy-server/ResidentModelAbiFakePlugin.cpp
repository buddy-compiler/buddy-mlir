//===- ResidentModelAbiFakePlugin.cpp - Resident ABI test fixtures --------===//
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

#include "buddy/runtime/core/ModelPlugin.h"

#include <cstdio>
#include <cstdlib>

#if !defined(BUDDY_RESIDENT_ABI_LEGACY)
#include "buddy/runtime/core/ResidentModelPlugin.h"
#endif

namespace {

void recordCall(const char *name) {
  if (const char *marker = std::getenv("BUDDY_RESIDENT_ABI_TEST_MARKER")) {
    if (FILE *file = std::fopen(marker, "a")) {
      std::fputs(name, file);
      std::fclose(file);
    }
  }
}

#if !defined(BUDDY_RESIDENT_ABI_LEGACY)
class FakeModel final : public buddy::runtime::ResidentModel {
public:
  void load(const buddy::runtime::ResidentModelConfig &config) override {
    statusValue.state = buddy::runtime::ModelLoadState::Ready;
    statusValue.modelName = config.modelName;
    statusValue.backend = "abi-test";
  }

  buddy::runtime::ModelStatus status() const override { return statusValue; }

  buddy::runtime::ResidentModelCapabilities capabilities() const override {
    buddy::runtime::ResidentModelCapabilities capabilities;
    if (statusValue.state == buddy::runtime::ModelLoadState::Ready) {
      capabilities.rawDetokenize = true;
      capabilities.chatRoles = {"user"};
    }
    return capabilities;
  }

  std::string
  renderChat(const buddy::runtime::ChatCompletionRequest &request) override {
    return request.input;
  }

  buddy::runtime::TokenizeResult
  tokenize(const buddy::runtime::TokenizeRequest &) override {
    return {{7, 9}, 2, {}};
  }

  buddy::runtime::DetokenizeResult
  detokenize(const buddy::runtime::DetokenizeRequest &request) override {
    buddy::runtime::DetokenizeResult result;
    for (int token : request.tokens) {
      if (!result.content.empty())
        result.content += ',';
      result.content += std::to_string(token);
    }
    return result;
  }

  buddy::runtime::CompletionResult
  complete(const buddy::runtime::CompletionRequest &) override {
    return {};
  }

  buddy::runtime::CompletionResult
  completeStream(const buddy::runtime::CompletionRequest &,
                 const buddy::runtime::CompletionStreamCallback &) override {
    return {};
  }

  buddy::runtime::CompletionResult
  chat(const buddy::runtime::ChatCompletionRequest &) override {
    return {};
  }

  buddy::runtime::CompletionResult
  chatStream(const buddy::runtime::ChatCompletionRequest &,
             const buddy::runtime::CompletionStreamCallback &) override {
    return {};
  }

private:
  buddy::runtime::ModelStatus statusValue;
};
#endif

} // namespace

#if defined(BUDDY_RESIDENT_ABI_LEGACY)
// Deliberately independent of ResidentModel.h: a legacy binary has no v2 C++
// interface. Abort if any old entry point is called, before an old vtable could
// be accessed through the new interface.
extern "C" void *buddy_create_resident_model_v1() {
  recordCall("create_v1\n");
  std::abort();
}

extern "C" void buddy_destroy_resident_model_v1(void *) {
  recordCall("destroy_v1\n");
  std::abort();
}

extern "C" const char *buddy_resident_model_type_v1() {
  recordCall("type_v1\n");
  std::abort();
}
#else
#if !defined(BUDDY_RESIDENT_ABI_MISSING_CREATE)
extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v2() {
  recordCall("create_v2\n");
#if defined(BUDDY_RESIDENT_ABI_NULL_FACTORY)
  return nullptr;
#else
  return new FakeModel();
#endif
}
#endif

#if !defined(BUDDY_RESIDENT_ABI_MISSING_DESTROY)
extern "C" void
buddy_destroy_resident_model_v2(buddy::runtime::ResidentModel *model) {
  recordCall("destroy_v2\n");
  delete model;
}
#endif

#if !defined(BUDDY_RESIDENT_ABI_MISSING_TYPE)
extern "C" const char *buddy_resident_model_type_v2() {
  recordCall("type_v2\n");
#if defined(BUDDY_RESIDENT_ABI_NULL_TYPE)
  return nullptr;
#elif defined(BUDDY_RESIDENT_ABI_EMPTY_TYPE)
  return "";
#else
  return "fake_resident_v2";
#endif
}
#endif
#endif

extern "C" const buddy::runtime::ModelPluginDescriptorV1 *
buddy_model_plugin_descriptor_v1() {
  static constexpr buddy::runtime::ModelPluginDescriptorV1 descriptor = {
      sizeof(descriptor), buddy::runtime::ModelPluginDescriptorAbiVersionV1,
      buddy::runtime::ModelServiceKind::Resident, 0};
  return &descriptor;
}
