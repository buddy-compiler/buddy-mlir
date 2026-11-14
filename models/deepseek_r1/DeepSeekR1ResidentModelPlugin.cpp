//===- DeepSeekR1ResidentModelPlugin.cpp - DeepSeek R1 serving plugin -----===//
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

#include "buddy/runtime/core/ServerModelPluginAdapters.h"
#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"

namespace {
struct Factory {
  static std::unique_ptr<buddy::runtime::ResidentModel> create() {
    return std::make_unique<buddy::runtime::DeepSeekR1ResidentModel>();
  }
};
using Adapter = buddy::runtime::server_abi::ResidentServerModelAdapter<Factory>;
} // namespace

extern "C" const BuddyServerModelDescriptorV1 *
buddy_server_model_descriptor_v1() {
  static const BuddyServerModelDescriptorV1 descriptor = {
      sizeof(descriptor),
      BUDDY_SERVER_MODEL_ABI_VERSION_V1,
      BUDDY_SERVER_SERVICE_RESIDENT,
      (1ull << BUDDY_SERVER_OPERATION_COMPLETION) |
          (1ull << BUDDY_SERVER_OPERATION_CHAT) |
          (1ull << BUDDY_SERVER_OPERATION_TOKENIZE),
      BUDDY_SERVER_CAPABILITY_STREAMING,
      BUDDY_SERVER_MODALITY_TEXT,
      BUDDY_SERVER_RESULT_TEXT,
      1,
      "deepseek_r1",
      Adapter::ops(),
      0};
  return &descriptor;
}
