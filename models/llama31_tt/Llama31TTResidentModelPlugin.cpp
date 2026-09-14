//===- Llama31TTResidentModelPlugin.cpp - Llama 3.1 TT serving plugin -----===//
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

#include "buddy/runtime/core/ResidentModelPlugin.h"
#include "buddy/runtime/models/Llama31TTNNBackend.h"
#include "buddy/runtime/models/Llama31TTResidentModel.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

bool fakeExecutionRequested() {
  const char *value = std::getenv("BUDDY_LLAMA31_TT_FAKE_EXECUTION");
  if (!value)
    return false;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized == "1" || normalized == "true" || normalized == "yes" ||
         normalized == "on";
}

} // namespace

extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v1() {
  auto execution = std::make_shared<buddy::runtime::Llama31TTExecution>();
  // Fake execution is an explicit test mode. Production plugin instances
  // always construct the hardware session during ResidentModel::load().
  if (!fakeExecutionRequested()) {
    execution->setBackendFactory(
        [](const buddy::runtime::Llama31TTExecution::Metadata &metadata) {
          return buddy::runtime::createLlama31TTNNBackend(metadata);
        });
  }
  return new buddy::runtime::Llama31TTResidentModel(std::move(execution));
}

extern "C" void
buddy_destroy_resident_model_v1(buddy::runtime::ResidentModel *model) {
  delete model;
}

extern "C" const char *buddy_resident_model_type_v1() { return "llama31_tt"; }
