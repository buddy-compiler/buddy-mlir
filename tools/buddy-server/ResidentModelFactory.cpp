//===- ResidentModelFactory.cpp - Resident model factory ------------------===//
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

#include "ResidentModelFactory.h"

#ifdef BUDDY_SERVER_HAVE_DEEPSEEK_R1
#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"
#endif

#include <sstream>
#include <stdexcept>

namespace buddy {
namespace server {

namespace {

std::string joinTypes(const std::vector<std::string> &types) {
  std::ostringstream os;
  for (size_t i = 0; i < types.size(); ++i) {
    if (i)
      os << ", ";
    os << types[i];
  }
  return os.str();
}

} // namespace

std::vector<std::string> availableResidentModelTypes() {
  std::vector<std::string> types;
#ifdef BUDDY_SERVER_HAVE_DEEPSEEK_R1
  types.push_back("deepseek_r1");
#endif
  return types;
}

std::unique_ptr<buddy::runtime::ResidentModel>
createResidentModel(const std::string &modelType) {
#ifdef BUDDY_SERVER_HAVE_DEEPSEEK_R1
  if (modelType == "deepseek_r1")
    return std::make_unique<buddy::runtime::DeepSeekR1ResidentModel>();
#endif

  const auto types = availableResidentModelTypes();
  if (types.empty()) {
    throw std::runtime_error(
        "buddy-server: no built-in resident model backends are available");
  }

  throw std::runtime_error("buddy-server: unsupported resident model type '" +
                           modelType + "'. Available: " + joinTypes(types));
}

} // namespace server
} // namespace buddy
