//===- ResidentModelFactory.h - Resident model factory ---------*- C++ -*-===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELFACTORY_H
#define BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELFACTORY_H

#include "buddy/runtime/core/ResidentModel.h"

#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace server {

std::unique_ptr<buddy::runtime::ResidentModel>
createResidentModel(const std::string &modelType);

std::vector<std::string> availableResidentModelTypes();

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_RESIDENTMODELFACTORY_H
