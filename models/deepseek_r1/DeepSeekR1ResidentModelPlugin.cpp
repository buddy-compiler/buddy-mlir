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

#include "buddy/runtime/core/ResidentModelPlugin.h"
#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"

extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v1() {
  return new buddy::runtime::DeepSeekR1ResidentModel();
}

extern "C" void
buddy_destroy_resident_model_v1(buddy::runtime::ResidentModel *model) {
  delete model;
}

extern "C" const char *buddy_resident_model_type_v1() { return "deepseek_r1"; }
