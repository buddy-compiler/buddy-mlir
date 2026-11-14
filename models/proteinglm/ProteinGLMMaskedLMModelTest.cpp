//===- ProteinGLMMaskedLMModelTest.cpp - ProteinGLM masked-LM tests ------===//
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

#include "buddy/runtime/models/ProteinGLMMaskedLMModel.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
  buddy::runtime::ProteinGLMMaskedLMModel model;
  bool loadFailed = false;
  try {
    model.load(buddy::runtime::MaskedLMModelConfig{});
  } catch (const std::exception &) {
    loadFailed = true;
  }
  assert(loadFailed);
  assert(model.status().state == buddy::runtime::ModelLoadState::Error);

  bool predictFailed = false;
  try {
    model.predict(buddy::runtime::MaskedLMRequest{});
  } catch (const std::exception &) {
    predictFailed = true;
  }
  assert(predictFailed);
  std::cout << "ProteinGLM masked-LM runtime tests passed\n";
  return 0;
}
