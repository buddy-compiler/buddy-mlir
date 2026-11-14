//===- Llama31TTBatchManifestTest.cpp - Batch manifest rejection test -----===//
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

#include "buddy/runtime/models/Llama31TTExecution.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef BUDDY_LLAMA31_TT_BATCH_TEST_RAX
#error "BUDDY_LLAMA31_TT_BATCH_TEST_RAX is required"
#endif

int main() {
  buddy::runtime::Llama31TTExecution execution;
  buddy::runtime::ResidentModelConfig config;
  config.raxPath = BUDDY_LLAMA31_TT_BATCH_TEST_RAX;
  try {
    execution.load(config);
  } catch (const std::exception &error) {
    assert(std::string(error.what())
               .find("supports only canonical batch_size=1") !=
           std::string::npos);
    std::cout << "Llama31TT batch manifest rejection passed\n";
    return 0;
  }
  assert(false && "batch manifest should be rejected");
}
