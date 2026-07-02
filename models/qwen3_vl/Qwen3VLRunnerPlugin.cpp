//===- Qwen3VLRunnerPlugin.cpp - Qwen3-VL runner plugin -------------------===//
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
//
// C-ABI boundary: buddy-cli dlsym's these two symbols to construct/destroy the
// runner. Fixed template (see models/deepseek_r1/DeepSeekR1RunnerPlugin.cpp).
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Qwen3VLRunner.h"

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new buddy::runtime::Qwen3VLRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}
