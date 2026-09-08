//===- Qwen3Runner.h - Qwen3 inference runner -----------------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H
#define BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Full inference runner for Qwen3 (Qwen/Qwen3-0.6B).
///
/// Implements the complete loop: load weights → tokenize → prefill → decode.
/// Handles both Mode A (.rax manifest) and Mode B (explicit paths).
class Qwen3Runner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H
