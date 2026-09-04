//===- Smolvlm2Runner.h - SmolVLM2 text-LM runner -------------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_SMOLVLM2RUNNER_H
#define BUDDY_RUNTIME_MODELS_SMOLVLM2RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// SmolVLM2 text-only LM runner.
///
/// The compiled kernel produces the text last_hidden_state (1 x max_seq_len x
/// hidden_size) for a fixed 64-token input. The runner owns byte-level BPE
/// tokenization (tokenizer.json), weight loading, kernel invocation, and
/// emission of the final-token hidden-state vector.
class Smolvlm2Runner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_SMOLVLM2RUNNER_H
