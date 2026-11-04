//===- PaligemmaRunner.h - PaliGemma vision-language runner ---------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_PALIGEMMARUNNER_H
#define BUDDY_RUNTIME_MODELS_PALIGEMMARUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// PaliGemma-3B-224 (google/paligemma-3b-mix-224) vision-language runner.
///
/// The compiled AOT kernel is the full VLM forward traced with a fixed
/// 1 x 3 x 224 x 224 zero image and a 280-token text sequence (256 <image>
/// tokens + 24 text tokens). The runner owns the deterministic input
/// construction (zero pixel_values, fixed token ids), weight loading, kernel
/// invocation via `_mlir_ciface_forward`, and emission of the resulting
/// logits / image features.
class PaligemmaRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_PALIGEMMARUNNER_H
