//===- PaddleocrRunner.h - PaddleOCR-VL inference runner ------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_PADDLEOCRRUNNER_H
#define BUDDY_RUNTIME_MODELS_PADDLEOCRRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// PaddleOCR-VL single-shot OCR vision-language runner.
///
/// The compiled kernel performs the full OCR forward pass (SigLIP vision
/// encoder + projector + ERNIE decoder + LM head) in one call. The runner owns
/// weight loading, fixed-shape input construction (image + text), kernel
/// invocation, and emission of the last-token logits.
class PaddleocrRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_PADDLEOCRRUNNER_H
