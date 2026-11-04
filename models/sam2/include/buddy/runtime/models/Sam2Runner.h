//===- Sam2Runner.h - SAM2-hiera-tiny vision encoder runner ---------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_SAM2RUNNER_H
#define BUDDY_RUNTIME_MODELS_SAM2RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// SAM2-hiera-tiny image-encoder runner.
///
/// Runs the AOT-compiled Sam2VisionModel forward graph over a fixed-shape
/// `1 x 3 x 256 x 256` f32 image tensor and emits the resulting `1 x 8 x 8 x
/// 768` image feature map (plus the six FPN feature maps). See Sam2Runner.cpp
/// for the exact `_mlir_ciface_forward` ABI.
class Sam2Runner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_SAM2RUNNER_H
