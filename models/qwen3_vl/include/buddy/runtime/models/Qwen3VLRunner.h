//===- Qwen3VLRunner.h - Qwen3-VL inference runner ------------------------===//
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
// Qwen3-VL is a vision-language model: a ViT vision encoder (with DeepStack
// multi-level feature injection) + patch merger feeding a dense Qwen3 text
// decoder that uses interleaved MRoPE. It is integrated as a self-contained
// InferenceRunner plugin (mirroring the models/whisper design), reusing the
// existing Qwen3 text-decoder compile path where possible.
//
// NOTE: run() is a stub during the feasibility-spike phase. The full multimodal
// loop (image preprocess -> ViT encode + deepstack -> embedding splice -> MRoPE
// position ids -> KV-cache decode) is implemented after the vision and decoder
// compile spikes pass; see models/qwen3_vl/codegen/ and SPIKE.md.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_QWEN3VLRUNNER_H
#define BUDDY_RUNTIME_MODELS_QWEN3VLRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Full inference runner for Qwen3-VL.
///
/// Implements the complete multimodal loop: image preprocess -> vision encode
/// (ViT + DeepStack) -> splice image embeddings into the text sequence ->
/// MRoPE position ids -> autoregressive decode.
class Qwen3VLRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_QWEN3VLRUNNER_H
