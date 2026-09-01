//===- MoLFormerRunner.h - MoLFormer molecular embedding runner ----------===//
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

#ifndef BUDDY_RUNTIME_MODELS_MOLFORMERRUNNER_H
#define BUDDY_RUNTIME_MODELS_MOLFORMERRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// MoLFormer chemistry Transformer encoder runner.
///
/// The compiled kernel is the full ibm/MoLFormer-XL-both-10pct encoder traced
/// as a single AOT graph (12-layer linear attention, deterministic random
/// Fourier features).  The runner owns SMILES tokenization (the checkpoint's
/// WordLevel tokenizer with the SMILES regex pre-tokenizer), weight loading,
/// kernel invocation and emission of the per-token hidden states plus the
/// pooled molecular embedding.
class MoLFormerRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_MOLFORMERRUNNER_H
