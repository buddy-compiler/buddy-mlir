//===- EmbeddinggemmaRunner.h - EmbeddingGemma sentence embedding runner -===//
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

#ifndef BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMARUNNER_H
#define BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMARUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// google/embeddinggemma-300m sentence embedding runner.
///
/// The compiled kernel is the SentenceTransformer pipeline (Gemma3TextModel ->
/// mean pooling -> two dense layers -> L2 normalize) traced as a single AOT
/// graph.  The runner owns Gemma byte-level BPE tokenization
/// (EmbeddinggemmaTokenizer.h), weight loading, kernel invocation and the
/// emission of the 768-dim embedding vector.
class EmbeddinggemmaRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMARUNNER_H
