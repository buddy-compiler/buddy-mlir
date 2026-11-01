//===- ServingGeneration.h - Compatible serving generation ---------------===//
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

#ifndef BUDDY_RUNTIME_LLM_SERVINGGENERATION_H
#define BUDDY_RUNTIME_LLM_SERVINGGENERATION_H

#include "buddy/runtime/core/ServingTypes.h"
#include "buddy/runtime/llm/LLMSession.h"

#include <functional>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Model-specific encoding and raw token bytes. Encoding must preserve the
/// complete prompt, including any model-required initial token, and must not
/// truncate it or apply a chat template. tokenBytes must not trim whitespace,
/// repair UTF-8, or stop at special tokens.
struct ServingTextCodec {
  std::function<std::vector<int>(const std::string &)> encode;
  std::function<std::string(int)> tokenBytes;
  int contextLength = 0;
  /// Decodable prefix of the token source vocabulary; zero uses its full
  /// vocabulary. A positive value excludes padding logits from sampling in
  /// the LLMSession overload.
  int vocabSize = 0;
};

/// Model execution and token selection, including backends that return a
/// device-selected token without exposing logits. All callbacks are required.
/// prefill receives the complete encoded prompt; decode consumes the previous
/// selected token. selectToken sees the prompt and all previously selected
/// tokens, and must return the next token from the latest execution result.
/// validToken identifies decodable IDs for prompt, stop, and generated tokens.
/// The adapter must reject sampling options its selectToken cannot implement.
struct ServingTokenSource {
  std::function<void()> reset;
  std::function<void(const std::vector<int> &)> prefill;
  std::function<void(int)> decode;
  std::function<int(const std::vector<int> &)> selectToken;
  std::function<bool(int)> validToken;
};

/// Generate with the serving token budget, stopping, seed, and usage contract.
/// The adapter merges model/template stop IDs into sampling.stopTokenIds.
/// Each request resets the source before prefill; cancellation and exceptions
/// also reset it before returning. No context shifting or prompt truncation is
/// performed. Empty encoded prompts are rejected because the source requires
/// an input token to produce the first prediction.
///
/// Chunks/results leave id and model empty for the adapter. A callback receives
/// deltas followed by one done summary, including for prefill-only requests.
/// Returning false cancels immediately without further callbacks. Exceptions
/// propagate to the transport after source cleanup.
CompletionResult runServingGeneration(
    const std::string &prompt, const ServingTokenSource &source,
    const SamplingParams &sampling, const ServingTextCodec &codec,
    const CompletionStreamCallback &callback = {});

/// Wrap an LLMSession with host sampling, padded prefill, and vocabulary and
/// position checks, using the same generation contract as a token source.
CompletionResult
runServingGeneration(const std::string &prompt, LLMSession &session,
                     const SamplingParams &sampling,
                     const ServingTextCodec &codec,
                     const CompletionStreamCallback &callback = {});

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_SERVINGGENERATION_H
