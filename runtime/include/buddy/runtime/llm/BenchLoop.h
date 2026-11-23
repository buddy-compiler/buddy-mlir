//===- BenchLoop.h - Model-agnostic benchmark loop ------------------------===//
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
// Prefill/decode benchmark loop shared by every autoregressive runner.
// A runner only supplies the model-specific parts (text codec, stop tokens,
// padding token and a loaded session); the timing boundaries follow
// llama-bench: tokenization, sampling and detokenization are excluded from
// the reported prefill and decode times.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_BENCHLOOP_H
#define BUDDY_RUNTIME_LLM_BENCHLOOP_H

#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/llm/LLMSession.h"
#include "buddy/runtime/llm/Sampler.h"
#include "buddy/runtime/llm/TextGeneration.h"

#include <functional>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Returns the token positions the last prefill() actually computed. Models
/// whose prefill rounds or pads the prompt (e.g. tiered KV caches) supply this
/// so throughput is divided by the work done, not the requested length.
/// An empty callback means "same as the token count passed to prefill()".
/// This is a callback instead of an LLMSession method because LLMSession's
/// vtable is shared with sessions compiled into already-shipped .rax packages.
using PrefillWorkLenFn = std::function<int()>;

/// One timed benchmark pass (prefill + decode) with llama-bench semantics:
///   promptLength ==  0  -> tokenize cfg.prompt, then decode maxNewTokens
///   promptLength == -1  -> decode-only (tg): prefill 1 seed token
///   promptLength  >  0  -> fixed prompt length (truncate/pad to that count)
/// `padTokenId` fills a prompt shorter than the requested length; callers
/// must pass a benign token so the first sampled token is not a stop token.
BenchmarkResult runBenchPass(LLMSession &session, const std::string &vocabPath,
                             const RunConfig &cfg, Sampler &sampler,
                             const std::vector<long long> &stopTokenIds,
                             const TextCodec &codec, int padTokenId,
                             const PrefillWorkLenFn &prefillWorkLen);

/// Run `warmupReps` unmeasured passes followed by `reps` measured ones, all on
/// the same session, and store the measured passes in `out`. Warmup passes
/// warm the allocator, thread pool and caches the measurement uses, but are
/// never included in the results.
void runBenchLoop(LLMSession &session, const std::string &vocabPath,
                  const RunConfig &cfg, Sampler &sampler,
                  const std::vector<long long> &stopTokenIds,
                  const TextCodec &codec, int padTokenId,
                  const PrefillWorkLenFn &prefillWorkLen, int warmupReps,
                  int reps, std::vector<BenchmarkResult> &out);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_BENCHLOOP_H
