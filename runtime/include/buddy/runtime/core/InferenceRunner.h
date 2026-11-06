//===- InferenceRunner.h - Generic model inference interface --------------===//
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
// Source tree: buddy-mlir/runtime/include/buddy/runtime/core/InferenceRunner.h
// Include as:  #include "buddy/runtime/core/InferenceRunner.h"
//
// InferenceRunner is the single extension point for buddy-cli. Each supported
// model implements a subclass and exports it from a runner plugin shared
// library through the C ABI declared below.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_INFERENCERUNNER_H
#define BUDDY_RUNTIME_CORE_INFERENCERUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "buddy/runtime/llm/Sampler.h"

namespace buddy {
namespace runtime {

/// Configuration passed from buddy-cli to the runner.
struct RunConfig {
  // Mode A: single manifest (all paths resolved from .rax)
  std::string raxPath;

  // Mode B: explicit paths (legacy / no-rax fallback)
  std::string modelSoPath;
  std::string weightsPath;
  std::string vocabPath;

  std::string prompt;
  /// Optional batch prompts. When set, runners may use one prompt per batch
  /// element instead of broadcasting `prompt`.
  std::vector<std::string> prompts;

  /// Path to an audio file (e.g. .wav) for speech models such as Whisper.
  /// Ignored by text-only LLM runners. Empty = let the runner pick a default.
  std::string audioPath;

  /// Optional fixed prompt/prefill length. 0 means use the encoded prompt
  /// length for single-prompt runs, or the longest encoded prompt for batched
  /// prompt-file runs. -1 is a benchmark-only hint for decode-only (tg) mode:
  /// prefill a single seed token and measure the decode loop only. Values > 0
  /// fix the prompt length to exactly that many tokens.
  int promptLength = 0;

  /// Upper bound on generated tokens. The first generated token comes from
  /// prefill; subsequent tokens come from decode. 0 = no limit.
  int maxNewTokens = 4096;

  /// Runtime batch size override. 0 means use the packaged model default.
  int batchSize = 0;

  /// Optional image input for vision-language models (ignored by text models).
  /// May be an image path or a pre-processed pixel_values blob, per the runner.
  std::string imagePath;

  // ── Sampling configuration ──
  buddy::SamplerConfig samplerConfig;

  // ── Chat template ──
  /// Path to chat template JSON config file (empty = disabled).
  std::string chatTemplatePath;

  // ── Output control ──
  /// Suppress performance statistics output.
  bool suppressStats = false;

  /// Print every batch element's prompt/output instead of only user0.
  bool printAllBatchOutputs = false;

  /// Defer device-side token-id readback until the end of the decode loop when
  /// the runner can feed each decode token tensor directly into the next step.
  bool deferDecodeTokenReadback = false;

  /// Emit machine-readable token events as JSON Lines on stdout.
  bool streamJsonl = false;

  /// Benchmark-only: keep decoding for the full `maxNewTokens` budget even if
  /// the model emits an end-of-sequence token. llama-bench's --ignore-eos.
  bool ignoreEos = false;

  // ── Interactive mode ──
  /// Enable REPL interactive mode for multi-turn conversation.
  bool interactive = false;

  // ── Perplexity mode ──
  /// File of token ids (e.g. the output of `llama-tokenize --ids`). When set,
  /// LLM runners that support it report the perplexity of the stream instead
  /// of generating text (buddy/runtime/llm/Perplexity.h).
  std::string perplexityIdsPath;
  /// Tokens per perplexity chunk.
  int perplexityContext = 512;
  /// At most this many chunks (0: all).
  int perplexityChunks = 0;
};

/// Raw timings for a single benchmark pass (one sample).
struct BenchmarkResult {
  int nPrompt = 0; ///< prefill token count for this pass
  int nGen = 0;    ///< generated (decode) token count for this pass
  /// Token positions the prefill actually computed. Runners whose prefill
  /// pads/rounds the prompt (e.g. tiered KV caches) set this so throughput is
  /// divided by the work done, not by the requested prompt length. 0 = use
  /// nPrompt.
  int prefillWorkLen = 0;
  double prefillSecs = 0.0;
  double decodeSecs = 0.0;
  /// Model-specific extra metrics (name -> per-pass value). Individual runners
  /// append their own metrics here; buddy-bench aggregates them by name across
  /// repetitions. This is the "on top of the common result structure"
  /// extension point for non-LLM runners (vision / audio / scientific / ...).
  std::vector<std::pair<std::string, double>> extraMetrics;
};

/// Abstract base class for model inference runners.
///
/// Each supported model implements run() which owns the full inference loop:
///   - session creation (dlopen, KV cache alloc)
///   - weight loading
///   - tokenization
///   - prefill → decode loop
///   - result printing
class InferenceRunner {
public:
  virtual ~InferenceRunner() = default;
  virtual void run(const RunConfig &cfg) = 0;

  /// Run @p warmupReps untimed passes and then @p reps measured passes with
  /// @p cfg, appending one BenchmarkResult per measured pass to @p out.
  ///
  /// The warmup passes must run on the same session as the measured ones
  /// (same loaded weights, same allocated buffers); otherwise they do not warm
  /// anything the measurement uses. Implementations should also make repeated
  /// runBench() calls reuse their session when @p cfg still points at the same
  /// model files, so a multi-configuration benchmark loads the weights once.
  ///
  /// The default implementation reports that the model does not support
  /// benchmarking (returns false).
  virtual bool runBench(const RunConfig &cfg, int warmupReps, int reps,
                        std::vector<BenchmarkResult> &out) {
    (void)cfg;
    (void)warmupReps;
    (void)reps;
    (void)out;
    return false;
  }
};

using CreateInferenceRunnerFn = InferenceRunner *(*)();
using DestroyInferenceRunnerFn = void (*)(InferenceRunner *);

/// Capability bits returned by a runner plugin's
/// buddy_inference_runner_capabilities_v1() export.
///
/// The export doubles as an ABI marker: every plugin built against this header
/// must provide it. A host that calls virtual methods added after an older
/// plugin was built can therefore reject that plugin up front, instead of
/// jumping through a vtable slot the old object does not have.
constexpr uint64_t kRunnerCapBench = 1ULL << 0;

using InferenceRunnerCapabilitiesFn = uint64_t (*)();

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_INFERENCERUNNER_H
