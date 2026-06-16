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
// InferenceRunner is the single extension point for buddy-cli:
//   - Each supported model implements a subclass of InferenceRunner
//   - buddy-cli reads the model_name from the .rax manifest and constructs
//     the right runner via the model name
//
// Current implementations:
//   deepseek_r1  →  DeepSeekR1Runner  (models/deepseek_r1/)
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_INFERENCERUNNER_H
#define BUDDY_RUNTIME_CORE_INFERENCERUNNER_H

#include <memory>
#include <string>
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

  /// Optional fixed prompt/prefill length. 0 means use the encoded prompt
  /// length for single-prompt runs, or the longest encoded prompt for batched
  /// prompt-file runs.
  int promptLength = 0;

  /// Upper bound on generated tokens. The first generated token comes from
  /// prefill; subsequent tokens come from decode. 0 = no limit.
  int maxNewTokens = 4096;

  /// Runtime batch size override. 0 means use the packaged model default.
  int batchSize = 0;

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

  // ── Interactive mode ──
  /// Enable REPL interactive mode for multi-turn conversation.
  bool interactive = false;
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
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_INFERENCERUNNER_H
