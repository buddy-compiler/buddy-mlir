//===- TextGeneration.h - Model-agnostic text generation ------------------===//
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
// Declares the single-pass generation loop (tokenize → prefill → decode) and
// its result type.  Model-specific tokenization is abstracted via TextCodec.
//
// Used by both single-shot and interactive modes.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_TEXTGENERATION_H
#define BUDDY_RUNTIME_LLM_TEXTGENERATION_H

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/LLMSession.h"
#include "buddy/runtime/llm/Sampler.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace buddy {

using buddy::Text;

namespace runtime {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Log a message to stderr with [Log] prefix.  Suppressed when @p suppress is
/// true (e.g. in quiet / piped mode).
void printLog(const std::string &msg, bool suppress = false);

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

/// Set to true by the SIGINT handler; checked inside the decode loop so that
/// Ctrl+C interrupts generation without killing the process.
extern std::atomic<bool> g_interrupted;

/// Minimal SIGINT handler — sets g_interrupted.
void interruptHandler(int);

//===----------------------------------------------------------------------===//
// TextCodec — model-specific tokenize / detokenize callbacks
//===----------------------------------------------------------------------===//

/// Abstracts model-specific text encoding/decoding so that the generation loop
/// remains model-agnostic.
struct TextCodec {
  /// Tokenize a prompt string into the given Text container.
  /// The implementation should call the appropriate model-specific tokenizer
  /// (e.g. `t.tokenizeDeepSeekR1(vocabPath, maxLen)`).
  std::function<void(Text<size_t, 2> &, const std::string &vocabPath)> tokenize;

  /// Decode a Text container of output token ids back to a string.
  /// The implementation should call the appropriate model-specific decoder
  /// (e.g. `return t.revertDeepSeekR1()`).
  std::function<std::string(Text<size_t, 2> &)> detokenize;

  /// Maximum context length in tokens.  Used for KV cache overflow detection.
  int maxTokenLen;
};

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

/// Result of a single generation pass.
struct GenerationResult {
  int generatedTokens = 0;
  double prefillSecs = 0.0;
  double decodeSecs = 0.0;
  std::string text;
};

/// Print prefill / decode throughput to stderr.
void printStats(const GenerationResult &result, bool verbose = false);

/// Run a single generation pass: tokenize → prefill → decode loop.
/// Resets session position before prefill (safe for multi-turn reuse).
/// Weights must already be loaded into the session via loadWeights().
GenerationResult runGeneration(const std::string &prompt, LLMSession &session,
                               const std::string &vocabPath, int maxNewTokens,
                               const std::vector<long long> &stopTokenIds,
                               buddy::Sampler &sampler, const TextCodec &codec,
                               bool suppress);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_TEXTGENERATION_H
