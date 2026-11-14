//===- ServingTypes.h - Model serving request and response types ----------===//
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
// Source tree: buddy-mlir/runtime/include/buddy/runtime/core/ServingTypes.h
// Include as:  #include "buddy/runtime/core/ServingTypes.h"
//
// These plain data types describe the model-serving boundary shared by
// buddy-server transport adapters and long-lived model implementations. They
// intentionally avoid HTTP, JSON, SSE, and model-family-specific dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_SERVINGTYPES_H
#define BUDDY_RUNTIME_CORE_SERVINGTYPES_H

#include "buddy/runtime/llm/Sampler.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Model lifecycle state exposed by /health.
enum class ModelLoadState {
  Unloaded,
  Loading,
  Ready,
  Error,
};

/// Reason a generation request finished.
enum class FinishReason {
  None,
  Stop,
  Length,
  Cancelled,
  Error,
};

/// Configuration used to initialize a resident model.
///
/// Mode A uses a .rax manifest. Mode B mirrors the legacy buddy-cli explicit
/// paths and is useful while migrating existing model artifacts.
struct ResidentModelConfig {
  // Mode A: single manifest (recommended).
  std::string raxPath;

  // Mode B: explicit paths.
  std::string modelSoPath;
  std::vector<std::string> weightPaths;
  std::string vocabPath;

  // Optional chat template JSON file.
  std::string chatTemplatePath;

  // Optional display name. Implementations may override this from a manifest.
  std::string modelName;
};

/// Current model status for health checks and diagnostics.
struct ModelStatus {
  ModelLoadState state = ModelLoadState::Unloaded;
  std::string modelName;
  std::string backend;
  std::string message;
  int contextLength = 0;
};

/// Serving seed policy, independent of the legacy SamplerConfig seed sentinel.
enum class SamplingSeedPolicy { Random, Fixed };

/// Request-time sampling and stopping configuration.
struct SamplingParams {
  /// Maximum total token count, including prompt tokens. A value of 0 means no
  /// explicit request limit; the model/session may still stop at context size.
  int maxTokens = 512;

  buddy::SamplerConfig samplerConfig;

  /// Optional string stops, matched against raw token bytes by serving
  /// generation before UTF-8 decoding.
  std::vector<std::string> stop;

  /// Optional token-id stops merged with model/template default stop ids.
  std::vector<long long> stopTokenIds;

  /// Generated-token budget for runServingGeneration, separate from the legacy
  /// maxTokens limit. Zero runs prefill only; -1 uses the remaining context.
  /// Positive limits count the first prediction and any EOS/stop token.
  int maxNewTokens = 512;

  /// Serving seed selection is independent of samplerConfig.seed. Fixed
  /// permits zero; UINT32_MAX is reserved for random selection.
  SamplingSeedPolicy seedPolicy = SamplingSeedPolicy::Random;
  /// Used only when seedPolicy is Fixed.
  uint32_t fixedSeed = 0;
};
/// One image supplied with a multimodal request.
///
/// The current Qwen3-VL server path accepts one local file URI/path. `bytes`
/// is reserved for a future in-memory transport and is intentionally optional.
/// A resident plugin and buddy-server must be rebuilt together if this DTO is
/// changed because the resident plugin boundary uses the C++ runtime ABI.
struct ImageInput {
  std::string uri;
  std::string mimeType;
  std::vector<uint8_t> bytes;
};

/// Completion request with a fully rendered prompt.
struct CompletionRequest {
  std::string prompt;
  SamplingParams sampling;
  std::vector<ImageInput> images;
};

/// Chat message before model-specific template rendering.
struct ChatMessage {
  std::string role;    // "system", "user", "assistant"
  std::string content; // Plain message text.
  std::vector<ImageInput> images;
};

/// Chat completion request. If messages is empty, input is treated as a single
/// user message by implementations that support bare-string chat requests.
struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::string input;
  SamplingParams sampling;
  std::vector<ImageInput> images;
};

/// Tokenization request.
struct TokenizeRequest {
  std::string content;
  bool addSpecial = true;
  bool countOnly = false;
};

/// Tokenization result.
struct TokenizeResult {
  std::vector<int> tokens;
  std::size_t count = 0;
};

/// Token usage accounting returned to HTTP adapters.
struct CompletionUsage {
  int promptTokens = 0;
  int completionTokens = 0;
  int totalTokens = 0;
};

/// Timing data in milliseconds.
struct CompletionTimings {
  double prefillMs = 0.0;
  double decodeMs = 0.0;
  double tokensPerSecond = 0.0;
};

/// Final result for non-streaming requests and the terminal summary for
/// streaming requests.
struct CompletionResult {
  std::string id;
  std::string model;
  std::string content;
  FinishReason finishReason = FinishReason::None;
  CompletionUsage usage;
  CompletionTimings timings;
};

/// Incremental generation event. The transport layer decides whether this
/// becomes a llama.cpp-style chunk, an OpenAI-compatible SSE chunk, or another
/// wire format.
struct CompletionChunk {
  std::string id;
  std::string model;
  std::string delta;
  int tokenId = -1;
  bool done = false;
  FinishReason finishReason = FinishReason::None;
  CompletionUsage usage;
  CompletionTimings timings;
};

/// Return false from the callback to cancel generation, for example when an
/// HTTP client disconnects.
using CompletionStreamCallback =
    std::function<bool(const CompletionChunk &chunk)>;

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_SERVINGTYPES_H
