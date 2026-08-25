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

/// Request-time sampling and stopping configuration.
struct SamplingParams {
  /// Maximum total token count, including prompt tokens. A value of 0 means no
  /// explicit request limit; the model/session may still stop at context size.
  int maxTokens = 512;

  buddy::SamplerConfig samplerConfig;

  /// Optional string stops supplied by the transport layer. The MVP may choose
  /// to support token-id stops first and leave string stops to the adapter.
  std::vector<std::string> stop;

  /// Optional token-id stops merged with model/template default stop ids.
  std::vector<long long> stopTokenIds;
};

/// Completion request with a fully rendered prompt.
struct CompletionRequest {
  std::string prompt;
  SamplingParams sampling;
};

/// Chat message before model-specific template rendering.
struct ChatMessage {
  std::string role;    // "system", "user", "assistant"
  std::string content; // Plain message text.
};

/// Chat completion request. If messages is empty, input is treated as a single
/// user message by implementations that support bare-string chat requests.
struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::string input;
  SamplingParams sampling;
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
