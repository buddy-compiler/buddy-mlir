//===- Llama31TTExecution.h - Resident Llama 3.1 TT context --------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_LLAMA31TTEXECUTION_H
#define BUDDY_RUNTIME_MODELS_LLAMA31TTEXECUTION_H

#include "buddy/runtime/core/ServingTypes.h"
#include "buddy/runtime/llm/ServingGeneration.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace buddy {
class ChatTemplate;
namespace runtime {

/// Long-lived, model-specific state used by the llama31_tt resident plugin.
///
/// The context deliberately owns manifest metadata and tokenizer state, while
/// the TTNN submission implementation is supplied as a small backend hook.
/// This keeps HTTP and model lifetime concerns out of the native runner and
/// also permits ABI tests to run without a Tenstorrent device.
class Llama31TTExecution {
public:
  struct Metadata {
    std::string modelName;
    std::string prefillPath;
    std::string decodePath;
    std::string artifactsPath;
    std::string tokenizerPath;
    std::string promptFormat = "chat";
    std::string prefillKVOutputOrder = "key_value";
    int maxCacheLen = 1024;
    int batchSize = 1;
    int eosTokenId = 128009;
    uint32_t programIndex = 0;
    bool ignoreEOS = false;
    bool disableStaticReuse = false;
  };

  struct BackendRequest {
    std::vector<int> promptTokens;
    SamplingParams sampling;
    int cachePosition = 0;
    int maxCacheLen = 0;
    bool ignoreEOS = false;
    int eosTokenId = 128009;
    std::function<std::string(int)> decodeToken;
  };

  using Backend = std::function<CompletionResult(
      const BackendRequest &, const CompletionStreamCallback &)>;
  using ResetCallback = std::function<void()>;
  using ServingBackend = std::function<CompletionResult(
      const std::string &, const SamplingParams &, const ServingTextCodec &,
      const CompletionStreamCallback &)>;
  struct BackendHooks {
    Backend generate;
    ResetCallback reset;
    ServingBackend serve;
  };
  using BackendFactory = std::function<BackendHooks(const Metadata &)>;

  Llama31TTExecution();
  ~Llama31TTExecution();
  Llama31TTExecution(const Llama31TTExecution &) = delete;
  Llama31TTExecution &operator=(const Llama31TTExecution &) = delete;

  /// Load and validate a canonical batch=1 llama31_tt .rax package.
  void load(const ResidentModelConfig &config);
  bool isLoaded() const { return loaded; }
  const Metadata &metadata() const { return metadataValue; }

  /// Reset per-request KV/cache position. This is safe to call repeatedly.
  void reset();
  int cachePosition() const { return cachePositionValue; }

  TokenizeResult tokenize(const std::string &text, bool countOnly,
                          bool addSpecial = true) const;
  TokenizeResult tokenizeRaw(const TokenizeRequest &request) const;
  DetokenizeResult detokenize(const DetokenizeRequest &request) const;
  ResidentModelCapabilities capabilities() const;
  std::string renderChat(const ChatCompletionRequest &request) const;

  CompletionResult generate(const std::string &prompt,
                            const SamplingParams &sampling,
                            const CompletionStreamCallback &callback = {});

  /// Compatible serving path. The legacy generate() entry retains its limits
  /// and output contract for existing non-HTTP callers.
  CompletionResult
  generateServing(const std::string &prompt, const SamplingParams &sampling,
                  const CompletionStreamCallback &callback = {},
                  bool addSpecial = true);

  /// Install a test/injected backend. A production plugin uses
  /// setBackendFactory() to create its hardware session during load().
  void setBackend(Backend backend) { backendValue = std::move(backend); }
  void setBackendFactory(BackendFactory factory) {
    backendFactory = std::move(factory);
  }
  void setServingBackend(ServingBackend backend) {
    servingBackend = std::move(backend);
  }
  void setResetCallback(ResetCallback callback) {
    resetCallback = std::move(callback);
  }

private:
  class Tokenizer;
  std::unique_ptr<Tokenizer> tokenizer;
  Metadata metadataValue;
  Backend backendValue;
  ServingBackend servingBackend;
  BackendFactory backendFactory;
  ResetCallback resetCallback;
  int cachePositionValue = 0;
  bool loaded = false;
  std::unique_ptr<buddy::ChatTemplate> chatTemplate;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_LLAMA31TTEXECUTION_H
