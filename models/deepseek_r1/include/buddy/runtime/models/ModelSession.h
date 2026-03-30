//===- ModelSession.h - Runtime session for DeepSeekR1 --------------------===//
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
// ModelSession owns all persistent state for one inference session:
//   - KV cache buffers (kv0..kv55), managed via BufferPool
//   - Logits output buffer
//   - Decode position counter
//   - Loaded model .so handle (dlopen/dlclose lifecycle)
//
// Host code only calls prefill() / decode().
// No raw MemRef, no _mlir_ciface_*, no kv0..kv55 fields in host.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_MODELSESSION_H
#define BUDDY_RUNTIME_MODELS_MODELSESSION_H

#include "buddy/runtime/core/BufferPool.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// Bring buddy::Text into scope alongside the global MemRef.
using buddy::Text;

namespace buddy {
namespace runtime {

/// Shape constants for DeepSeekR1.
/// Override at compile time via CMake definitions.
#ifndef BUDDY_DSR1_HEAD_NUM
#define BUDDY_DSR1_HEAD_NUM 2
#endif
#ifndef BUDDY_DSR1_MAX_TOKEN_LEN
#define BUDDY_DSR1_MAX_TOKEN_LEN 1024
#endif
#ifndef BUDDY_DSR1_HIDDEN_SIZE
#define BUDDY_DSR1_HIDDEN_SIZE 128
#endif
#ifndef BUDDY_DSR1_VOCAB_SIZE
#define BUDDY_DSR1_VOCAB_SIZE 129280
#endif
#ifndef BUDDY_DSR1_KV_LAYERS
#define BUDDY_DSR1_KV_LAYERS 56
#endif
#ifndef BUDDY_DSR1_PARAMS_SIZE
#define BUDDY_DSR1_PARAMS_SIZE 1777088064LL
#endif

/// Token sampler callback.
/// Receives logits pointer + vocab size, returns selected token id.
using TokenSampler = std::function<int(const float *, int)>;

/// Default greedy (argmax) sampler.
int greedySample(const float *logits, int vocabSize);

//===----------------------------------------------------------------------===//
// ModelSession
//===----------------------------------------------------------------------===//

/// Encapsulates one inference session for DeepSeekR1.
///
/// Lifecycle:
///   1. Fill Config, set modelSoPath = "/path/to/deepseek_r1_model.so"
///   2. session = ModelSession::create(cfg)   — dlopen + symbol resolve
///   3. session->prefill(weights, tokens)     — returns first token
///   4. session->decode(weights, token)       — returns next token (loop)
///   5. ~ModelSession()                       — dlclose
///
/// The session does NOT own model weights (arg0) — caller loads and passes
/// them.
///
class ModelSession {
public:
  struct Config {
    /// Path to the runtime-loadable model shared library.
    /// Must be set before calling create().
    /// Example: "/path/to/build/deepseek_r1_model.so"
    std::string modelSoPath;

    int headNum = BUDDY_DSR1_HEAD_NUM;
    int maxTokenLen = BUDDY_DSR1_MAX_TOKEN_LEN;
    int hiddenSize = BUDDY_DSR1_HIDDEN_SIZE;
    int vocabSize = BUDDY_DSR1_VOCAB_SIZE;
    int kvLayers = BUDDY_DSR1_KV_LAYERS;

    /// Custom sampler (nullptr → greedySample)
    TokenSampler sampler;
  };

  /// Create a session: allocates KV cache, then dlopen + dlsym.
  /// Throws std::runtime_error on any failure.
  static std::unique_ptr<ModelSession> create(const Config &cfg);

  /// Create a session from a packed .rax manifest.
  /// The manifest encodes the .so URI, weights URI, and vocab URI.
  /// resolvedManifest is filled with the resolved paths so the caller
  /// can load weights and initialise the tokeniser.
  static std::unique_ptr<ModelSession>
  createFromRax(const std::string &raxPath, ModelManifest &resolvedManifest);

  ~ModelSession();

  // Non-copyable
  ModelSession(const ModelSession &) = delete;
  ModelSession &operator=(const ModelSession &) = delete;

  /// Run prefill pass.
  /// @param weights  Flat model parameter buffer (Constant, caller-owned)
  /// @param tokens   Input token sequence
  /// @return         First generated token id
  int prefill(MemRef<float, 1> &weights, Text<size_t, 2> &tokens);

  /// Run one decode step.
  /// @param weights  Model parameter buffer
  /// @param tokenId  Current input token
  /// @return         Next token id
  int decode(MemRef<float, 1> &weights, int tokenId);

  /// Reset decode position (allows session reuse with a new prompt).
  void resetPosition();

  /// Handle KV cache overflow: discard half of non-essential tokens and
  /// adjust RoPE on surviving key cache entries.
  /// @param keepTokenNum  Number of initial tokens to preserve.
  /// @param ropeTheta     RoPE theta (default 10000.0 for DeepSeek/Qwen).
  /// @return true if overflow was handled, false if no overflow.
  bool handleKVCacheOverflow(int keepTokenNum, float ropeTheta = 10000.0f);

  /// Current position (tokens processed so far).
  int position() const { return position_; }

  /// Direct logits access for custom sampling.
  const float *logitsData() const;
  int vocabSize() const { return cfg_.vocabSize; }

  /// Path to the .so that was loaded by this session.
  std::string loadedSoPath() const;

  /// Access the buffer pool (for debugging / inspection).
  const BufferPool &bufferPool() const { return pool_; }

private:
  explicit ModelSession(const Config &cfg);
  void allocateKVCache();

  Config cfg_;
  BufferPool pool_;
  int position_ = 0;

  // Pimpl: hides KVCacheContainer, dlopen handle, and function pointers.
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Decode step inputs (reused every call)
  std::unique_ptr<MemRef<long long, 2>> decodeTokenInput_; // {1,1}
  std::unique_ptr<MemRef<long long, 1>> cachePosition_;    // {1}
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_MODELSESSION_H
