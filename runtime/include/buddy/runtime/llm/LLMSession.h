//===- LLMSession.h - Abstract LLM inference session ---------------------===//
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
// Pure virtual interface for an LLM inference session.  Model-specific
// implementations (e.g. DeepSeekR1 ModelSession) inherit from this so that
// the generation loop and interactive REPL remain model-agnostic.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_LLMSESSION_H
#define BUDDY_RUNTIME_LLM_LLMSESSION_H

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"

namespace buddy {
namespace runtime {

/// Abstract inference session for autoregressive LLM generation.
///
/// Concrete implementations own model-specific state (KV cache, loaded .so,
/// function pointers) and expose a uniform prefill/decode interface.
class LLMSession {
public:
  virtual ~LLMSession() = default;

  // Non-copyable.
  LLMSession(const LLMSession &) = delete;
  LLMSession &operator=(const LLMSession &) = delete;

  /// Run prefill pass over the full input token sequence.
  /// Populates internal logits buffer accessible via logitsData().
  virtual void prefill(MemRef<float, 1> &weights, Text<size_t, 2> &tokens) = 0;

  /// Run one decode step for a single token.
  /// Populates internal logits buffer accessible via logitsData().
  virtual void decode(MemRef<float, 1> &weights, int tokenId) = 0;

  /// Reset decode position (allows session reuse with a new prompt).
  virtual void resetPosition() = 0;

  /// Current position (tokens processed so far).
  virtual int position() const = 0;

  /// Pointer to the logits output buffer (vocabSize floats per position).
  virtual const float *logitsData() const = 0;

  /// Vocabulary size (number of logits per position).
  virtual int vocabSize() const = 0;

  /// Handle KV cache overflow: discard tokens and adjust RoPE.
  /// @return true if overflow was handled, false if no overflow.
  virtual bool handleKVCacheOverflow(int keepTokenNum,
                                     float ropeTheta = 10000.0f) = 0;

protected:
  LLMSession() = default;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_LLMSESSION_H
