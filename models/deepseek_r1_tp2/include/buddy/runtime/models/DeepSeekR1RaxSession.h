//===- DeepSeekR1RaxSession.h - DeepSeek RAX resource session ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXSESSION_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXSESSION_H

#include "buddy/runtime/communication/Communicator.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/LLMSession.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace buddy {
namespace runtime {

/// Owns the resources needed to execute an existing DeepSeek R1 RAX schedule.
///
/// External parameter packs are mapped from the paths resolved by the RAX
/// manifest. Rank-local buffers are allocated from their manifest shapes and
/// remain valid for the session lifetime. This class does not implement text
/// generation policy, tokenization, or sampling.
class DeepSeekR1RaxSession : public LLMSession {
public:
  explicit DeepSeekR1RaxSession(const std::string &raxPath);
  DeepSeekR1RaxSession(const std::string &raxPath, Communicator &communicator);
  ~DeepSeekR1RaxSession() override;

  DeepSeekR1RaxSession(const DeepSeekR1RaxSession &) = delete;
  DeepSeekR1RaxSession &operator=(const DeepSeekR1RaxSession &) = delete;

  /// Replace an automatically resolved parameter pack with an owned mapping.
  void bindParameterPack(uint32_t id, const std::string &path);
  void bindParameterPack(const std::string &name, const std::string &path);

  /// Copy call input data into a session-owned RAX input buffer.
  void bindRuntimeInput(uint32_t id, const void *data, size_t bytes);
  void bindRuntimeInput(const std::string &name, const void *data,
                        size_t bytes);

  /// Copy initial or updated KV state into a persistent session buffer.
  void bindKVCacheBuffer(uint32_t id, const void *data, size_t bytes);
  void bindKVCacheBuffer(const std::string &name, const void *data,
                         size_t bytes);

  void forwardPrefill();
  void forwardDecode();

  void loadWeights(const std::vector<std::string> &weightPaths) override;
  void prefill(Text<size_t, 2> &tokens) override;
  void decode(int tokenId) override;
  void resetPosition() override;
  int position() const override;
  const float *logitsData(int tokenOffset = 0) const override;
  int vocabSize() const override;
  bool handleKVCacheOverflow(int keepTokenNum,
                             float ropeTheta = 10000.0f) override;

  void *bufferData(uint32_t id);
  void *bufferData(const std::string &name);
  size_t bufferSize(uint32_t id) const;
  size_t bufferSize(const std::string &name) const;
  const ModelManifest &manifest() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAXSESSION_H
