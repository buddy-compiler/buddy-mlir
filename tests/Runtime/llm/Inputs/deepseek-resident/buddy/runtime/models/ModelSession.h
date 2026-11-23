//===- ModelSession.h - Test session for DeepSeek serving ----------------===//
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

#ifndef BUDDY_TEST_DEEPSEEK_MODELSESSION_H
#define BUDDY_TEST_DEEPSEEK_MODELSESSION_H

#include "buddy/runtime/llm/LLMSession.h"

#include <memory>
#include <stdexcept>

#define BUDDY_DSR1_MAX_TOKEN_LEN 64

namespace buddy::runtime {

struct ModelManifest;

// Replace only the generated model session. The test compiles the production
// resident model and exercises its real renderer, codec, usage and generation
// loop without loading model weights. Every prefill immediately produces EOS.
class ModelSession final : public LLMSession {
public:
  struct Config {
    std::string modelSoPath;
  };

  static std::unique_ptr<ModelSession> create(const Config &) {
    return std::make_unique<ModelSession>();
  }
  static std::unique_ptr<ModelSession> createFromRax(const std::string &,
                                                     ModelManifest &) {
    throw std::runtime_error("test does not load RAX files");
  }

  ModelSession() : logits(151665, -1000.0f) { logits[151643] = 10.0f; }
  void loadWeights(const std::vector<std::string> &) override {}
  void resetPosition() override { promptTokens.clear(); }
  void prefill(Text<size_t, 2> &tokens) override {
    promptTokens.assign(tokens.getData(),
                        tokens.getData() + tokens.getTokenCnt());
  }
  void decode(int) override {
    throw std::runtime_error("test must stop at first-token EOS");
  }
  int position() const override {
    return static_cast<int>(promptTokens.size());
  }
  int vocabSize() const override { return static_cast<int>(logits.size()); }
  const float *logitsData(int = 0) const override { return logits.data(); }
  bool handleKVCacheOverflow(int, float) override {
    throw std::runtime_error("test must not overflow the context");
  }

  inline static std::vector<size_t> promptTokens;

private:
  std::vector<float> logits;
};

} // namespace buddy::runtime

#endif
