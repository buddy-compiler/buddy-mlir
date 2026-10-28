//===- DeepSeekR1RawCodec.h - Raw DeepSeek vocabulary codec -------*- C++
//-*-===//
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

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAWCODEC_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAWCODEC_H

#include "buddy/runtime/core/ServingTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Raw codec for Buddy's line-based DeepSeek vocabulary. The vocabulary does
/// not contain HF merge ranks: encoding uses deterministic longest-score
/// segmentation of its byte pieces, without claiming HF tokenizer equivalence.
/// It never inserts a chat template, collapses whitespace, or truncates input.
class DeepSeekR1RawCodec {
public:
  explicit DeepSeekR1RawCodec(const std::string &vocabPath);
  ~DeepSeekR1RawCodec();

  DeepSeekR1RawCodec(const DeepSeekR1RawCodec &) = delete;
  DeepSeekR1RawCodec &operator=(const DeepSeekR1RawCodec &) = delete;

  std::vector<int> encode(const std::string &content, bool addSpecial = false,
                          bool parseSpecial = true) const;
  std::string tokenBytes(int id) const;
  int tokenId(const std::string &piece) const;
  int vocabSize() const;
  TokenizeResult tokenize(const TokenizeRequest &request) const;
  DetokenizeResult detokenize(const DetokenizeRequest &request) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1RAWCODEC_H
