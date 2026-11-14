//===- MaskedLMTypes.h - Masked language model serving DTOs --------------===//
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

#ifndef BUDDY_RUNTIME_CORE_MASKEDLMTYPES_H
#define BUDDY_RUNTIME_CORE_MASKEDLMTYPES_H

#include "buddy/runtime/core/ServingTypes.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

struct MaskedLMModelConfig {
  std::string raxPath;
  std::string modelSoPath;
  std::vector<std::string> weightPaths;
  std::string vocabPath;
  std::string modelName;
  std::size_t topK = 5;
  std::size_t contextLength = 0;
};

struct MaskedLMRequest {
  std::string model;
  std::string input;
  std::size_t topK = 0; // zero means use the manifest default
};

struct MaskedLMToken {
  std::int64_t tokenId = 0;
  std::string token;
  float score = 0.0f;
};

struct MaskedLMPrediction {
  std::size_t position = 0;
  std::vector<MaskedLMToken> tokens;
};

struct MaskedLMResult {
  std::string model;
  std::size_t sequenceLength = 0;
  std::vector<MaskedLMPrediction> predictions;
  std::size_t promptTokens = 0;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MASKEDLMTYPES_H
