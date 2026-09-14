//===- EmbeddingTypes.h - Embedding serving data types -------------------===//
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

#ifndef BUDDY_RUNTIME_CORE_EMBEDDINGTYPES_H
#define BUDDY_RUNTIME_CORE_EMBEDDINGTYPES_H

#include "buddy/runtime/core/ServingTypes.h"

#include <cstddef>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

struct EmbeddingModelConfig {
  std::string raxPath;
  std::string modelSoPath;
  std::vector<std::string> weightPaths;
  std::string vocabPath;
  std::string modelName;
};

struct EmbeddingRequest {
  std::string model;
  std::string input;
};

struct EmbeddingResult {
  std::string model;
  std::vector<float> embedding;
  std::size_t promptTokens = 0;
  std::size_t totalTokens = 0;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_EMBEDDINGTYPES_H
