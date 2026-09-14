//===- BgeM3Runtime.h - Reusable BGE-M3 inference runtime -----------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_BGEM3RUNTIME_H
#define BUDDY_RUNTIME_MODELS_BGEM3RUNTIME_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

class BgeM3Runtime {
public:
  static std::unique_ptr<BgeM3Runtime> loadFromRax(const std::string &raxPath);
  static std::unique_ptr<BgeM3Runtime>
  loadLegacy(const std::string &modelSoPath, const std::string &weightsPath,
             const std::string &vocabPath, size_t maxSeqLen = 512,
             size_t maxPositionEmbeddings = 8194, size_t hiddenSize = 1024,
             const std::string &modelName = "bge_m3");

  ~BgeM3Runtime();
  BgeM3Runtime(const BgeM3Runtime &) = delete;
  BgeM3Runtime &operator=(const BgeM3Runtime &) = delete;

  std::vector<float> embed(const std::string &text,
                           size_t *tokenCount = nullptr);
  const std::string &modelName() const;
  size_t contextLength() const;
  size_t embeddingDimension() const;
  size_t tokenCount(const std::string &text) const;

private:
  BgeM3Runtime() = default;
  struct Impl;
  std::unique_ptr<Impl> impl;

  static std::unique_ptr<BgeM3Runtime>
  load(const std::string &modelSoPath, const std::string &weightsPath,
       const std::string &vocabPath, size_t maxSeqLen,
       size_t maxPositionEmbeddings, size_t hiddenSize,
       const std::string &modelName,
       const std::vector<std::string> &dependentSoPaths = {});
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_BGEM3RUNTIME_H
