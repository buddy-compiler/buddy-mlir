//===- Qwen3VLRuntime.h - Resident Qwen3-VL runtime -----------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_QWEN3VLRUNTIME_H
#define BUDDY_RUNTIME_MODELS_QWEN3VLRUNTIME_H

#include "buddy/runtime/core/ServingTypes.h"
#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

class Qwen3VLRuntime {
public:
  class Impl;
  Qwen3VLRuntime();
  ~Qwen3VLRuntime();
  Qwen3VLRuntime(const Qwen3VLRuntime &) = delete;
  Qwen3VLRuntime &operator=(const Qwen3VLRuntime &) = delete;

  void load(const std::string &raxPath);
  bool isLoaded() const;
  int contextLength() const;
  std::size_t imageTokenCount() const;
  const std::string &modelName() const;
  TokenizeResult tokenize(const std::string &prompt, bool countOnly) const;
  CompletionResult generate(const std::string &prompt,
                            const std::vector<ImageInput> &images,
                            const SamplingParams &sampling,
                            const CompletionStreamCallback &callback = {});

private:
  std::unique_ptr<Impl> impl;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_QWEN3VLRUNTIME_H
