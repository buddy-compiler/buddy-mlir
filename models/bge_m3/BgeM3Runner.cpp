//===- BgeM3Runner.cpp - BGE-M3 embedding runner --------------------------===//
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

#include "buddy/runtime/models/BgeM3Runner.h"
#include "buddy/runtime/models/BgeM3Runtime.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace buddy {
namespace runtime {

void BgeM3Runner::run(const RunConfig &cfg) {
  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error("BgeM3Runner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "BgeM3Runner: only single-prompt inference is implemented for BGE-M3");
  const std::string prompt =
      cfg.prompts.empty() ? cfg.prompt : cfg.prompts.front();

  std::unique_ptr<BgeM3Runtime> runtime;
  if (!cfg.raxPath.empty()) {
    runtime = BgeM3Runtime::loadFromRax(cfg.raxPath);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("BgeM3Runner: legacy mode requires --model-so, "
                               "--weights, and --vocab");
    const size_t maxSeqLen =
        cfg.promptLength > 0 ? static_cast<size_t>(cfg.promptLength) : 512;
    runtime = BgeM3Runtime::loadLegacy(cfg.modelSoPath, cfg.weightsPath,
                                       cfg.vocabPath, maxSeqLen);
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  size_t tokenCount = 0;
  std::vector<float> embedding = runtime->embed(prompt, &tokenCount);
  const auto t1 = std::chrono::high_resolution_clock::now();

  if (!cfg.suppressStats) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mBGE-M3 Dense Embedding\033[0m\n";
    std::cerr << "  dim: " << embedding.size() << "\n";
    std::cerr << "  time: " << seconds << "s\n";
  }

  std::cout << "[";
  for (size_t i = 0; i < embedding.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << embedding[i];
  }
  std::cout << "]\n";
}

} // namespace runtime
} // namespace buddy
