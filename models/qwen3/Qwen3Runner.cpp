//===- Qwen3Runner.cpp - Qwen3 runner plugin -------------===//
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

#include "buddy/runtime/models/Qwen3Runner.h"
#include "buddy/runtime/llm/Perplexity.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/ModelSession.h"

#include <filesystem>
#include <memory>
#include <stdexcept>

namespace buddy::runtime {
void Qwen3Runner::run(const RunConfig &cfg) {
  ModelManifest manifest;
  std::unique_ptr<ModelSession> session;
  std::vector<std::string> weights;
  std::string vocab;
  if (!cfg.raxPath.empty()) {
    session = ModelSession::createFromRax(cfg.raxPath, manifest);
    weights = manifest.weightPaths;
    vocab = manifest.vocabPath;
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error(
          "Qwen3 requires --model or --model-so and --weights");
    ModelSession::Config modelConfig;
    modelConfig.modelSoPath = cfg.modelSoPath;
    session = ModelSession::create(modelConfig);
    weights = {cfg.weightsPath};
    vocab = cfg.vocabPath.empty()
                ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                   "vocab.txt")
                      .string()
                : cfg.vocabPath;
  }
  session->loadWeights(weights);
  const bool quiet = cfg.suppressStats || cfg.streamJsonl;
  if (!cfg.perplexityIdsPath.empty()) {
    if (cfg.perplexityContext > BUDDY_QWEN3_MAX_TOKEN_LEN)
      throw std::runtime_error(
          "--ppl-context exceeds the compiled Qwen3 cache");
    PerplexityOptions options;
    options.context = cfg.perplexityContext;
    options.maxChunks = cfg.perplexityChunks;
    options.quiet = quiet;
    options.prefillCapacity = BUDDY_QWEN3_MAX_TOKEN_LEN;
    runPerplexity(*session, readTokenIds(cfg.perplexityIdsPath), options);
    return;
  }
  // tokenizeQwen3 supplies the Qwen user/assistant framing itself.
  if (cfg.interactive || !cfg.chatTemplatePath.empty())
    throw std::runtime_error("Qwen3 currently accepts a single prompt with its "
                             "built-in chat framing");
  TextCodec codec;
  codec.maxTokenLen = BUDDY_QWEN3_MAX_TOKEN_LEN;
  codec.tokenize = [](Text<size_t, 2> &text, const std::string &path) {
    text.tokenizeQwen3(path, BUDDY_QWEN3_MAX_TOKEN_LEN);
  };
  codec.detokenize = [](Text<size_t, 2> &text) { return text.revertQwen3(); };
  buddy::Sampler sampler(cfg.samplerConfig);
  auto result = runGeneration(cfg.prompt, *session, vocab, cfg.maxNewTokens,
                              {151645}, sampler, codec, quiet, cfg.streamJsonl);
  if (!quiet)
    printStats(result, true);
}
} // namespace buddy::runtime
