//===- DeepSeekR1TPRunner.cpp - DeepSeek R1 TP=2 inference ----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1TPRunner.h"

#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/DeepSeekR1RaxRunner.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using buddy::Text;

namespace buddy {
namespace runtime {
namespace {

static constexpr int kEosToken = 151643; // <|end▁of▁sentence|>
static constexpr int kEotToken = 151647; // <|EOT|>

} // namespace

void DeepSeekR1TPRunner::run(const RunConfig &cfg) {
  if (cfg.raxPath.empty())
    throw std::runtime_error(
        "DeepSeek R1 TP runner requires --model <rank-local.rax>");
  if (cfg.interactive)
    throw std::runtime_error(
        "DeepSeek R1 tensor-parallel interactive mode is not supported");

  const auto &samplerConfig = cfg.samplerConfig;
  if (samplerConfig.temperature != 0.0f || samplerConfig.topK != 0 ||
      samplerConfig.topP != 1.0f || samplerConfig.minP != 0.0f ||
      samplerConfig.repeatPenalty != 1.0f)
    throw std::runtime_error(
        "DeepSeek R1 tensor-parallel execution requires greedy sampling");

  std::vector<long long> stopTokenIds = {kEosToken, kEotToken};
  std::unique_ptr<buddy::ChatTemplate> chatTemplate;
  if (!cfg.chatTemplatePath.empty()) {
    chatTemplate = std::make_unique<buddy::ChatTemplate>(
        buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
    for (int id : chatTemplate->stopTokenIds())
      if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
          stopTokenIds.end())
        stopTokenIds.push_back(static_cast<long long>(id));
  }

  TextCodec codec;
  codec.tokenize = [](Text<size_t, 2> &tokens, const std::string &vocab) {
    tokens.tokenizeDeepSeekR1(vocab, BUDDY_DSR1_MAX_TOKEN_LEN);
  };
  codec.detokenize = [](Text<size_t, 2> &tokens) {
    return tokens.revertDeepSeekR1();
  };
  codec.maxTokenLen = BUDDY_DSR1_MAX_TOKEN_LEN;
  buddy::Sampler sampler(samplerConfig);

  runDeepSeekR1Rax(cfg.raxPath, [&](DeepSeekR1RaxSession &session, int rank) {
    const ModelManifest &manifest = session.manifest();
    const std::string vocabPath =
        manifest.vocabPath.empty()
            ? (std::filesystem::path(manifest.soPath).parent_path() /
               "vocab.txt")
                  .string()
            : manifest.vocabPath;
    const bool emitOutput = rank == 0;
    const bool suppress = cfg.suppressStats || cfg.streamJsonl || !emitOutput;

    if (!suppress) {
      std::cerr << "\033[33;1mDeepSeekR1 TP=2 Inference (buddy-cli / "
                   "BuddyRuntime)\033[0m\n";
      printLog("Manifest: " + cfg.raxPath, false);
      printLog("  .so     = " + manifest.soPath, false);
      for (const auto &path : manifest.weightPaths)
        printLog("  weights = " + path, false);
      printLog("  vocab   = " + vocabPath, false);
    }

    session.loadWeights(manifest.weightPaths);
    printLog("Weights loaded.", suppress);
    printLog("Vocab: " + vocabPath, suppress);
    printLog("KV cache: " + std::to_string(BUDDY_DSR1_KV_LAYERS) + " x {1," +
                 std::to_string(BUDDY_DSR1_HEAD_NUM) + "," +
                 std::to_string(BUDDY_DSR1_MAX_TOKEN_LEN) + "," +
                 std::to_string(BUDDY_DSR1_HIDDEN_SIZE) + "} f32",
             suppress);

    std::string finalPrompt = cfg.prompt;
    if (chatTemplate) {
      std::vector<buddy::Message> messages = {{"user", cfg.prompt}};
      finalPrompt = chatTemplate->apply(messages);
    }

    GenerationResult result = runGeneration(
        finalPrompt, session, vocabPath, cfg.maxNewTokens, stopTokenIds,
        sampler, codec, suppress, cfg.streamJsonl, emitOutput);
    if (!suppress)
      printStats(result, /*verbose=*/true);
  });
}

} // namespace runtime
} // namespace buddy
