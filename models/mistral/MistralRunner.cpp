//===- MistralRunner.cpp - Mistral full inference loop --------------------===//
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
//
// Implements MistralRunner::run().  This file owns session setup and
// dispatch:
//   - weight file loading
//   - chat template loading
//   - sampler creation
//   - dispatch to single-shot generation or interactive REPL
//
// Mistral-7B uses a Llama-style BPE tokenizer, so tokenization goes through
// Text::tokenizeLlama / Text::revertLlama and the EOS token is </s> (id 2).
//
// buddy-cli calls this through the generic InferenceRunner interface.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/MistralRunner.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/InteractiveSession.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

using buddy::Text;

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {
static constexpr int kEosToken = 2; // </s>
} // namespace

//===----------------------------------------------------------------------===//
// MistralRunner::run
//===----------------------------------------------------------------------===//

void MistralRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  const bool suppress = cfg.suppressStats || cfg.streamJsonl;

  if (!suppress)
    std::cerr << "\033[32;1mMistral Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  // ── Chat template: load if provided ─────────────────────────────────────
  std::vector<long long> stopTokenIds = {kEosToken};
  std::unique_ptr<buddy::ChatTemplate> chatTmpl;

  if (!cfg.chatTemplatePath.empty()) {
    chatTmpl = std::make_unique<buddy::ChatTemplate>(
        buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
    for (int id : chatTmpl->stopTokenIds()) {
      if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
          stopTokenIds.end()) {
        stopTokenIds.push_back(static_cast<long long>(id));
      }
    }
    printLog("Chat template loaded: " + cfg.chatTemplatePath, suppress);
  }

  // ── Create session ───────────────────────────────────────────────────────
  std::unique_ptr<ModelSession> session;
  std::vector<std::string> weightPaths;
  std::string vocabPath;

  if (!cfg.raxPath.empty()) {
    printLog("Manifest: " + cfg.raxPath, suppress);
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);

    weightPaths = manifest.weightPaths;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;

    printLog("  .so     = " + manifest.soPath, suppress);
    for (const auto &wp : weightPaths)
      printLog("  weights = " + wp, suppress);
    printLog("  vocab   = " + vocabPath, suppress);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath and weightsPath.");

    weightPaths.push_back(cfg.weightsPath);
    vocabPath = cfg.vocabPath.empty()
                    ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : cfg.vocabPath;

    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    printLog("Loading model: " + cfg.modelSoPath, suppress);
    session = ModelSession::create(mcfg);
  }

  // ── Load weights into session ───────────────────────────────────────────
  // Reads weight files from disk into session-owned MemRefs (layout and
  // element types match the compiled variant; see manifest constant order).
  session->loadWeights(weightPaths);
  printLog("Weights loaded.", suppress);

  printLog("Vocab: " + vocabPath, suppress);
  printLog("KV cache: " + std::to_string(BUDDY_MISTRAL_KV_LAYERS) + " x {1," +
               std::to_string(BUDDY_MISTRAL_HEAD_NUM) + "," +
               std::to_string(BUDDY_MISTRAL_MAX_TOKEN_LEN) + "," +
               std::to_string(BUDDY_MISTRAL_HIDDEN_SIZE) + "} f32",
           suppress);

  // ── Model-specific text codec ───────────────────────────────────────────
  TextCodec codec;
  codec.tokenize = [](Text<size_t, 2> &t, const std::string &vocab) {
    t.tokenizeLlama(vocab, BUDDY_MISTRAL_MAX_TOKEN_LEN);
  };
  codec.detokenize = [](Text<size_t, 2> &t) { return t.revertLlama(); };
  codec.maxTokenLen = BUDDY_MISTRAL_MAX_TOKEN_LEN;

  // ── Create Sampler ───────────────────────────────────────────────────────
  buddy::Sampler sampler(cfg.samplerConfig);

  // ── Interactive or single-shot mode ──────────────────────────────────────
  if (cfg.interactive) {
    if (!chatTmpl) {
      throw std::runtime_error(
          "--interactive requires --chat-template <path.json>");
    }
    buddy::ConversationManager conv(
        std::move(*chatTmpl), [&vocabPath](const std::string &text) -> size_t {
          Text<size_t, 2> tmp(text);
          tmp.tokenizeLlama(vocabPath, BUDDY_MISTRAL_MAX_TOKEN_LEN);
          return tmp.getTokenCnt();
        });
    if (!cfg.prompt.empty())
      conv.setSystemPrompt(cfg.prompt);

    runInteractiveSession(*session, vocabPath, cfg, stopTokenIds, conv, codec,
                          sampler);
  } else {
    // Single-shot: format prompt with chat template if available.
    std::string finalPrompt = cfg.prompt;
    if (chatTmpl) {
      std::vector<buddy::Message> msgs = {{"user", cfg.prompt}};
      finalPrompt = chatTmpl->apply(msgs);
    }

    GenerationResult result =
        runGeneration(finalPrompt, *session, vocabPath, cfg.maxNewTokens,
                      stopTokenIds, sampler, codec, suppress, cfg.streamJsonl);

    if (!suppress)
      printStats(result, /*verbose=*/true);
  }
}

} // namespace runtime
} // namespace buddy
