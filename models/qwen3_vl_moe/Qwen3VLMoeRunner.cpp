//===- Qwen3VLMoeRunner.cpp - Qwen3-VL-MoE text inference loop ------------===//
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
// Implements Qwen3VLMoeRunner::run() for the text decoder of a Qwen3-VL-MoE
// model (text-only prompts): weight loading, the Qwen3 tokenizer and chat
// scaffold, perplexity, interactive and single-shot generation.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Qwen3VLMoeRunner.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/InteractiveSession.h"
#include "buddy/runtime/llm/Perplexity.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

using buddy::Text;

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace buddy {
namespace runtime {

namespace {
static constexpr int kEndOfText = 151643; // <|endoftext|>
static constexpr int kImEnd = 151645;     // <|im_end|>
} // namespace

void Qwen3VLMoeRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  const bool suppress = cfg.suppressStats || cfg.streamJsonl;

  if (!suppress)
    std::cerr << "\033[33;1mQwen3-VL-MoE Inference (buddy-cli / "
                 "BuddyRuntime)\033[0m\n";

  // ── Chat template: load if provided ─────────────────────────────────────
  std::vector<long long> stopTokenIds = {kImEnd, kEndOfText};
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
  if (cfg.raxPath.empty())
    throw std::runtime_error("Qwen3-VL-MoE needs a .rax model (--model).");
  printLog("Manifest: " + cfg.raxPath, suppress);
  ModelManifest manifest;
  std::unique_ptr<ModelSession> session =
      ModelSession::createFromRax(cfg.raxPath, manifest);
  std::vector<std::string> weightPaths = manifest.weightPaths;
  std::string vocabPath =
      manifest.vocabPath.empty()
          ? (std::filesystem::path(manifest.soPath).parent_path() / "vocab.txt")
                .string()
          : manifest.vocabPath;
  printLog("  .so     = " + manifest.soPath, suppress);
  for (const auto &wp : weightPaths)
    printLog("  weights = " + wp, suppress);
  printLog("  vocab   = " + vocabPath, suppress);

  session->loadWeights(weightPaths);
  printLog("Weights loaded.", suppress);

  // ── Perplexity mode: score a token stream instead of generating ─────────
  if (!cfg.perplexityIdsPath.empty()) {
    if (cfg.perplexityContext > BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN)
      throw std::runtime_error(
          "--ppl-context " + std::to_string(cfg.perplexityContext) +
          " exceeds the KV cache (" +
          std::to_string(BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN) + " tokens)");
    PerplexityOptions opts;
    opts.context = cfg.perplexityContext;
    opts.maxChunks = cfg.perplexityChunks;
    opts.quiet = suppress;
    runPerplexity(*session, readTokenIds(cfg.perplexityIdsPath), opts);
    return;
  }

  printLog("KV cache: " + std::to_string(BUDDY_QWEN3_VL_MOE_KV_LAYERS) +
               " x {1," + std::to_string(BUDDY_QWEN3_VL_MOE_HEAD_NUM) + "," +
               std::to_string(BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN) + "," +
               std::to_string(BUDDY_QWEN3_VL_MOE_HIDDEN_SIZE) + "} f32",
           suppress);

  // ── Text codec: Qwen3 byte-level BPE, with the user / assistant scaffold
  TextCodec codec;
  codec.tokenize = [](Text<size_t, 2> &t, const std::string &vocab) {
    t.tokenizeQwen3(vocab, BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN);
  };
  codec.detokenize = [](Text<size_t, 2> &t) { return t.revertQwen3(); };
  codec.maxTokenLen = BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN;

  buddy::Sampler sampler(cfg.samplerConfig);

  if (cfg.interactive) {
    if (!chatTmpl) {
      throw std::runtime_error(
          "--interactive requires --chat-template <path.json>");
    }
    buddy::ConversationManager conv(
        std::move(*chatTmpl), [&vocabPath](const std::string &text) -> size_t {
          Text<size_t, 2> tmp(text);
          tmp.tokenizeQwen3(vocabPath, BUDDY_QWEN3_VL_MOE_MAX_TOKEN_LEN);
          return tmp.getTokenCnt();
        });
    if (!cfg.prompt.empty())
      conv.setSystemPrompt(cfg.prompt);

    runInteractiveSession(*session, vocabPath, cfg, stopTokenIds, conv, codec,
                          sampler);
  } else {
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
