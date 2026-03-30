//===- DeepSeekR1Runner.cpp - DeepSeekR1 full inference loop --------------===//
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
// Implements DeepSeekR1Runner::run().  This file owns everything that is
// model-specific for text generation:
//   - weight file loading
//   - tokenization (buddy::Text / TextContainer)
//   - prefill + decode loop
//   - sampling (via buddy::Sampler)
//   - chat template formatting (via buddy::ChatTemplate)
//   - interactive REPL mode (via buddy::ConversationManager)
//   - performance counters and result printing
//
// buddy-cli calls this through the generic InferenceRunner interface.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1Runner.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/Sampler.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"

using buddy::Text;

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS
#define BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS 1024
#endif

namespace buddy {
namespace runtime {

namespace {

static constexpr int kEosToken = 151643; // <|end▁of▁sentence|>
static constexpr int kKeepTokenNum = BUDDY_DSR1_MAX_TOKEN_LEN / 4;
static constexpr float kRopeTheta = 10000.0f;

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

static std::atomic<bool> g_interrupted{false};

void interruptHandler(int) { g_interrupted = true; }

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void printLog(const std::string &msg, bool suppress = false) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log]\033[0m " << msg << "\n";
}

void loadWeights(const std::string &path, MemRef<float, 1> &weights,
                 bool suppress = false) {
  printLog("Weights: " + std::filesystem::canonical(path).string(), suppress);
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open weights: " + path);
  f.read(reinterpret_cast<char *>(weights.getData()),
         sizeof(float) * weights.getSize());
  if (f.fail())
    throw std::runtime_error("Read failed: " + path);

  const double secs = std::chrono::duration<double>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
  printLog("Weights loaded in " + std::to_string(secs) + "s", suppress);
}

/// Stream decoded text incrementally to stdout.
void streamNewText(Text<size_t, 2> &outputContainer, std::string &lastPrinted) {
  std::string current = outputContainer.revertDeepSeekR1();
  if (current.size() > lastPrinted.size()) {
    std::cout.write(current.data() + lastPrinted.size(),
                    current.size() - lastPrinted.size());
    std::cout.flush();
  }
  lastPrinted = std::move(current);
}

//===----------------------------------------------------------------------===//
// Single generation pass
//===----------------------------------------------------------------------===//

struct GenerationResult {
  int generatedTokens = 0;
  double prefillSecs = 0.0;
  double decodeSecs = 0.0;
  std::string text;
};

void printStats(const GenerationResult &result, bool verbose = false) {
  const double decodeTPS =
      result.decodeSecs > 0 ? (double)result.generatedTokens / result.decodeSecs
                            : 0;
  if (verbose) {
    std::cerr << "\n";
    std::cerr << "\033[33;1m[Total time]\033[0m "
              << (result.prefillSecs + result.decodeSecs) << "s\n";
  }
  std::cerr << "\033[33;1m[Prefill]\033[0m " << result.prefillSecs << "s";
  if (verbose)
    std::cerr << "\n";
  else
    std::cerr << "  ";
  std::cerr << "\033[33;1m[Decode]\033[0m " << decodeTPS << " tokens/s ("
            << result.generatedTokens << " tokens)\n";
}

/// Run a single generation pass: tokenize → prefill → decode loop.
/// Resets session position before prefill (safe for multi-turn reuse).
static GenerationResult
runGeneration(const std::string &prompt, ModelSession &session,
              MemRef<float, 1> &weights, const std::string &vocabPath,
              int maxNewTokens, const std::vector<long long> &stopTokenIds,
              buddy::Sampler &sampler, bool suppress) {
  GenerationResult result;

  auto isStopToken = [&](int tokenId) -> bool {
    return std::find(stopTokenIds.begin(), stopTokenIds.end(), tokenId) !=
           stopTokenIds.end();
  };

  // Reset session for this generation pass.
  session.resetPosition();

  Text<size_t, 2> inputTokens(prompt);
  inputTokens.tokenizeDeepSeekR1(vocabPath, BUDDY_DSR1_MAX_TOKEN_LEN);

  Text<size_t, 2> outputTokens;
  outputTokens.loadVocab(vocabPath);

  std::vector<int> recentTokens;

  // ── Prefill ─────────────────────────────────────────────────────────────
  const auto t0 = std::chrono::high_resolution_clock::now();
  session.prefill(weights, inputTokens);
  result.prefillSecs = std::chrono::duration<double>(
                           std::chrono::high_resolution_clock::now() - t0)
                           .count();

  // Sample first token from prefill logits.
  const int tokenIndex = static_cast<int>(inputTokens.getTokenCnt()) - 1;
  const float *prefillLogits =
      session.logitsData() + tokenIndex * session.vocabSize();
  int firstToken =
      sampler.sample(prefillLogits, session.vocabSize(), recentTokens);
  recentTokens.push_back(firstToken);

  if (isStopToken(firstToken)) {
    std::cout << std::endl;
    return result;
  }

  outputTokens.appendTokenIdx(firstToken);
  std::string lastPrinted;
  streamNewText(outputTokens, lastPrinted);

  // ── Decode loop ─────────────────────────────────────────────────────────
  int curToken = firstToken;
  const int maxSteps = (maxNewTokens <= 0)
                           ? std::numeric_limits<int>::max()
                           : maxNewTokens - (int)inputTokens.getTokenCnt();
  double decodeAccumMs = 0.0;
  int decodeCount = 0;

  for (int step = 1; step <= maxSteps; ++step) {
    // Check for user interrupt (Ctrl+C).
    if (g_interrupted.load(std::memory_order_relaxed)) {
      g_interrupted = false;
      std::cerr << "\n[Generation interrupted]\n";
      break;
    }

    // Handle KV cache overflow before decode.
    if (session.position() >= BUDDY_DSR1_MAX_TOKEN_LEN) {
      printLog("KV cache overflow at position " +
                   std::to_string(session.position()) + ", discarding...",
               suppress);
      session.handleKVCacheOverflow(kKeepTokenNum, kRopeTheta);
      printLog("New position: " + std::to_string(session.position()), suppress);
    }

    const auto ds = std::chrono::high_resolution_clock::now();
    session.decode(weights, curToken);
    const double stepMs = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - ds)
                              .count();
    decodeAccumMs += stepMs;
    decodeCount += 1;

    int nextToken =
        sampler.sample(session.logitsData(), session.vocabSize(), recentTokens);
    recentTokens.push_back(nextToken);
    // Cap to sampler's repeat penalty window to avoid unbounded growth.
    if ((int)recentTokens.size() > sampler.config().repeatLastN * 2)
      recentTokens.erase(recentTokens.begin(),
                         recentTokens.end() - sampler.config().repeatLastN);

    if (isStopToken(nextToken))
      break;

    outputTokens.appendTokenIdx(nextToken);
    streamNewText(outputTokens, lastPrinted);
    curToken = nextToken;
  }

  std::cout << std::endl;

  result.generatedTokens = decodeCount;
  result.decodeSecs = decodeAccumMs / 1000.0;
  result.text = outputTokens.revertDeepSeekR1();
  return result;
}

//===----------------------------------------------------------------------===//
// Interactive REPL session
//===----------------------------------------------------------------------===//

static void runInteractiveSession(ModelSession &session,
                                  MemRef<float, 1> &weights,
                                  const std::string &vocabPath,
                                  const RunConfig &cfg,
                                  const std::vector<long long> &stopTokenIds,
                                  buddy::ConversationManager &conv,
                                  buddy::Sampler &sampler) {
  const bool suppress = cfg.suppressStats;

  std::cerr << "\033[33;1mInteractive mode\033[0m (type :exit to quit)\n"
            << "  :exit / :quit   Exit\n"
            << "  :clear          Clear conversation history\n"
            << "  /history        Show conversation history\n"
            << "  /system <text>  Set system prompt\n"
            << "  Line ending \\  Multi-line continuation\n\n";

  // Install SIGINT handler (Ctrl+C interrupts generation, not process).
  struct sigaction sa, oldSa;
  sa.sa_handler = interruptHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0; // Don't restart getline on signal.
  sigaction(SIGINT, &sa, &oldSa);

  std::string bufferedInput;

  while (true) {
    // Prompt.
    std::cerr << (bufferedInput.empty() ? ">>> " : "... ");
    std::string line;
    if (!std::getline(std::cin, line)) {
      // EOF or interrupted.
      if (g_interrupted) {
        g_interrupted = false;
        std::cin.clear();
        std::cerr << "\n";
        bufferedInput.clear();
        continue;
      }
      break; // Real EOF.
    }

    if (g_interrupted) {
      g_interrupted = false;
      bufferedInput.clear();
      continue;
    }

    // ── Commands (only when not in multi-line continuation) ──
    if (bufferedInput.empty()) {
      if (line == ":exit" || line == ":quit")
        break;

      if (line == ":clear") {
        conv.clearHistory();
        std::cerr << "Conversation history cleared.\n";
        continue;
      }

      if (line == "/history") {
        const auto &msgs = conv.messages();
        if (msgs.empty()) {
          std::cerr << "(empty)\n";
        } else {
          for (const auto &m : msgs)
            std::cerr << "\033[36m[" << m.role << "]\033[0m " << m.content
                      << "\n";
        }
        if (!conv.systemPrompt().empty())
          std::cerr << "\033[33m[system]\033[0m " << conv.systemPrompt()
                    << "\n";
        continue;
      }

      if (line.rfind("/system ", 0) == 0) {
        conv.setSystemPrompt(line.substr(8));
        std::cerr << "System prompt updated.\n";
        continue;
      }
    }

    // ── Multi-line continuation with backslash ──
    if (!line.empty() && line.back() == '\\') {
      line.pop_back();
      if (!bufferedInput.empty())
        bufferedInput += '\n';
      bufferedInput += line;
      continue;
    }

    if (!bufferedInput.empty()) {
      bufferedInput += '\n';
      bufferedInput += line;
    } else {
      bufferedInput = line;
    }

    if (bufferedInput.empty())
      continue;

    // ── Generate response ──
    conv.addMessage("user", bufferedInput);
    // Build single-turn prompt (system + current user message only).
    std::vector<buddy::Message> msgs = {{"user", bufferedInput}};
    if (!conv.systemPrompt().empty())
      msgs.insert(msgs.begin(), {"system", conv.systemPrompt()});
    std::string fullPrompt = conv.chatTemplate().apply(msgs);

    GenerationResult result =
        runGeneration(fullPrompt, session, weights, vocabPath, cfg.maxNewTokens,
                      stopTokenIds, sampler, suppress);

    conv.addMessage("assistant", result.text);

    if (!suppress)
      printStats(result);

    bufferedInput.clear();
  }

  // Restore original signal handler.
  sigaction(SIGINT, &oldSa, nullptr);
  std::cerr << "\nBye.\n";
}

} // namespace

//===----------------------------------------------------------------------===//
// DeepSeekR1Runner::run
//===----------------------------------------------------------------------===//

void DeepSeekR1Runner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  const bool suppress = cfg.suppressStats;

  if (!suppress)
    std::cerr
        << "\033[33;1mDeepSeekR1 Inference (buddy-cli / BuddyRuntime)\033[0m\n";

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
  std::string weightsPath;
  std::string vocabPath;

  if (!cfg.raxPath.empty()) {
    printLog("Manifest: " + cfg.raxPath, suppress);
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);

    weightsPath = manifest.weightsPath;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;

    printLog("  .so     = " + manifest.soPath, suppress);
    printLog("  weights = " + manifest.weightsPath, suppress);
    printLog("  vocab   = " + vocabPath, suppress);
  } else {
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath (--model-so).");
    if (cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires weightsPath (--weights).");

    weightsPath = cfg.weightsPath;
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

  // ── Load weights ─────────────────────────────────────────────────────────
  intptr_t weightsShape[1] = {BUDDY_DSR1_PARAMS_SIZE};
  MemRef<float, 1> weights(weightsShape);
  loadWeights(weightsPath, weights, suppress);

  printLog("Vocab: " + vocabPath, suppress);
  printLog("KV cache: " + std::to_string(BUDDY_DSR1_KV_LAYERS) + " x {1," +
               std::to_string(BUDDY_DSR1_HEAD_NUM) + "," +
               std::to_string(BUDDY_DSR1_MAX_TOKEN_LEN) + "," +
               std::to_string(BUDDY_DSR1_HIDDEN_SIZE) + "} f32",
           suppress);

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
          tmp.tokenizeDeepSeekR1(vocabPath, BUDDY_DSR1_MAX_TOKEN_LEN);
          return tmp.getTokenCnt();
        });
    if (!cfg.prompt.empty())
      conv.setSystemPrompt(cfg.prompt);

    runInteractiveSession(*session, weights, vocabPath, cfg, stopTokenIds, conv,
                          sampler);
  } else {
    // Single-shot: format prompt with chat template if available.
    std::string finalPrompt = cfg.prompt;
    if (chatTmpl) {
      std::vector<buddy::Message> msgs = {{"user", cfg.prompt}};
      finalPrompt = chatTmpl->apply(msgs);
    }

    GenerationResult result =
        runGeneration(finalPrompt, *session, weights, vocabPath,
                      cfg.maxNewTokens, stopTokenIds, sampler, suppress);

    if (!suppress)
      printStats(result, /*verbose=*/true);
  }
}

} // namespace runtime
} // namespace buddy
