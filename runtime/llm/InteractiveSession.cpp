//===- InteractiveSession.cpp - REPL interactive mode ---------------------===//
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

#include "buddy/runtime/llm/InteractiveSession.h"
#include "buddy/runtime/llm/TextGeneration.h"

#include <csignal>
#include <iostream>
#include <string>

namespace buddy {
namespace runtime {

void runInteractiveSession(LLMSession &session, const std::string &vocabPath,
                           const RunConfig &cfg,
                           const std::vector<long long> &stopTokenIds,
                           buddy::ConversationManager &conv,
                           const TextCodec &codec, buddy::Sampler &sampler) {
  const bool suppress = cfg.suppressStats;
  int turnNumber = 0;

  static constexpr int kThinkTokenId = 151648;
  static constexpr int kEndThinkTokenId = 151649;
  static constexpr float kRopeTheta = 10000.0f;

  TextCodec rawCodec;
  const int maxTokLen = codec.maxTokenLen;
  rawCodec.tokenize = [maxTokLen](Text<size_t, 2> &t,
                                  const std::string &vocab) {
    t.tokenizeDeepSeekR1Raw(vocab, maxTokLen);
  };
  rawCodec.detokenize = codec.detokenize;
  rawCodec.maxTokenLen = codec.maxTokenLen;

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
        session.resetPosition();
        turnNumber = 0;
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
    turnNumber++;

    std::string prompt;
    bool resetKV;

    if (turnNumber == 1) {
      std::vector<buddy::Message> msgs = {{"user", bufferedInput}};
      if (!conv.systemPrompt().empty())
        msgs.insert(msgs.begin(), {"system", conv.systemPrompt()});
      prompt = conv.chatTemplate().apply(msgs);
      resetKV = true;
    } else {
      prompt = conv.buildIncrementalPrompt(bufferedInput);
      resetKV = false;

      Text<size_t, 2> probe(prompt);
      rawCodec.tokenize(probe, vocabPath);
      const int newTokens = static_cast<int>(probe.getTokenCnt());
      if (session.position() + newTokens >= codec.maxTokenLen) {
        session.handleKVCacheOverflow(codec.maxTokenLen / 4, kRopeTheta);
        if (session.position() + newTokens >= codec.maxTokenLen) {
          std::vector<buddy::Message> msgs = {{"user", bufferedInput}};
          if (!conv.systemPrompt().empty())
            msgs.insert(msgs.begin(), {"system", conv.systemPrompt()});
          prompt = conv.chatTemplate().apply(msgs);
          resetKV = true;
        }
      }
    }

    const TextCodec &activeCodec = resetKV ? codec : rawCodec;
    GenerationResult result =
        runGeneration(prompt, session, vocabPath, cfg.maxNewTokens,
                      stopTokenIds, sampler, activeCodec, suppress, resetKV,
                      kThinkTokenId, kEndThinkTokenId);

    conv.addMessage("assistant", result.text);

    if (result.thinkingRange.isComplete()) {
      const int keepNum = result.thinkingRange.startPos;
      const int discardLen =
          result.thinkingRange.endPos - result.thinkingRange.startPos + 1;
      printLog("Discarding thinking tokens [" +
                   std::to_string(result.thinkingRange.startPos) + ", " +
                   std::to_string(result.thinkingRange.endPos) + "] (" +
                   std::to_string(discardLen) + " tokens)",
               suppress);
      session.discardKVRange(keepNum, discardLen, kRopeTheta);
      printLog("Position after discard: " + std::to_string(session.position()),
               suppress);
    }

    if (!suppress)
      printStats(result);

    bufferedInput.clear();
  }

  // Restore original signal handler.
  sigaction(SIGINT, &oldSa, nullptr);
  std::cerr << "\nBye.\n";
}

} // namespace runtime
} // namespace buddy
