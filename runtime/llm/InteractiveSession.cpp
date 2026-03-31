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

void runInteractiveSession(LLMSession &session, MemRef<float, 1> &weights,
                           const std::string &vocabPath, const RunConfig &cfg,
                           const std::vector<long long> &stopTokenIds,
                           buddy::ConversationManager &conv,
                           const TextCodec &codec, buddy::Sampler &sampler) {
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
                      stopTokenIds, sampler, codec, suppress);

    conv.addMessage("assistant", result.text);

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
