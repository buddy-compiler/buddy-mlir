//===- ConversationManager.h
//-----------------------------------------------===//
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
// Conversation history manager for multi-turn LLM chat.
//
// Maintains a list of messages (system, user, assistant), applies the chat
// template to produce the prompt string, and handles history manipulation
// (clear, regenerate, oldest-turn truncation on context overflow).
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_CONVERSATIONMANAGER
#define FRONTEND_INTERFACES_BUDDY_LLM_CONVERSATIONMANAGER

#include "buddy/LLM/ChatTemplate.h"
#include <functional>
#include <string>
#include <vector>

namespace buddy {

class ConversationManager {
public:
  /// Construct with a chat template and an optional token-counting function.
  /// tokenCounter takes a string and returns its token count. If not provided,
  /// a rough byte-based estimate is used (1 token per 4 bytes).
  explicit ConversationManager(
      ChatTemplate tmpl,
      std::function<size_t(const std::string &)> tokenCounter = nullptr)
      : template_(std::move(tmpl)), tokenCounter_(std::move(tokenCounter)) {}

  void setSystemPrompt(const std::string &content) { systemPrompt_ = content; }

  const std::string &systemPrompt() const { return systemPrompt_; }

  void addMessage(const std::string &role, const std::string &content) {
    history_.push_back({role, content});
  }

  /// Build the full prompt by applying the chat template to all messages.
  std::string buildPrompt() const {
    std::vector<Message> all;
    if (!systemPrompt_.empty()) {
      all.push_back({"system", systemPrompt_});
    }
    all.insert(all.end(), history_.begin(), history_.end());
    return template_.apply(all);
  }

  /// Build a prompt that fits within maxTokens by dropping oldest turns.
  /// Returns the prompt string. If even the current turn alone exceeds
  /// maxTokens, returns it anyway (caller handles the overflow).
  std::string buildPromptWithLimit(size_t maxTokens) const {
    // Try full history first.
    std::string prompt = buildPrompt();
    if (countTokens(prompt) <= maxTokens) {
      return prompt;
    }

    // Progressively drop oldest user+assistant turn pairs.
    // Find how many turns (pairs) we have.
    size_t dropCount = 0;
    size_t totalPairs = countTurnPairs();

    while (dropCount < totalPairs) {
      ++dropCount;
      std::vector<Message> all;
      if (!systemPrompt_.empty()) {
        all.push_back({"system", systemPrompt_});
      }
      // Skip the oldest dropCount turn pairs.
      size_t skipped = 0;
      for (const auto &msg : history_) {
        if (skipped < dropCount * 2) {
          // Skip user+assistant pairs from the front.
          if (msg.role == "user" || msg.role == "assistant") {
            ++skipped;
            continue;
          }
        }
        all.push_back(msg);
      }
      prompt = template_.apply(all);
      if (countTokens(prompt) <= maxTokens) {
        return prompt;
      }
    }

    // Nothing left to drop; return whatever we have.
    return prompt;
  }

  /// Remove the last assistant message for regeneration.
  /// Returns false if there is no assistant message to remove.
  bool removeLastAssistantMessage() {
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
      if (it->role == "assistant") {
        history_.erase(std::next(it).base());
        return true;
      }
    }
    return false;
  }

  /// Remove the last message regardless of role.
  bool removeLastMessage() {
    if (history_.empty())
      return false;
    history_.pop_back();
    return true;
  }

  /// Clear all messages (keeps system prompt).
  void clearHistory() { history_.clear(); }

  size_t turnCount() const { return history_.size(); }

  const std::vector<Message> &messages() const { return history_; }

  const ChatTemplate &chatTemplate() const { return template_; }

private:
  ChatTemplate template_;
  std::string systemPrompt_;
  std::vector<Message> history_;
  std::function<size_t(const std::string &)> tokenCounter_;

  size_t countTokens(const std::string &text) const {
    if (tokenCounter_) {
      return tokenCounter_(text);
    }
    // Rough estimate: 1 token per 4 bytes.
    return (text.size() + 3) / 4;
  }

  /// Count user+assistant turn pairs in history.
  size_t countTurnPairs() const {
    size_t users = 0, assistants = 0;
    for (const auto &msg : history_) {
      if (msg.role == "user")
        ++users;
      else if (msg.role == "assistant")
        ++assistants;
    }
    return std::min(users, assistants);
  }
};

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_CONVERSATIONMANAGER
