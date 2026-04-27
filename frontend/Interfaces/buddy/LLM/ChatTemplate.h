//===- ChatTemplate.h -----------------------------------------------------===//
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
// Chat template for formatting multi-turn conversations.
//
// Loads a JSON config that defines how role markers and special tokens are
// arranged, then applies it to a message list to produce the prompt string
// expected by the model.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE
#define FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {

struct Message {
  std::string role; // "system", "user", "assistant"
  std::string content;
};

class ChatTemplate {
public:
  /// Load a chat template from a JSON config file.
  /// Throws std::runtime_error on failure.
  static ChatTemplate fromFile(const std::string &path) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr) {
      throw std::runtime_error("Failed to open chat template file: " + path);
    }
    auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
    if (!parsed) {
      throw std::runtime_error("Failed to parse chat template JSON: " +
                               llvm::toString(parsed.takeError()));
    }
    return fromJSON(*parsed);
  }

  /// Apply the template to a message list, producing the full prompt string.
  std::string apply(const std::vector<Message> &messages) const {
    std::string result;

    if (addBos_) {
      result += bosToken_;
    }

    std::string pendingSystem;

    for (const auto &msg : messages) {
      // If systemInFirstUser is set, defer the system message content.
      if (msg.role == "system" && systemInFirstUser_) {
        pendingSystem = msg.content;
        continue;
      }

      auto it = rolePrefixes_.find(msg.role);
      if (it != rolePrefixes_.end()) {
        result += it->second;
      }

      // Prepend deferred system content into the first user message.
      if (msg.role == "user" && !pendingSystem.empty()) {
        result += pendingSystem;
        result += "\n\n";
        pendingSystem.clear();
      }

      result += msg.content;
      result += roleSuffix_;
      result += turnSuffix_;
    }

    if (addGenerationPrompt_) {
      auto it = rolePrefixes_.find("assistant");
      if (it != rolePrefixes_.end()) {
        result += it->second;
      }
    }

    return result;
  }

  const std::vector<int> &stopTokenIds() const { return stopTokenIds_; }
  const std::vector<std::string> &stopTokens() const { return stopTokens_; }
  const std::string &bosToken() const { return bosToken_; }
  const std::string &eosToken() const { return eosToken_; }

private:
  std::string bosToken_;
  std::string eosToken_;
  std::map<std::string, std::string> rolePrefixes_;
  std::string roleSuffix_;
  std::string turnSuffix_;
  std::vector<std::string> stopTokens_;
  std::vector<int> stopTokenIds_;
  bool addBos_ = true;
  bool addGenerationPrompt_ = true;
  bool systemInFirstUser_ = false;

  static std::string getStr(const llvm::json::Object &obj, llvm::StringRef key,
                            llvm::StringRef defaultVal = "") {
    if (auto v = obj.getString(key))
      return v->str();
    return defaultVal.str();
  }

  static bool getBool(const llvm::json::Object &obj, llvm::StringRef key,
                      bool defaultVal = false) {
    if (auto v = obj.getBoolean(key))
      return *v;
    return defaultVal;
  }

  static ChatTemplate fromJSON(const llvm::json::Value &root) {
    ChatTemplate tmpl;
    auto *obj = root.getAsObject();
    if (!obj) {
      throw std::runtime_error("Chat template JSON root must be an object");
    }

    tmpl.bosToken_ = getStr(*obj, "bos_token");
    tmpl.eosToken_ = getStr(*obj, "eos_token");
    tmpl.roleSuffix_ = getStr(*obj, "role_suffix");
    tmpl.turnSuffix_ = getStr(*obj, "turn_suffix");
    tmpl.addBos_ = getBool(*obj, "add_bos", true);
    tmpl.addGenerationPrompt_ = getBool(*obj, "add_generation_prompt", true);
    tmpl.systemInFirstUser_ = getBool(*obj, "system_in_first_user", false);

    if (auto *roles = obj->getObject("roles")) {
      for (const auto &kv : *roles) {
        if (auto str = kv.second.getAsString()) {
          tmpl.rolePrefixes_[kv.first.str()] = str->str();
        }
      }
    }

    if (auto *arr = obj->getArray("stop_tokens")) {
      for (const auto &v : *arr) {
        if (auto s = v.getAsString()) {
          tmpl.stopTokens_.push_back(s->str());
        }
      }
    }

    if (auto *arr = obj->getArray("stop_token_ids")) {
      for (const auto &v : *arr) {
        if (auto n = v.getAsInteger()) {
          tmpl.stopTokenIds_.push_back(static_cast<int>(*n));
        }
      }
    }

    return tmpl;
  }
};

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE
