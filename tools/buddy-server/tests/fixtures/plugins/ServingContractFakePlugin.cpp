//===- ServingContractFakePlugin.cpp - Synthetic HTTP contract fixture ----===//
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

// Token bytes and scripted logits belong only to the synthetic contract. The
// normal path executes the production serving controller, not a second budget
// or stopping implementation. Named adversarial prompts exercise transport
// behavior that a well-behaved controller does not normally produce.

#include "buddy/runtime/core/ModelPlugin.h"

#if defined(BUDDY_CONTRACT_EMBEDDING)
#include "buddy/runtime/core/EmbeddingModelPlugin.h"
#else
#include "buddy/runtime/core/ResidentModelPlugin.h"
#include "buddy/runtime/llm/ServingGeneration.h"
#include "buddy/runtime/llm/TextGeneration.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace buddy::runtime;

std::string setting(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}

#if !defined(BUDDY_CONTRACT_EMBEDDING)
void record(const std::string &event) {
  const auto marker = setting("BUDDY_CONTRACT_MARKER");
  if (!marker.empty()) {
    if (FILE *file = std::fopen(marker.c_str(), "a")) {
      std::fputs((event + "\n").c_str(), file);
      std::fclose(file);
    }
  }
}
#endif

void loadStatus(ModelStatus &status, const std::string &name) {
  status.modelName = name.empty() ? "issue912_fixture" : name;
  status.backend = "synthetic-contract";
  status.contextLength = 16;
  const auto state = setting("BUDDY_CONTRACT_LOAD_STATE");
  if (state == "loading")
    std::this_thread::sleep_for(std::chrono::seconds(30));
  if (state == "error")
    throw std::runtime_error("fixture load failure");
  status.state =
      state == "unloaded" ? ModelLoadState::Unloaded : ModelLoadState::Ready;
}

#if defined(BUDDY_CONTRACT_EMBEDDING)
class FakeModel final : public EmbeddingModel {
public:
  void load(const EmbeddingModelConfig &config) override {
    loadStatus(statusValue, config.modelName);
  }

  ModelStatus status() const override { return statusValue; }

  EmbeddingResult embed(const EmbeddingRequest &) override {
    return {statusValue.modelName, {0.25F, 0.75F}, 1, 1};
  }

private:
  ModelStatus statusValue;
};
#else
std::string hex(const std::string &text) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : text) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

std::string tokenBytes(int token) {
  static const std::array<std::string, 13> bytes = {"<eos>",
                                                    "A",
                                                    "B",
                                                    std::string("\xe4\xb8", 2),
                                                    std::string("\xe4", 1),
                                                    std::string("\xb8", 1),
                                                    std::string("\xad", 1),
                                                    " ",
                                                    "\n",
                                                    "\"",
                                                    std::string(1, '\0'),
                                                    std::string("\xff", 1),
                                                    "<bos>"};
  if (token < 0 || token >= static_cast<int>(bytes.size()))
    throw std::invalid_argument("tokens: token ID is out of range");
  return bytes[token];
}

std::vector<int> encode(const std::string &text, bool parseSpecial = true) {
  if (text.empty())
    return {};
  if (text == "AB")
    return {1, 2};
  if (text == "<eos>" && parseSpecial)
    return {0};
  if (text == "overflow")
    return std::vector<int>(17, 1);
  if (text == "almost-full")
    return std::vector<int>(15, 1);
  return {1};
}

class ScriptedSession final : public LLMSession {
public:
  std::string mode;
  std::vector<int> script = {1, 2, 0};

  void loadWeights(const std::vector<std::string> &) override {}

  void prefill(buddy::Text<std::size_t, 2> &tokens) override {
    record("prefill");
    if (mode == "disconnect-before-first")
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (mode == "throw-prefill")
      throw std::runtime_error("fixture prefill failure");
    currentPosition = static_cast<int>(tokens.getTokenCnt());
    scriptIndex = 0;
  }

  void decode(int) override {
    if (mode == "throw-midstream")
      throw std::runtime_error("fixture midstream failure");
    if (mode == "disconnect" || mode == "disconnect-before-first")
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++scriptIndex;
    ++currentPosition;
  }

  void resetPosition() override {
    currentPosition = 0;
    scriptIndex = 0;
    record("reset-position");
  }

  int position() const override { return currentPosition; }

  const float *logitsData(int = 0) const override {
    record("sample");
    logits.fill(-1000);
    const int token = mode == "disconnect" || mode == "disconnect-before-first"
                          ? 1
                          : script[std::min(scriptIndex, script.size() - 1)];
    logits[token] = 1000;
    return logits.data();
  }

  int vocabSize() const override { return 13; }
  bool handleKVCacheOverflow(int, float = 10000) override { return false; }

private:
  int currentPosition = 0;
  std::size_t scriptIndex = 0;
  mutable std::array<float, 13> logits;
};

class FakeModel final : public ResidentModel {
public:
  void load(const ResidentModelConfig &config) override {
    loadStatus(statusValue, config.modelName);
    const auto script = setting("BUDDY_CONTRACT_TOKEN_SCRIPT");
    if (!script.empty()) {
      session.script.clear();
      std::istringstream input(script);
      std::string token;
      while (std::getline(input, token, ',')) {
        const int id = std::stoi(token);
        (void)tokenBytes(id);
        session.script.push_back(id);
      }
    }
  }

  ModelStatus status() const override { return statusValue; }

  ResidentModelCapabilities capabilities() const override {
    ResidentModelCapabilities caps;
    caps.rawTokenize = caps.rawDetokenize = true;
    caps.tokenizeAddSpecial = caps.tokenizeParseSpecial =
        caps.tokenizeWithPieces = true;
    caps.chatRoles = {"system", "user", "assistant"};
    caps.chatMultiTurn = caps.streaming = caps.maxNewTokens = true;
    caps.stop = caps.stopTokenIds = true;
    caps.sampling = {true, true, true, true, true, true, true};
    if (setting("BUDDY_CONTRACT_UNSUPPORTED") == "temperature")
      caps.sampling.temperature = false;
    return caps;
  }

  std::string renderChat(const ChatCompletionRequest &request) override {
    record("template");
    for (const auto &message : request.messages)
      record("render:" + message.role + ":" + hex(message.content));
    // The contract's synthetic chat template always has one prompt token.
    // Preserve adversarial test modes so both HTTP generation routes run them.
    if (!request.messages.empty())
      return request.messages.back().content;
    return request.input;
  }

  TokenizeResult tokenize(const TokenizeRequest &request) override {
    TokenizeResult result;
    result.tokens = encode(request.content, request.parseSpecial);
    if (request.addSpecial)
      result.tokens.insert(result.tokens.begin(), 12);
    result.count = result.tokens.size();
    if (request.withPieces)
      for (int token : result.tokens)
        result.pieces.push_back(tokenBytes(token));
    return result;
  }

  DetokenizeResult detokenize(const DetokenizeRequest &request) override {
    std::string bytes;
    for (int token : request.tokens)
      bytes += tokenBytes(token);
    Utf8StreamDecoder decoder;
    std::string content = decoder.append(bytes);
    content += decoder.finish();
    return {content};
  }

  CompletionResult complete(const CompletionRequest &request) override {
    std::lock_guard<std::mutex> lock(generationMutex);
    return generate(request, {});
  }

  CompletionResult
  completeStream(const CompletionRequest &request,
                 const CompletionStreamCallback &callback) override {
    std::lock_guard<std::mutex> lock(generationMutex);
    return generate(request, callback);
  }

  CompletionResult chat(const ChatCompletionRequest &request) override {
    CompletionRequest completion;
    completion.prompt = renderChat(request);
    completion.sampling = request.sampling;
    return complete(completion);
  }

  CompletionResult
  chatStream(const ChatCompletionRequest &request,
             const CompletionStreamCallback &callback) override {
    CompletionRequest completion;
    completion.prompt = renderChat(request);
    completion.sampling = request.sampling;
    return completeStream(completion, callback);
  }

private:
  CompletionResult generate(const CompletionRequest &request,
                            const CompletionStreamCallback &callback) {
    record("request");
    record(callback ? "stream-request" : "plain-request");
    record(request.sampling.seedPolicy == SamplingSeedPolicy::Fixed
               ? "seed:fixed:" + std::to_string(request.sampling.fixedSeed)
               : "seed:random");
    const auto before = setting("BUDDY_CONTRACT_THROW_BEFORE_STREAM");
    if (!before.empty())
      throw std::runtime_error(before);
    if (request.prompt == "final-delta") {
      CompletionResult result;
      result.content = "tail";
      result.finishReason = FinishReason::Length;
      result.usage = {1, 1, 2};
      if (callback) {
        CompletionChunk chunk;
        chunk.delta = "tail";
        chunk.done = true;
        chunk.finishReason = result.finishReason;
        chunk.usage = result.usage;
        if (!callback(chunk))
          result.finishReason = FinishReason::Cancelled;
      }
      return result;
    }
    session.mode = request.prompt;
    auto sampling = request.sampling;
    sampling.stopTokenIds.push_back(0);
    ServingTextCodec codec;
    codec.contextLength = 16;
    codec.encode = [](const std::string &text) { return encode(text); };
    codec.tokenBytes = [](int token) { return tokenBytes(token); };
    CompletionStreamCallback observe;
    if (callback)
      observe = [&](const CompletionChunk &chunk) {
        record("callback");
        const bool keepGoing = callback(chunk);
        if (!keepGoing)
          record("callback-false");
        return keepGoing;
      };
    auto result =
        runServingGeneration(request.prompt, session, sampling, codec, observe);
    if (result.finishReason == FinishReason::Cancelled)
      record("return-cancelled:" +
             std::to_string(result.usage.completionTokens));
    return result;
  }

  ModelStatus statusValue;
  ScriptedSession session;
  std::mutex generationMutex;
};
#endif

} // namespace

#if defined(BUDDY_CONTRACT_EMBEDDING)
extern "C" buddy::runtime::EmbeddingModel *buddy_create_embedding_model_v1() {
  return new FakeModel();
}
extern "C" void
buddy_destroy_embedding_model_v1(buddy::runtime::EmbeddingModel *model) {
  delete model;
}
extern "C" const char *buddy_embedding_model_type_v1() {
  return "issue912_fixture";
}
#else
extern "C" buddy::runtime::ResidentModel *buddy_create_resident_model_v2() {
  return new FakeModel();
}
extern "C" void
buddy_destroy_resident_model_v2(buddy::runtime::ResidentModel *model) {
  delete model;
}
extern "C" const char *buddy_resident_model_type_v2() {
  return "issue912_fixture";
}
#endif

extern "C" const buddy::runtime::ModelPluginDescriptorV1 *
buddy_model_plugin_descriptor_v1() {
  static constexpr buddy::runtime::ModelPluginDescriptorV1 descriptor = {
      sizeof(descriptor), buddy::runtime::ModelPluginDescriptorAbiVersionV1,
#if defined(BUDDY_CONTRACT_EMBEDDING)
      buddy::runtime::ModelServiceKind::Embedding,
#else
      buddy::runtime::ModelServiceKind::Resident,
#endif
      0};
  return &descriptor;
}
