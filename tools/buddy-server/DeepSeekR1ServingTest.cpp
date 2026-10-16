//===- DeepSeekR1ServingTest.cpp - Production DeepSeek serving checks -----===//
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

#include "buddy/runtime/models/DeepSeekR1RawCodec.h"
#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace buddy::runtime;

namespace {

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

template <typename Function> void reject(Function function) {
  bool failed = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  require(failed, "expected an invalid argument");
}

std::string utf8(unsigned int code) {
  if (code < 128)
    return std::string(1, static_cast<char>(code));
  return {static_cast<char>(0xc0 | (code >> 6)),
          static_cast<char>(0x80 | (code & 0x3f))};
}

std::string bytePiece(unsigned int value) {
  unsigned int extra = 0;
  for (unsigned int byte = 0; byte < 256; ++byte) {
    const bool direct = (byte >= 33 && byte <= 126) ||
                        (byte >= 161 && byte <= 172) || byte >= 174;
    const unsigned int code = direct ? byte : 256 + extra++;
    if (byte == value)
      return utf8(code);
  }
  throw std::runtime_error("invalid fixture byte");
}

class Fixture {
public:
  Fixture() {
    auto pattern = (std::filesystem::temp_directory_path() /
                    "buddy-deepseek-serving-XXXXXX")
                       .string();
    const char *directory = mkdtemp(pattern.data());
    require(directory, "could not create fixture directory");
    root = directory;
    vocab = root / "vocab.txt";
    chat = root / "chat.json";
    std::ofstream output(vocab, std::ios::binary);
    for (unsigned int byte = 0; byte < 256; ++byte)
      output << bytePiece(byte) << '\n';
    output << "<|end▁of▁sentence|>\n<|begin▁of▁sentence|>\n"
              "<|User|>\n<|Assistant|>\n<|EOT|>\n<think>\n";
    output << bytePiece(0xe4) << bytePiece(0xb8) << '\n';
    output << "中\n";
    output.close();
    std::ofstream config(chat);
    config << R"({
      "bos_token":"<|begin▁of▁sentence|>",
      "roles":{"system":"","user":"<|User|>",
               "assistant":"<|Assistant|>"},
      "turn_suffix":"<|end▁of▁sentence|>",
      "stop_token_ids":[256,260],
      "add_bos":true,"add_generation_prompt":true,
      "system_in_first_user":true
    })";
  }

  ~Fixture() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }

  std::filesystem::path root;
  std::filesystem::path vocab;
  std::filesystem::path chat;
};

void checkCodec(const Fixture &fixture, const std::string &productionVocab) {
  DeepSeekR1RawCodec codec(fixture.vocab.string());
  const std::string replacement = "\xef\xbf\xbd";
  require(codec.detokenize({{262}}).content == replacement,
          "incomplete token must finish as one replacement");
  require(codec.detokenize({{0xe4, 0xb8, 0xad}}).content == "中",
          "token bytes must be joined before UTF-8 repair");
  require(codec.detokenize({{263, 262}}).content == "中" + replacement,
          "complete character and incomplete tail changed");
  require(codec.detokenize({{0xff}}).content == replacement,
          "invalid raw bytes must be repaired");
  require(codec.detokenize({}).content.empty(), "empty decoding changed");
  std::string preserved = "  <|end▁of▁sentence|>A";
  preserved += '\0';
  preserved += '\n';
  require(codec.detokenize({{32, 32, 256, 'A', 0, '\n'}}).content == preserved,
          "raw decoding lost spaces, special tokens, EOS tail, or NUL");
  reject([&] { (void)codec.detokenize({{-1}}); });
  reject([&] { (void)codec.detokenize({{256, codec.vocabSize()}}); });

  TokenizeRequest request;
  request.content = "  A\n中\t";
  request.addSpecial = false;
  auto encoded = codec.tokenize(request);
  require(codec.detokenize({encoded.tokens}).content == request.content,
          "raw tokenization changed whitespace or Unicode");
  require(codec.encode("", false).empty(), "empty input gained chat tokens");
  require(codec.encode(std::string(2048, 'A')).size() == 2048,
          "raw tokenization must not truncate input at the model context");
  require(codec.encode("A", true) == std::vector<int>({257, 'A'}),
          "add_special must add only model BOS");
  require(codec.encode("<|begin▁of▁sentence|>A", true) ==
              std::vector<int>({257, 'A'}),
          "existing BOS was added twice");
  require(codec.encode("<|User|>", false, true) == std::vector<int>({258}),
          "parse_special failed to recognize a vocabulary special token");
  auto ordinary = codec.encode("<|User|>", false, false);
  require(ordinary != std::vector<int>({258}) &&
              codec.detokenize({ordinary}).content == "<|User|>",
          "parse_special=false must encode literal ordinary text");
  request.countOnly = true;
  auto counted = codec.tokenize(request);
  require(counted.tokens.empty() && counted.count == encoded.count,
          "count_only does not count the actual encoding");
  request.withPieces = true;
  reject([&] { (void)codec.tokenize(request); });

  DeepSeekR1RawCodec production(productionVocab);
  // Recorded from the official DeepSeek-R1-Distill-Qwen-1.5B tokenizer at
  // revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 with no added BOS.
  // These samples pin the tested vocabulary without claiming BPE equivalence.
  const std::vector<std::pair<std::string, std::vector<int>>> examples = {
      {"  Hello\n\tworld", {220, 21927, 198, 76508}},
      {"你好，中\n", {108386, 3837, 15946, 198}},
      {"<think> A", {151648, 362}},
      {"Hello, world!", {9707, 11, 1879, 0}},
      {"The capital of France is Paris.",
       {785, 6722, 315, 9625, 374, 12095, 13}},
      {"A simple test with repeated words words and  spaces.",
       {32, 4285, 1273, 448, 11504, 4244, 4244, 323, 220, 12621, 13}},
      {"2 + 2 = 4.", {17, 488, 220, 17, 284, 220, 19, 13}},
      {"你好，世界！", {108386, 3837, 99489, 6313}},
      {"I'm learning C++ and MLIR.",
       {40, 2776, 6832, 356, 1027, 323, 19614, 2801, 13}}};
  for (const auto &example : examples) {
    auto tokens = production.encode(example.first);
    require(tokens == example.second,
            "production vocabulary token IDs changed for " + example.first);
    require(production.detokenize({tokens}).content == example.first,
            "production vocabulary raw round trip changed text");
  }
  const std::string differing = "antidisestablishmentarianism";
  auto differingTokens = production.encode(differing);
  require(differingTokens == std::vector<int>(
                                 {15359, 4243, 33400, 76, 79680, 72, 67813}) &&
              production.detokenize({differingTokens}).content == differing,
          "documented deterministic segmentation changed");
  const auto eos = production.tokenId("<|end▁of▁sentence|>");
  const auto a = production.tokenId("A");
  require(eos >= 0 && a >= 0 &&
              production.detokenize({{eos, a}}).content ==
                  "<|end▁of▁sentence|>A",
          "production decoding stopped at EOS");
}

class ScriptedSession final : public LLMSession {
public:
  explicit ScriptedSession(int vocabulary) : logits(vocabulary, -100.0f) {}

  void loadWeights(const std::vector<std::string> &) override { ++weightLoads; }

  void prefill(buddy::Text<std::size_t, 2> &tokens) override {
    ++prefills;
    prompt.clear();
    for (std::size_t i = 0; i < tokens.getTokenCnt(); ++i)
      prompt.push_back(static_cast<int>(tokens.getData()[i]));
    currentPosition = static_cast<int>(prompt.size());
    if (failPrefill)
      throw std::runtime_error("scripted prefill failure");
    setLogits();
  }

  void decode(int) override {
    ++decodes;
    ++currentPosition;
    ++scriptPosition;
    setLogits();
  }

  void resetPosition() override {
    ++resets;
    currentPosition = 0;
    scriptPosition = 0;
  }

  int position() const override { return currentPosition; }
  const float *logitsData(int = 0) const override { return logits.data(); }
  int vocabSize() const override { return static_cast<int>(logits.size()); }
  bool handleKVCacheOverflow(int, float = 10000.0f) override {
    throw std::runtime_error("serving must not move context");
  }

  void setLogits() {
    std::fill(logits.begin(), logits.end(), -100.0f);
    if (sampleAlternatives) {
      logits['A'] = 0.0f;
      logits['B'] = 0.0f;
    } else {
      const int next =
          scriptPosition < script.size() ? script[scriptPosition] : 256;
      logits[next] = 100.0f;
    }
    // The generated session may expose a padded logits width. These rows do
    // not have raw vocabulary pieces and must never become sampled tokens.
    logits.back() = 1000.0f;
  }

  std::vector<int> script = {'A', 'B', 256};
  std::vector<int> prompt;
  std::vector<float> logits;
  std::size_t scriptPosition = 0;
  int currentPosition = 0;
  int weightLoads = 0;
  int prefills = 0;
  int decodes = 0;
  int resets = 0;
  bool sampleAlternatives = false;
  bool failPrefill = false;
};

void checkResident(const Fixture &fixture) {
  ScriptedSession *session = nullptr;
  DeepSeekR1ResidentModel model([&](const ResidentModelConfig &) {
    DeepSeekR1ResidentModel::SessionResources resources;
    auto scripted = std::make_unique<ScriptedSession>(266);
    session = scripted.get();
    resources.session = std::move(scripted);
    resources.contextLength = 16;
    resources.vocabPath = fixture.vocab.string();
    resources.modelName = "loaded-artifact-name";
    return resources;
  });
  require(!model.capabilities().rawTokenize, "unloaded model advertised codec");
  ResidentModelConfig config;
  config.chatTemplatePath = fixture.chat.string();
  model.load(config);
  const auto caps = model.capabilities();
  require(caps.rawTokenize && caps.rawDetokenize && caps.maxNewTokens &&
              caps.streaming && caps.sampling.seed && caps.chatMultiTurn &&
              !caps.tokenizeWithPieces && caps.maxImages == 0 &&
              model.status().modelName == "loaded-artifact-name" &&
              session->weightLoads == 1,
          "loaded artifact capabilities/status changed");
  TokenizeRequest tokenize;
  tokenize.content = " A";
  tokenize.addSpecial = false;
  require(model.detokenize({model.tokenize(tokenize).tokens}).content == " A",
          "production resident did not use its raw codec");

  CompletionRequest request;
  request.prompt = "A";
  request.sampling.maxNewTokens = 0;
  auto zero = model.complete(request);
  require(zero.content.empty() && zero.usage.promptTokens == 2 &&
              zero.usage.completionTokens == 0 && session->prefills == 1 &&
              session->decodes == 0,
          "zero budget must prefill actual raw prompt without sampling");
  request.sampling.maxNewTokens = 1;
  auto one = model.complete(request);
  require(one.content == "A" && one.usage.completionTokens == 1 &&
              one.finishReason == FinishReason::Length,
          "one-token serving budget changed");
  request.sampling.maxNewTokens = 8;
  std::string streamed;
  int terminals = 0;
  auto result =
      model.completeStream(request, [&](const CompletionChunk &chunk) {
        require(!chunk.id.empty() && chunk.model == "loaded-artifact-name",
                "stream identity is missing");
        streamed += chunk.delta;
        terminals += chunk.done;
        return true;
      });
  require(result.content == "AB" && streamed == result.content &&
              terminals == 1 && result.usage.completionTokens == 3 &&
              result.finishReason == FinishReason::Stop,
          "resident stream content/EOS accounting/terminal event changed");
  session->script = {256};
  result = model.complete(request);
  require(result.content.empty() && result.usage.completionTokens == 1 &&
              result.finishReason == FinishReason::Stop,
          "first-token EOS accounting changed");
  session->script = {'A', 'B', 'C', 'D'};
  request.sampling.maxNewTokens = 3;
  request.sampling.stop = {"BC"};
  result = model.complete(request);
  require(result.content == "A" && result.usage.completionTokens == 3 &&
              result.finishReason == FinishReason::Stop,
          "string stop must win over coincident length limit");
  request.sampling.stop.clear();
  request.sampling.maxNewTokens = 8;
  auto cancelled = model.completeStream(
      request, [](const CompletionChunk &chunk) { return chunk.done; });
  require(cancelled.finishReason == FinishReason::Cancelled,
          "disconnect must cancel production generation");
  session->script = {'A', 256};
  require(model.complete(request).content == "A",
          "cancelled session was not reusable");

  session->failPrefill = true;
  bool failed = false;
  try {
    (void)model.complete(request);
  } catch (const std::runtime_error &) {
    failed = true;
  }
  session->failPrefill = false;
  require(failed && model.complete(request).content == "A",
          "failed prefill did not leave a reusable session");

  session->script.assign(20, 'A');
  request.sampling.maxNewTokens = -1;
  auto bounded = model.complete(request);
  require(bounded.usage.completionTokens == 14 &&
              bounded.finishReason == FinishReason::Length,
          "unlimited generation must stop at remaining context");
  request.prompt.assign(16, 'A');
  reject([&] { (void)model.complete(request); });
  request.prompt = "A";
  request.sampling.maxNewTokens = 4;
  request.sampling.samplerConfig.temperature = 1.0f;
  request.sampling.seedPolicy = SamplingSeedPolicy::Fixed;
  request.sampling.fixedSeed = 0;
  session->sampleAlternatives = true;
  require(model.complete(request).content == model.complete(request).content,
          "explicit seed zero must be reproducible across requests");
  session->sampleAlternatives = false;
  request.sampling.samplerConfig.temperature = 0.0f;
  request.sampling.stopTokenIds = {264};
  reject([&] { (void)model.complete(request); });
  request.sampling.stopTokenIds.clear();
  request.images = {{"image.png", "", {}}};
  reject([&] { (void)model.complete(request); });

  ChatCompletionRequest chat;
  chat.messages = {{"system", "S", {}},
                   {"user", "U", {}},
                   {"assistant", "A", {}},
                   {"user", "V", {}}};
  chat.sampling.maxNewTokens = 0;
  chat.model = "ignored-name";
  const std::string rendered = model.renderChat(chat);
  require(rendered == "<|begin▁of▁sentence|><|User|>S\n\nU"
                      "<|end▁of▁sentence|><|Assistant|>A"
                      "<|end▁of▁sentence|><|User|>V"
                      "<|end▁of▁sentence|><|Assistant|>",
          "chat rendering lost supported roles or turns");
  // The full multi-turn prompt is larger than this deliberately tiny context;
  // validate its exact rendering separately and use a short prompt for prefill.
  chat.messages = {{"user", "U", {}}};
  auto chatResult = model.chat(chat);
  const auto &tokens = session->prompt;
  require(tokens == std::vector<int>({257, 258, 'U', 256, 259}) &&
              chatResult.usage.promptTokens == 5,
          "chat template was applied more than once");
  chat.messages = {{"tool", "U", {}}};
  reject([&] { (void)model.chat(chat); });
  chat.messages = {{"user", "U", {}}, {"system", "S", {}}};
  reject([&] { (void)model.chat(chat); });
  chat.messages = {{"system", "S", {}}};
  reject([&] { (void)model.chat(chat); });
  chat.messages = {{"user", "U", {{"image.png", "", {}}}}};
  reject([&] { (void)model.chat(chat); });

  const auto noBosTemplate = fixture.root / "no-bos.json";
  {
    std::ofstream output(noBosTemplate);
    output << R"({"roles":{"user":"<|User|>",
                             "assistant":"<|Assistant|>"},
                   "turn_suffix":"<|end▁of▁sentence|>",
                   "add_bos":false,"add_generation_prompt":true})";
  }
  config.chatTemplatePath = noBosTemplate.string();
  model.load(config);
  chat.messages = {{"user", "U", {}}};
  (void)model.chat(chat);
  require(session->prompt == std::vector<int>({258, 'U', 256, 259}),
          "chat encoding must honor the template's add_bos=false setting");

  config.chatTemplatePath.clear();
  model.load(config);
  require(model.capabilities().chatRoles.empty() &&
              !model.capabilities().chatMultiTurn &&
              model.capabilities().rawTokenize,
          "chat capability must depend on configured template");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected production DeepSeek vocabulary path");
    Fixture fixture;
    checkCodec(fixture, argv[1]);
    checkResident(fixture);
    std::cout
        << "DeepSeek raw codec and production resident serving checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DeepSeek serving check failed: " << error.what() << '\n';
    return 1;
  }
}
