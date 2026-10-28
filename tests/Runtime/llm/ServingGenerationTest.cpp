//===- ServingGenerationTest.cpp ----------------------------------------===//
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

// RUN: buddy-serving-generation-test 2>&1 | FileCheck %s

#include <buddy/runtime/llm/ServingGeneration.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

using buddy::Sampler;
using buddy::SamplerConfig;
using buddy::Text;
using namespace buddy::runtime;

namespace {

constexpr int kVocab = 16;
constexpr int kPrompt = 14;
constexpr int kEos = 15;
using Frame = std::vector<float>;
using Script = std::vector<Frame>;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function>
void requireFailure(Function &&function, const std::string &part) {
  try {
    function();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(part) != std::string::npos,
            "unexpected failure: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected failure containing " + part);
}

Frame logits(std::initializer_list<std::pair<int, float>> values) {
  Frame result(kVocab, -1000.0f);
  for (const auto &value : values)
    result.at(value.first) = value.second;
  return result;
}

class FakeSession final : public LLMSession {
public:
  explicit FakeSession(Script script) : script(std::move(script)) {}

  void loadWeights(const std::vector<std::string> &) override {
    throw std::runtime_error("test must not load weights");
  }
  void resetPosition() override {
    ++resets;
    currentPosition = 0;
    step = 0;
    decoded.clear();
    offsets.clear();
  }
  void prefill(Text<std::size_t, 2> &tokens) override {
    ++prefills;
    require(currentPosition == 0, "prefill did not follow reset");
    if (throwPrefill)
      throw std::runtime_error("scripted prefill failure");
    prompt.assign(tokens.getData(), tokens.getData() + tokens.getTokenCnt());
    require(tokens.getSizes()[0] == 1 &&
                tokens.getSizes()[1] == expectedContext &&
                tokens.getStrides()[0] == expectedContext &&
                tokens.getStrides()[1] == 1,
            "input does not have the exact padded prefill memref shape");
    for (std::size_t index = tokens.getTokenCnt();
         index < static_cast<std::size_t>(expectedContext); ++index)
      require(tokens.getData()[index] == 0,
              "prefill padding is not initialized");
    currentPosition = static_cast<int>(tokens.getTokenCnt());
    if (vocabularyAfterPrefill > 0)
      vocabulary = vocabularyAfterPrefill;
  }
  void decode(int token) override {
    ++decodes;
    if (throwDecode)
      throw std::runtime_error("scripted decode failure");
    decoded.push_back(token);
    ++currentPosition;
    require(++step < script.size(), "decode exceeded the script");
  }
  int position() const override { return currentPosition; }
  int vocabSize() const override { return vocabulary; }
  const float *logitsData(int offset = 0) const override {
    offsets.push_back(offset);
    return nullLogits ? nullptr : script.at(step).data();
  }
  bool handleKVCacheOverflow(int, float) override {
    throw std::runtime_error("serving must not shift the context");
  }

  int resets = 0;
  int prefills = 0;
  int decodes = 0;
  int expectedContext = 64;
  int vocabulary = kVocab;
  int vocabularyAfterPrefill = 0;
  bool throwPrefill = false;
  bool throwDecode = false;
  bool nullLogits = false;
  std::vector<std::size_t> prompt;
  std::vector<int> decoded;
  mutable std::vector<int> offsets;

private:
  Script script;
  std::size_t step = 0;
  int currentPosition = 23;
};

// A backend can select IDs on a device without exposing any host logits.
class FakeTokenSource {
public:
  explicit FakeTokenSource(std::vector<int> tokens)
      : tokens(std::move(tokens)) {}

  ServingTokenSource callbacks() {
    ServingTokenSource source;
    source.reset = [&] {
      ++resets;
      step = 0;
      histories.clear();
      decoded.clear();
    };
    source.prefill = [&](const std::vector<int> &tokens) {
      ++prefills;
      prompt = tokens;
    };
    source.decode = [&](int token) {
      require(token == tokens.at(step), "source decoded the wrong token");
      decoded.push_back(token);
      ++step;
    };
    source.selectToken = [&](const std::vector<int> &history) {
      histories.push_back(history);
      return tokens.at(step);
    };
    source.validToken = [](int token) { return token >= 0 && token < kVocab; };
    return source;
  }

  int resets = 0;
  int prefills = 0;
  std::vector<int> prompt;
  std::vector<int> decoded;
  std::vector<std::vector<int>> histories;

private:
  std::vector<int> tokens;
  std::size_t step = 0;
};

ServingTextCodec codecFor(std::vector<int> prompt = {kPrompt},
                          std::vector<std::string> pieces = {"A", "B", "C"},
                          int context = 64) {
  ServingTextCodec codec;
  codec.encode = [prompt](const std::string &) { return prompt; };
  codec.tokenBytes = [pieces](int token) { return pieces.at(token); };
  codec.contextLength = context;
  return codec;
}

SamplingParams params(int budget = -1) {
  SamplingParams result;
  result.maxNewTokens = budget;
  result.stopTokenIds = {kEos};
  return result;
}

struct Events {
  bool operator()(const CompletionChunk &chunk) {
    require(chunk.id.empty() && chunk.model.empty(),
            "the public controller must leave adapter identifiers empty");
    if (chunk.done) {
      ++terminals;
      terminal = chunk;
      require(chunk.delta.empty(), "terminal summary contains unflushed text");
    } else {
      require(terminals == 0, "content arrived after the terminal summary");
      text += chunk.delta;
    }
    return true;
  }
  void check(const CompletionResult &result) const {
    require(terminals == 1 && text == result.content,
            "streamed content/terminal disagrees with the result");
    require(terminal.finishReason == result.finishReason &&
                terminal.usage.promptTokens == result.usage.promptTokens &&
                terminal.usage.completionTokens ==
                    result.usage.completionTokens &&
                terminal.usage.totalTokens == result.usage.totalTokens,
            "terminal usage or finish reason disagrees with the result");
    require(result.id.empty() && result.model.empty(),
            "result contains adapter-owned identifiers");
  }
  int terminals = 0;
  std::string text;
  CompletionChunk terminal;
};

void checkBudgets() {
  const Script script = {logits({{0, 10}}), logits({{1, 10}}),
                         logits({{2, 10}}), logits({{kEos, 10}})};
  struct Case {
    int budget;
    const char *text;
    int sampled;
    int decodes;
    FinishReason reason;
  };
  for (const auto &test : {Case{0, "", 0, 0, FinishReason::Length},
                           Case{1, "A", 1, 0, FinishReason::Length},
                           Case{2, "AB", 2, 1, FinishReason::Length},
                           Case{4, "ABC", 4, 3, FinishReason::Stop},
                           Case{32, "ABC", 4, 3, FinishReason::Stop},
                           Case{-1, "ABC", 4, 3, FinishReason::Stop}}) {
    FakeSession session(script);
    Events events;
    auto result = runServingGeneration(
        "prompt", session, params(test.budget), codecFor(),
        [&](const CompletionChunk &chunk) { return events(chunk); });
    events.check(result);
    require(result.content == test.text && result.finishReason == test.reason,
            "budget output/finish reason is wrong");
    require(result.usage.promptTokens == 1 &&
                result.usage.completionTokens == test.sampled &&
                result.usage.totalTokens == test.sampled + 1,
            "first/EOS samples are not counted consistently");
    require(session.prefills == 1 && session.decodes == test.decodes &&
                session.resets == 1,
            "budget reset/prefill/decode counts are wrong");
    require(session.offsets.size() == static_cast<std::size_t>(test.sampled),
            "a zero budget sampled logits or an expected sample was skipped");
    if (!session.offsets.empty())
      require(session.offsets.front() == 0, "wrong initial logits offset");
    auto nonstream = runServingGeneration("prompt", session,
                                          params(test.budget), codecFor());
    require(nonstream.content == result.content &&
                nonstream.finishReason == result.finishReason &&
                nonstream.usage.completionTokens ==
                    result.usage.completionTokens,
            "streaming and non-streaming generation differ");
  }

  FakeSession firstEos({logits({{kEos, 10}})});
  auto eos = runServingGeneration("", firstEos, params(1), codecFor());
  require(eos.content.empty() && eos.usage.completionTokens == 1 &&
              eos.finishReason == FinishReason::Stop && firstEos.decodes == 0,
          "first-token EOS must take priority over the budget");
  FakeSession stopped(script);
  auto stopParams = params(2);
  stopParams.stopTokenIds.push_back(1);
  auto stop = runServingGeneration("", stopped, stopParams, codecFor());
  require(stop.content == "A" && stop.usage.completionTokens == 2 &&
              stop.finishReason == FinishReason::Stop,
          "stop ID must be counted and take priority over the budget");
}

void checkContextAndValidation() {
  const Script script = {logits({{0, 10}}), logits({{1, 10}}),
                         logits({{kEos, 10}})};
  for (int budget : {-1, 10, std::numeric_limits<int>::max()}) {
    FakeSession session(script);
    session.expectedContext = 4;
    auto result =
        runServingGeneration("", session, params(budget),
                             codecFor({kPrompt, kPrompt}, {"A", "B"}, 4));
    require(result.content == "AB" &&
                result.finishReason == FinishReason::Length &&
                result.usage.promptTokens == 2 &&
                result.usage.completionTokens == 2 &&
                result.usage.totalTokens == 4 && session.decodes == 1 &&
                session.offsets.front() == 1,
            "remaining context budget or prefill logits offset is wrong");
  }
  FakeSession full(script);
  full.expectedContext = 2;
  auto fullResult = runServingGeneration(
      "", full, params(), codecFor({kPrompt, kPrompt}, {"A", "B"}, 2));
  require(fullResult.finishReason == FinishReason::Length &&
              fullResult.usage.completionTokens == 0 && full.prefills == 1 &&
              full.decodes == 0 && full.offsets.empty(),
          "a prompt filling the context must prefill without sampling");

  auto invalid = [&](const SamplingParams &sampling,
                     const ServingTextCodec &codec, const std::string &field) {
    FakeSession session(script);
    requireFailure([&] { runServingGeneration("", session, sampling, codec); },
                   field);
    require(session.prefills == 0 && session.decodes == 0 &&
                session.position() == 0,
            "invalid request reached execution or was not cleaned up");
  };
  invalid(params(), codecFor({}, {"A"}), "prompt");
  invalid(params(), codecFor({kPrompt, kPrompt}, {"A"}, 1), "context length");
  invalid(params(), codecFor({-1}), "prompt");
  invalid(params(), codecFor({kVocab}), "prompt");
  invalid(params(), codecFor({kPrompt}, {"A"}, 0), "codec");
  auto missingEncoder = codecFor();
  missingEncoder.encode = {};
  invalid(params(), missingEncoder, "codec");
  auto missingDecoder = codecFor();
  missingDecoder.tokenBytes = {};
  invalid(params(), missingDecoder, "codec");
  for (long long token : {-1, kVocab}) {
    auto sampling = params(0);
    sampling.stopTokenIds = {token};
    invalid(sampling, codecFor(), "stop_token_ids");
  }
  invalid(params(-2), codecFor(), "max_tokens");
  auto sampling = params();
  sampling.samplerConfig.temperature = std::numeric_limits<float>::infinity();
  invalid(sampling, codecFor(), "temperature");
  sampling = params();
  sampling.samplerConfig.topP = 0;
  invalid(sampling, codecFor(), "top_p");
  sampling = params();
  sampling.samplerConfig.minP = 2;
  invalid(sampling, codecFor(), "min_p");
  sampling = params();
  sampling.samplerConfig.repeatPenalty = 0;
  invalid(sampling, codecFor(), "repeat_penalty");
  sampling = params();
  sampling.samplerConfig.repeatLastN = -1;
  invalid(sampling, codecFor(), "repeat_last_n");
  sampling = params();
  sampling.seedPolicy = SamplingSeedPolicy::Fixed;
  sampling.fixedSeed = std::numeric_limits<uint32_t>::max();
  invalid(sampling, codecFor(), "seed");
  sampling = params();
  sampling.stop = {""};
  invalid(sampling, codecFor(), "stop");
  FakeSession noVocabulary(script);
  noVocabulary.vocabulary = 0;
  requireFailure(
      [&] { runServingGeneration("", noVocabulary, params(), codecFor()); },
      "vocabulary size");
  require(noVocabulary.prefills == 0 && noVocabulary.position() == 0,
          "invalid vocabulary reached prefill");
}

void checkEffectiveVocabulary() {
  // The kernel vocabulary can be padded beyond the tokenizer vocabulary.
  // Even a winning padding logit must never reach tokenBytes or the history.
  const Script script = {logits({{1, 10}, {kVocab - 1, 1000}}),
                         logits({{2, 10}, {kVocab - 1, 1000}})};
  auto codec = codecFor({0}, {"A", "B", ""});
  codec.vocabSize = 3;
  auto sampling = params(2);
  sampling.stopTokenIds = {2};
  for (float temperature : {0.0f, 1.0f}) {
    FakeSession session(script);
    sampling.samplerConfig.temperature = temperature;
    sampling.seedPolicy = SamplingSeedPolicy::Fixed;
    sampling.fixedSeed = 0;
    auto result = runServingGeneration("", session, sampling, codec);
    require(result.content == "B" && result.usage.completionTokens == 2 &&
                result.finishReason == FinishReason::Stop &&
                session.decoded == std::vector<int>{1},
            "padding logits were sampled outside the decodable vocabulary");
  }

  auto invalid = [&](const ServingTextCodec &invalidCodec,
                     const SamplingParams &invalidSampling,
                     const std::string &field) {
    FakeSession session(script);
    requireFailure(
        [&] {
          runServingGeneration("", session, invalidSampling, invalidCodec);
        },
        field);
    require(session.prefills == 0 && session.position() == 0,
            "invalid effective vocabulary request reached prefill");
  };
  for (int size : {-1, kVocab + 1}) {
    auto invalidCodec = codec;
    invalidCodec.vocabSize = size;
    invalid(invalidCodec, sampling, "codec: vocabulary size");
  }
  auto invalidSampling = sampling;
  invalidSampling.stopTokenIds = {3};
  invalid(codec, invalidSampling, "stop_token_ids");
  auto invalidCodec = codec;
  invalidCodec.encode = [](const std::string &) { return std::vector<int>{3}; };
  invalid(invalidCodec, sampling, "prompt");

  // A smaller codec prefix must not hide a changed physical logits shape.
  FakeSession changed(script);
  changed.vocabularyAfterPrefill = kVocab - 1;
  requireFailure([&] { runServingGeneration("", changed, sampling, codec); },
                 "vocabulary size changed");
  require(changed.position() == 0,
          "changed session vocabulary did not clean up after failure");
}

CompletionResult runPieces(const std::vector<std::string> &pieces,
                           std::vector<std::string> stops = {},
                           int budget = -1) {
  require(pieces.size() < kPrompt, "too many test pieces");
  Script script;
  for (std::size_t index = 0; index < pieces.size(); ++index)
    script.push_back(logits({{static_cast<int>(index), 10}}));
  script.push_back(logits({{kEos, 10}}));
  FakeSession session(script);
  auto sampling = params(budget);
  sampling.stop = std::move(stops);
  Events events;
  auto result = runServingGeneration(
      "", session, sampling, codecFor({kPrompt}, pieces),
      [&](const CompletionChunk &chunk) { return events(chunk); });
  events.check(result);
  return result;
}

void checkStopsAndUtf8() {
  auto crossing = runPieces({"prefix ", "ST", "OP", "leak"}, {"STOP"});
  require(crossing.content == "prefix " &&
              crossing.usage.completionTokens == 3 &&
              crossing.finishReason == FinishReason::Stop,
          "cross-token stop leaked or counted the wrong token");
  auto overlap = runPieces({"leftx", "ab", "ab", "leak"}, {"abab", "bab"});
  require(overlap.content == "leftx" && overlap.usage.completionTokens == 3,
          "overlapping stop prefixes leaked");
  auto earliest = runPieces({"a stop later"}, {"later", "stop"}, 1);
  require(earliest.content == "a " &&
              earliest.finishReason == FinishReason::Stop,
          "earliest stop or stop/budget priority is wrong");
  auto chineseStop =
      runPieces({"prefix ", "\xE4", "\xB8", "\xAD", "leak"}, {"中"});
  require(chineseStop.content == "prefix " &&
              chineseStop.usage.completionTokens == 4,
          "a stop split inside UTF-8 leaked");
  auto partial = runPieces({"abcST"}, {"STOP"}, 1);
  require(partial.content == "abcST" &&
              partial.finishReason == FinishReason::Length,
          "partial stop tail was not flushed at the budget");
  auto eosTail = runPieces({"abcST"}, {"STOP"});
  require(eosTail.content == "abcST" && eosTail.usage.completionTokens == 2,
          "partial stop tail was not flushed at EOS");
  require(runPieces({"\xE4", "\xB8", "\xAD"}).content == "中",
          "UTF-8 bytes were completed token by token");
  require(runPieces({"中", "\xE4", "\xB8"}).content == "中�",
          "truncated UTF-8 tail was not replaced once");
  require(runPieces({"\x80", "A"}).content == "�A",
          "malformed UTF-8 was not repaired");
  require(runPieces({std::string(" \n\"\0", 4)}).content ==
              std::string(" \n\"\0", 4),
          "raw whitespace, quotes, or NUL were changed");
}

void checkRepetition() {
  const Frame repeated = logits({{0, 8}, {1, 6}});
  auto sampling = params(1);
  sampling.samplerConfig.repeatPenalty = 2;
  sampling.samplerConfig.repeatLastN = 2;
  FakeSession window({repeated});
  auto result = runServingGeneration("", window, sampling, codecFor({1, 0, 0}));
  require(result.content == "B",
          "repetition must select the window before dedup");
  sampling.samplerConfig.repeatLastN = 0;
  result = runServingGeneration("", window, sampling, codecFor({1, 0, 0}));
  require(result.content == "A", "repeat_last_n=0 did not disable penalties");
  sampling.samplerConfig.repeatLastN = 64;
  sampling.maxNewTokens = 2;
  FakeSession generated({repeated, repeated});
  result = runServingGeneration("", generated, sampling, codecFor());
  require(result.content == "AB",
          "generated tokens are absent from repeat history");
  sampling.maxNewTokens = 1;
  FakeSession prompt({logits({{0, 4}, {1, 3}})});
  result = runServingGeneration("", prompt, sampling, codecFor({0}));
  require(result.content == "B",
          "prompt tokens are absent from repeat history");
  FakeSession duplicates({logits({{0, 8}, {1, 3}})});
  result = runServingGeneration("", duplicates, sampling, codecFor({0, 0}));
  require(result.content == "A",
          "duplicate token IDs applied the penalty twice");
}

void checkCancellationAndFailures() {
  const Script script = {logits({{0, 10}}), logits({{1, 10}}),
                         logits({{kEos, 10}})};
  FakeSession session(script);
  auto sampling = params();
  auto codec = codecFor();
  int callbacks = 0;
  auto result = runServingGeneration("", session, sampling, codec,
                                     [&](const CompletionChunk &) {
                                       ++callbacks;
                                       return false;
                                     });
  require(result.finishReason == FinishReason::Cancelled && callbacks == 1 &&
              result.usage.completionTokens == 1 && session.decodes == 0 &&
              session.resets == 2 && session.position() == 0,
          "callback cancellation continued execution or failed to clean up");
  result = runServingGeneration("", session, sampling, codec);
  require(result.content == "AB" && result.finishReason == FinishReason::Stop,
          "session could not be reused after cancellation");
  for (int failure = 0; failure < 5; ++failure) {
    const int resetsBefore = session.resets;
    session.throwPrefill = failure == 0;
    session.throwDecode = failure == 1;
    session.nullLogits = failure == 2;
    auto failingCodec = codec;
    if (failure == 3)
      failingCodec.tokenBytes = [](int) -> std::string {
        throw std::runtime_error("scripted codec failure");
      };
    requireFailure(
        [&] {
          runServingGeneration("", session, sampling, failingCodec,
                               [&](const CompletionChunk &) -> bool {
                                 if (failure == 4)
                                   throw std::runtime_error(
                                       "scripted callback failure");
                                 return true;
                               });
        },
        failure == 2 ? "logits pointer" : "scripted");
    require(session.resets == resetsBefore + 2 && session.position() == 0,
            "exception did not reset the session");
    session.throwPrefill = session.throwDecode = session.nullLogits = false;
    result = runServingGeneration("", session, sampling, codec);
    require(result.content == "AB",
            "session could not be reused after an exception");
  }
  // A transport can disconnect while the buffered stop/UTF-8 tail is flushed.
  FakeSession tail({logits({{0, 10}})});
  sampling = params(1);
  sampling.stop = {"STOP"};
  callbacks = 0;
  result = runServingGeneration(
      "", tail, sampling, codecFor({kPrompt}, {"ST"}),
      [&](const CompletionChunk &) { return ++callbacks < 2; });
  require(result.finishReason == FinishReason::Cancelled && callbacks == 2 &&
              tail.position() == 0,
          "tail cancellation emitted a terminal event or did not reset");
}

void checkSeeds() {
  SamplerConfig config;
  config.temperature = 1;
  config.seed = 42;
  const std::vector<float> values = {0, -1, -2};
  Sampler legacy(config);
  Sampler explicit42(config, 42);
  for (int index = 0; index < 32; ++index)
    require(legacy.sample(values.data(), values.size(), {}) ==
                explicit42.sample(values.data(), values.size(), {}),
            "explicit seed changed the legacy nonzero seed sequence");
  Sampler explicitZero(config, 0);
  std::mt19937 rng(0);
  std::discrete_distribution<int> reference(
      {1.0, std::exp(-1.0), std::exp(-2.0)});
  for (int index = 0; index < 32; ++index)
    require(explicitZero.sample(values.data(), values.size(), {}) ==
                reference(rng),
            "explicit zero did not match a fixed zero seed");

  const Frame random = logits({{0, 0}, {1, -1}, {2, -2}});
  FakeSession session(Script(16, random));
  auto sampling = params(12);
  sampling.samplerConfig.temperature = 1;
  sampling.seedPolicy = SamplingSeedPolicy::Fixed;
  sampling.fixedSeed = 0;
  auto first = runServingGeneration("", session, sampling, codecFor());
  auto second = runServingGeneration("", session, sampling, codecFor());
  require(first.content == second.content && first.usage.completionTokens == 12,
          "serving fixed seed zero is not repeatable across requests");
}

void checkTokenSource() {
  FakeTokenSource device({0, 1, 2, 3, 4, 5});
  auto source = device.callbacks();
  auto codec = codecFor({kPrompt, kPrompt},
                        {"\xE4", "\xB8", "\xAD", "ST", "OP", "leak"});
  auto sampling = params(10);
  sampling.stop = {"STOP"};
  Events events;
  auto result = runServingGeneration(
      "", source, sampling, codec,
      [&](const CompletionChunk &chunk) { return events(chunk); });
  events.check(result);
  require(result.content == "中" && result.finishReason == FinishReason::Stop &&
              result.usage.promptTokens == 2 &&
              result.usage.completionTokens == 5 &&
              result.usage.totalTokens == 7 && device.resets == 1 &&
              device.prefills == 1 &&
              device.decoded == std::vector<int>({0, 1, 2, 3}),
          "direct token source bypassed stop, UTF-8, usage, or decode rules");
  require(device.prompt == std::vector<int>({kPrompt, kPrompt}) &&
              device.histories.front() == device.prompt &&
              device.histories.back() ==
                  std::vector<int>({kPrompt, kPrompt, 0, 1, 2, 3}),
          "token source did not receive the complete prompt and token history");

  // A direct token source shares prefill-only and remaining-context limits.
  FakeTokenSource bounded({0, 1, kEos});
  auto boundedSource = bounded.callbacks();
  auto boundedCodec = codecFor({kPrompt, kPrompt}, {"A", "B"}, 4);
  for (int budget : {0, 1, -1}) {
    result =
        runServingGeneration("", boundedSource, params(budget), boundedCodec);
    const int count = budget < 0 ? 2 : budget;
    require(result.content == std::string("AB").substr(0, count) &&
                result.finishReason == FinishReason::Length &&
                result.usage.completionTokens == count &&
                bounded.histories.size() == static_cast<std::size_t>(count) &&
                bounded.decoded.size() ==
                    static_cast<std::size_t>(std::max(0, count - 1)),
            "direct token source ignored the budget or context limit");
  }

  int resetsBefore = device.resets;
  int callbacks = 0;
  result = runServingGeneration("", source, sampling, codec,
                                [&](const CompletionChunk &) {
                                  ++callbacks;
                                  return false;
                                });
  require(result.finishReason == FinishReason::Cancelled && callbacks == 1 &&
              result.usage.completionTokens == 1 &&
              device.resets == resetsBefore + 2 && device.histories.empty(),
          "direct token source cancellation did not reset immediately");

  // Generated IDs must be checked before they reach the text codec.
  bool decodedInvalidToken = false;
  auto invalidCodec = codecFor();
  invalidCodec.tokenBytes = [&](int) {
    decodedInvalidToken = true;
    return "invalid";
  };
  for (int token : {-1, kVocab}) {
    auto invalidSource = source;
    invalidSource.selectToken = [token](const std::vector<int> &) {
      return token;
    };
    resetsBefore = device.resets;
    requireFailure(
        [&] {
          runServingGeneration("", invalidSource, params(1), invalidCodec);
        },
        "selected token ID");
    require(!decodedInvalidToken && device.resets == resetsBefore + 2,
            "invalid selected ID reached tokenBytes or did not reset");
  }
  auto prefixCodec = codecFor({0});
  prefixCodec.vocabSize = 1;
  auto prefixSampling = params(1);
  prefixSampling.stopTokenIds.clear();
  auto paddedSource = source;
  paddedSource.selectToken = [](const std::vector<int> &) { return 1; };
  requireFailure(
      [&] {
        runServingGeneration("", paddedSource, prefixSampling, prefixCodec);
      },
      "selected token ID");

  auto incompleteSource = source;
  incompleteSource.selectToken = {};
  const int prefillsBefore = device.prefills;
  requireFailure(
      [&] { runServingGeneration("", incompleteSource, sampling, codec); },
      "incomplete execution");
  auto invalidSampling = sampling;
  invalidSampling.stopTokenIds = {std::numeric_limits<long long>::max()};
  requireFailure(
      [&] { runServingGeneration("", source, invalidSampling, codec); },
      "stop_token_ids");
  require(device.prefills == prefillsBefore,
          "invalid token source configuration reached prefill");

  auto throwingSource = source;
  throwingSource.decode = [](int) {
    throw std::runtime_error("source failure");
  };
  resetsBefore = device.resets;
  requireFailure(
      [&] { runServingGeneration("", throwingSource, params(2), codec); },
      "source failure");
  require(device.resets == resetsBefore + 2,
          "direct token source exception did not reset");
}

} // namespace

int main() {
  try {
    checkBudgets();
    // CHECK: serving budgets and sampled-token usage: passed
    std::cout << "serving budgets and sampled-token usage: passed\n";
    checkContextAndValidation();
    // CHECK: serving context and preflight validation: passed
    std::cout << "serving context and preflight validation: passed\n";
    checkEffectiveVocabulary();
    // CHECK: serving decodable vocabulary and padding logits: passed
    std::cout << "serving decodable vocabulary and padding logits: passed\n";
    checkStopsAndUtf8();
    // CHECK: serving stops and raw UTF-8: passed
    std::cout << "serving stops and raw UTF-8: passed\n";
    checkRepetition();
    // CHECK: serving repetition window and unique IDs: passed
    std::cout << "serving repetition window and unique IDs: passed\n";
    checkCancellationAndFailures();
    // CHECK: serving cancellation, failures, and reuse: passed
    std::cout << "serving cancellation, failures, and reuse: passed\n";
    checkSeeds();
    // CHECK: serving explicit seed and legacy sampler: passed
    std::cout << "serving explicit seed and legacy sampler: passed\n";
    checkTokenSource();
    // CHECK: serving direct token source contract: passed
    std::cout << "serving direct token source contract: passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
