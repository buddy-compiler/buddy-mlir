//===- Llama31TTServingTest.cpp - No-device TT serving contract checks ----===//
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

#include "buddy/runtime/models/Llama31TTResidentModel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace buddy::runtime;

namespace {

constexpr int kBos = 128000;
constexpr int kEndOfText = 128001;
constexpr int kStartHeader = 128006;
constexpr int kEndHeader = 128007;
constexpr int kEom = 128008;
constexpr int kEot = 128009;

void require(bool value, const std::string &message) {
  if (!value)
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

SamplingParams params(int budget) {
  SamplingParams result;
  result.maxNewTokens = budget;
  return result;
}

// A device-style backend supplies selected token IDs, never artificial logits.
class ScriptedBackend {
public:
  enum class Failure { None, Prefill, Decode, Select };

  void reset() {
    step = 0;
    position = 0;
  }

  CompletionResult serve(const std::string &text,
                         const SamplingParams &sampling,
                         const ServingTextCodec &codec,
                         const CompletionStreamCallback &callback) {
    ++starts;
    rendered = text;
    ServingTokenSource source;
    source.reset = [&] {
      ++sourceResets;
      reset();
    };
    source.prefill = [&](const std::vector<int> &tokens) {
      ++prefills;
      prompt = tokens;
      decoded.clear();
      histories.clear();
      position = static_cast<int>(tokens.size());
      if (failure == Failure::Prefill)
        throw std::runtime_error("scripted prefill failure");
    };
    source.decode = [&](int token) {
      if (failure == Failure::Decode)
        throw std::runtime_error("scripted decode failure");
      require(token == script.at(step), "TT decoded the wrong previous token");
      decoded.push_back(token);
      ++step;
      ++position;
    };
    source.selectToken = [&](const std::vector<int> &history) {
      if (failure == Failure::Select)
        throw std::runtime_error("scripted token selection failure");
      require(history.size() == prompt.size() + step &&
                  std::equal(prompt.begin(), prompt.end(), history.begin()),
              "TT token source did not receive the full prompt/history");
      histories.push_back(history);
      return script.at(step);
    };
    source.validToken = [&](int token) {
      try {
        (void)codec.tokenBytes(token);
        return true;
      } catch (const std::invalid_argument &) {
        return false;
      }
    };
    return runServingGeneration(text, source, sampling, codec, callback);
  }

  std::vector<int> script = {'A', 'B', kEot};
  std::vector<int> prompt;
  std::vector<int> decoded;
  std::vector<std::vector<int>> histories;
  std::string rendered;
  Failure failure = Failure::None;
  int starts = 0;
  int prefills = 0;
  int sourceResets = 0;
  int executionResets = 0;
  int position = 0;

private:
  std::size_t step = 0;
};

class Fixture {
public:
  explicit Fixture(const std::string &rax, const std::string &chatTemplate = {})
      : execution(std::make_shared<Llama31TTExecution>()), model(execution) {
    require(!model.capabilities().rawTokenize &&
                !model.capabilities().maxNewTokens,
            "unloaded TT model advertised serving capabilities");
    execution->setBackend(
        [](const Llama31TTExecution::BackendRequest &,
           const CompletionStreamCallback &) -> CompletionResult {
          throw std::runtime_error(
              "resident called the legacy TT generation path");
        });
    execution->setServingBackend([&](const std::string &prompt,
                                     const SamplingParams &sampling,
                                     const ServingTextCodec &codec,
                                     const CompletionStreamCallback &callback) {
      return backend.serve(prompt, sampling, codec, callback);
    });
    execution->setResetCallback([&] {
      ++backend.executionResets;
      backend.reset();
    });
    ResidentModelConfig config;
    config.raxPath = rax;
    config.chatTemplatePath = chatTemplate;
    model.load(config);
    require(model.isLoaded() && model.status().contextLength == 128 &&
                execution->metadata().batchSize == 1,
            "unexpected TT serving fixture metadata");
  }

  ScriptedBackend backend;
  std::shared_ptr<Llama31TTExecution> execution;
  Llama31TTResidentModel model;
};

struct Events {
  bool operator()(const CompletionChunk &chunk) {
    require(!chunk.id.empty() && !chunk.model.empty(),
            "TT chunk is missing adapter-owned identifiers");
    require(terminals == 0, "TT emitted an event after its terminal summary");
    if (id.empty()) {
      id = chunk.id;
      model = chunk.model;
    }
    require(id == chunk.id && model == chunk.model,
            "TT stream identifiers changed within a request");
    if (chunk.done) {
      ++terminals;
      terminal = chunk;
      require(chunk.delta.empty(), "TT terminal summary contains content");
    } else {
      text += chunk.delta;
      ++deltas;
    }
    return true;
  }

  void check(const CompletionResult &result) const {
    require(terminals == 1 && text == result.content && id == result.id &&
                model == result.model,
            "TT stream content or identifiers disagree with the result");
    require(terminal.finishReason == result.finishReason &&
                terminal.usage.promptTokens == result.usage.promptTokens &&
                terminal.usage.completionTokens ==
                    result.usage.completionTokens &&
                terminal.usage.totalTokens == result.usage.totalTokens,
            "TT stream finish reason or usage disagrees with the result");
    require(std::isfinite(result.timings.prefillMs) &&
                std::isfinite(result.timings.decodeMs) &&
                std::isfinite(result.timings.tokensPerSecond) &&
                result.timings.prefillMs >= 0 && result.timings.decodeMs >= 0 &&
                result.timings.tokensPerSecond >= 0,
            "TT timings are invalid");
  }

  int deltas = 0;
  int terminals = 0;
  std::string text;
  std::string id;
  std::string model;
  CompletionChunk terminal;
};

CompletionResult complete(Fixture &fixture, const std::string &prompt,
                          const SamplingParams &sampling,
                          Events *events = nullptr) {
  CompletionRequest request;
  request.prompt = prompt;
  request.sampling = sampling;
  if (!events)
    return fixture.model.complete(request);
  auto result = fixture.model.completeStream(
      request, [&](const CompletionChunk &chunk) { return (*events)(chunk); });
  events->check(result);
  return result;
}

void checkCapabilities(Fixture &fixture, const std::string &rax) {
  const auto caps = fixture.model.capabilities();
  require(caps.rawTokenize && caps.rawDetokenize && caps.tokenizeAddSpecial &&
              caps.tokenizeParseSpecial && caps.tokenizeWithPieces &&
              caps.streaming && caps.maxNewTokens && caps.stop &&
              caps.stopTokenIds && caps.chatMultiTurn && caps.maxImages == 0 &&
              !caps.imagePaths && !caps.imageBytes,
          "loaded TT serving capabilities are incomplete or overstate images");
  for (const char *role : {"system", "user", "assistant"})
    require(std::find(caps.chatRoles.begin(), caps.chatRoles.end(), role) !=
                caps.chatRoles.end(),
            "TT omitted a supported chat role");
  const auto sampling = caps.sampling;
  require(!sampling.temperature && !sampling.topK && !sampling.topP &&
              !sampling.minP && !sampling.repeatPenalty &&
              !sampling.repeatLastN && !sampling.seed,
          "device argmax advertised host sampling support");

  Llama31TTExecution withoutBackend;
  ResidentModelConfig config;
  config.raxPath = rax;
  withoutBackend.load(config);
  require(withoutBackend.capabilities().rawTokenize &&
              !withoutBackend.capabilities().maxNewTokens &&
              withoutBackend.capabilities().chatRoles.empty(),
          "a tokenizer-only TT load advertised an unavailable backend");
  requireFailure([&] { withoutBackend.generateServing("abc", params(1)); },
                 "unavailable");
}

void checkCodec(Fixture &fixture) {
  TokenizeRequest request;
  request.content = "abc";
  request.addSpecial = false;
  request.withPieces = true;
  auto result = fixture.model.tokenize(request);
  require(result.tokens == std::vector<int>{257} && result.count == 1 &&
              result.pieces == std::vector<std::string>{"abc"},
          "TT raw BPE ranks or pieces changed");
  request.content = "ab";
  require(fixture.model.tokenize(request).tokens == std::vector<int>{256},
          "TT fixture merge rank was not applied");
  request.content = "abc";
  request.addSpecial = true;
  result = fixture.model.tokenize(request);
  require(result.tokens == std::vector<int>({kBos, 257}) && result.count == 2 &&
              result.pieces ==
                  std::vector<std::string>({"<|begin_of_text|>", "abc"}),
          "TT add_special added a template or omitted BOS");
  request.countOnly = true;
  result = fixture.model.tokenize(request);
  require(result.count == 2 && result.tokens.empty() && result.pieces.empty(),
          "TT count_only did not suppress tokens and pieces");
  request.countOnly = false;
  request.addSpecial = false;
  request.content.clear();
  result = fixture.model.tokenize(request);
  require(result.count == 0 && result.tokens.empty() && result.pieces.empty(),
          "empty raw TT tokenization gained a prompt");
  request.addSpecial = true;
  require(fixture.model.tokenize(request).tokens == std::vector<int>{kBos},
          "empty TT add_special did not return only BOS");

  request.addSpecial = false;
  request.content = "<|eot_id|>";
  request.parseSpecial = true;
  require(fixture.model.tokenize(request).tokens == std::vector<int>{kEot},
          "TT parse_special did not recognize a special token");
  request.parseSpecial = false;
  result = fixture.model.tokenize(request);
  require(result.tokens != std::vector<int>{kEot} &&
              fixture.model.detokenize({result.tokens}).content ==
                  request.content,
          "TT parse_special=false did not preserve literal text");
  request.parseSpecial = true;
  request.content = "<|python_tag|>";
  require(fixture.model.tokenize(request).tokens == std::vector<int>{128010},
          "TT discarded an artifact-defined special token");
  request.content += "suffix";
  require(
      fixture.model.tokenize(request).tokens == std::vector<int>{128011} &&
          fixture.model.detokenize({{128011, 'A'}}).content ==
              "<|python_tag|>suffixA",
      "TT special token matching did not select the longest registry entry");
  request.parseSpecial = false;
  result = fixture.model.tokenize(request);
  require(std::find(result.tokens.begin(), result.tokens.end(), 128011) ==
                  result.tokens.end() &&
              fixture.model.detokenize({result.tokens}).content ==
                  request.content,
          "TT parse_special=false parsed an artifact-defined special token");
  request.content = "  abc\n中\t";
  request.content += '\0';
  result = fixture.model.tokenize(request);
  require(result.tokens.size() == result.pieces.size() &&
              fixture.model.detokenize({result.tokens}).content ==
                  request.content,
          "TT raw tokenization changed whitespace, Unicode, or NUL");
  request.content.assign(256, 'x');
  require(fixture.model.tokenize(request).count == 256,
          "TT raw tokenization truncated at the execution context");

  std::string preserved = "  <|begin_of_text|><|end_of_text|>A";
  preserved += '\0';
  preserved += '\n';
  require(fixture.model.detokenize({{32, 32, kBos, kEndOfText, 'A', 0, '\n'}})
                  .content == preserved,
          "TT raw decoding trimmed spaces, special tokens, EOS tail, or NUL");
  require(fixture.model.detokenize({}).content.empty(),
          "TT empty decoding changed");
  require(fixture.model.detokenize({{0xe4, 0xb8, 0xad}}).content == "中" &&
              fixture.model.detokenize({{0xe4, 0xb8}}).content == "�" &&
              fixture.model.detokenize({{0x80, 'A'}}).content == "�A",
          "TT raw bytes were not joined before UTF-8 repair");
  for (int token : {-1, 258, kBos - 1, 128012})
    requireFailure([&] { fixture.model.detokenize({{token}}); },
                   "out of range");
}

void checkBudgets(Fixture &fixture) {
  struct Case {
    int budget;
    const char *content;
    int sampled;
    FinishReason reason;
  };
  for (const auto &test : {Case{0, "", 0, FinishReason::Length},
                           Case{1, "A", 1, FinishReason::Length},
                           Case{2, "AB", 2, FinishReason::Length},
                           Case{-1, "AB", 3, FinishReason::Stop}}) {
    Events events;
    auto result = complete(fixture, "abc", params(test.budget), &events);
    require(result.content == test.content &&
                result.finishReason == test.reason &&
                result.usage.promptTokens == 2 &&
                result.usage.completionTokens == test.sampled &&
                result.usage.totalTokens == 2 + test.sampled &&
                fixture.backend.prompt == std::vector<int>({kBos, 257}) &&
                fixture.backend.histories.size() ==
                    static_cast<size_t>(test.sampled) &&
                fixture.backend.decoded.size() ==
                    static_cast<size_t>(std::max(0, test.sampled - 1)) &&
                fixture.execution->cachePosition() == 0 &&
                fixture.backend.position == 0,
            "TT serving budget, EOS usage, or session reset changed");
    auto nonstream = complete(fixture, "abc", params(test.budget));
    require(nonstream.content == result.content &&
                nonstream.finishReason == result.finishReason &&
                nonstream.usage.completionTokens ==
                    result.usage.completionTokens &&
                nonstream.id != result.id,
            "TT stream and non-stream contracts differ or reused an ID");
  }
  auto full = complete(fixture, std::string(127, 'x'), params(2));
  require(full.usage.promptTokens == 128 && full.usage.completionTokens == 0 &&
              full.finishReason == FinishReason::Length &&
              fixture.backend.histories.empty(),
          "TT full context did not prefill without sampling");
  auto remaining = complete(fixture, std::string(126, 'x'), params(-1));
  require(remaining.content == "A" && remaining.usage.completionTokens == 1 &&
              remaining.finishReason == FinishReason::Length &&
              fixture.backend.decoded.empty(),
          "TT unlimited generation exceeded the remaining context");
  const int prefills = fixture.backend.prefills;
  requireFailure([&] { complete(fixture, std::string(128, 'x'), params(1)); },
                 "context length");
  require(fixture.backend.prefills == prefills && fixture.backend.position == 0,
          "oversized TT prompt reached execution or was not reset");
  requireFailure([&] { complete(fixture, "abc", params(-2)); }, "max_tokens");

  for (int token : {kEndOfText, kEom, kEot}) {
    fixture.backend.script = {token};
    auto eos = complete(fixture, "abc", params(1));
    require(eos.content.empty() && eos.finishReason == FinishReason::Stop &&
                eos.usage.completionTokens == 1 &&
                fixture.backend.decoded.empty(),
            "TT first EOS was not counted or lost priority over the budget");
  }
  fixture.backend.script = {'A', 'B', kEot};
  auto customStop = params(2);
  customStop.stopTokenIds = {'B'};
  auto stopped = complete(fixture, "abc", customStop);
  require(stopped.content == "A" &&
              stopped.finishReason == FinishReason::Stop &&
              stopped.usage.completionTokens == 2,
          "TT explicit stop ID leaked or lost priority over the budget");
  for (long long token : {-1LL, 258LL, std::numeric_limits<long long>::max()}) {
    customStop.stopTokenIds = {token};
    requireFailure([&] { complete(fixture, "abc", customStop); }, "token");
  }
}

void checkStopsAndUtf8(Fixture &fixture) {
  fixture.backend.script = {'p', 'S', 'T', 'O', 'P', 'x'};
  auto sampling = params(-1);
  sampling.stop = {"STOP"};
  Events events;
  auto stopped = complete(fixture, "abc", sampling, &events);
  require(stopped.content == "p" &&
              stopped.finishReason == FinishReason::Stop &&
              stopped.usage.completionTokens == 5,
          "TT split string stop leaked bytes or counted the wrong tokens");
  fixture.backend.script = {0xe4, 0xb8, 0xad, kEot};
  Events unicodeEvents;
  auto unicode = complete(fixture, "abc", params(-1), &unicodeEvents);
  require(unicode.content == "中" && unicode.usage.completionTokens == 4,
          "TT split UTF-8 generated replacement characters");
  sampling.stop = {"中"};
  auto unicodeStop = complete(fixture, "abc", sampling);
  require(unicodeStop.content.empty() &&
              unicodeStop.usage.completionTokens == 3 &&
              unicodeStop.finishReason == FinishReason::Stop,
          "TT stop split inside UTF-8 leaked text");
  fixture.backend.script = {0xe4, 0xb8};
  require(complete(fixture, "abc", params(2)).content == "�",
          "TT truncated UTF-8 tail was not repaired once");
  fixture.backend.script = {'S', 'T'};
  sampling.maxNewTokens = 2;
  sampling.stop = {"STOP"};
  Events tailEvents;
  auto tail = complete(fixture, "abc", sampling, &tailEvents);
  require(tail.content == "ST" && tail.finishReason == FinishReason::Length,
          "TT partial stop tail was not flushed at the budget");
  fixture.backend.script = {'A', 'B', kEot};
}

void checkUnsupportedSampling(Fixture &fixture) {
  using Member = bool SamplingParameterPresence::*;
  for (const auto &field :
       {std::make_pair("temperature",
                       Member(&SamplingParameterPresence::temperature)),
        std::make_pair("top_k", Member(&SamplingParameterPresence::topK)),
        std::make_pair("top_p", Member(&SamplingParameterPresence::topP)),
        std::make_pair("min_p", Member(&SamplingParameterPresence::minP)),
        std::make_pair("repeat_penalty",
                       Member(&SamplingParameterPresence::repeatPenalty)),
        std::make_pair("repeat_last_n",
                       Member(&SamplingParameterPresence::repeatLastN)),
        std::make_pair("seed", Member(&SamplingParameterPresence::seed))}) {
    auto sampling = params(1);
    sampling.explicitParameters.*field.second = true;
    const int starts = fixture.backend.starts;
    requireFailure([&] { complete(fixture, "abc", sampling); }, field.first);
    require(fixture.backend.starts == starts,
            "unsupported explicit sampling reached TT execution");
  }
  using Change = std::function<void(SamplingParams &)>;
  const std::vector<std::pair<std::string, Change>> changes = {
      {"temperature",
       [](SamplingParams &p) { p.samplerConfig.temperature = 1; }},
      {"top_k", [](SamplingParams &p) { p.samplerConfig.topK = 4; }},
      {"top_p", [](SamplingParams &p) { p.samplerConfig.topP = 0.5f; }},
      {"min_p", [](SamplingParams &p) { p.samplerConfig.minP = 0.1f; }},
      {"repeat_penalty",
       [](SamplingParams &p) { p.samplerConfig.repeatPenalty = 1.1f; }},
      {"repeat_last_n",
       [](SamplingParams &p) { p.samplerConfig.repeatLastN = 0; }},
      {"seed", [](SamplingParams &p) { p.samplerConfig.seed = 42; }},
      {"seed", [](SamplingParams &p) {
         p.seedPolicy = SamplingSeedPolicy::Fixed;
         p.fixedSeed = 0;
       }}};
  for (const auto &change : changes) {
    auto sampling = params(1);
    change.second(sampling);
    const int starts = fixture.backend.starts;
    requireFailure([&] { complete(fixture, "abc", sampling); }, change.first);
    require(fixture.backend.starts == starts,
            "unsupported sampling value reached TT execution");
  }
}

void checkChat(Fixture &fixture) {
  ChatCompletionRequest chat;
  chat.model = "client-visible-alias";
  chat.messages = {{"system", "s", {}},
                   {"user", "u", {}},
                   {"assistant", "a", {}},
                   {"user", "v", {}}};
  chat.sampling = params(1);
  std::string expected = "<|begin_of_text|>";
  for (const auto &message : chat.messages)
    expected += "<|start_header_id|>" + message.role + "<|end_header_id|>\n\n" +
                message.content + "<|eot_id|>";
  expected += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  require(fixture.model.renderChat(chat) == expected,
          "TT chat alias, roles, or multi-turn rendering changed");
  Events events;
  auto result = fixture.model.chatStream(
      chat, [&](const CompletionChunk &chunk) { return events(chunk); });
  events.check(result);
  require(result.content == "A" && fixture.backend.rendered == expected &&
              result.model == fixture.model.status().modelName &&
              result.model != chat.model &&
              std::count(fixture.backend.prompt.begin(),
                         fixture.backend.prompt.end(), kBos) == 1 &&
              std::count(fixture.backend.prompt.begin(),
                         fixture.backend.prompt.end(), kStartHeader) == 5 &&
              std::count(fixture.backend.prompt.begin(),
                         fixture.backend.prompt.end(), kEndHeader) == 5,
          "TT chat applied its template twice, duplicated BOS, or returned the "
          "request alias");
  TokenizeRequest encoded;
  encoded.content = expected;
  encoded.addSpecial = false;
  require(result.usage.promptTokens ==
              static_cast<int>(fixture.model.tokenize(encoded).count),
          "TT chat usage includes a second template or BOS");
  auto completion = complete(fixture, expected, params(1));
  require(completion.content == result.content &&
              fixture.backend.rendered == expected &&
              completion.usage.promptTokens == result.usage.promptTokens,
          "TT completion re-rendered an already formatted prompt");
  chat.messages.clear();
  chat.input = "abc";
  require(fixture.model.renderChat(chat) ==
              "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nabc"
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
          "TT bare-string chat did not create exactly one user turn");
  chat.messages = {{"tool", "x", {}}};
  requireFailure([&] { fixture.model.chat(chat); }, "role");
  chat.messages = {{"user", "x", {ImageInput{}}}};
  requireFailure([&] { fixture.model.chat(chat); }, "image");
  CompletionRequest images;
  images.prompt = "abc";
  images.sampling = params(1);
  images.images.push_back({});
  requireFailure([&] { fixture.model.complete(images); }, "image");
}

void checkCancellationAndFailures(Fixture &fixture) {
  CompletionRequest request;
  request.prompt = "abc";
  request.sampling = params(-1);
  int callbacks = 0;
  int terminals = 0;
  auto cancelled =
      fixture.model.completeStream(request, [&](const CompletionChunk &chunk) {
        ++callbacks;
        terminals += chunk.done;
        return false;
      });
  require(cancelled.finishReason == FinishReason::Cancelled && callbacks == 1 &&
              terminals == 0 && cancelled.usage.completionTokens == 1 &&
              fixture.backend.decoded.empty() &&
              fixture.backend.position == 0 &&
              fixture.execution->cachePosition() == 0,
          "TT cancellation emitted a terminal event or continued execution");
  Events reused;
  require(complete(fixture, "abc", params(-1), &reused).content == "AB",
          "TT session was not reusable after cancellation");
  for (auto failure :
       {ScriptedBackend::Failure::Prefill, ScriptedBackend::Failure::Decode,
        ScriptedBackend::Failure::Select}) {
    fixture.backend.failure = failure;
    const int resets = fixture.backend.sourceResets;
    const int executionResets = fixture.backend.executionResets;
    requireFailure([&] { fixture.model.complete(request); }, "scripted");
    require(fixture.backend.sourceResets >= resets + 2 &&
                fixture.backend.executionResets >= executionResets + 2 &&
                fixture.backend.position == 0 &&
                fixture.execution->cachePosition() == 0,
            "TT backend exception did not reset execution and token source");
    fixture.backend.failure = ScriptedBackend::Failure::None;
    require(complete(fixture, "abc", params(-1)).content == "AB",
            "TT session was not reusable after a backend exception");
  }
  callbacks = 0;
  requireFailure(
      [&] {
        fixture.model.completeStream(
            request, [&](const CompletionChunk &) -> bool {
              ++callbacks;
              throw std::runtime_error("scripted transport failure");
            });
      },
      "transport failure");
  require(
      callbacks == 1 && fixture.backend.position == 0 &&
          fixture.execution->cachePosition() == 0,
      "TT transport exception continued callbacks or retained request state");
  require(complete(fixture, "abc", params(-1)).content == "AB",
          "TT session was not reusable after a transport exception");
}

void checkCustomTemplate(const std::string &rax) {
  struct TemplateFile {
    ~TemplateFile() { std::remove(path.c_str()); }
    std::string path;
  } file{rax + ".serving-template-test.json"};
  {
    std::ofstream output(file.path);
    output << R"({
      "bos_token":"<|begin_of_text|>",
      "roles":{"user":"U:","assistant":"A:"},
      "turn_suffix":"<|eot_id|>",
      "stop_tokens":["B"],"stop_token_ids":[128010],
      "add_bos":true,"add_generation_prompt":true,
      "system_in_first_user":true
    })";
    require(output.good(), "could not create TT custom template fixture");
  }
  Fixture fixture(rax, file.path);
  require(std::remove(file.path.c_str()) == 0,
          "could not remove loaded TT custom template fixture");
  ChatCompletionRequest chat;
  chat.model = "custom-template-alias";
  chat.messages = {{"system", "s", {}}, {"user", "u", {}}};
  chat.sampling = params(2);
  const std::string expected = "<|begin_of_text|>U:s\n\nu<|eot_id|>A:";
  require(fixture.model.renderChat(chat) == expected &&
              fixture.model.capabilities().chatMultiTurn,
          "TT custom template was not cached or lost system-in-user rendering");
  auto result = fixture.model.chat(chat);
  require(result.content == "A" && result.usage.completionTokens == 2 &&
              result.finishReason == FinishReason::Stop &&
              fixture.backend.rendered == expected &&
              std::count(fixture.backend.prompt.begin(),
                         fixture.backend.prompt.end(), kBos) == 1,
          "TT custom template stop or single rendering contract changed");
  fixture.backend.script = {'A', 128010};
  result = fixture.model.chat(chat);
  require(result.content == "A" && result.usage.completionTokens == 2 &&
              result.finishReason == FinishReason::Stop,
          "TT custom template stop IDs were not merged");
  for (const std::vector<ChatMessage> &messages :
       {std::vector<ChatMessage>{
            {"system", "s", {}}, {"assistant", "a", {}}, {"user", "u", {}}},
        std::vector<ChatMessage>{{"user", "u", {}}, {"system", "s", {}}},
        std::vector<ChatMessage>{
            {"system", "s", {}}, {"system", "s", {}}, {"user", "u", {}}},
        std::vector<ChatMessage>{{"system", "s", {}}}}) {
    chat.messages = messages;
    requireFailure([&] { fixture.model.renderChat(chat); }, "system");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected the no-weight TT serving .rax fixture");
    Fixture fixture(argv[1]);
    checkCapabilities(fixture, argv[1]);
    checkCodec(fixture);
    checkBudgets(fixture);
    checkStopsAndUtf8(fixture);
    checkUnsupportedSampling(fixture);
    checkChat(fixture);
    checkCancellationAndFailures(fixture);
    checkCustomTemplate(argv[1]);
    std::cout << "Llama31TT raw codec and greedy serving contract: passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
