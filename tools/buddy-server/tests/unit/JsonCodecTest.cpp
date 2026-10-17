//===- JsonCodecTest.cpp - buddy-server JSON codec tests -----------------===//
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

#include "JsonCodec.h"

#include "llvm/Support/JSON.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using buddy::server::parseAudioTranscriptionRequest;
using buddy::server::parseChatCompletionRequest;
using buddy::server::parseCompletionRequest;
using buddy::server::parseEmbeddingRequest;
using buddy::server::parseMaskedLMRequest;
using buddy::server::toOpenAIAudioTranscriptionJson;
using buddy::server::toOpenAIEmbeddingJson;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

#define CHECK(condition) require(static_cast<bool>(condition), #condition)

template <typename Fn>
void expectFailure(Fn &&fn, const std::string &field = "") {
  try {
    fn();
  } catch (const buddy::server::JsonCodecError &error) {
    CHECK(std::string(error.what()).find(field) != std::string::npos);
    return;
  }
  throw std::runtime_error("expected JsonCodecError for " + field);
}

using namespace buddy::runtime;
using namespace buddy::server;
namespace json = llvm::json;

json::Value parseJson(const std::string &text) {
  auto value = json::parse(text);
  if (!value)
    throw std::runtime_error(llvm::toString(value.takeError()));
  return std::move(*value);
}

const json::Object &object(const json::Value &value) {
  CHECK(value.getAsObject());
  return *value.getAsObject();
}

const json::Value &member(const json::Value &value, llvm::StringRef key) {
  const auto *result = object(value).get(key);
  CHECK(result);
  return *result;
}

const json::Value &firstChoice(const json::Value &value) {
  const auto *choices = member(value, "choices").getAsArray();
  CHECK(choices && choices->size() == 1);
  return choices->front();
}

void testNormalization() {
  auto defaults = parseCompletionRequest(R"({"prompt":""})");
  CHECK(defaults.request.prompt.empty());
  const auto &sampling = defaults.request.sampling;
  CHECK(sampling.maxNewTokens == 512);
  CHECK(sampling.maxTokens == 512);
  CHECK(sampling.samplerConfig.temperature == 0);
  CHECK(sampling.samplerConfig.topK == 0);
  CHECK(sampling.samplerConfig.topP == 1);
  CHECK(sampling.samplerConfig.minP == 0);
  CHECK(sampling.samplerConfig.repeatPenalty == 1);
  CHECK(sampling.samplerConfig.repeatLastN == 64);
  CHECK(sampling.seedPolicy == SamplingSeedPolicy::Random);
  CHECK(!sampling.explicitParameters.maxNewTokens);
  CHECK(!defaults.stream && !defaults.includeUsage);

  auto aliases = parseCompletionRequest(
      R"({"prompt":"A","n_predict":1,"max_completion_tokens":2,"max_tokens":3})");
  CHECK(aliases.request.sampling.maxNewTokens == 1);
  CHECK(aliases.request.sampling.explicitParameters.maxNewTokens);
  auto nullAlias = parseCompletionRequest(
      R"({"prompt":"A","n_predict":null,"max_completion_tokens":2,"max_tokens":1})");
  CHECK(nullAlias.request.sampling.maxNewTokens == 2);
  auto lastAlias = parseCompletionRequest(
      R"({"prompt":"A","n_predict":null,"max_completion_tokens":null,"max_tokens":0})");
  CHECK(lastAlias.request.sampling.maxNewTokens == 0);
  auto noAlias = parseCompletionRequest(
      R"({"prompt":"A","n_predict":null,"max_completion_tokens":null,"max_tokens":null})");
  CHECK(noAlias.request.sampling.maxNewTokens == 512);
  CHECK(!noAlias.request.sampling.explicitParameters.maxNewTokens);
  auto ignored = parseCompletionRequest(
      R"({"prompt":"A","n_predict":1,"max_completion_tokens":{},"max_tokens":"ignored"})");
  CHECK(ignored.request.sampling.maxNewTokens == 1);
  for (int value : {-1, 0, 1, std::numeric_limits<int32_t>::max()}) {
    auto request = parseCompletionRequest(
        "{\"prompt\":\"A\",\"n_predict\":" + std::to_string(value) + "}");
    CHECK(request.request.sampling.maxNewTokens == value);
  }
  for (const char *value : {"-2", "2147483648", "1.5", "true", "\"1\""})
    expectFailure(
        [&] {
          parseCompletionRequest(
              std::string("{\"prompt\":\"A\",\"n_predict\":") + value + "}");
        },
        "n_predict");

  auto greedy = parseCompletionRequest(
      R"({"prompt":"A","temperature":-1,"top_k":-2147483648,"repeat_last_n":0})");
  CHECK(greedy.request.sampling.samplerConfig.temperature == 0);
  CHECK(greedy.request.sampling.samplerConfig.topK == 0);
  CHECK(greedy.request.sampling.samplerConfig.repeatLastN == 0);
  CHECK(greedy.request.sampling.explicitParameters.temperature);
  CHECK(greedy.request.sampling.explicitParameters.topK);

  const std::vector<std::pair<std::string, std::string>> invalid = {
      {"temperature", "\"hot\""},
      {"temperature", "true"},
      {"temperature", "1e100"},
      {"top_k", "2147483648"},
      {"top_k", "-2147483649"},
      {"top_k", "0.5"},
      {"top_p", "0"},
      {"top_p", "1.01"},
      {"top_p", "-1"},
      {"top_p", "true"},
      {"min_p", "1.01"},
      {"min_p", "-0.01"},
      {"repeat_penalty", "0"},
      {"repeat_penalty", "-1"},
      {"repeat_last_n", "-1"},
      {"repeat_last_n", "2147483648"},
      {"repeat_last_n", "false"},
      {"stream", "0"},
      {"model", "[]"},
      {"image_path", "12"}};
  for (const auto &entry : invalid)
    expectFailure(
        [&] {
          parseCompletionRequest("{\"prompt\":\"A\",\"" + entry.first +
                                 "\":" + entry.second + "}");
        },
        entry.first);

  for (const char *field :
       {"temperature", "top_k", "top_p", "min_p", "repeat_penalty",
        "repeat_last_n", "seed", "stop", "stop_token_ids", "stream",
        "stream_options", "model", "image_path", "grammar", "tools",
        "response_format"}) {
    auto request = parseCompletionRequest(std::string("{\"prompt\":\"A\",\"") +
                                          field + "\":null}");
    CHECK(request.request.sampling.maxNewTokens == 512);
    CHECK(!request.request.sampling.explicitParameters.temperature);
    CHECK(!request.request.sampling.explicitParameters.seed);
    CHECK(!request.stream && !request.includeUsage);
  }
  for (const char *field : {"unexpected", "include_usage"})
    expectFailure(
        [&] {
          parseCompletionRequest(std::string("{\"prompt\":\"A\",\"") + field +
                                 "\":null}");
        },
        field);
  expectFailure(
      [] {
        parseCompletionRequest(R"({"prompt":"A","grammar":"root ::= A"})");
      },
      "grammar");
  expectFailure([] { parseCompletionRequest(R"({"prompt":"A","tools":[]})"); },
                "tools");
  expectFailure([] { parseCompletionRequest(R"({"prompt":null})"); }, "prompt");
  expectFailure([] { parseCompletionRequest(R"({"prompt":[]})"); }, "prompt");
  expectFailure([] { parseCompletionRequest("{}"); }, "prompt");
  expectFailure([] { parseCompletionRequest("[]"); });
  expectFailure([] { parseCompletionRequest("not-json"); });

  auto included = parseCompletionRequest(
      R"({"prompt":"A","stream_options":{"include_usage":true}})");
  CHECK(!included.stream && included.includeUsage);
  auto includedChat = parseChatCompletionRequest(
      R"({"input":"A","stream":true,"stream_options":{"include_usage":true}})");
  CHECK(includedChat.stream && includedChat.includeUsage);
  auto nullUsage = parseCompletionRequest(
      R"({"prompt":"A","stream_options":{"include_usage":null}})");
  CHECK(!nullUsage.includeUsage);
  expectFailure(
      [] { parseCompletionRequest(R"({"prompt":"A","stream_options":[]})"); },
      "stream_options");
  expectFailure(
      [] {
        parseCompletionRequest(
            R"({"prompt":"A","stream_options":{"include_usage":1}})");
      },
      "stream_options.include_usage");
  expectFailure(
      [] {
        parseCompletionRequest(
            R"({"prompt":"A","stream_options":{"unknown":null}})");
      },
      "stream_options.unknown");
}

void testSeedsAndStops() {
  for (const char *seed : {"null", "-1", "4294967295"}) {
    auto decoded = parseCompletionRequest(
        std::string("{\"prompt\":\"A\",\"seed\":") + seed + "}");
    CHECK(decoded.request.sampling.seedPolicy == SamplingSeedPolicy::Random);
    CHECK(decoded.request.sampling.explicitParameters.seed ==
          (std::string(seed) != "null"));
  }
  for (uint32_t seed : {uint32_t(0), uint32_t(23), uint32_t(4294967294)}) {
    auto decoded = parseCompletionRequest(
        "{\"prompt\":\"A\",\"seed\":" + std::to_string(seed) + "}");
    CHECK(decoded.request.sampling.seedPolicy == SamplingSeedPolicy::Fixed);
    CHECK(decoded.request.sampling.fixedSeed == seed);
    CHECK(decoded.request.sampling.explicitParameters.seed);
  }
  for (const char *seed :
       {"-2", "4294967296", "18446744073709551615", "1.5", "true", "\"0\""})
    expectFailure(
        [&] {
          parseCompletionRequest(std::string("{\"prompt\":\"A\",\"seed\":") +
                                 seed + "}");
        },
        "seed");

  auto single = parseCompletionRequest(R"({"prompt":"A","stop":"中"})");
  CHECK(single.request.sampling.stop == std::vector<std::string>({"中"}));
  auto array = parseCompletionRequest(
      R"({"prompt":"A","stop":["AB","A"],"stop_token_ids":[0,7]})");
  CHECK(array.request.sampling.stop == std::vector<std::string>({"AB", "A"}));
  CHECK(array.request.sampling.stopTokenIds == std::vector<long long>({0, 7}));
  CHECK(array.request.sampling.explicitParameters.stop);
  CHECK(array.request.sampling.explicitParameters.stopTokenIds);
  for (const char *stop : {"\"\"", "[\"\"]", "[1]", "[null]", "{}", "false"})
    expectFailure(
        [&] {
          parseCompletionRequest(std::string("{\"prompt\":\"A\",\"stop\":") +
                                 stop + "}");
        },
        "stop");
  for (const char *ids : {"1", "[-1]", "[1.5]", "[true]", "[2147483648]"})
    expectFailure(
        [&] {
          parseCompletionRequest(
              std::string("{\"prompt\":\"A\",\"stop_token_ids\":") + ids + "}");
        },
        "stop_token_ids");
}

void testCodecsAndChat() {
  auto raw = parseTokenizeRequest(R"({"content":""})");
  CHECK(raw.content.empty() && !raw.addSpecial && raw.parseSpecial);
  CHECK(parseTokenizeRequest(R"({"content":"A","text":null})").content == "A");
  CHECK(!raw.withPieces && !raw.countOnly);
  auto text = parseTokenizeRequest(
      R"({"text":" A\n\"\u0000","add_special":null,"parse_special":null,"with_pieces":true,"count_only":true})");
  CHECK(text.content == std::string(" A\n\"\0", 5));
  CHECK(!text.addSpecial && text.parseSpecial && text.withPieces &&
        text.countOnly);
  expectFailure([] { parseTokenizeRequest(R"({"content":null,"text":"A"})"); },
                "content");
  expectFailure(
      [] { parseTokenizeRequest(R"({"content":"A","parse_special":1})"); },
      "parse_special");
  expectFailure(
      [] { parseTokenizeRequest(R"({"content":"A","with_pieces":"yes"})"); },
      "with_pieces");
  expectFailure(
      [] { parseTokenizeRequest(R"({"content":"A","count_only":0})"); },
      "count_only");
  expectFailure([] { parseTokenizeRequest(R"({"content":"A","extra":null})"); },
                "extra");
  CHECK(parseDetokenizeRequest(R"({"tokens":[]})").tokens.empty());
  CHECK(parseDetokenizeRequest(R"({"tokens":[0,7,2147483647]})").tokens ==
        std::vector<int>({0, 7, 2147483647}));
  for (const char *tokens :
       {"null", "0", "[-1]", "[1.5]", "[true]", "[\"1\"]", "[2147483648]"})
    expectFailure(
        [&] {
          parseDetokenizeRequest(std::string("{\"tokens\":") + tokens + "}");
        },
        "tokens");
  expectFailure([] { parseDetokenizeRequest("{}"); }, "tokens");
  expectFailure([] { parseDetokenizeRequest(R"({"tokens":[],"extra":null})"); },
                "extra");

  auto chat = parseChatCompletionRequest(
      R"({"input":"ignored","prompt":"ignored too","model":"requested","messages":[{"role":"system","content":"policy"},{"role":"user","content":[{"type":"text","text":" A"},{"type":"text","text":"B "}]}]})");
  CHECK(chat.request.input.empty());
  CHECK(chat.request.model == "requested");
  CHECK(chat.request.messages.size() == 2);
  CHECK(chat.request.messages[0].role == "system");
  CHECK(chat.request.messages[1].content == " AB ");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":[{"type":"text","text":"A","image_url":"/tmp/a.png"}]}]})");
      },
      "image_url");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"/tmp/a.png","text":"lost"}]}]})");
      },
      "text");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":[{"type":"image","image_url":"/tmp/a.png","image":"/tmp/b.png"}]}]})");
      },
      "image");
  auto nullCrossFields = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"A","image_url":null},{"type":"image_url","image_url":"/tmp/a.png","text":null}]}]})");
  CHECK(nullCrossFields.request.messages.front().content == "A");
  CHECK(nullCrossFields.request.images.size() == 1);
  CHECK(nullCrossFields.request.messages.front().imagesAfterText);
  auto prefixImage = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":""},{"type":"image_url","image_url":"image.png"},{"type":"text","text":"caption"}]}]})");
  CHECK(!prefixImage.request.messages.front().imagesAfterText);
  CHECK(prefixImage.request.messages.front().content == "caption");
  auto interleavedImage = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":"image.png"},{"type":"text","text":"after"}]}]})");
  CHECK(interleavedImage.request.messages.front().imagesAfterText);
  CHECK(interleavedImage.request.messages.front().content == "beforeafter");
  auto nullImageAlias = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":null,"uri":"/tmp/a.png"}}]}]})");
  CHECK(nullImageAlias.request.images.front().uri == "/tmp/a.png");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/tmp/a.png","uri":"/tmp/b.png"}}]}]})");
      },
      "url/uri");
  for (const char *messages :
       {"null", "[]", "{}", "[null]", "[{\"role\":\"user\"}]",
        "[{\"role\":\"user\",\"content\":null}]"})
    expectFailure([&] {
      parseChatCompletionRequest(
          std::string("{\"input\":\"fallback\",\"messages\":") + messages +
          "}");
    });
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":"A","tool_calls":[]}]})");
      },
      "tool_calls");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"messages":[{"role":"user","content":"A","unknown":null}]})");
      },
      "unknown");
  expectFailure(
      [] {
        parseChatCompletionRequest(
            R"({"input":2,"messages":[{"role":"user","content":"A"}]})");
      },
      "input");
  CHECK(parseChatCompletionRequest(R"({"prompt":"A"})").request.input == "A");
  CHECK(parseChatCompletionRequest(R"({"input":"A","prompt":"B"})")
            .request.input == "A");

  auto embedding = parseEmbeddingRequest(
      R"({"input":"A","model":null,"encoding_format":"float"})");
  CHECK(embedding.request.input == "A" && embedding.request.model.empty());
  for (const char *encoding : {"\"base64\"", "0", "[]"})
    expectFailure(
        [&] {
          parseEmbeddingRequest(
              std::string("{\"input\":\"A\",\"encoding_format\":") + encoding +
              "}");
        },
        "encoding_format");
  expectFailure(
      [] { parseEmbeddingRequest(R"({"input":"A","stream":"false"})"); },
      "stream");
  expectFailure([] { parseEmbeddingRequest(R"({"input":"A","extra":null})"); },
                "extra");
  expectFailure(
      [] { parseEmbeddingRequest(R"({"input":"A","dimensions":3})"); },
      "dimensions");
  CHECK(parseEmbeddingRequest(
            R"({"input":"A","dimensions":null,"encoding_format":null})")
            .request.input == "A");
}

void testCapabilities() {
  ResidentModelCapabilities caps;
  auto completion = parseCompletionRequest(R"({"prompt":"A"})");
  expectFailure([&] { validateCapabilities(completion, caps); }, "max_tokens");
  caps.maxNewTokens = true;
  validateCapabilities(completion, caps);
  for (const auto &entry : std::vector<std::pair<std::string, std::string>>{
           {"temperature", "0"},
           {"top_k", "0"},
           {"top_p", "1"},
           {"min_p", "0"},
           {"repeat_penalty", "1"},
           {"repeat_last_n", "64"},
           {"seed", "-1"},
           {"stop", "[]"},
           {"stop_token_ids", "[]"}}) {
    auto requested = parseCompletionRequest(
        "{\"prompt\":\"A\",\"" + entry.first + "\":" + entry.second + "}");
    expectFailure([&] { validateCapabilities(requested, caps); }, entry.first);
    auto absent = parseCompletionRequest("{\"prompt\":\"A\",\"" + entry.first +
                                         "\":null}");
    validateCapabilities(absent, caps);
  }
  auto streaming = parseCompletionRequest(R"({"prompt":"A","stream":true})");
  expectFailure([&] { validateCapabilities(streaming, caps); }, "stream");
  caps.streaming = true;
  validateCapabilities(streaming, caps);
  auto image =
      parseCompletionRequest(R"({"prompt":"A","image_path":"/tmp/a.png"})");
  expectFailure([&] { validateCapabilities(image, caps); }, "image");
  caps.maxImages = 1;
  expectFailure([&] { validateCapabilities(image, caps); }, "URI");
  caps.imagePaths = true;
  validateCapabilities(image, caps);
  image.request.images.front().uri.clear();
  image.request.images.front().bytes = {1, 2};
  expectFailure([&] { validateCapabilities(image, caps); }, "bytes");
  caps.imageBytes = true;
  validateCapabilities(image, caps);

  auto chat = parseChatCompletionRequest(
      R"({"messages":[{"role":"system","content":"S"},{"role":"user","content":"A"}]})");
  expectFailure([&] { validateCapabilities(chat, caps); }, "role");
  caps.chatRoles = {"system", "user", "assistant"};
  validateCapabilities(chat, caps);
  auto multi = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"B"},{"role":"user","content":"C"}]})");
  expectFailure([&] { validateCapabilities(multi, caps); }, "multi-turn");
  caps.chatMultiTurn = true;
  validateCapabilities(multi, caps);
  auto unsupportedRole = parseChatCompletionRequest(
      R"({"messages":[{"role":"developer","content":"A"}]})");
  expectFailure([&] { validateCapabilities(unsupportedRole, caps); },
                "developer");

  auto raw = parseTokenizeRequest(R"({"content":"A"})");
  expectFailure([&] { validateCapabilities(raw, caps); }, "raw");
  caps.rawTokenize = true;
  validateCapabilities(raw, caps);
  raw.addSpecial = true;
  expectFailure([&] { validateCapabilities(raw, caps); }, "add_special");
  caps.tokenizeAddSpecial = true;
  raw.parseSpecial = false;
  expectFailure([&] { validateCapabilities(raw, caps); }, "parse_special");
  caps.tokenizeParseSpecial = true;
  raw.withPieces = true;
  expectFailure([&] { validateCapabilities(raw, caps); }, "with_pieces");
  caps.tokenizeWithPieces = true;
  validateCapabilities(raw, caps);
  DetokenizeRequest tokens;
  expectFailure([&] { validateCapabilities(tokens, caps); }, "detokenize");
  caps.rawDetokenize = true;
  validateCapabilities(tokens, caps);
}

void testSerialization() {
  const std::string content = std::string(" A\n\"\0", 5) + "中";
  CompletionResult result;
  result.id = "stable-id";
  result.model = "actual-model";
  result.content = content;
  result.usage = {2, 3, 5};
  result.timings = {1, 2, 3};
  for (auto reason : {FinishReason::Stop, FinishReason::Length,
                      FinishReason::Cancelled, FinishReason::Error}) {
    result.finishReason = reason;
    auto native = parseJson(toJson(result));
    CHECK(member(native, "stop").getAsBoolean() == true);
    CHECK(member(native, "content").getAsString() == content);
    CHECK(member(member(native, "usage"), "completion_tokens").getAsInteger() ==
          3);
  }
  auto chat = parseJson(toOpenAIChatJson(result, 123));
  CHECK(member(chat, "created").getAsInteger() == 123);
  CHECK(member(chat, "id").getAsString() == result.id);
  CHECK(member(member(firstChoice(chat), "message"), "content").getAsString() ==
        content);
  CHECK(member(parseJson(toOpenAIChatJson(result)), "created")
            .getAsInteger()
            .value_or(0) > 0);

  CompletionChunk chunk;
  chunk.id = result.id;
  chunk.model = result.model;
  chunk.delta = content;
  chunk.usage = result.usage;
  auto role =
      parseJson(toOpenAIChatRoleChunkJson(result.id, result.model, 123));
  CHECK(member(member(firstChoice(role), "delta"), "role").getAsString() ==
        "assistant");
  CHECK(member(firstChoice(role), "finish_reason").getAsNull().has_value());
  auto incremental = parseJson(toOpenAIChatChunkJson(chunk, 123));
  CHECK(member(firstChoice(incremental), "finish_reason")
            .getAsNull()
            .has_value());
  CHECK(member(member(firstChoice(incremental), "delta"), "content")
            .getAsString() == content);
  CHECK(!object(incremental).get("usage"));
  chunk.done = true;
  chunk.finishReason = FinishReason::Length;
  auto terminal = parseJson(toOpenAIChatChunkJson(chunk, 123));
  CHECK(member(firstChoice(terminal), "finish_reason").getAsString() ==
        "length");
  CHECK(
      member(member(firstChoice(terminal), "delta"), "content").getAsString() ==
      content);
  CHECK(!object(terminal).get("usage"));
  auto nativeTerminal = parseJson(toCompletionChunkJson(chunk));
  CHECK(member(nativeTerminal, "content").getAsString() == content);
  CHECK(member(nativeTerminal, "stop").getAsBoolean() == true);
  auto usage = parseJson(toOpenAIChatUsageChunkJson(result, 123));
  CHECK(member(usage, "choices").getAsArray()->empty());
  CHECK(member(member(usage, "usage"), "total_tokens").getAsInteger() == 5);
  for (const auto *event : {&role, &incremental, &terminal, &usage}) {
    CHECK(member(*event, "id").getAsString() == result.id);
    CHECK(member(*event, "model").getAsString() == result.model);
    CHECK(member(*event, "created").getAsInteger() == 123);
  }

  TokenizeResult tokens;
  tokens.tokens = {3, 4};
  tokens.count = 2;
  tokens.pieces = {std::string("\xe4\xb8", 2), std::string("\n\"\0", 3)};
  auto pieces = parseJson(toJson(tokens, false, true));
  const auto *entries = member(pieces, "tokens").getAsArray();
  CHECK(entries && entries->size() == 2);
  CHECK(member((*entries)[0], "id").getAsInteger() == 3);
  CHECK(member((*entries)[0], "piece").getAsString() == "�");
  CHECK(member((*entries)[1], "piece").getAsString() == tokens.pieces[1]);
  auto ids = parseJson(toJson(tokens, false));
  CHECK(member(ids, "tokens").getAsArray()->front().getAsInteger() == 3);
  auto countOnly = parseJson(toJson(tokens, true, true));
  CHECK(member(countOnly, "count").getAsInteger() == 2);
  CHECK(!object(countOnly).get("tokens"));
  CHECK(member(parseJson(toJson(DetokenizeResult{content})), "content")
            .getAsString() == content);
  ModelStatus status;
  status.modelName = "actual-model";
  auto models = parseJson(toModelListJson(status, 123));
  const auto *data = member(models, "data").getAsArray();
  CHECK(data && data->size() == 1);
  CHECK(member(data->front(), "id").getAsString() == status.modelName);
  CHECK(member(data->front(), "object").getAsString() == "model");
}

} // namespace

int main() {
  testNormalization();
  testSeedsAndStops();
  testCodecsAndChat();
  testCapabilities();
  testSerialization();
  auto completion = parseCompletionRequest(
      R"({"prompt":"read","image_path":"file:/tmp/a.png"})");
  CHECK(completion.request.prompt == "read");
  CHECK(completion.request.images.size() == 1);
  CHECK(completion.request.images.front().uri == "/tmp/a.png");

  auto textChat = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"hello"}]})");
  CHECK(textChat.request.messages.size() == 1);
  CHECK(textChat.request.messages.front().content == "hello");
  CHECK(textChat.request.images.empty());

  auto multimodal = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"read"},{"type":"image_url","image_url":{"url":"/tmp/a.png"}}]}]})");
  CHECK(multimodal.request.messages.front().content == "read");
  CHECK(multimodal.request.images.size() == 1);
  CHECK(multimodal.request.messages.front().images.size() == 1);
  CHECK(multimodal.request.messages.front().images.front().uri == "/tmp/a.png");
  CHECK(multimodal.request.images.front().uri == "/tmp/a.png");

  expectFailure([] {
    parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://example/a.png"}},{"type":"image_url","image_url":{"url":"/tmp/b.png"}}]}]})");
  });
  expectFailure([] {
    auto images = parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/tmp/a.png"}},{"type":"image_url","image_url":{"url":"/tmp/b.png"}}]}]})");
    buddy::runtime::ResidentModelCapabilities caps;
    caps.maxNewTokens = true;
    caps.maxImages = 1;
    buddy::server::validateCapabilities(images, caps);
  });
  expectFailure([] {
    parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})");
  });

  auto embedding =
      parseEmbeddingRequest(R"({"model":"bge_m3_base","input":"hello world"})");
  CHECK(embedding.request.model == "bge_m3_base");
  CHECK(embedding.request.input == "hello world");
  expectFailure([] { parseEmbeddingRequest(R"({"model":"bge_m3_base"})"); });
  expectFailure([] {
    parseEmbeddingRequest(R"({"model":"bge_m3_base","input":["a","b"]})");
  });
  expectFailure(
      [] { parseEmbeddingRequest(R"({"model":"bge_m3_base","input":3})"); });
  expectFailure([] {
    parseEmbeddingRequest(
        R"({"model":"bge_m3_base","input":"a","stream":true})");
  });
  buddy::runtime::EmbeddingResult result;
  result.model = "bge_m3_base";
  result.embedding = {0.5f, -0.5f};
  result.promptTokens = 2;
  result.totalTokens = 2;
  const std::string json = toOpenAIEmbeddingJson(result);
  CHECK(json.find("\"object\":\"list\"") != std::string::npos);
  CHECK(json.find("\"index\":0") != std::string::npos);
  CHECK(json.find("\"prompt_tokens\":2") != std::string::npos);

  auto masked = parseMaskedLMRequest(
      R"({"model":"proteinglm_1b_mlm","prompt":"A <mask> C","top_k":3,"extra":true})");
  CHECK(masked.request.model == "proteinglm_1b_mlm");
  CHECK(masked.request.input == "A <mask> C");
  CHECK(masked.request.topK == 3);
  expectFailure([] { parseMaskedLMRequest(R"({"input":"x","prompt":"y"})"); });
  expectFailure([] { parseMaskedLMRequest(R"({"input":"x","top_k":0})"); });
  buddy::runtime::MaskedLMResult maskedResult;
  maskedResult.model = "proteinglm_1b_mlm";
  maskedResult.sequenceLength = 4;
  maskedResult.promptTokens = 4;
  maskedResult.predictions.push_back({1, {{3, "G", 12.3f}}});
  const std::string maskedJson = buddy::server::toJson(maskedResult);
  CHECK(maskedJson.find("\"object\":\"masked_lm\"") != std::string::npos);
  CHECK(maskedJson.find("\"position\":1") != std::string::npos);

  auto audio = parseAudioTranscriptionRequest(
      R"({"model":"whisper_base","file":"file:/tmp/sample.wav","max_tokens":7})");
  CHECK(audio.request.model == "whisper_base");
  CHECK(audio.request.audio.uri == "/tmp/sample.wav");
  CHECK(audio.request.audio.mimeType == "audio/wav");
  CHECK(audio.request.audio.bytes.empty());
  CHECK(audio.request.maxTokens == 7);
  auto alias =
      parseAudioTranscriptionRequest(R"({"audio_path":"/tmp/sample.wav"})");
  CHECK(alias.request.audio.uri == "/tmp/sample.wav");
  CHECK(alias.request.maxTokens == 64);

  expectFailure([] { parseAudioTranscriptionRequest(R"({})"); });
  expectFailure([] { parseAudioTranscriptionRequest(R"({"file":""})"); });
  expectFailure([] { parseAudioTranscriptionRequest(R"({"file":[]})"); });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"a.wav","audio_path":"b.wav"})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"http://example/a.wav"})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"data:audio/wav;base64,AA=="})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"a.wav","stream":true})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"a.wav","max_tokens":0})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"a.wav","max_tokens":448})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(R"({"file":"a.wav","audio_base64":"AA=="})");
  });
  expectFailure([] {
    parseAudioTranscriptionRequest(std::string("{\"file\":\"a") + '\0' +
                                   ".wav\"}");
  });

  buddy::runtime::AudioTranscriptionResult transcriptionResult;
  transcriptionResult.model = "whisper_base";
  transcriptionResult.text = "hello";
  transcriptionResult.generatedTokens = 2;
  transcriptionResult.timings.preprocessMs = 1.25;
  transcriptionResult.timings.inferenceMs = 3.5;
  transcriptionResult.timings.totalMs = 4.75;
  const std::string audioJson =
      toOpenAIAudioTranscriptionJson(transcriptionResult);
  CHECK(audioJson.find("\"text\":\"hello\"") != std::string::npos);
  CHECK(audioJson.find("\"model\":\"whisper_base\"") != std::string::npos);
  CHECK(audioJson.find("\"generated_tokens\":2") != std::string::npos);
  CHECK(audioJson.find("\"preprocess_ms\":1.25") != std::string::npos);

  std::cout << "JsonCodec tests passed\n";
  return 0;
}
