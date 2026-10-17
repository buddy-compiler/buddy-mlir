//===- Qwen3VLServingTest.cpp - Production Qwen3-VL serving checks --------===//
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
#include "buddy/runtime/models/Qwen3VLResidentModel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace buddy::runtime;
namespace fs = std::filesystem;

namespace {

constexpr int kEndOfText = 256;
constexpr int kStart = 257;
constexpr int kEnd = 258;
constexpr int kVisionStart = 259;
constexpr int kVisionEnd = 260;
constexpr int kImage = 261;

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

std::string imageBlock() {
  std::string result = "<|vision_start|>";
  for (int i = 0; i < 98; ++i)
    result += "<|image_pad|>";
  return result + "<|vision_end|>";
}

std::string canonicalPrompt(const std::string &text = "Hi") {
  return "<|im_start|>user\n" + imageBlock() + text +
         "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<int> canonicalTokens() {
  std::vector<int> result = {kStart, 'u', 's', 'e', 'r', '\n', kVisionStart};
  result.insert(result.end(), 98, kImage);
  for (int token : {kVisionEnd, int('H'), int('i'), kEnd, int('\n'), kStart,
                    int('a'), int('s'), int('s'), int('i'), int('s'), int('t'),
                    int('a'), int('n'), int('t'), int('\n')})
    result.push_back(token);
  require(result.size() == 121,
          "incorrect independent canonical token fixture");
  return result;
}

struct Events {
  bool operator()(const CompletionChunk &chunk) {
    require(terminals == 0, "Qwen callback followed its terminal event");
    require(!chunk.id.empty() && chunk.model == "resident-alias",
            "Qwen callbacks lost resident IDs or the configured model alias");
    if (id.empty())
      id = chunk.id;
    require(id == chunk.id, "Qwen callback ID changed within a request");
    if (chunk.done) {
      ++terminals;
      terminal = chunk;
      require(chunk.delta.empty(), "Qwen terminal event contains text");
    } else {
      ++deltas;
      text += chunk.delta;
    }
    return true;
  }

  void check(const CompletionResult &result) const {
    require(terminals == 1 && text == result.content && id == result.id &&
                result.model == "resident-alias" &&
                terminal.finishReason == result.finishReason &&
                terminal.usage.promptTokens == result.usage.promptTokens &&
                terminal.usage.completionTokens ==
                    result.usage.completionTokens &&
                terminal.usage.totalTokens == result.usage.totalTokens,
            "Qwen stream and final result disagree");
    require(std::isfinite(result.timings.prefillMs) &&
                std::isfinite(result.timings.decodeMs) &&
                std::isfinite(result.timings.tokensPerSecond) &&
                result.timings.prefillMs >= 0 && result.timings.decodeMs >= 0 &&
                result.timings.tokensPerSecond >= 0,
            "Qwen returned invalid timings");
  }

  int deltas = 0;
  int terminals = 0;
  std::string id;
  std::string text;
  CompletionChunk terminal;
};

class KernelControl {
public:
  explicit KernelControl(const std::string &path) {
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    require(handle != nullptr, "could not open test kernel control library");
    configure = symbol<Configure>("qwen3vl_test_configure");
    stat = symbol<Stat>("qwen3vl_test_stat");
  }
  ~KernelControl() { dlclose(handle); }

  void reset(const std::vector<int> &script = {'A', 'B', kEnd},
             const std::vector<int> &prompt = canonicalTokens(),
             int failureCall = 0) {
    configure(prompt.data(), static_cast<long>(prompt.size()), script.data(),
              static_cast<long>(script.size()), 7, failureCall);
  }
  int visions() const { return stat(0); }
  int decoders() const { return stat(1); }
  int previousChecks() const { return stat(2); }

private:
  using Configure = void (*)(const int *, long, const int *, long, long, int);
  using Stat = int (*)(int);
  template <typename Function> Function symbol(const char *name) {
    void *address = dlsym(handle, name);
    require(address != nullptr, std::string("missing kernel control: ") + name);
    return reinterpret_cast<Function>(address);
  }
  void *handle = nullptr;
  Configure configure = nullptr;
  Stat stat = nullptr;
};

void load(Qwen3VLResidentModel &model, const fs::path &directory,
          const std::string &profile = "canonical",
          const std::string &chatTemplate = {}) {
  ResidentModelConfig config;
  config.raxPath = (directory / (profile + ".rax")).string();
  config.modelName = "resident-alias";
  config.chatTemplatePath = chatTemplate;
  model.load(config);
  require(model.status().state == ModelLoadState::Ready &&
              model.status().contextLength == 512 &&
              model.status().modelName == "resident-alias",
          "Qwen fixture did not load through the production runtime");
}

CompletionResult complete(Qwen3VLResidentModel &model,
                          const SamplingParams &sampling,
                          Events *events = nullptr,
                          const std::string &prompt = canonicalPrompt()) {
  CompletionRequest request;
  request.prompt = prompt;
  request.sampling = sampling;
  if (!events)
    return model.complete(request);
  auto result = model.completeStream(
      request, [&](const CompletionChunk &chunk) { return (*events)(chunk); });
  events->check(result);
  return result;
}

void checkCapabilitiesAndRaw(Qwen3VLResidentModel &model) {
  const auto caps = model.capabilities();
  require(caps.rawTokenize && caps.rawDetokenize && caps.tokenizeAddSpecial &&
              caps.tokenizeParseSpecial && caps.tokenizeWithPieces &&
              caps.maxNewTokens && caps.streaming && caps.stop &&
              caps.stopTokenIds && caps.maxImages == 1 && caps.imagePaths &&
              !caps.imageBytes,
          "Qwen serving capabilities are incomplete or overstate images");
  require(!caps.sampling.temperature && !caps.sampling.topK &&
              !caps.sampling.topP && !caps.sampling.minP &&
              !caps.sampling.repeatPenalty && !caps.sampling.repeatLastN &&
              !caps.sampling.seed,
          "Qwen greedy runtime advertised unsupported sampling");
  for (const char *role : {"system", "user", "assistant"})
    require(std::find(caps.chatRoles.begin(), caps.chatRoles.end(), role) !=
                caps.chatRoles.end(),
            "Qwen omitted a renderable role");

  TokenizeRequest request;
  request.content = "Hi";
  request.addSpecial = false;
  request.withPieces = true;
  auto tokens = model.tokenize(request);
  require(tokens.tokens == std::vector<int>({'H', 'i'}) && tokens.count == 2 &&
              tokens.pieces == std::vector<std::string>({"H", "i"}),
          "raw Qwen tokenize added a chat/image scaffold");
  request.addSpecial = true;
  require(model.tokenize(request).tokens == tokens.tokens,
          "Qwen add_special invented an automatic BOS");
  request.countOnly = true;
  tokens = model.tokenize(request);
  require(tokens.count == 2 && tokens.tokens.empty() && tokens.pieces.empty(),
          "Qwen count_only retained tokens/pieces");
  request.countOnly = false;
  request.addSpecial = false;
  request.content = "<|im_end|>";
  require(model.tokenize(request).tokens == std::vector<int>{kEnd},
          "Qwen parse_special ignored artifact-defined IDs");
  request.parseSpecial = false;
  tokens = model.tokenize(request);
  require(tokens.tokens != std::vector<int>{kEnd} &&
              model.detokenize({tokens.tokens}).content == request.content,
          "Qwen parse_special=false did not preserve literal special text");
  request.content = "  \n\t中";
  request.content += '\0';
  tokens = model.tokenize(request);
  require(model.detokenize({tokens.tokens}).content == request.content,
          "Qwen raw round trip changed whitespace, Unicode or NUL");
  request.content.assign(600, 'x');
  require(model.tokenize(request).count == 600,
          "Qwen raw tokenizer truncated at the execution context");
  request.content.clear();
  require(model.tokenize(request).count == 0 &&
              model.detokenize({}).content.empty(),
          "empty raw Qwen codec inserted a template");
  std::string preserved = " <|endoftext|><|im_end|>A";
  preserved += '\0';
  require(model.detokenize({{32, kEndOfText, kEnd, 'A', 0}}).content ==
              preserved,
          "Qwen detokenize trimmed whitespace, EOS tail or NUL");
  require(model.detokenize({{0xe4, 0xb8, 0xad}}).content == "中" &&
              model.detokenize({{0xe4, 0xb8}}).content == "�" &&
              model.detokenize({{0x80, 'A'}}).content == "�A",
          "Qwen raw decoder repaired bytes before concatenation");
  for (int token : {-1, 262})
    requireFailure([&] { model.detokenize({{token}}); },
                   "outside the vocabulary");
  request.content = canonicalPrompt();
  request.parseSpecial = true;
  require(model.tokenize(request).tokens == canonicalTokens(),
          "fixture token IDs disagree with the canonical resident template");
}

void checkBudgets(Qwen3VLResidentModel &model, KernelControl &kernel) {
  struct Case {
    int budget;
    const char *content;
    int sampled;
    FinishReason reason;
  };
  for (const auto &test : {Case{0, "", 0, FinishReason::Length},
                           Case{1, "A", 1, FinishReason::Length},
                           Case{2, "AB", 2, FinishReason::Length},
                           Case{8, "AB", 3, FinishReason::Stop},
                           Case{-1, "AB", 3, FinishReason::Stop}}) {
    kernel.reset();
    Events events;
    auto result = complete(model, params(test.budget), &events);
    require(result.content == test.content &&
                result.finishReason == test.reason &&
                result.usage.promptTokens == 121 &&
                result.usage.completionTokens == test.sampled &&
                result.usage.totalTokens == 121 + test.sampled &&
                kernel.visions() == 1 &&
                kernel.decoders() == std::max(1, test.sampled),
            "Qwen budget, prefill, decoder row, EOS usage or template count "
            "changed");
    if (test.sampled > 1)
      require(kernel.previousChecks() > 0,
              "Qwen never embedded the previously selected token");
    // Reuse without resetting kernel control: selection must depend on row,
    // not hidden global state or stale embeddings from the previous request.
    const auto repeated = complete(model, params(test.budget));
    require(
        repeated.content == result.content &&
            repeated.finishReason == result.finishReason &&
            repeated.usage.completionTokens == result.usage.completionTokens &&
            repeated.id != result.id,
        "Qwen repeated/non-streamed request changed results or reused an ID");
  }
  for (int eos : {kEnd, kEndOfText}) {
    kernel.reset({eos});
    auto result = complete(model, params(1));
    require(result.content.empty() && result.usage.completionTokens == 1 &&
                result.finishReason == FinishReason::Stop &&
                kernel.decoders() == 1,
            "Qwen first-token EOS was not counted or lost stop precedence");
  }
}

void checkStopsAndUtf8(Qwen3VLResidentModel &model, KernelControl &kernel) {
  kernel.reset({'p', 'S', 'T', 'O', 'P', 'x'});
  auto sampling = params(-1);
  sampling.stop = {"STOP", "TOP"};
  Events events;
  auto result = complete(model, sampling, &events);
  require(result.content == "p" && result.usage.completionTokens == 5 &&
              result.finishReason == FinishReason::Stop,
          "Qwen cross-token/overlapping string stop leaked text or usage");
  kernel.reset({0xe4, 0xb8, 0xad, kEnd});
  Events unicode;
  result = complete(model, params(-1), &unicode);
  require(result.content == "中" && result.usage.completionTokens == 4,
          "Qwen emitted replacement characters inside split UTF-8");
  sampling.stop = {"中"};
  kernel.reset({0xe4, 0xb8, 0xad, kEnd});
  result = complete(model, sampling);
  require(result.content.empty() && result.usage.completionTokens == 3 &&
              result.finishReason == FinishReason::Stop,
          "Qwen stop inside split UTF-8 leaked text");
  kernel.reset({0xe4, 0xb8});
  require(complete(model, params(2)).content == "�",
          "Qwen incomplete UTF-8 tail was not repaired once");
  kernel.reset({'S', 'T'});
  sampling.maxNewTokens = 2;
  sampling.stop = {"STOP"};
  require(complete(model, sampling).content == "ST",
          "Qwen did not flush a partial stop at the token budget");
  kernel.reset({'A', 'B', 'x'});
  sampling = params(2);
  sampling.stopTokenIds = {'B'};
  result = complete(model, sampling);
  require(result.content == "A" && result.usage.completionTokens == 2 &&
              result.finishReason == FinishReason::Stop,
          "Qwen token stop lost precedence to the same-step budget");
}

void checkContextLimits(const fs::path &directory, KernelControl &kernel) {
  for (const auto &profile :
       {std::make_pair("full", 512), std::make_pair("near", 511)}) {
    Qwen3VLResidentModel model;
    load(model, directory, profile.first);
    const int promptCount = profile.second;
    const std::string prompt =
        canonicalPrompt() + std::string(promptCount - 121, 'x');
    auto tokens = canonicalTokens();
    tokens.insert(tokens.end(), promptCount - 121, 'x');
    for (int budget : {0, 2, -1}) {
      kernel.reset({'A', 'B', kEnd}, tokens);
      Events events;
      const auto result = complete(model, params(budget), &events, prompt);
      const int generated = promptCount == 512 || budget == 0 ? 0 : 1;
      require(result.content == (generated ? "A" : "") &&
                  result.finishReason == FinishReason::Length &&
                  result.usage.promptTokens == promptCount &&
                  result.usage.completionTokens == generated &&
                  result.usage.totalTokens == promptCount + generated &&
                  kernel.visions() == 1 && kernel.decoders() == 1 &&
                  kernel.previousChecks() == 0,
              "Qwen full/near context skipped prefill, exceeded its remaining "
              "position or decoded an unusable token");
    }
  }
}

void checkUnsupportedSampling(Qwen3VLResidentModel &model,
                              KernelControl &kernel) {
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
    kernel.reset();
    auto sampling = params(1);
    sampling.explicitParameters.*field.second = true;
    requireFailure([&] { complete(model, sampling); }, field.first);
    require(kernel.visions() == 0 && kernel.decoders() == 0,
            "Qwen unsupported explicit sampling reached a kernel");
  }
  using Change = std::function<void(SamplingParams &)>;
  for (const auto &change : std::vector<std::pair<std::string, Change>>{
           {"temperature",
            [](SamplingParams &p) { p.samplerConfig.temperature = 1; }},
           {"top_k", [](SamplingParams &p) { p.samplerConfig.topK = 2; }},
           {"top_p", [](SamplingParams &p) { p.samplerConfig.topP = 0.5f; }},
           {"min_p", [](SamplingParams &p) { p.samplerConfig.minP = 0.1f; }},
           {"repeat_penalty",
            [](SamplingParams &p) { p.samplerConfig.repeatPenalty = 1.1f; }},
           {"repeat_last_n",
            [](SamplingParams &p) { p.samplerConfig.repeatLastN = 0; }},
           {"seed", [](SamplingParams &p) {
              p.seedPolicy = SamplingSeedPolicy::Fixed;
            }}}) {
    kernel.reset();
    auto sampling = params(1);
    change.second(sampling);
    requireFailure([&] { complete(model, sampling); }, change.first);
    require(kernel.visions() == 0,
            "Qwen unsupported direct sampling ran vision");
  }
  kernel.reset();
  for (int budget : {-2})
    requireFailure([&] { complete(model, params(budget)); }, "max_tokens");
  for (long long token : {-1LL, 262LL}) {
    auto sampling = params(1);
    sampling.stopTokenIds = {token};
    requireFailure([&] { complete(model, sampling); }, "stop_token_ids");
  }
  auto sampling = params(1);
  sampling.stop = {""};
  requireFailure([&] { complete(model, sampling); }, "stop");
}

void checkCancellationAndFailures(Qwen3VLResidentModel &model,
                                  KernelControl &kernel) {
  CompletionRequest request;
  request.prompt = canonicalPrompt();
  request.sampling = params(-1);
  kernel.reset();
  int callbacks = 0;
  int terminals = 0;
  auto result =
      model.completeStream(request, [&](const CompletionChunk &chunk) {
        ++callbacks;
        terminals += chunk.done;
        return false;
      });
  require(result.finishReason == FinishReason::Cancelled && callbacks == 1 &&
              terminals == 0 && result.usage.completionTokens == 1 &&
              kernel.decoders() == 1,
          "Qwen cancellation emitted done or continued decoding");
  require(complete(model, params(-1)).content == "AB",
          "Qwen was not reusable after cancellation");
  for (int failureCall : {1, 2}) {
    kernel.reset({'A', 'B', kEnd}, canonicalTokens(), failureCall);
    requireFailure([&] { model.complete(request); },
                   "scripted Qwen decoder failure");
    kernel.reset();
    require(complete(model, params(-1)).content == "AB",
            "Qwen was not reusable after prefill/decode failure");
  }
  kernel.reset();
  callbacks = 0;
  requireFailure(
      [&] {
        model.completeStream(request, [&](const CompletionChunk &) -> bool {
          ++callbacks;
          throw std::runtime_error("scripted transport failure");
        });
      },
      "transport failure");
  require(callbacks == 1 && kernel.decoders() == 1,
          "Qwen transport exception continued callbacks or decoding");
  require(complete(model, params(-1)).content == "AB",
          "Qwen was not reusable after a transport exception");
}

void checkChatAndImages(Qwen3VLResidentModel &model, KernelControl &kernel,
                        const fs::path &directory) {
  ChatCompletionRequest request;
  request.model = "unrelated-client-alias";
  request.messages = {{"user", "Hi", {}}};
  request.sampling = params(-1);
  require(model.renderChat(request) == canonicalPrompt(),
          "Qwen canonical chat template or bundled image fallback changed");
  kernel.reset();
  Events events;
  auto result = model.chatStream(
      request, [&](const CompletionChunk &chunk) { return events(chunk); });
  events.check(result);
  require(result.content == "AB" && result.usage.promptTokens == 121,
          "Qwen chat applied its template twice");
  request.messages.clear();
  request.input = "Hi";
  require(model.renderChat(request) == canonicalPrompt(),
          "Qwen bare input did not become one user turn");
  request.messages = {{"system", "s", {}},
                      {"user", "", {}},
                      {"assistant", "a", {}},
                      {"user", "v", {}}};
  const auto rendered = model.renderChat(request);
  require(
      rendered.find("<|im_start|>system\ns<|im_end|>\n") == 0 &&
          rendered.find("<|im_start|>user\n" + imageBlock() +
                        "<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n") !=
              std::string::npos &&
          rendered.find("<|im_start|>user\nv<|im_end|>\n") != std::string::npos,
      "Qwen flattened roles, empty turns or multiple user turns");
  kernel.reset();
  requireFailure([&] { model.chat(request); }, "packaged");
  require(kernel.visions() == 0, "Qwen static chat mismatch reached vision");
  request.messages = {{"tool", "Hi", {}}};
  requireFailure([&] { model.renderChat(request); }, "role");

  const auto path = (directory / "image.ppm").string();
  ImageInput image;
  image.uri = path;
  request.messages = {{"user", "Hi", {image}}};
  request.images = {image}; // Exactly the mirror produced by JsonCodec.
  kernel.reset();
  require(
      model.renderChat(request) == canonicalPrompt() &&
          model.chat(request).content == "AB",
      "Qwen counted a DTO image mirror twice or failed local preprocessing");
  request.images.clear();
  kernel.reset();
  require(model.chat(request).content == "AB",
          "Qwen direct message image needed a transport mirror");
  request.messages[0].images.push_back(image);
  requireFailure([&] { model.chat(request); }, "image");
  request.messages = {{"user", "Hi", {image}}, {"user", "Hi", {image}}};
  requireFailure([&] { model.chat(request); }, "image");
  request.messages = {{"assistant", "Hi", {image}}};
  requireFailure([&] { model.chat(request); }, "user");
  request.messages = {{"user", "Hi", {image}}};
  request.messages[0].imagesAfterText = true;
  requireFailure([&] { model.chat(request); }, "image");
  request.messages[0].imagesAfterText = false;
  ImageInput other = image;
  other.uri += ".different";
  request.images = {other};
  requireFailure([&] { model.chat(request); }, "image");

  CompletionRequest completion;
  completion.prompt = canonicalPrompt();
  completion.sampling = params(1);
  for (const auto &uri :
       {std::string(), std::string("https://example.invalid/x"),
        std::string("data:image/png;base64,AA"),
        std::string("file://remote/image"), std::string("image\0tail", 10)}) {
    image.uri = uri;
    completion.images = {image};
    kernel.reset();
    requireFailure([&] { model.complete(completion); }, "image");
    require(kernel.visions() == 0, "Qwen invalid image reached vision");
  }
  image.uri = path;
  image.bytes = {1};
  completion.images = {image};
  requireFailure([&] { model.complete(completion); }, "bytes");
  image.bytes.clear();
  completion.images = {image, image};
  requireFailure([&] { model.complete(completion); }, "one image");

  const std::string jsonPath =
      path; // Build paths contain no JSON metacharacters.
  const std::string prefix = "{\"messages\":[{\"role\":\"user\",\"content\":[";
  const std::string imagePart =
      "{\"type\":\"image_url\",\"image_url\":\"" + jsonPath + "\"}";
  const std::string textPart = "{\"type\":\"text\",\"text\":\"Hi\"}";
  auto decoded = buddy::server::parseChatCompletionRequest(
      prefix + imagePart + "," + textPart + "]}],\"n_predict\":1}");
  require(decoded.request.images.size() == 1 &&
              decoded.request.messages[0].images.size() == 1,
          "Qwen test did not exercise the HTTP image mirror");
  kernel.reset();
  require(model.chat(decoded.request).content == "A",
          "Qwen HTTP message image mirror was counted as a second image");
  decoded = buddy::server::parseChatCompletionRequest(
      prefix + textPart + "," + imagePart + "]}],\"n_predict\":1}");
  requireFailure([&] { model.chat(decoded.request); }, "image");
}

void checkStaticValidation(Qwen3VLResidentModel &model, KernelControl &kernel,
                           const fs::path &directory) {
  kernel.reset();
  requireFailure(
      [&] { complete(model, params(1), nullptr, canonicalPrompt("Hey")); },
      "packaged");
  // Same token count, different image offset: length checking alone is unsafe.
  auto shiftedPrompt = canonicalPrompt();
  shiftedPrompt.erase(shiftedPrompt.find("Hi"), 1);
  shiftedPrompt.insert(shiftedPrompt.find("<|vision_start|>"), "x");
  requireFailure([&] { complete(model, params(1), nullptr, shiftedPrompt); },
                 "image positions");
  require(kernel.visions() == 0,
          "Qwen static position mismatch reached vision");
  for (const char *profile : {"shifted", "length"}) {
    Qwen3VLResidentModel mismatched;
    load(mismatched, directory, profile);
    requireFailure([&] { complete(mismatched, params(1)); },
                   std::string(profile) == "shifted" ? "image positions"
                                                     : "packaged");
  }
  requireFailure(
      [&] { complete(model, params(1), nullptr, std::string(513, 'x')); },
      "context");
  kernel.reset();
  require(complete(model, params(-1)).content == "AB",
          "Qwen failed to recover after static validation errors");
}

void checkAliases(Qwen3VLResidentModel &model, KernelControl &kernel) {
  for (const std::string &fields :
       {std::string(
            "\"n_predict\":1,\"max_completion_tokens\":2,\"max_tokens\":3"),
        std::string(
            "\"n_predict\":null,\"max_completion_tokens\":1,\"max_tokens\":3"),
        std::string("\"n_predict\":null,\"max_completion_tokens\":null,\"max_"
                    "tokens\":1")}) {
    auto decoded = buddy::server::parseChatCompletionRequest(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," + fields +
        "}");
    kernel.reset();
    const auto result = model.chat(decoded.request);
    require(result.content == "A" && result.usage.completionTokens == 1,
            "Qwen did not execute the selected generated-token length alias");
  }
}

void checkCustomTemplate(const fs::path &directory, KernelControl &kernel) {
  Qwen3VLResidentModel model;
  load(model, directory, "canonical",
       (directory / "custom-template.json").string());
  ChatCompletionRequest request;
  request.messages = {{"system", "s", {}}, {"user", "Hi", {}}};
  request.sampling = params(1);
  require(
      model.renderChat(request) ==
          "S:s<|im_end|>U:" + imageBlock() + "Hi<|im_end|>A:",
      "Qwen ignored the configured template or lost system/image placement");
  requireFailure([&] { model.chat(request); }, "packaged");

  const auto read = [](const fs::path &path) {
    std::ifstream input(path);
    require(input.good(), "could not read Qwen template fixture");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  };
  const std::string canonical = read(directory / "canonical-template.json");
  const std::string changed = read(directory / "custom-template.json");
  const auto withField = [&](const std::string &field) {
    const auto end = canonical.find_last_of('}');
    require(end != std::string::npos,
            "canonical template is not a JSON object");
    return canonical.substr(0, end) + "," + field + "}\n";
  };
  struct TemplateFile {
    ~TemplateFile() { std::remove(path.c_str()); }
    void write(const std::string &text) const {
      std::ofstream output(path);
      output << text;
      require(output.good(), "could not write Qwen template fixture");
    }
    std::string path;
  } file{(directory / "active-template-test.json").string()};

  request.messages = {{"user", "Hi", {}}};
  request.sampling = params(-1);
  for (const auto &field : {std::string("\"stop_token_ids\":[66]"),
                            std::string("\"stop_tokens\":[\"B\"]")}) {
    file.write(withField(field));
    load(model, directory, "canonical", file.path);
    require(model.renderChat(request) == canonicalPrompt(),
            "configured Qwen template differs from its canonical layout");
    kernel.reset();
    Events events;
    auto result = model.chatStream(
        request, [&](const CompletionChunk &chunk) { return events(chunk); });
    events.check(result);
    require(
        result.content == "A" && result.finishReason == FinishReason::Stop &&
            result.usage.promptTokens == 121 &&
            result.usage.completionTokens == 2 && kernel.decoders() == 2,
        "Qwen configured template was applied twice or its stop was ignored");

    // Editing the file must not change either its layout or configured stop
    // during an already loaded resident's lifetime.
    file.write(changed);
    require(model.renderChat(request) == canonicalPrompt(),
            "Qwen re-read its chat template after loading");
    kernel.reset();
    result = model.chat(request);
    require(
        result.content == "A" && result.usage.completionTokens == 2 &&
            result.finishReason == FinishReason::Stop,
        "Qwen lost its cached configured stop after the template file changed");
    load(model, directory, "canonical", file.path);
    require(model.renderChat(request) == "U:" + imageBlock() + "Hi<|im_end|>A:",
            "Qwen reload did not pick up the changed template");
    requireFailure([&] { model.chat(request); }, "packaged");
  }

  for (const auto &invalid :
       {std::make_pair("\"stop_token_ids\":[2147483648]", "stop_token_ids"),
        std::make_pair("\"stop_token_ids\":[262]", "vocabulary"),
        std::make_pair("\"stop_token_ids\":[66.5]", "stop_token_ids"),
        std::make_pair("\"stop_token_ids\":[\"66\"]", "stop_token_ids"),
        std::make_pair("\"stop_tokens\":[\"\"]", "stop_tokens")}) {
    file.write(canonical);
    load(model, directory, "canonical", file.path);
    file.write(withField(invalid.first));
    kernel.reset();
    requireFailure([&] { load(model, directory, "canonical", file.path); },
                   invalid.second);
    const auto caps = model.capabilities();
    require(
        model.status().state == ModelLoadState::Error && !caps.rawTokenize &&
            !caps.rawDetokenize && !caps.maxNewTokens &&
            caps.chatRoles.empty() && kernel.visions() == 0,
        "Qwen failed template reload retained a ready model or capabilities");
    TokenizeRequest raw;
    raw.content = "Hi";
    requireFailure([&] { model.tokenize(raw); }, "not loaded");
    requireFailure([&] { model.detokenize({{'A'}}); }, "not loaded");
    requireFailure([&] { model.chat(request); }, "not loaded");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "expected Qwen fixture directory and kernel library");
    const fs::path directory(argv[1]);
    KernelControl kernel(argv[2]);
    Qwen3VLResidentModel model;
    require(!model.capabilities().rawTokenize &&
                !model.capabilities().maxNewTokens,
            "unloaded Qwen advertised serving capabilities");
    load(model, directory);
    checkCapabilitiesAndRaw(model);
    checkBudgets(model, kernel);
    checkContextLimits(directory, kernel);
    checkStopsAndUtf8(model, kernel);
    checkUnsupportedSampling(model, kernel);
    checkCancellationAndFailures(model, kernel);
    checkChatAndImages(model, kernel, directory);
    checkStaticValidation(model, kernel, directory);
    checkAliases(model, kernel);
    checkCustomTemplate(directory, kernel);
    std::cout
        << "Qwen3-VL production runtime/raw codec/greedy serving: passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
