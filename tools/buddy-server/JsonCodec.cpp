//===- JsonCodec.cpp - buddy-server JSON conversion helpers ---------------===//
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

#include "buddy/runtime/llm/TextGeneration.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace buddy {
namespace server {

namespace {

using namespace buddy::runtime;
namespace json = llvm::json;

std::string serialize(json::Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << value;
  return os.str();
}

json::Value parseValue(const std::string &body) {
  auto parsed = json::parse(body);
  if (!parsed)
    throw JsonCodecError("invalid JSON: " + llvm::toString(parsed.takeError()));
  return std::move(*parsed);
}

const json::Object &asObject(const json::Value &value) {
  const auto *obj = value.getAsObject();
  if (!obj)
    throw JsonCodecError("JSON root must be an object");
  return *obj;
}

const json::Value *optionalValue(const json::Object &obj, llvm::StringRef key) {
  const auto *value = obj.get(key);
  return value && value->kind() != json::Value::Null ? value : nullptr;
}

const json::Value &requiredValue(const json::Object &obj, llvm::StringRef key) {
  const auto *value = obj.get(key);
  if (!value)
    throw JsonCodecError("missing required field: " + key.str());
  return *value;
}

std::string checkedString(const json::Value &value, llvm::StringRef key) {
  auto string = value.getAsString();
  if (!string)
    throw JsonCodecError(key.str() + " must be a string");
  return string->str();
}

std::string optionalString(const json::Object &obj, llvm::StringRef key) {
  if (const auto *value = optionalValue(obj, key))
    return checkedString(*value, key);
  return "";
}

bool checkedBool(const json::Object &obj, llvm::StringRef key, bool fallback) {
  if (const auto *value = optionalValue(obj, key)) {
    auto boolean = value->getAsBoolean();
    if (!boolean)
      throw JsonCodecError(key.str() + " must be a boolean");
    return *boolean;
  }
  return fallback;
}

int64_t checkedInteger(const json::Value &value, llvm::StringRef key,
                       int64_t minimum, int64_t maximum) {
  // Check the small DTO range before LLVM's int64 conversion: large JSON
  // doubles must never be narrowed before range validation.
  auto number = value.getAsNumber();
  if (!number || !std::isfinite(*number) || *number < minimum ||
      *number > maximum)
    throw JsonCodecError(key.str() + " must be an integer in [" +
                         std::to_string(minimum) + ", " +
                         std::to_string(maximum) + "]");
  auto integer = value.getAsInteger();
  if (!integer || *integer < minimum || *integer > maximum)
    throw JsonCodecError(key.str() + " must be an integer in [" +
                         std::to_string(minimum) + ", " +
                         std::to_string(maximum) + "]");
  return *integer;
}

double checkedNumber(const json::Value &value, llvm::StringRef key) {
  auto number = value.getAsNumber();
  if (!number || !std::isfinite(*number))
    throw JsonCodecError(key.str() + " must be a finite number");
  return *number;
}

float checkedFloat(double value, llvm::StringRef key) {
  if (value > std::numeric_limits<float>::max() ||
      value < -std::numeric_limits<float>::max())
    throw JsonCodecError(key.str() + " is outside the supported numeric range");
  const float converted = static_cast<float>(value);
  if (!std::isfinite(converted) || (value != 0 && converted == 0))
    throw JsonCodecError(key.str() + " is outside the supported numeric range");
  return converted;
}

bool contains(std::initializer_list<llvm::StringRef> fields,
              llvm::StringRef key) {
  return std::find(fields.begin(), fields.end(), key) != fields.end();
}

void checkFields(const json::Object &obj,
                 std::initializer_list<llvm::StringRef> supported,
                 std::initializer_list<llvm::StringRef> unsupported = {}) {
  for (const auto &entry : obj)
    if (!contains(supported, entry.first) &&
        !contains(unsupported, entry.first))
      throw JsonCodecError("unknown parameter: " + entry.first.str());
  for (const auto &entry : obj)
    if (contains(unsupported, entry.first) &&
        entry.second.kind() != json::Value::Null)
      throw JsonCodecError("unsupported parameter: " + entry.first.str());
}

void checkGenerationFields(const json::Object &obj, bool chat) {
  static const std::initializer_list<llvm::StringRef> supported = {
      "prompt",
      "model",
      "image_path",
      "n_predict",
      "max_completion_tokens",
      "max_tokens",
      "temperature",
      "top_k",
      "top_p",
      "min_p",
      "repeat_penalty",
      "repeat_last_n",
      "seed",
      "stop",
      "stop_token_ids",
      "stream",
      "stream_options"};
  static const std::initializer_list<llvm::StringRef> unsupported = {
      "n",
      "best_of",
      "logprobs",
      "top_logprobs",
      "logit_bias",
      "presence_penalty",
      "frequency_penalty",
      "grammar",
      "json_schema",
      "response_format",
      "tools",
      "tool_choice",
      "parallel_tool_calls",
      "functions",
      "function_call",
      "user",
      "suffix",
      "cache_prompt",
      "id_slot",
      "samplers",
      "typical_p",
      "tfs_z",
      "mirostat",
      "mirostat_tau",
      "mirostat_eta",
      "xtc_probability",
      "xtc_threshold",
      "dynatemp_range",
      "dynatemp_exponent",
      "ignore_eos",
      "penalize_nl",
      "return_tokens",
      "prompt_cache_all",
      "min_keep",
      "n_probs"};
  for (const auto &entry : obj) {
    bool chatField =
        chat && (entry.first == "messages" || entry.first == "input");
    if (!chatField && !contains(supported, entry.first) &&
        !contains(unsupported, entry.first))
      throw JsonCodecError("unknown parameter: " + entry.first.str());
  }
  for (const auto &entry : obj)
    if (contains(unsupported, entry.first) &&
        entry.second.kind() != json::Value::Null)
      throw JsonCodecError("unsupported parameter: " + entry.first.str());
}

ImageInput parseImageUri(const std::string &raw, const std::string &field) {
  if (raw.empty())
    throw JsonCodecError("image URI is empty: " + field);
  if (raw.size() > 4096)
    throw JsonCodecError("image URI is too long: " + field);
  if (raw.find('\0') != std::string::npos)
    throw JsonCodecError("image URI contains NUL: " + field);

  std::string uri = raw;
  if (uri.rfind("file:", 0) == 0)
    uri = uri.substr(5);
  if (uri.empty())
    throw JsonCodecError("image file URI is empty: " + field);
  if (uri.rfind("http://", 0) == 0 || uri.rfind("https://", 0) == 0)
    throw JsonCodecError("remote image URLs are not supported: " + field);
  if (uri.rfind("data:", 0) == 0)
    throw JsonCodecError("data URI images are not supported: " + field);

  ImageInput image;
  image.uri = std::move(uri);
  return image;
}

ImageInput parseImageValue(const json::Value &value, const std::string &field) {
  std::string uri;
  if (auto s = value.getAsString()) {
    uri = s->str();
  } else if (const auto *obj = value.getAsObject()) {
    checkFields(*obj, {"url", "uri"}, {"detail"});
    const auto *url = optionalValue(*obj, "url");
    const auto *path = optionalValue(*obj, "uri");
    if (url && path)
      throw JsonCodecError("multiple image sources: " + field + ".url/uri");
    if (!url && !path)
      throw JsonCodecError("missing image source: " + field + ".url/uri");
    uri = checkedString(url ? *url : *path, field + (url ? ".url" : ".uri"));
  } else {
    throw JsonCodecError("image value must be a string or object: " + field);
  }
  return parseImageUri(uri, field);
}

void appendImage(std::vector<ImageInput> &images, ImageInput image,
                 const std::string &field) {
  (void)field;
  images.push_back(std::move(image));
}

void parseMessageContent(const json::Value &value, ChatMessage &message,
                         const std::string &field) {
  if (auto text = value.getAsString()) {
    message.content = text->str();
    return;
  }

  const auto *parts = value.getAsArray();
  if (!parts)
    throw JsonCodecError("message content must be a string or array: " + field);

  for (std::size_t i = 0; i < parts->size(); ++i) {
    const auto *part = (*parts)[i].getAsObject();
    if (!part)
      throw JsonCodecError("content entries must be objects: " + field);
    checkFields(*part, {"type", "text", "image_url", "image", "url"});
    const std::string type =
        checkedString(requiredValue(*part, "type"), field + ".type");
    if (type == "text") {
      checkFields(*part, {"type", "text"}, {"image_url", "image", "url"});
      const std::string text =
          checkedString(requiredValue(*part, "text"), field + ".text");
      message.content += text;
      continue;
    } else if (type == "image_url" || type == "image") {
      checkFields(*part, {"type", "image_url", "image", "url"}, {"text"});
      const json::Value *imageValue = nullptr;
      for (const char *key : {"image_url", "image", "url"})
        if (const auto *candidate = optionalValue(*part, key)) {
          if (imageValue)
            throw JsonCodecError("multiple image sources: " + field + "." +
                                 key);
          imageValue = candidate;
        }
      if (!imageValue)
        throw JsonCodecError("image content part requires image_url: " + field);
      appendImage(message.images,
                  parseImageValue(*imageValue, field + ".image_url"), field);
      continue;
    }
    throw JsonCodecError("unsupported content part type '" + type +
                         "': " + field);
  }
}

void fillSampling(const json::Object &obj, SamplingParams &params) {
  constexpr int64_t maximum = std::numeric_limits<int32_t>::max();
  auto &presence = params.explicitParameters;
  for (const char *key : {"n_predict", "max_completion_tokens", "max_tokens"})
    if (const auto *value = optionalValue(obj, key)) {
      params.maxNewTokens =
          static_cast<int>(checkedInteger(*value, key, -1, maximum));
      presence.maxNewTokens = true;
      break;
    }
  auto &sampler = params.samplerConfig;
  if (const auto *value = optionalValue(obj, "temperature")) {
    double number = checkedNumber(*value, "temperature");
    sampler.temperature = number <= 0 ? 0 : checkedFloat(number, "temperature");
    presence.temperature = true;
  }
  if (const auto *value = optionalValue(obj, "top_k")) {
    int64_t integer = checkedInteger(
        *value, "top_k", std::numeric_limits<int32_t>::min(), maximum);
    sampler.topK = static_cast<int>(std::max<int64_t>(0, integer));
    presence.topK = true;
  }
  auto fraction = [&](const char *key, float &target, bool positive,
                      bool &provided) {
    if (const auto *value = optionalValue(obj, key)) {
      double number = checkedNumber(*value, key);
      if (number < 0 || number > 1 || (positive && number == 0))
        throw JsonCodecError(std::string(key) + (positive
                                                     ? " must be in (0, 1]"
                                                     : " must be in [0, 1]"));
      target = checkedFloat(number, key);
      provided = true;
    }
  };
  fraction("top_p", sampler.topP, true, presence.topP);
  fraction("min_p", sampler.minP, false, presence.minP);
  if (const auto *value = optionalValue(obj, "repeat_penalty")) {
    double number = checkedNumber(*value, "repeat_penalty");
    if (number <= 0)
      throw JsonCodecError("repeat_penalty must be greater than zero");
    sampler.repeatPenalty = checkedFloat(number, "repeat_penalty");
    presence.repeatPenalty = true;
  }
  if (const auto *value = optionalValue(obj, "repeat_last_n")) {
    sampler.repeatLastN =
        static_cast<int>(checkedInteger(*value, "repeat_last_n", 0, maximum));
    presence.repeatLastN = true;
  }
  if (const auto *value = optionalValue(obj, "seed")) {
    int64_t seed = checkedInteger(*value, "seed", -1,
                                  std::numeric_limits<uint32_t>::max());
    presence.seed = true;
    if (seed != -1 && seed != std::numeric_limits<uint32_t>::max()) {
      params.seedPolicy = SamplingSeedPolicy::Fixed;
      params.fixedSeed = static_cast<uint32_t>(seed);
    }
  }
  if (const auto *value = optionalValue(obj, "stop_token_ids")) {
    const auto *array = value->getAsArray();
    if (!array)
      throw JsonCodecError("stop_token_ids must be an integer array");
    for (const auto &id : *array)
      params.stopTokenIds.push_back(
          checkedInteger(id, "stop_token_ids", 0, maximum));
    presence.stopTokenIds = true;
  }
  if (const auto *value = optionalValue(obj, "stop")) {
    if (auto stop = value->getAsString())
      params.stop.push_back(stop->str());
    else if (const auto *array = value->getAsArray()) {
      for (const auto &entry : *array)
        params.stop.push_back(checkedString(entry, "stop"));
    } else
      throw JsonCodecError("stop must be a string or string array");
    for (const auto &stop : params.stop)
      if (stop.empty())
        throw JsonCodecError("stop strings must not be empty");
    presence.stop = true;
  }
}

bool parseIncludeUsage(const json::Object &obj) {
  const auto *value = optionalValue(obj, "stream_options");
  if (!value)
    return false;
  const auto *options = value->getAsObject();
  if (!options)
    throw JsonCodecError("stream_options must be an object");
  for (const auto &entry : *options)
    if (entry.first != "include_usage")
      throw JsonCodecError("unknown parameter: stream_options." +
                           entry.first.str());
  try {
    return checkedBool(*options, "include_usage", false);
  } catch (const JsonCodecError &) {
    throw JsonCodecError("stream_options.include_usage must be a boolean");
  }
}

void requireCapability(bool supported, const std::string &field) {
  if (!supported)
    throw JsonCodecError(field + " is unsupported by this resident model");
}

void validateSampling(const SamplingParams &params,
                      const ResidentModelCapabilities &capabilities) {
  requireCapability(capabilities.maxNewTokens,
                    "n_predict/max_completion_tokens/max_tokens");
  const auto &provided = params.explicitParameters;
  const auto &sampling = capabilities.sampling;
  for (const auto &field :
       {std::make_pair(provided.temperature && !sampling.temperature,
                       "temperature"),
        std::make_pair(provided.topK && !sampling.topK, "top_k"),
        std::make_pair(provided.topP && !sampling.topP, "top_p"),
        std::make_pair(provided.minP && !sampling.minP, "min_p"),
        std::make_pair(provided.repeatPenalty && !sampling.repeatPenalty,
                       "repeat_penalty"),
        std::make_pair(provided.repeatLastN && !sampling.repeatLastN,
                       "repeat_last_n"),
        std::make_pair(provided.seed && !sampling.seed, "seed"),
        std::make_pair(provided.stop && !capabilities.stop, "stop"),
        std::make_pair(provided.stopTokenIds && !capabilities.stopTokenIds,
                       "stop_token_ids")})
    requireCapability(!field.first, field.second);
}

void validateImages(const std::vector<ImageInput> &images,
                    const ResidentModelCapabilities &capabilities) {
  requireCapability(images.size() <= capabilities.maxImages,
                    "image_path/images");
  for (const auto &image : images) {
    if (!image.uri.empty())
      requireCapability(capabilities.imagePaths, "image_path/images URI");
    if (!image.bytes.empty())
      requireCapability(capabilities.imageBytes, "images bytes");
  }
}

int64_t timestamp(int64_t created) {
  return created ? created
                 : std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
}

std::string validUtf8(const std::string &bytes) {
  Utf8StreamDecoder decoder;
  std::string result = decoder.append(bytes);
  result += decoder.finish();
  return result;
}

std::string finishReason(FinishReason reason) {
  switch (reason) {
  case FinishReason::None:
    return "";
  case FinishReason::Stop:
    return "stop";
  case FinishReason::Length:
    return "length";
  case FinishReason::Cancelled:
    return "cancelled";
  case FinishReason::Error:
    return "error";
  }
  return "stop";
}

std::string statusString(ModelLoadState state) {
  switch (state) {
  case ModelLoadState::Unloaded:
    return "unloaded";
  case ModelLoadState::Loading:
    return "loading";
  case ModelLoadState::Ready:
    return "ok";
  case ModelLoadState::Error:
    return "error";
  }
  return "unknown";
}

json::Object usageJson(const CompletionUsage &usage) {
  return json::Object{{"prompt_tokens", usage.promptTokens},
                      {"completion_tokens", usage.completionTokens},
                      {"total_tokens", usage.totalTokens}};
}

json::Object timingsJson(const CompletionTimings &timings) {
  return json::Object{{"prefill_ms", timings.prefillMs},
                      {"decode_ms", timings.decodeMs},
                      {"tokens_per_second", timings.tokensPerSecond}};
}

} // namespace

DecodedCompletionRequest parseCompletionRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  checkGenerationFields(obj, false);
  DecodedCompletionRequest decoded;
  decoded.request.prompt =
      checkedString(requiredValue(obj, "prompt"), "prompt");
  (void)optionalString(obj, "model");
  if (const auto *imagePath = optionalValue(obj, "image_path"))
    appendImage(
        decoded.request.images,
        parseImageUri(checkedString(*imagePath, "image_path"), "image_path"),
        "image_path");

  fillSampling(obj, decoded.request.sampling);
  decoded.stream = checkedBool(obj, "stream", false);
  decoded.includeUsage = parseIncludeUsage(obj);
  return decoded;
}

DecodedEmbeddingRequest parseEmbeddingRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  checkFields(obj, {"model", "input", "stream", "encoding_format"},
              {"dimensions", "user"});
  DecodedEmbeddingRequest decoded;
  decoded.request.model = optionalString(obj, "model");

  auto inputIt = obj.find("input");
  if (inputIt == obj.end())
    throw JsonCodecError("missing required field: input");
  if (checkedBool(obj, "stream", false))
    throw JsonCodecError("streaming embeddings are not supported");
  if (const auto *encoding = optionalValue(obj, "encoding_format"))
    if (checkedString(*encoding, "encoding_format") != "float")
      throw JsonCodecError("encoding_format supports only float");
  if (auto input = inputIt->second.getAsString()) {
    decoded.request.input = input->str();
    if (decoded.request.input.empty())
      throw JsonCodecError("input must not be empty");
  } else if (inputIt->second.getAsArray()) {
    throw JsonCodecError("input arrays are not supported; provide one string");
  } else {
    throw JsonCodecError("input must be a string");
  }
  return decoded;
}

DecodedMaskedLMRequest parseMaskedLMRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);
  DecodedMaskedLMRequest decoded;
  if (auto model = obj.getString("model"))
    decoded.request.model = model->str();
  else if (obj.find("model") != obj.end())
    throw JsonCodecError("model must be a string");
  const bool hasInput = obj.find("input") != obj.end();
  const bool hasPrompt = obj.find("prompt") != obj.end();
  if (hasInput && hasPrompt)
    throw JsonCodecError("provide only one of input or prompt");
  const char *key = hasInput ? "input" : "prompt";
  if (!hasInput && !hasPrompt)
    throw JsonCodecError("missing required field: input");
  auto it = obj.find(key);
  auto input = it->second.getAsString();
  if (!input)
    throw JsonCodecError("input must be a string");
  decoded.request.input = input->str();
  if (decoded.request.input.empty())
    throw JsonCodecError("input must not be empty");
  if (auto top = obj.getInteger("top_k")) {
    if (*top <= 0)
      throw JsonCodecError("top_k must be positive");
    decoded.request.topK = static_cast<std::size_t>(*top);
  } else if (obj.find("top_k") != obj.end()) {
    throw JsonCodecError("top_k must be a positive integer");
  }
  if (auto stream = obj.getBoolean("stream"); stream && *stream)
    throw JsonCodecError("streaming masked-LM is not supported");
  return decoded;
}

DecodedChatRequest parseChatCompletionRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  checkGenerationFields(obj, true);
  DecodedChatRequest decoded;
  decoded.request.model = optionalString(obj, "model");
  const std::string input = optionalString(obj, "input");
  const std::string prompt = optionalString(obj, "prompt");

  if (const auto *messageValue = obj.get("messages")) {
    const auto *messages = messageValue->getAsArray();
    if (!messages || messages->empty())
      throw JsonCodecError("messages must be a nonempty array");
    for (const auto &messageValue : *messages) {
      const auto *messageObj = messageValue.getAsObject();
      if (!messageObj)
        throw JsonCodecError("messages entries must be objects");
      checkFields(*messageObj, {"role", "content"},
                  {"name", "tool_call_id", "tool_calls", "refusal", "audio"});
      ChatMessage msg;
      msg.role =
          checkedString(requiredValue(*messageObj, "role"), "messages.role");
      if (msg.role.empty())
        throw JsonCodecError("chat message requires role");
      parseMessageContent(requiredValue(*messageObj, "content"), msg,
                          "messages[" +
                              std::to_string(decoded.request.messages.size()) +
                              "].content");
      for (const auto &image : msg.images)
        appendImage(decoded.request.images, image, "messages");
      decoded.request.messages.push_back(std::move(msg));
    }
  } else {
    if (!optionalValue(obj, "input") && !optionalValue(obj, "prompt"))
      throw JsonCodecError("missing required messages or input/prompt");
    decoded.request.input = optionalValue(obj, "input") ? input : prompt;
  }
  if (const auto *imagePath = optionalValue(obj, "image_path"))
    appendImage(
        decoded.request.images,
        parseImageUri(checkedString(*imagePath, "image_path"), "image_path"),
        "image_path");

  fillSampling(obj, decoded.request.sampling);
  decoded.stream = checkedBool(obj, "stream", false);
  decoded.includeUsage = parseIncludeUsage(obj);
  return decoded;
}

TokenizeRequest parseTokenizeRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  checkFields(obj, {"content", "text", "add_special", "parse_special",
                    "with_pieces", "count_only"});
  TokenizeRequest request;
  const char *key = obj.get("content") ? "content" : "text";
  request.content = checkedString(requiredValue(obj, key), key);
  if (obj.get("content"))
    if (const auto *text = optionalValue(obj, "text"))
      (void)checkedString(*text, "text");
  request.addSpecial = checkedBool(obj, "add_special", false);
  request.parseSpecial = checkedBool(obj, "parse_special", true);
  request.withPieces = checkedBool(obj, "with_pieces", false);
  request.countOnly = checkedBool(obj, "count_only", false);
  return request;
}

DetokenizeRequest parseDetokenizeRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);
  checkFields(obj, {"tokens"});
  const auto *tokens = requiredValue(obj, "tokens").getAsArray();
  if (!tokens)
    throw JsonCodecError("tokens must be an integer array");
  DetokenizeRequest request;
  for (const auto &token : *tokens)
    request.tokens.push_back(static_cast<int>(checkedInteger(
        token, "tokens", 0, std::numeric_limits<int32_t>::max())));
  return request;
}

void validateCapabilities(const DecodedCompletionRequest &decoded,
                          const ResidentModelCapabilities &capabilities) {
  validateSampling(decoded.request.sampling, capabilities);
  if (decoded.stream)
    requireCapability(capabilities.streaming, "stream");
  validateImages(decoded.request.images, capabilities);
}

void validateCapabilities(const DecodedChatRequest &decoded,
                          const ResidentModelCapabilities &capabilities) {
  validateSampling(decoded.request.sampling, capabilities);
  if (decoded.stream)
    requireCapability(capabilities.streaming, "stream");
  validateImages(decoded.request.images, capabilities);
  std::size_t conversationalMessages = 0;
  for (const auto &message : decoded.request.messages) {
    requireCapability(std::find(capabilities.chatRoles.begin(),
                                capabilities.chatRoles.end(),
                                message.role) != capabilities.chatRoles.end(),
                      "messages.role '" + message.role + "'");
    if (message.role != "system")
      ++conversationalMessages;
  }
  if (decoded.request.messages.empty())
    requireCapability(std::find(capabilities.chatRoles.begin(),
                                capabilities.chatRoles.end(),
                                "user") != capabilities.chatRoles.end(),
                      "input/user role");
  if (conversationalMessages > 1)
    requireCapability(capabilities.chatMultiTurn, "messages/multi-turn chat");
}

void validateCapabilities(const TokenizeRequest &request,
                          const ResidentModelCapabilities &capabilities) {
  requireCapability(capabilities.rawTokenize, "tokenize/raw content");
  if (request.addSpecial)
    requireCapability(capabilities.tokenizeAddSpecial, "add_special");
  if (!request.parseSpecial)
    requireCapability(capabilities.tokenizeParseSpecial, "parse_special");
  if (request.withPieces)
    requireCapability(capabilities.tokenizeWithPieces, "with_pieces");
}

void validateCapabilities(const DetokenizeRequest &,
                          const ResidentModelCapabilities &capabilities) {
  requireCapability(capabilities.rawDetokenize, "detokenize/tokens");
}

std::string toJson(const ModelStatus &status) {
  return serialize(
      json::Object{{"status", statusString(status.state)},
                   {"model_loaded", status.state == ModelLoadState::Ready},
                   {"model", status.modelName},
                   {"backend", status.backend},
                   {"context_length", status.contextLength},
                   {"message", status.message}});
}

std::string toJson(const CompletionResult &result) {
  return serialize(
      json::Object{{"id", result.id},
                   {"object", "text_completion"},
                   {"model", result.model},
                   {"content", result.content},
                   {"stop", true},
                   {"stop_reason", finishReason(result.finishReason)},
                   {"usage", usageJson(result.usage)},
                   {"timings", timingsJson(result.timings)}});
}

std::string toOpenAIEmbeddingJson(const EmbeddingResult &result) {
  json::Array embedding;
  embedding.reserve(result.embedding.size());
  for (float value : result.embedding)
    embedding.emplace_back(value);

  json::Array data;
  data.emplace_back(json::Object{{"object", "embedding"},
                                 {"embedding", std::move(embedding)},
                                 {"index", 0}});
  return serialize(json::Object{
      {"object", "list"},
      {"data", std::move(data)},
      {"model", result.model},
      {"usage",
       json::Object{
           {"prompt_tokens", static_cast<int64_t>(result.promptTokens)},
           {"total_tokens", static_cast<int64_t>(result.totalTokens)}}}});
}

std::string toJson(const MaskedLMResult &result) {
  json::Array predictions;
  for (const auto &prediction : result.predictions) {
    json::Array tokens;
    for (const auto &token : prediction.tokens)
      tokens.emplace_back(json::Object{{"token_id", token.tokenId},
                                       {"token", token.token},
                                       {"score", token.score}});
    predictions.emplace_back(
        json::Object{{"position", static_cast<int64_t>(prediction.position)},
                     {"tokens", std::move(tokens)}});
  }
  return serialize(json::Object{
      {"object", "masked_lm"},
      {"model", result.model},
      {"sequence_length", static_cast<int64_t>(result.sequenceLength)},
      {"predictions", std::move(predictions)},
      {"usage", json::Object{{"prompt_tokens",
                              static_cast<int64_t>(result.promptTokens)}}}});
}

std::string toOpenAIChatJson(const CompletionResult &result, int64_t created) {
  json::Array choices;
  choices.emplace_back(
      json::Object{{"index", 0},
                   {"message", json::Object{{"role", "assistant"},
                                            {"content", result.content}}},
                   {"finish_reason", finishReason(result.finishReason)}});

  return serialize(json::Object{{"id", result.id},
                                {"object", "chat.completion"},
                                {"created", timestamp(created)},
                                {"model", result.model},
                                {"choices", std::move(choices)},
                                {"usage", usageJson(result.usage)}});
}

std::string toJson(const TokenizeResult &result, bool countOnly,
                   bool withPieces) {
  json::Object obj{{"count", static_cast<int64_t>(result.count)}};
  if (!countOnly) {
    json::Array tokens;
    if (withPieces && result.pieces.size() != result.tokens.size())
      throw std::runtime_error(
          "tokenize returned inconsistent with_pieces data");
    for (std::size_t i = 0; i < result.tokens.size(); ++i)
      if (withPieces)
        tokens.emplace_back(json::Object{
            {"id", result.tokens[i]}, {"piece", validUtf8(result.pieces[i])}});
      else
        tokens.emplace_back(result.tokens[i]);
    obj["tokens"] = std::move(tokens);
  }
  return serialize(std::move(obj));
}

std::string toJson(const DetokenizeResult &result) {
  return serialize(json::Object{{"content", result.content}});
}

std::string toModelListJson(const ModelStatus &status, int64_t created) {
  json::Array models;
  models.emplace_back(json::Object{{"id", status.modelName},
                                   {"object", "model"},
                                   {"created", timestamp(created)},
                                   {"owned_by", "buddy"}});
  return serialize(
      json::Object{{"object", "list"}, {"data", std::move(models)}});
}

std::string toCompletionChunkJson(const CompletionChunk &chunk) {
  json::Object obj{{"id", chunk.id},
                   {"object", "text_completion.chunk"},
                   {"model", chunk.model},
                   {"content", chunk.delta},
                   {"stop", chunk.done}};
  if (chunk.tokenId >= 0)
    obj["token_id"] = chunk.tokenId;
  if (chunk.done) {
    obj["stop_reason"] = finishReason(chunk.finishReason);
    obj["usage"] = usageJson(chunk.usage);
    obj["timings"] = timingsJson(chunk.timings);
  }
  return serialize(std::move(obj));
}

std::string toOpenAIChatChunkJson(const CompletionChunk &chunk,
                                  int64_t created) {
  json::Object delta;
  if (!chunk.delta.empty())
    delta["content"] = chunk.delta;

  json::Array choices;
  choices.emplace_back(
      json::Object{{"index", 0},
                   {"delta", std::move(delta)},
                   {"finish_reason",
                    chunk.done ? json::Value(finishReason(chunk.finishReason))
                               : json::Value(nullptr)}});

  json::Object obj{{"id", chunk.id},
                   {"object", "chat.completion.chunk"},
                   {"created", timestamp(created)},
                   {"model", chunk.model},
                   {"choices", std::move(choices)}};
  return serialize(std::move(obj));
}

std::string toOpenAIChatRoleChunkJson(const std::string &id,
                                      const std::string &model,
                                      int64_t created) {
  json::Array choices;
  choices.emplace_back(
      json::Object{{"index", 0},
                   {"delta", json::Object{{"role", "assistant"}}},
                   {"finish_reason", nullptr}});
  return serialize(json::Object{{"id", id},
                                {"object", "chat.completion.chunk"},
                                {"created", timestamp(created)},
                                {"model", model},
                                {"choices", std::move(choices)}});
}

std::string toOpenAIChatUsageChunkJson(const CompletionResult &result,
                                       int64_t created) {
  return serialize(json::Object{{"id", result.id},
                                {"object", "chat.completion.chunk"},
                                {"created", timestamp(created)},
                                {"model", result.model},
                                {"choices", json::Array{}},
                                {"usage", usageJson(result.usage)}});
}

std::string errorJson(const std::string &message, const std::string &type,
                      int code) {
  return serialize(json::Object{
      {"error",
       json::Object{{"message", message}, {"type", type}, {"code", code}}}});
}

} // namespace server
} // namespace buddy
