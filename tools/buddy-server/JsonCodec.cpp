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

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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

std::string getString(const json::Object &obj, llvm::StringRef key,
                      const std::string &fallback = "") {
  if (auto value = obj.getString(key))
    return value->str();
  return fallback;
}

bool getBool(const json::Object &obj, llvm::StringRef key, bool fallback) {
  if (auto value = obj.getBoolean(key))
    return *value;
  return fallback;
}

int getInt(const json::Object &obj, llvm::StringRef key, int fallback) {
  if (auto value = obj.getInteger(key))
    return static_cast<int>(*value);
  return fallback;
}

uint64_t getUInt64(const json::Object &obj, llvm::StringRef key,
                   uint64_t fallback) {
  if (auto value = obj.getInteger(key))
    return static_cast<uint64_t>(*value);
  return fallback;
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
    uri = getString(*obj, "url", getString(*obj, "uri"));
  } else {
    throw JsonCodecError("image value must be a string or object: " + field);
  }
  return parseImageUri(uri, field);
}

void appendImage(std::vector<ImageInput> &images, ImageInput image,
                 const std::string &field) {
  if (!images.empty())
    throw JsonCodecError("only one image is supported: " + field);
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
    const std::string type = getString(*part, "type");
    if (type == "text") {
      const std::string text = getString(*part, "text");
      if (text.empty())
        throw JsonCodecError("text content part is empty: " + field);
      if (!message.content.empty())
        message.content += "\n";
      message.content += text;
      continue;
    } else if (type == "image_url" || type == "image") {
      const json::Value *imageValue = nullptr;
      if (auto it = part->find("image_url"); it != part->end())
        imageValue = &it->second;
      else if (auto it = part->find("image"); it != part->end())
        imageValue = &it->second;
      else if (auto it = part->find("url"); it != part->end())
        imageValue = &it->second;
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

float getFloat(const json::Object &obj, llvm::StringRef key, float fallback) {
  if (auto value = obj.getNumber(key))
    return static_cast<float>(*value);
  return fallback;
}

void fillSampling(const json::Object &obj, SamplingParams &params) {
  params.maxTokens = getInt(obj, "max_tokens", params.maxTokens);
  params.samplerConfig.temperature =
      getFloat(obj, "temperature", params.samplerConfig.temperature);
  params.samplerConfig.topK = getInt(obj, "top_k", params.samplerConfig.topK);
  params.samplerConfig.topP = getFloat(obj, "top_p", params.samplerConfig.topP);
  params.samplerConfig.minP = getFloat(obj, "min_p", params.samplerConfig.minP);
  params.samplerConfig.repeatPenalty =
      getFloat(obj, "repeat_penalty", params.samplerConfig.repeatPenalty);
  params.samplerConfig.repeatLastN =
      getInt(obj, "repeat_last_n", params.samplerConfig.repeatLastN);
  params.samplerConfig.seed = getUInt64(obj, "seed", params.samplerConfig.seed);

  if (const auto *arr = obj.getArray("stop_token_ids")) {
    for (const auto &value : *arr) {
      if (auto id = value.getAsInteger())
        params.stopTokenIds.push_back(static_cast<long long>(*id));
    }
  }
  if (const auto *arr = obj.getArray("stop")) {
    for (const auto &value : *arr) {
      if (auto stop = value.getAsString())
        params.stop.push_back(stop->str());
    }
  }
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

  DecodedCompletionRequest decoded;
  decoded.request.prompt = getString(obj, "prompt");
  if (decoded.request.prompt.empty())
    throw JsonCodecError("missing required field: prompt");
  if (auto imagePath = obj.getString("image_path"))
    appendImage(decoded.request.images,
                parseImageUri(imagePath->str(), "image_path"), "image_path");

  fillSampling(obj, decoded.request.sampling);
  decoded.stream = getBool(obj, "stream", false);
  return decoded;
}

DecodedEmbeddingRequest parseEmbeddingRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  DecodedEmbeddingRequest decoded;
  if (auto model = obj.getString("model"))
    decoded.request.model = model->str();
  else if (obj.find("model") != obj.end())
    throw JsonCodecError("model must be a string");

  auto inputIt = obj.find("input");
  if (inputIt == obj.end())
    throw JsonCodecError("missing required field: input");
  if (auto stream = obj.getBoolean("stream"); stream && *stream)
    throw JsonCodecError("streaming embeddings are not supported");
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

DecodedChatRequest parseChatCompletionRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  DecodedChatRequest decoded;
  decoded.request.model = getString(obj, "model");
  decoded.request.input = getString(obj, "input", getString(obj, "prompt"));

  if (const auto *messages = obj.getArray("messages")) {
    for (const auto &messageValue : *messages) {
      const auto *messageObj = messageValue.getAsObject();
      if (!messageObj)
        throw JsonCodecError("messages entries must be objects");
      ChatMessage msg;
      msg.role = getString(*messageObj, "role");
      if (msg.role.empty())
        throw JsonCodecError("chat message requires role");
      auto contentIt = messageObj->find("content");
      if (contentIt != messageObj->end())
        parseMessageContent(
            contentIt->second, msg,
            "messages[" + std::to_string(decoded.request.messages.size()) +
                "].content");
      if (msg.content.empty() && msg.images.empty())
        throw JsonCodecError("chat message requires content or image");
      for (const auto &image : msg.images)
        appendImage(decoded.request.images, image, "messages");
      decoded.request.messages.push_back(std::move(msg));
    }
  }
  if (auto imagePath = obj.getString("image_path"))
    appendImage(decoded.request.images,
                parseImageUri(imagePath->str(), "image_path"), "image_path");

  if (decoded.request.messages.empty() && decoded.request.input.empty() &&
      decoded.request.images.empty())
    throw JsonCodecError("missing messages or input");

  fillSampling(obj, decoded.request.sampling);
  decoded.stream = getBool(obj, "stream", false);
  return decoded;
}

TokenizeRequest parseTokenizeRequest(const std::string &body) {
  json::Value value = parseValue(body);
  const json::Object &obj = asObject(value);

  TokenizeRequest request;
  request.content = getString(obj, "content", getString(obj, "text"));
  if (request.content.empty())
    throw JsonCodecError("missing required field: content");
  request.addSpecial = getBool(obj, "add_special", request.addSpecial);
  request.countOnly = getBool(obj, "count_only", request.countOnly);
  return request;
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
                   {"stop", result.finishReason == FinishReason::Stop},
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

std::string toOpenAIChatJson(const CompletionResult &result) {
  json::Array choices;
  choices.emplace_back(
      json::Object{{"index", 0},
                   {"message", json::Object{{"role", "assistant"},
                                            {"content", result.content}}},
                   {"finish_reason", finishReason(result.finishReason)}});

  return serialize(json::Object{{"id", result.id},
                                {"object", "chat.completion"},
                                {"model", result.model},
                                {"choices", std::move(choices)},
                                {"usage", usageJson(result.usage)}});
}

std::string toJson(const TokenizeResult &result, bool countOnly) {
  json::Object obj{{"count", static_cast<int64_t>(result.count)}};
  if (!countOnly) {
    json::Array tokens;
    for (int token : result.tokens)
      tokens.emplace_back(token);
    obj["tokens"] = std::move(tokens);
  }
  return serialize(std::move(obj));
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

std::string toOpenAIChatChunkJson(const CompletionChunk &chunk) {
  json::Object delta;
  if (!chunk.done) {
    if (!chunk.delta.empty())
      delta["content"] = chunk.delta;
  }

  json::Array choices;
  choices.emplace_back(json::Object{
      {"index", 0},
      {"delta", std::move(delta)},
      {"finish_reason", chunk.done ? finishReason(chunk.finishReason) : ""}});

  json::Object obj{{"id", chunk.id},
                   {"object", "chat.completion.chunk"},
                   {"model", chunk.model},
                   {"choices", std::move(choices)}};
  if (chunk.done)
    obj["usage"] = usageJson(chunk.usage);
  return serialize(std::move(obj));
}

std::string errorJson(const std::string &message, const std::string &type,
                      int code) {
  return serialize(json::Object{
      {"error",
       json::Object{{"message", message}, {"type", type}, {"code", code}}}});
}

} // namespace server
} // namespace buddy
