//===- AudioTranscriptionJsonCodec.cpp - Transcription JSON adapter ------===//
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

#include <string>

namespace buddy {
namespace server {
namespace {

namespace json = llvm::json;

std::string serializeAudioJson(json::Value value) {
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << value;
  return output;
}

bool hasPrefix(const std::string &value, const char *prefix) {
  return value.rfind(prefix, 0) == 0;
}

} // namespace

DecodedAudioTranscriptionRequest
parseAudioTranscriptionRequest(const std::string &body) {
  auto parsed = json::parse(body);
  if (!parsed)
    throw JsonCodecError("invalid JSON: " + llvm::toString(parsed.takeError()));
  const json::Object *object = parsed->getAsObject();
  if (!object)
    throw JsonCodecError("JSON root must be an object");

  DecodedAudioTranscriptionRequest decoded;
  if (auto model = object->getString("model"))
    decoded.request.model = model->str();
  else if (object->find("model") != object->end())
    throw JsonCodecError("model must be a string");

  const auto file = object->find("file");
  const auto alias = object->find("audio_path");
  if (file != object->end() && alias != object->end())
    throw JsonCodecError("provide only one of file or audio_path");
  if (file == object->end() && alias == object->end())
    throw JsonCodecError("missing required field: file");

  const json::Value &pathValue =
      file != object->end() ? file->second : alias->second;
  auto pathString = pathValue.getAsString();
  if (!pathString)
    throw JsonCodecError("file must be a string");
  std::string path = pathString->str();
  if (path.empty())
    throw JsonCodecError("file must not be empty");
  if (path.size() > 4096)
    throw JsonCodecError("file path is too long");
  if (path.find('\0') != std::string::npos)
    throw JsonCodecError("file path contains NUL");
  if (hasPrefix(path, "http://") || hasPrefix(path, "https://"))
    throw JsonCodecError("remote audio URLs are not supported");
  if (hasPrefix(path, "data:"))
    throw JsonCodecError("data URI audio is not supported");
  if (hasPrefix(path, "file:"))
    path.erase(0, 5);
  if (path.empty())
    throw JsonCodecError("file URI path must not be empty");
  decoded.request.audio.uri = std::move(path);
  decoded.request.audio.mimeType = "audio/wav";

  if (auto maxTokens = object->getInteger("max_tokens")) {
    if (*maxTokens <= 0 || *maxTokens > 447)
      throw JsonCodecError("max_tokens must be between 1 and 447");
    decoded.request.maxTokens = static_cast<int>(*maxTokens);
  } else if (object->find("max_tokens") != object->end()) {
    throw JsonCodecError("max_tokens must be a positive integer");
  } else {
    decoded.request.maxTokens = 64;
  }

  if (auto stream = object->getBoolean("stream"); stream && *stream)
    throw JsonCodecError("streaming audio transcription is not supported");
  if (object->find("audio_bytes") != object->end() ||
      object->find("audio_base64") != object->end())
    throw JsonCodecError("inline audio bytes are not supported");
  return decoded;
}

std::string toOpenAIAudioTranscriptionJson(
    const buddy::runtime::AudioTranscriptionResult &result) {
  return serializeAudioJson(json::Object{
      {"text", result.text},
      {"model", result.model},
      {"generated_tokens", static_cast<std::int64_t>(result.generatedTokens)},
      {"timings", json::Object{{"preprocess_ms", result.timings.preprocessMs},
                               {"inference_ms", result.timings.inferenceMs},
                               {"total_ms", result.timings.totalMs}}}});
}

} // namespace server
} // namespace buddy
