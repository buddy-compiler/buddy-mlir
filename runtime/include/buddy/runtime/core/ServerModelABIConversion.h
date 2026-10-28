//===- ServerModelABIConversion.h - Server ABI C++ conversions ------------===//
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

#ifndef BUDDY_RUNTIME_CORE_SERVERMODELABICONVERSION_H
#define BUDDY_RUNTIME_CORE_SERVERMODELABICONVERSION_H

#include "buddy/runtime/core/AudioTranscriptionTypes.h"
#include "buddy/runtime/core/EmbeddingTypes.h"
#include "buddy/runtime/core/MaskedLMTypes.h"
#include "buddy/runtime/core/ServerModelABI.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {
namespace server_abi {

inline std::string copy(BuddyServerStringViewV1 value) {
  if (!value.data || value.size == 0)
    return {};
  return std::string(value.data, static_cast<size_t>(value.size));
}

inline BuddyServerStringViewV1 view(const std::string &value) {
  return {value.data(), static_cast<uint64_t>(value.size())};
}

inline ImageInput image(const BuddyServerImageInputV1 &value) {
  ImageInput result;
  result.uri = copy(value.uri);
  result.mimeType = copy(value.mimeType);
  if (value.bytes.data && value.bytes.size)
    result.bytes.assign(value.bytes.data, value.bytes.data + value.bytes.size);
  return result;
}

inline std::vector<ImageInput> images(const BuddyServerImageInputV1 *values,
                                      uint64_t count) {
  std::vector<ImageInput> result;
  if (!values && count != 0)
    throw std::invalid_argument("server ABI image array is null");
  result.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i)
    result.push_back(image(values[i]));
  return result;
}

inline SamplingParams sampling(const BuddyServerSamplingV1 &value) {
  SamplingParams result;
  result.maxTokens = value.maxTokens;
  result.samplerConfig.temperature = value.temperature;
  result.samplerConfig.topK = value.topK;
  result.samplerConfig.topP = value.topP;
  result.samplerConfig.minP = value.minP;
  result.samplerConfig.repeatPenalty = value.repeatPenalty;
  result.samplerConfig.repeatLastN = value.repeatLastN;
  result.samplerConfig.seed = value.seed;
  if (value.stopTokenIds && value.stopTokenIdCount)
    result.stopTokenIds.assign(value.stopTokenIds,
                               value.stopTokenIds + value.stopTokenIdCount);
  if (value.stop && value.stopCount) {
    result.stop.reserve(static_cast<size_t>(value.stopCount));
    for (uint64_t i = 0; i < value.stopCount; ++i)
      result.stop.push_back(copy(value.stop[i]));
  }
  return result;
}

inline CompletionRequest completion(const BuddyServerCompletionRequestV1 &v) {
  CompletionRequest result;
  result.prompt = copy(v.prompt);
  result.sampling = sampling(v.sampling);
  result.images = images(v.images, v.imageCount);
  return result;
}

inline ChatMessage chatMessage(const BuddyServerChatMessageV1 &v) {
  ChatMessage result;
  result.role = copy(v.role);
  result.content = copy(v.content);
  result.images = images(v.images, v.imageCount);
  return result;
}

inline ChatCompletionRequest chat(const BuddyServerChatRequestV1 &v) {
  ChatCompletionRequest result;
  result.model = copy(v.model);
  result.input = copy(v.input);
  result.sampling = sampling(v.sampling);
  result.images = images(v.images, v.imageCount);
  if (!v.messages && v.messageCount != 0)
    throw std::invalid_argument("server ABI message array is null");
  result.messages.reserve(static_cast<size_t>(v.messageCount));
  for (uint64_t i = 0; i < v.messageCount; ++i)
    result.messages.push_back(chatMessage(v.messages[i]));
  return result;
}

inline TokenizeRequest tokenize(const BuddyServerTokenizeRequestV1 &v) {
  TokenizeRequest result;
  result.content = copy(v.content);
  result.addSpecial = v.addSpecial != 0;
  result.countOnly = v.countOnly != 0;
  return result;
}

inline EmbeddingRequest embedding(const BuddyServerEmbeddingRequestV1 &v) {
  return {copy(v.model), copy(v.input)};
}

inline MaskedLMRequest maskedLM(const BuddyServerMaskedLMRequestV1 &v) {
  return {copy(v.model), copy(v.input), static_cast<size_t>(v.topK)};
}

inline AudioInput audio(const BuddyServerAudioInputV1 &v) {
  AudioInput result;
  result.uri = copy(v.uri);
  result.mimeType = copy(v.mimeType);
  if (v.bytes.data && v.bytes.size)
    result.bytes.assign(v.bytes.data, v.bytes.data + v.bytes.size);
  return result;
}

inline AudioTranscriptionRequest
transcription(const BuddyServerTranscriptionRequestV1 &v) {
  AudioTranscriptionRequest result;
  result.model = copy(v.model);
  result.audio = audio(v.audio);
  result.maxTokens = v.maxTokens;
  return result;
}

inline BuddyServerUsageV1 usage(const CompletionUsage &value) {
  return {value.promptTokens, value.completionTokens, value.totalTokens};
}

inline BuddyServerTimingsV1 timings(const CompletionTimings &value) {
  BuddyServerTimingsV1 result{};
  result.prefillMs = value.prefillMs;
  result.decodeMs = value.decodeMs;
  result.tokensPerSecond = value.tokensPerSecond;
  return result;
}

inline BuddyServerTimingsV1 timings(const AudioTranscriptionTimings &value) {
  BuddyServerTimingsV1 result{};
  result.preprocessMs = value.preprocessMs;
  result.inferenceMs = value.inferenceMs;
  result.totalMs = value.totalMs;
  return result;
}

inline BuddyServerStringViewV1 viewOrEmpty(const std::string *value) {
  return value ? view(*value) : BuddyServerStringViewV1{nullptr, 0};
}

} // namespace server_abi
} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_SERVERMODELABICONVERSION_H
