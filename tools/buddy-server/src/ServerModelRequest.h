//===- ServerModelRequest.h - Server ABI request views --------------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELREQUEST_H
#define BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELREQUEST_H

#include "buddy/runtime/core/ServerModelABI.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace buddy {
namespace server {

class ServerModelRequest {
public:
  const BuddyServerRequestV1 &request() const { return value; }
  void setRequestId(uint64_t requestId) { value.requestId = requestId; }

  void setCompletion(const buddy::runtime::CompletionRequest &source,
                     bool stream) {
    reset(BUDDY_SERVER_OPERATION_COMPLETION);
    value.completion.prompt = view(source.prompt);
    value.completion.sampling = sampling(source.sampling);
    imageViews = makeImages(source.images);
    value.completion.images = imageViews.data();
    value.completion.imageCount = imageViews.size();
    value.completion.stream = stream ? 1u : 0u;
  }

  void setChat(const buddy::runtime::ChatCompletionRequest &source,
               bool stream) {
    reset(BUDDY_SERVER_OPERATION_CHAT);
    value.chat.model = view(source.model);
    value.chat.input = view(source.input);
    value.chat.sampling = sampling(source.sampling);
    imageViews = makeImages(source.images);
    value.chat.images = imageViews.data();
    value.chat.imageCount = imageViews.size();
    chatImages.reserve(source.messages.size());
    chatViews.reserve(source.messages.size());
    for (const auto &message : source.messages) {
      chatImages.push_back({});
      auto &messageImages = chatImages.back();
      messageImages = makeImages(message.images);
      BuddyServerChatMessageV1 valueMessage{};
      valueMessage.role = view(message.role);
      valueMessage.content = view(message.content);
      valueMessage.images = messageImages.data();
      valueMessage.imageCount = messageImages.size();
      chatViews.push_back(valueMessage);
    }
    value.chat.messages = chatViews.data();
    value.chat.messageCount = chatViews.size();
    value.chat.stream = stream ? 1u : 0u;
  }

  void setTokenize(const buddy::runtime::TokenizeRequest &source) {
    reset(BUDDY_SERVER_OPERATION_TOKENIZE);
    value.tokenize.content = view(source.content);
    value.tokenize.addSpecial = source.addSpecial ? 1u : 0u;
    value.tokenize.countOnly = source.countOnly ? 1u : 0u;
  }

  void setEmbedding(const buddy::runtime::EmbeddingRequest &source) {
    reset(BUDDY_SERVER_OPERATION_EMBEDDING);
    value.embedding.model = view(source.model);
    value.embedding.input = view(source.input);
  }

  void setMaskedLM(const buddy::runtime::MaskedLMRequest &source) {
    reset(BUDDY_SERVER_OPERATION_MASKED_LM);
    value.maskedLM.model = view(source.model);
    value.maskedLM.input = view(source.input);
    value.maskedLM.topK = source.topK;
  }

  void
  setTranscription(const buddy::runtime::AudioTranscriptionRequest &source) {
    reset(BUDDY_SERVER_OPERATION_TRANSCRIPTION);
    value.transcription.model = view(source.model);
    value.transcription.audio.uri = view(source.audio.uri);
    value.transcription.audio.mimeType = view(source.audio.mimeType);
    value.transcription.audio.bytes = {
        source.audio.bytes.data(),
        static_cast<uint64_t>(source.audio.bytes.size())};
    value.transcription.maxTokens = source.maxTokens;
  }

private:
  static BuddyServerStringViewV1 view(const std::string &source) {
    return {source.data(), static_cast<uint64_t>(source.size())};
  }

  BuddyServerSamplingV1 sampling(const buddy::runtime::SamplingParams &source) {
    BuddyServerSamplingV1 result{};
    result.maxTokens = source.maxTokens;
    result.temperature = source.samplerConfig.temperature;
    result.topK = source.samplerConfig.topK;
    result.topP = source.samplerConfig.topP;
    result.minP = source.samplerConfig.minP;
    result.repeatPenalty = source.samplerConfig.repeatPenalty;
    result.repeatLastN = source.samplerConfig.repeatLastN;
    result.seed = source.samplerConfig.seed;
    stopTokenIds.assign(source.stopTokenIds.begin(), source.stopTokenIds.end());
    result.stopTokenIds = stopTokenIds.data();
    result.stopTokenIdCount = stopTokenIds.size();
    stopViews.reserve(source.stop.size());
    for (const auto &stop : source.stop)
      stopViews.push_back(view(stop));
    result.stop = stopViews.data();
    result.stopCount = stopViews.size();
    return result;
  }

  static std::vector<BuddyServerImageInputV1>
  makeImages(const std::vector<buddy::runtime::ImageInput> &source) {
    std::vector<BuddyServerImageInputV1> result;
    result.reserve(source.size());
    for (const auto &image : source) {
      BuddyServerImageInputV1 valueImage{};
      valueImage.uri = view(image.uri);
      valueImage.mimeType = view(image.mimeType);
      valueImage.bytes = {image.bytes.data(),
                          static_cast<uint64_t>(image.bytes.size())};
      result.push_back(valueImage);
    }
    return result;
  }

  void reset(uint32_t operation) {
    value = {};
    value.structSize = sizeof(value);
    value.operation = operation;
    stopViews.clear();
    stopTokenIds.clear();
    imageViews.clear();
    chatImages.clear();
    chatViews.clear();
  }

  BuddyServerRequestV1 value{};
  std::vector<BuddyServerStringViewV1> stopViews;
  std::vector<int64_t> stopTokenIds;
  std::vector<BuddyServerImageInputV1> imageViews;
  std::vector<std::vector<BuddyServerImageInputV1>> chatImages;
  std::vector<BuddyServerChatMessageV1> chatViews;
};

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_SERVERMODELREQUEST_H
