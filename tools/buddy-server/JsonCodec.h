//===- JsonCodec.h - buddy-server JSON conversion helpers -----------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_JSONCODEC_H
#define BUDDY_TOOLS_BUDDY_SERVER_JSONCODEC_H

#include "buddy/runtime/core/EmbeddingTypes.h"
#include "buddy/runtime/core/MaskedLMTypes.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <stdexcept>
#include <string>

namespace buddy {
namespace server {

class JsonCodecError : public std::runtime_error {
public:
  explicit JsonCodecError(const std::string &message)
      : std::runtime_error(message) {}
};

struct DecodedCompletionRequest {
  buddy::runtime::CompletionRequest request;
  bool stream = false;
};

struct DecodedChatRequest {
  buddy::runtime::ChatCompletionRequest request;
  bool stream = false;
};

struct DecodedEmbeddingRequest {
  buddy::runtime::EmbeddingRequest request;
};
struct DecodedMaskedLMRequest {
  buddy::runtime::MaskedLMRequest request;
};

DecodedCompletionRequest parseCompletionRequest(const std::string &body);
DecodedChatRequest parseChatCompletionRequest(const std::string &body);
DecodedEmbeddingRequest parseEmbeddingRequest(const std::string &body);
DecodedMaskedLMRequest parseMaskedLMRequest(const std::string &body);
buddy::runtime::TokenizeRequest parseTokenizeRequest(const std::string &body);

std::string toJson(const buddy::runtime::ModelStatus &status);
std::string toJson(const buddy::runtime::CompletionResult &result);
std::string
toOpenAIEmbeddingJson(const buddy::runtime::EmbeddingResult &result);
std::string toJson(const buddy::runtime::MaskedLMResult &result);
std::string toOpenAIChatJson(const buddy::runtime::CompletionResult &result);
std::string toJson(const buddy::runtime::TokenizeResult &result,
                   bool countOnly);

std::string toCompletionChunkJson(const buddy::runtime::CompletionChunk &chunk);
std::string toOpenAIChatChunkJson(const buddy::runtime::CompletionChunk &chunk);

std::string errorJson(const std::string &message, const std::string &type,
                      int code);

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_JSONCODEC_H
