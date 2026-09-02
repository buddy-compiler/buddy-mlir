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

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using buddy::server::parseChatCompletionRequest;
using buddy::server::parseCompletionRequest;
using buddy::server::parseEmbeddingRequest;
using buddy::server::toOpenAIEmbeddingJson;

namespace {

template <typename Fn> void expectFailure(Fn &&fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception &) {
    failed = true;
  }
  assert(failed);
}

} // namespace

int main() {
  auto completion = parseCompletionRequest(
      R"({"prompt":"read","image_path":"file:/tmp/a.png"})");
  assert(completion.request.prompt == "read");
  assert(completion.request.images.size() == 1);
  assert(completion.request.images.front().uri == "/tmp/a.png");

  auto textChat = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"hello"}]})");
  assert(textChat.request.messages.size() == 1);
  assert(textChat.request.messages.front().content == "hello");
  assert(textChat.request.images.empty());

  auto multimodal = parseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"read"},{"type":"image_url","image_url":{"url":"/tmp/a.png"}}]}]})");
  assert(multimodal.request.messages.front().content == "read");
  assert(multimodal.request.images.size() == 1);
  assert(multimodal.request.messages.front().images.size() == 1);
  assert(multimodal.request.messages.front().images.front().uri ==
         "/tmp/a.png");
  assert(multimodal.request.images.front().uri == "/tmp/a.png");

  expectFailure([] {
    parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://example/a.png"}},{"type":"image_url","image_url":{"url":"/tmp/b.png"}}]}]})");
  });
  expectFailure([] {
    parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/tmp/a.png"}},{"type":"image_url","image_url":{"url":"/tmp/b.png"}}]}]})");
  });
  expectFailure([] {
    parseChatCompletionRequest(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})");
  });

  auto embedding =
      parseEmbeddingRequest(R"({"model":"bge_m3_base","input":"hello world"})");
  assert(embedding.request.model == "bge_m3_base");
  assert(embedding.request.input == "hello world");
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
  assert(json.find("\"object\":\"list\"") != std::string::npos);
  assert(json.find("\"index\":0") != std::string::npos);
  assert(json.find("\"prompt_tokens\":2") != std::string::npos);

  std::cout << "JsonCodec tests passed\n";
  return 0;
}
