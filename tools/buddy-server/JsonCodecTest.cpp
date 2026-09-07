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

using buddy::server::parseAudioTranscriptionRequest;
using buddy::server::parseChatCompletionRequest;
using buddy::server::parseCompletionRequest;
using buddy::server::parseEmbeddingRequest;
using buddy::server::parseMaskedLMRequest;
using buddy::server::toOpenAIAudioTranscriptionJson;
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

  auto masked = parseMaskedLMRequest(
      R"({"model":"proteinglm_1b_mlm","prompt":"A <mask> C","top_k":3,"extra":true})");
  assert(masked.request.model == "proteinglm_1b_mlm");
  assert(masked.request.input == "A <mask> C");
  assert(masked.request.topK == 3);
  expectFailure([] { parseMaskedLMRequest(R"({"input":"x","prompt":"y"})"); });
  expectFailure([] { parseMaskedLMRequest(R"({"input":"x","top_k":0})"); });
  buddy::runtime::MaskedLMResult maskedResult;
  maskedResult.model = "proteinglm_1b_mlm";
  maskedResult.sequenceLength = 4;
  maskedResult.promptTokens = 4;
  maskedResult.predictions.push_back({1, {{3, "G", 12.3f}}});
  const std::string maskedJson = buddy::server::toJson(maskedResult);
  assert(maskedJson.find("\"object\":\"masked_lm\"") != std::string::npos);
  assert(maskedJson.find("\"position\":1") != std::string::npos);

  auto audio = parseAudioTranscriptionRequest(
      R"({"model":"whisper_base","file":"file:/tmp/sample.wav","max_tokens":7})");
  assert(audio.request.model == "whisper_base");
  assert(audio.request.audio.uri == "/tmp/sample.wav");
  assert(audio.request.audio.mimeType == "audio/wav");
  assert(audio.request.audio.bytes.empty());
  assert(audio.request.maxTokens == 7);
  auto alias =
      parseAudioTranscriptionRequest(R"({"audio_path":"/tmp/sample.wav"})");
  assert(alias.request.audio.uri == "/tmp/sample.wav");
  assert(alias.request.maxTokens == 64);

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
  assert(audioJson.find("\"text\":\"hello\"") != std::string::npos);
  assert(audioJson.find("\"model\":\"whisper_base\"") != std::string::npos);
  assert(audioJson.find("\"generated_tokens\":2") != std::string::npos);
  assert(audioJson.find("\"preprocess_ms\":1.25") != std::string::npos);

  std::cout << "JsonCodec tests passed\n";
  return 0;
}
