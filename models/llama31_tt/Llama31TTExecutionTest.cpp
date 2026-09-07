//===- Llama31TTExecutionTest.cpp - No-device serving context tests -------===//
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

#include "buddy/runtime/models/Llama31TTResidentModel.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef BUDDY_LLAMA31_TT_TEST_RAX
#error "BUDDY_LLAMA31_TT_TEST_RAX is required"
#endif

namespace {

template <typename Function>
void expectFailure(Function function, const std::string &needle) {
  try {
    function();
  } catch (const std::exception &error) {
    assert(std::string(error.what()).find(needle) != std::string::npos);
    return;
  }
  assert(false && "expected failure");
}

} // namespace

int main() {
  using namespace buddy::runtime;

  auto execution = std::make_shared<Llama31TTExecution>();
  ResidentModelConfig config;
  config.raxPath = BUDDY_LLAMA31_TT_TEST_RAX;
  execution->load(config);
  assert(execution->isLoaded());
  assert(execution->metadata().batchSize == 1);
  assert(execution->metadata().maxCacheLen == 64);

  TokenizeResult tokens = execution->tokenize("abc", false);
  assert(tokens.count == 2);
  assert(tokens.tokens.size() == tokens.count);
  TokenizeResult plain = execution->tokenize("abc", false, false);
  assert(plain.count == 1);
  assert(plain.tokens.front() == 3);

  ChatCompletionRequest chat;
  chat.messages.push_back({"system", "brief", {}});
  chat.messages.push_back({"user", "abc", {}});
  const std::string rendered = execution->renderChat(chat);
  assert(rendered.find("<|start_header_id|>system") != std::string::npos);
  assert(rendered.find("<|start_header_id|>assistant") != std::string::npos);

  int starts = 0;
  int resets = 0;
  execution->setResetCallback([&] { ++resets; });
  execution->setBackend([&](const Llama31TTExecution::BackendRequest &request,
                            const CompletionStreamCallback &callback) {
    assert(request.cachePosition == 2);
    assert(execution->cachePosition() == 2);
    ++starts;
    CompletionChunk chunk;
    chunk.delta = "x";
    chunk.tokenId = 42;
    CompletionResult result;
    if (!callback(chunk)) {
      result.finishReason = FinishReason::Cancelled;
      return result;
    }
    result.content = "x";
    result.finishReason = FinishReason::Stop;
    result.usage.completionTokens = 1;
    return result;
  });

  SamplingParams sampling;
  CompletionResult first = execution->generate("abc", sampling);
  assert(first.content == "x");
  assert(execution->cachePosition() == 0);
  assert(resets >= 2);
  CompletionResult second = execution->generate("abc", sampling);
  assert(second.content == "x");
  assert(starts == 2);

  bool sawChunk = false;
  CompletionResult cancelled =
      execution->generate("abc", sampling, [&](const CompletionChunk &chunk) {
        sawChunk = true;
        assert(!chunk.id.empty());
        assert(chunk.model == "llama31_tt");
        return false;
      });
  assert(sawChunk);
  assert(cancelled.finishReason == FinishReason::Cancelled);
  assert(execution->cachePosition() == 0);

  auto modelExecution = std::make_shared<Llama31TTExecution>();
  modelExecution->setBackend([](const Llama31TTExecution::BackendRequest &,
                                const CompletionStreamCallback &) {
    CompletionResult result;
    result.content = "resident";
    result.finishReason = FinishReason::Stop;
    result.usage.completionTokens = 1;
    return result;
  });
  Llama31TTResidentModel model(modelExecution);
  ResidentModelConfig modelConfig;
  modelConfig.raxPath = BUDDY_LLAMA31_TT_TEST_RAX;
  model.load(modelConfig);
  assert(model.status().state == ModelLoadState::Ready);
  CompletionRequest request;
  request.prompt = "abc";
  assert(model.complete(request).content == "resident");

  expectFailure([&] { execution->generate("", sampling); }, "prompt is empty");
  assert(execution->cachePosition() == 0);
  std::cout << "Llama31TTExecution tests passed\n";
  return 0;
}
