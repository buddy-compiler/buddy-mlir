//===- ServerModelPluginHandleTest.cpp - Unified server ABI tests --------===//
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

#include "ServerModelPluginHandle.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

BuddyServerEmitResultV1 captureCompletion(void *userData,
                                          const BuddyServerEventV1 *event) {
  auto *content = static_cast<std::string *>(userData);
  assert(event);
  assert(event->kind == BUDDY_SERVER_EVENT_FINAL);
  assert(event->operation == BUDDY_SERVER_OPERATION_COMPLETION);
  content->assign(event->completion.content.data,
                  static_cast<size_t>(event->completion.content.size));
  return BUDDY_SERVER_EMIT_CONTINUE;
}

struct StreamCapture {
  int chunks = 0;
  int finals = 0;
  bool cancel = false;
};

BuddyServerEmitResultV1 captureStream(void *userData,
                                      const BuddyServerEventV1 *event) {
  auto *capture = static_cast<StreamCapture *>(userData);
  assert(event);
  assert(event->operation == BUDDY_SERVER_OPERATION_COMPLETION);
  if (event->kind == BUDDY_SERVER_EVENT_CHUNK)
    ++capture->chunks;
  else if (event->kind == BUDDY_SERVER_EVENT_FINAL)
    ++capture->finals;
  else
    assert(false && "unexpected streaming event");
  return capture->cancel ? BUDDY_SERVER_EMIT_CANCEL
                         : BUDDY_SERVER_EMIT_CONTINUE;
}

void expectFailure(const std::string &path, const std::string &needle) {
  try {
    buddy::server::ServerModelPluginHandle plugin(path);
    assert(false && "expected server plugin load failure");
  } catch (const std::exception &error) {
    assert(std::string(error.what()).find(needle) != std::string::npos);
  }
}

} // namespace

int main() {
  expectFailure("", "model_library path is empty");
  expectFailure(BUDDY_TEST_LEGACY_SERVER_PLUGIN, "legacy server ABI");

  buddy::server::ServerModelPluginHandle plugin(BUDDY_TEST_SERVER_PLUGIN);
  assert(plugin.serviceKind() == BUDDY_SERVER_SERVICE_RESIDENT);
  assert(plugin.modelType() == "fake_model");
  assert((plugin.operationMask() &
          (uint64_t{1} << BUDDY_SERVER_OPERATION_COMPLETION)) != 0);
  assert(plugin.inputModalityMask() == BUDDY_SERVER_MODALITY_TEXT);
  assert(plugin.resultKind() == BUDDY_SERVER_RESULT_TEXT);

  plugin.create("fake_model");
  const std::string modelName = "fake_model";
  BuddyServerLoadArgsV1 load{};
  load.structSize = sizeof(load);
  load.modelName = {modelName.data(), modelName.size()};
  plugin.load(load);
  const BuddyServerStatusV1 status = plugin.status();
  assert(status.state == BUDDY_SERVER_MODEL_READY);

  const std::string prompt = "hello";
  BuddyServerRequestV1 request{};
  request.structSize = sizeof(request);
  request.operation = BUDDY_SERVER_OPERATION_COMPLETION;
  request.completion.prompt = {prompt.data(), prompt.size()};
  request.completion.sampling.maxTokens = 1;

  BuddyServerCallV1 call{};
  call.structSize = sizeof(call);
  call.requestId = 1;
  call.request = &request;

  std::string content;
  assert(plugin.invoke(call, &captureCompletion, &content) ==
         BUDDY_SERVER_STATUS_OK);
  assert(content == "fake completion");

  request.completion.stream = 1;
  StreamCapture stream;
  assert(plugin.invoke(call, &captureStream, &stream) ==
         BUDDY_SERVER_STATUS_OK);
  assert(stream.chunks == 1);
  assert(stream.finals == 1);

  StreamCapture cancelled;
  cancelled.cancel = true;
  assert(plugin.invoke(call, &captureStream, &cancelled) ==
         BUDDY_SERVER_STATUS_CANCELLED);
  assert(cancelled.chunks == 1);
  assert(cancelled.finals == 0);

  std::cout << "ServerModelPluginHandle tests passed\n";
  return 0;
}
