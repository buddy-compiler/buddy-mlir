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

#include <dlfcn.h>

#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

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

class ConcurrencyControls {
public:
  explicit ConcurrencyControls(const char *path) {
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    assert(library);
    resetFn = lookup<void (*)()>("buddy_test_concurrency_reset");
    setBlockedFn =
        lookup<void (*)(uint32_t)>("buddy_test_concurrency_set_blocked");
    callCountFn = lookup<uint32_t (*)()>("buddy_test_concurrency_call_count");
    activeCallsFn =
        lookup<uint32_t (*)()>("buddy_test_concurrency_active_calls");
    maximumActiveCallsFn =
        lookup<uint32_t (*)()>("buddy_test_concurrency_maximum_active_calls");
    destroyCountFn =
        lookup<uint32_t (*)()>("buddy_test_concurrency_destroy_count");
  }

  ~ConcurrencyControls() { dlclose(library); }

  void reset() const {
    setBlocked(false);
    resetFn();
  }
  void setBlocked(bool value) const { setBlockedFn(value ? 1 : 0); }
  uint32_t callCount() const { return callCountFn(); }
  uint32_t activeCalls() const { return activeCallsFn(); }
  uint32_t maximumActiveCalls() const { return maximumActiveCallsFn(); }
  uint32_t destroyCount() const { return destroyCountFn(); }

private:
  template <typename Function> Function lookup(const char *name) {
    dlerror();
    void *symbol = dlsym(library, name);
    assert(symbol && !dlerror());
    return reinterpret_cast<Function>(symbol);
  }

  void *library = nullptr;
  void (*resetFn)() = nullptr;
  void (*setBlockedFn)(uint32_t) = nullptr;
  uint32_t (*callCountFn)() = nullptr;
  uint32_t (*activeCallsFn)() = nullptr;
  uint32_t (*maximumActiveCallsFn)() = nullptr;
  uint32_t (*destroyCountFn)() = nullptr;
};

void initializePlugin(buddy::server::ServerModelPluginHandle &plugin) {
  plugin.create("concurrency_fake");
  BuddyServerLoadArgsV1 load{};
  load.structSize = sizeof(load);
  plugin.load(load);
}

BuddyServerCallV1 makeCompletionCall(BuddyServerRequestV1 &request) {
  request = {};
  request.structSize = sizeof(request);
  request.operation = BUDDY_SERVER_OPERATION_COMPLETION;
  BuddyServerCallV1 call{};
  call.structSize = sizeof(call);
  call.request = &request;
  return call;
}

void waitForCallCount(const ConcurrencyControls &controls, uint32_t expected) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (controls.callCount() < expected &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(controls.callCount() >= expected);
}

void testNonThreadSafeCallSerialization() {
  ConcurrencyControls controls(BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  buddy::server::ServerModelPluginHandle plugin(
      BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  initializePlugin(plugin);

  BuddyServerRequestV1 request{};
  BuddyServerCallV1 call = makeCompletionCall(request);
  controls.reset();
  controls.setBlocked(true);
  auto invoke = std::async(std::launch::async,
                           [&] { plugin.invoke(call, nullptr, nullptr); });
  waitForCallCount(controls, 1);
  auto status = std::async(std::launch::async, [&] { return plugin.status(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  assert(controls.callCount() == 1);
  assert(controls.activeCalls() == 1);
  controls.setBlocked(false);
  invoke.get();
  status.get();
  assert(controls.callCount() == 2);
  assert(controls.maximumActiveCalls() == 1);

  controls.reset();
  controls.setBlocked(true);
  auto firstStatus =
      std::async(std::launch::async, [&] { return plugin.status(); });
  waitForCallCount(controls, 1);
  auto secondStatus =
      std::async(std::launch::async, [&] { return plugin.status(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  assert(controls.callCount() == 1);
  controls.setBlocked(false);
  firstStatus.get();
  secondStatus.get();
  assert(controls.callCount() == 2);
  assert(controls.maximumActiveCalls() == 1);
}

void testThreadSafeCallConcurrency() {
  ConcurrencyControls controls(BUDDY_TEST_THREAD_SAFE_SERVER_PLUGIN);
  buddy::server::ServerModelPluginHandle plugin(
      BUDDY_TEST_THREAD_SAFE_SERVER_PLUGIN);
  initializePlugin(plugin);

  controls.reset();
  controls.setBlocked(true);
  auto firstStatus =
      std::async(std::launch::async, [&] { return plugin.status(); });
  waitForCallCount(controls, 1);
  auto secondStatus =
      std::async(std::launch::async, [&] { return plugin.status(); });
  waitForCallCount(controls, 2);
  assert(controls.activeCalls() == 2);
  assert(controls.maximumActiveCalls() == 2);
  controls.setBlocked(false);
  firstStatus.get();
  secondStatus.get();
}

void testCloseWaitsForStatus() {
  ConcurrencyControls controls(BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  auto *plugin = new buddy::server::ServerModelPluginHandle(
      BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  initializePlugin(*plugin);

  controls.reset();
  controls.setBlocked(true);
  auto status =
      std::async(std::launch::async, [plugin] { return plugin->status(); });
  waitForCallCount(controls, 1);
  auto close = std::async(std::launch::async, [plugin] { delete plugin; });
  assert(close.wait_for(std::chrono::milliseconds(50)) ==
         std::future_status::timeout);
  assert(controls.destroyCount() == 0);
  controls.setBlocked(false);
  status.get();
  close.get();
  assert(controls.destroyCount() == 1);
}

void testStatusOwnsStrings() {
  ConcurrencyControls controls(BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  buddy::server::ServerModelPluginHandle plugin(
      BUDDY_TEST_NON_THREAD_SAFE_SERVER_PLUGIN);
  initializePlugin(plugin);
  controls.reset();

  const buddy::server::ServerModelStatus first = plugin.status();
  const buddy::server::ServerModelStatus second = plugin.status();
  assert(first.message == "status-1");
  assert(second.message == "status-2");
  assert(first.modelName == "concurrency_fake");
  assert(first.backend == "test");
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
  const buddy::server::ServerModelStatus status = plugin.status();
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
  plugin.invoke(call, &captureCompletion, &content);
  assert(content == "fake completion");

  request.completion.stream = 1;
  StreamCapture stream;
  plugin.invoke(call, &captureStream, &stream);
  assert(stream.chunks == 1);
  assert(stream.finals == 1);

  StreamCapture cancelled;
  cancelled.cancel = true;
  bool cancelledError = false;
  try {
    plugin.invoke(call, &captureStream, &cancelled);
  } catch (const buddy::server::ServerModelError &error) {
    cancelledError = true;
    assert(error.status() == BUDDY_SERVER_STATUS_CANCELLED);
  }
  assert(cancelledError);
  assert(cancelled.chunks == 1);
  assert(cancelled.finals == 0);

  buddy::server::ServerModelPluginHandle nonStreaming(
      BUDDY_TEST_NON_STREAMING_SERVER_PLUGIN);
  assert((nonStreaming.capabilityMask() & BUDDY_SERVER_CAPABILITY_STREAMING) ==
         0);
  nonStreaming.create("fake_model");
  nonStreaming.load(load);
  StreamCapture rejected;
  bool notSupported = false;
  try {
    nonStreaming.invoke(call, &captureStream, &rejected);
  } catch (const buddy::server::ServerModelError &error) {
    notSupported = true;
    assert(error.status() == BUDDY_SERVER_STATUS_NOT_SUPPORTED);
    assert(std::string(error.what()).find("streaming is not supported") !=
           std::string::npos);
  }
  assert(notSupported);
  assert(rejected.chunks == 0);
  assert(rejected.finals == 0);

  testNonThreadSafeCallSerialization();
  testThreadSafeCallConcurrency();
  testCloseWaitsForStatus();
  testStatusOwnsStrings();

  std::cout << "ServerModelPluginHandle tests passed\n";
  return 0;
}
