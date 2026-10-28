//===- ServerModelConcurrencyFakePlugin.cpp - ABI call gate fixture -------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/ServerModelABI.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>

#ifndef BUDDY_FAKE_THREAD_SAFE
#define BUDDY_FAKE_THREAD_SAFE 0
#endif

namespace {

struct Model {
  std::atomic<uint32_t> statusSequence{0};
};

std::atomic<uint32_t> callCount{0};
std::atomic<uint32_t> activeCalls{0};
std::atomic<uint32_t> maximumActiveCalls{0};
std::atomic<uint32_t> destroyCount{0};
std::mutex blockMutex;
std::condition_variable blockCondition;
bool callsBlocked = false;

void enterCall() {
  callCount.fetch_add(1, std::memory_order_acq_rel);
  const uint32_t active =
      activeCalls.fetch_add(1, std::memory_order_acq_rel) + 1;
  uint32_t maximum = maximumActiveCalls.load(std::memory_order_acquire);
  while (maximum < active && !maximumActiveCalls.compare_exchange_weak(
                                 maximum, active, std::memory_order_acq_rel,
                                 std::memory_order_acquire)) {
  }

  std::unique_lock<std::mutex> lock(blockMutex);
  blockCondition.wait(lock, [] { return !callsBlocked; });
}

void leaveCall() { activeCalls.fetch_sub(1, std::memory_order_acq_rel); }

void *create(const BuddyServerCreateArgsV1 *, BuddyServerErrorV1 *) {
  return new Model();
}

BuddyServerStatusCodeV1 load(void *, const BuddyServerLoadArgsV1 *,
                             BuddyServerErrorV1 *) {
  return BUDDY_SERVER_STATUS_OK;
}

BuddyServerStatusCodeV1 getStatus(void *value, BuddyServerStatusV1 *status,
                                  BuddyServerErrorV1 *) {
  if (!value || !status)
    return BUDDY_SERVER_STATUS_INVALID_ARGUMENT;
  enterCall();
  auto *model = static_cast<Model *>(value);
  const uint32_t sequence =
      model->statusSequence.fetch_add(1, std::memory_order_acq_rel) + 1;
  static constexpr char modelName[] = "concurrency_fake";
  static constexpr char backend[] = "test";
  static thread_local char message[32];
  const int messageSize =
      std::snprintf(message, sizeof(message), "status-%u", sequence);
  status->structSize = sizeof(*status);
  status->state = BUDDY_SERVER_MODEL_READY;
  status->modelName = {modelName, sizeof(modelName) - 1};
  status->backend = {backend, sizeof(backend) - 1};
  status->message = {message,
                     messageSize > 0 ? static_cast<uint64_t>(messageSize) : 0};
  status->contextLength = 128;
  leaveCall();
  return BUDDY_SERVER_STATUS_OK;
}

BuddyServerStatusCodeV1 invoke(void *, const BuddyServerCallV1 *,
                               BuddyServerEmitFnV1, void *,
                               BuddyServerErrorV1 *) {
  enterCall();
  leaveCall();
  return BUDDY_SERVER_STATUS_OK;
}

void destroy(void *value) {
  delete static_cast<Model *>(value);
  destroyCount.fetch_add(1, std::memory_order_release);
}

} // namespace

extern "C" void buddy_test_concurrency_reset() {
  callCount.store(0, std::memory_order_release);
  activeCalls.store(0, std::memory_order_release);
  maximumActiveCalls.store(0, std::memory_order_release);
  destroyCount.store(0, std::memory_order_release);
}

extern "C" void buddy_test_concurrency_set_blocked(uint32_t blocked) {
  {
    std::lock_guard<std::mutex> lock(blockMutex);
    callsBlocked = blocked != 0;
  }
  if (!blocked)
    blockCondition.notify_all();
}

extern "C" uint32_t buddy_test_concurrency_call_count() {
  return callCount.load(std::memory_order_acquire);
}

extern "C" uint32_t buddy_test_concurrency_active_calls() {
  return activeCalls.load(std::memory_order_acquire);
}

extern "C" uint32_t buddy_test_concurrency_maximum_active_calls() {
  return maximumActiveCalls.load(std::memory_order_acquire);
}

extern "C" uint32_t buddy_test_concurrency_destroy_count() {
  return destroyCount.load(std::memory_order_acquire);
}

extern "C" const BuddyServerModelDescriptorV1 *
buddy_server_model_descriptor_v1() {
  static const BuddyServerModelOpsV1 ops = {
      sizeof(ops), BUDDY_SERVER_MODEL_ABI_VERSION_V1,
      &create,     &load,
      &getStatus,  &invoke,
      &destroy};
  static const BuddyServerModelDescriptorV1 descriptor = {
      sizeof(descriptor),
      BUDDY_SERVER_MODEL_ABI_VERSION_V1,
      BUDDY_SERVER_SERVICE_RESIDENT,
      (1ull << BUDDY_SERVER_OPERATION_COMPLETION),
#if BUDDY_FAKE_THREAD_SAFE
      BUDDY_SERVER_CAPABILITY_THREAD_SAFE,
#else
      0,
#endif
      BUDDY_SERVER_MODALITY_TEXT,
      BUDDY_SERVER_RESULT_TEXT,
      2,
      "concurrency_fake",
      &ops,
      0};
  return &descriptor;
}
