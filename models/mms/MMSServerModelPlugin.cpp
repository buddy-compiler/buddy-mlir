//===- MMSServerModelPlugin.cpp - Resident MMS vocoder service ------------===//
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

#include "buddy/runtime/core/ServerModelPluginAdapters.h"
#include "buddy/runtime/models/MMSRunner.h"

#include <chrono>
#include <memory>

using namespace buddy::runtime;
using namespace buddy::runtime::server_abi;

namespace {

struct MMSModel {
  std::unique_ptr<MMSRunner> runner;
  ModelStatus status;
};

void *create(const BuddyServerCreateArgsV1 *args, BuddyServerErrorV1 *error) {
  clearError(error);
  try {
    if (!args || args->structSize < sizeof(*args))
      throw std::invalid_argument("MMS: invalid create arguments");
    auto model = std::make_unique<MMSModel>();
    model->status.modelName = copy(args->modelName);
    model->status.backend = "cpu";
    return model.release();
  } catch (...) {
    setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                      "MMS create failed");
    return nullptr;
  }
}

BuddyServerStatusCodeV1 load(void *value, const BuddyServerLoadArgsV1 *args,
                             BuddyServerErrorV1 *error) {
  clearError(error);
  if (!value || !args || args->structSize < sizeof(*args))
    return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                    "MMS: invalid load arguments");
  auto &model = *static_cast<MMSModel *>(value);
  model.runner.reset();
  model.status.state = ModelLoadState::Loading;
  try {
    RunConfig config;
    config.raxPath = copy(args->raxPath);
    config.suppressStats = true;
    if (config.raxPath.empty())
      throw std::invalid_argument("MMS server requires a .rax package");
    auto runner = std::make_unique<MMSRunner>();
    runner->load(config);
    model.runner = std::move(runner);
    model.status.state = ModelLoadState::Ready;
    model.status.message.clear();
    return BUDDY_SERVER_STATUS_OK;
  } catch (...) {
    model.status.state = ModelLoadState::Error;
    return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                             "MMS load failed");
  }
}

BuddyServerStatusCodeV1 status(void *value, BuddyServerStatusV1 *result,
                               BuddyServerErrorV1 *error) {
  clearError(error);
  if (!value || !result || result->structSize < sizeof(*result))
    return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                    "MMS: invalid status arguments");
  fillStatus(static_cast<MMSModel *>(value)->status, result);
  return BUDDY_SERVER_STATUS_OK;
}

bool deadlineExceeded(const BuddyServerCallV1 &call) {
  if (!call.deadlineUnixNs)
    return false;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                 .count()) >= call.deadlineUnixNs;
}

BuddyServerStatusCodeV1 invoke(void *value, const BuddyServerCallV1 *call,
                               BuddyServerEmitFnV1 emit, void *data,
                               BuddyServerErrorV1 *error) {
  clearError(error);
  if (!value || !call || call->structSize < sizeof(*call) || !call->request ||
      call->request->structSize < sizeof(*call->request) || !emit)
    return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                    "MMS: invalid invoke arguments");
  if (call->request->operation != BUDDY_SERVER_OPERATION_VOCODER)
    return setError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED,
                    "MMS serves vocoder latents only");
  auto &model = *static_cast<MMSModel *>(value);
  if (!model.runner)
    return setError(error, BUDDY_SERVER_STATUS_MODEL_NOT_READY,
                    "MMS is not loaded");
  if (cancelled(*call))
    return BUDDY_SERVER_STATUS_CANCELLED;
  if (deadlineExceeded(*call))
    return BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED;
  try {
    const auto &request = call->request->vocoder;
    const auto requestedModel = copy(request.model);
    if (!requestedModel.empty() && requestedModel != model.status.modelName)
      throw std::invalid_argument(
          "MMS: requested model does not match loaded model");
    const auto start = std::chrono::steady_clock::now();
    const auto samples =
        model.runner->synthesize(request.latents, request.latentCount);
    const double elapsed = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (cancelled(*call))
      return BUDDY_SERVER_STATUS_CANCELLED;
    if (deadlineExceeded(*call))
      return BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED;
    BuddyServerEventV1 event{};
    event.structSize = sizeof(event);
    event.kind = BUDDY_SERVER_EVENT_FINAL;
    event.operation = BUDDY_SERVER_OPERATION_VOCODER;
    event.vocoder = {view(model.status.modelName), samples.data(),
                     samples.size(), model.runner->sampleRate(), elapsed};
    switch (emit(data, &event)) {
    case BUDDY_SERVER_EMIT_CONTINUE:
      return BUDDY_SERVER_STATUS_OK;
    case BUDDY_SERVER_EMIT_CANCEL:
      return BUDDY_SERVER_STATUS_CANCELLED;
    default:
      return setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                      "MMS: result callback failed");
    }
  } catch (...) {
    return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                             "MMS inference failed");
  }
}

void destroy(void *value) { delete static_cast<MMSModel *>(value); }

const BuddyServerModelOpsV1 operations = {sizeof(BuddyServerModelOpsV1),
                                          BUDDY_SERVER_MODEL_ABI_VERSION_V1,
                                          &create,
                                          &load,
                                          &status,
                                          &invoke,
                                          &destroy};

const BuddyServerModelDescriptorV1 descriptor = {
    sizeof(BuddyServerModelDescriptorV1),
    BUDDY_SERVER_MODEL_ABI_VERSION_V1,
    BUDDY_SERVER_SERVICE_VOCODER,
    uint64_t{1} << BUDDY_SERVER_OPERATION_VOCODER,
    0,
    BUDDY_SERVER_MODALITY_TENSOR,
    BUDDY_SERVER_RESULT_AUDIO,
    1,
    "facebook/mms-tts-eng",
    &operations,
    0};

} // namespace

extern "C" const BuddyServerModelDescriptorV1 *
buddy_server_model_descriptor_v1() {
  return &descriptor;
}
