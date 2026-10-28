//===- ServerModelPluginHandle.cpp - Server Model ABI loader --------------===//
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

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace buddy {
namespace server {
namespace {

constexpr const char *kLegacyDescriptor = "buddy_model_plugin_descriptor_v1";

[[noreturn]] void throwPluginError(const std::string &path,
                                   const std::string &message) {
  throw std::runtime_error("server model plugin " + path + ": " + message);
}

void *lookup(void *library, const char *symbol) {
  dlerror();
  void *value = dlsym(library, symbol);
  if (dlerror())
    return nullptr;
  return value;
}

bool isKnownServiceKind(uint32_t kind) {
  return kind >= BUDDY_SERVER_SERVICE_RESIDENT &&
         kind <= BUDDY_SERVER_SERVICE_TRANSCRIPTION;
}

bool isKnownOperation(uint32_t operation) {
  return operation >= BUDDY_SERVER_OPERATION_COMPLETION &&
         operation <= BUDDY_SERVER_OPERATION_TRANSCRIPTION;
}

bool requestsStreaming(const BuddyServerRequestV1 &request) {
  switch (request.operation) {
  case BUDDY_SERVER_OPERATION_COMPLETION:
    return request.completion.stream != 0;
  case BUDDY_SERVER_OPERATION_CHAT:
    return request.chat.stream != 0;
  default:
    return false;
  }
}

std::string copyView(BuddyServerStringViewV1 value) {
  if (!value.data || value.size == 0)
    return {};
  return {value.data, static_cast<size_t>(value.size)};
}

} // namespace

ServerModelPluginHandle::ServerModelPluginHandle(const std::string &path)
    : pluginPath(path) {
  if (path.empty())
    throw std::invalid_argument("buddy-server: model_library path is empty");

  library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    const char *error = dlerror();
    throwPluginError(path, "failed to open: " +
                               std::string(error ? error : "unknown error"));
  }

  void *rawDescriptor =
      lookup(library, BUDDY_SERVER_MODEL_DESCRIPTOR_SYMBOL_V1);
  if (!rawDescriptor) {
    if (lookup(library, kLegacyDescriptor)) {
      close();
      throwPluginError(
          path, "model package uses the legacy server ABI; please use a .rax "
                "package produced with the latest Buddy-MLIR build");
    }
    close();
    throwPluginError(path,
                     "is missing " BUDDY_SERVER_MODEL_DESCRIPTOR_SYMBOL_V1);
  }

  auto descriptorFn =
      reinterpret_cast<BuddyServerModelDescriptorFnV1>(rawDescriptor);
  descriptor = descriptorFn();
  if (!descriptor) {
    close();
    throwPluginError(path, "returned a null server model descriptor");
  }

  if (descriptor->structSize < sizeof(BuddyServerModelDescriptorV1)) {
    close();
    throwPluginError(path, "server model descriptor structSize is too small");
  }
  if (descriptor->abiVersion != BUDDY_SERVER_MODEL_ABI_VERSION_V1) {
    const uint32_t version = descriptor->abiVersion;
    close();
    throwPluginError(path, "server model descriptor ABI version " +
                               std::to_string(version) + " is unsupported");
  }
  if (descriptor->reserved != 0) {
    close();
    throwPluginError(path,
                     "server model descriptor reserved field is non-zero");
  }
  if (!isKnownServiceKind(descriptor->serviceKind)) {
    close();
    throwPluginError(path,
                     "server model descriptor has an unknown service kind");
  }
  if (!descriptor->ops) {
    close();
    throwPluginError(path, "server model descriptor has null operations");
  }
  if (descriptor->ops->structSize < sizeof(BuddyServerModelOpsV1)) {
    close();
    throwPluginError(path, "server model operations structSize is too small");
  }
  if (descriptor->ops->abiVersion != BUDDY_SERVER_MODEL_ABI_VERSION_V1) {
    const uint32_t version = descriptor->ops->abiVersion;
    close();
    throwPluginError(path, "server model operations ABI version " +
                               std::to_string(version) + " is unsupported");
  }
  if (!descriptor->ops->create || !descriptor->ops->load ||
      !descriptor->ops->getStatus || !descriptor->ops->invoke ||
      !descriptor->ops->destroy) {
    close();
    throwPluginError(path, "server model operations are incomplete");
  }

  if (descriptor->modelType)
    pluginModelType = descriptor->modelType;
}

ServerModelPluginHandle::~ServerModelPluginHandle() { close(); }

void ServerModelPluginHandle::close() {
  {
    std::unique_lock<std::mutex> lock(callMutex);
    closing = true;
    callCondition.notify_all();
    callCondition.wait(lock, [&] { return liveCalls == 0; });
  }
  if (model && descriptor && descriptor->ops && descriptor->ops->destroy)
    descriptor->ops->destroy(model);
  model = nullptr;
  if (library)
    dlclose(library);
  library = nullptr;
  descriptor = nullptr;
}

void ServerModelPluginHandle::create(const std::string &modelName) {
  if (model)
    throw std::logic_error("buddy-server: server model already created");

  BuddyServerCreateArgsV1 args{};
  args.structSize = sizeof(args);
  args.modelName = {modelName.data(), static_cast<uint64_t>(modelName.size())};
  BuddyServerErrorV1 error{};
  model = descriptor->ops->create(&args, &error);
  if (!model) {
    const std::string message = errorMessage(error);
    throwPluginError(pluginPath,
                     message.empty() ? "create returned null" : message);
  }
}

void ServerModelPluginHandle::load(const BuddyServerLoadArgsV1 &args) {
  if (!model)
    throw std::logic_error("buddy-server: server model was not created");
  BuddyServerErrorV1 error{};
  const BuddyServerStatusCodeV1 status =
      descriptor->ops->load(model, &args, &error);
  checkError(error, status);
}

ServerModelStatus ServerModelPluginHandle::status() const {
  acquireCall();
  struct CallGuard {
    const ServerModelPluginHandle *owner;
    ~CallGuard() { owner->releaseCall(); }
  } guard{this};

  if (!model)
    throw std::logic_error("buddy-server: server model was not created");
  BuddyServerStatusV1 abiStatus{};
  abiStatus.structSize = sizeof(abiStatus);
  BuddyServerErrorV1 error{};
  const BuddyServerStatusCodeV1 status =
      descriptor->ops->getStatus(model, &abiStatus, &error);
  checkError(error, status);

  ServerModelStatus result;
  result.state = abiStatus.state;
  result.modelName = copyView(abiStatus.modelName);
  result.backend = copyView(abiStatus.backend);
  result.message = copyView(abiStatus.message);
  result.contextLength = abiStatus.contextLength;
  return result;
}

void ServerModelPluginHandle::invoke(const BuddyServerCallV1 &call,
                                     BuddyServerEmitFnV1 emit, void *userData) {
  if (!call.request)
    throw std::invalid_argument("buddy-server: server model request is null");
  if (!isKnownOperation(call.request->operation))
    throw std::invalid_argument("buddy-server: unknown server model operation");

  acquireCall();
  struct CallGuard {
    const ServerModelPluginHandle *owner;
    ~CallGuard() { owner->releaseCall(); }
  } guard{this};

  if (!model)
    throw std::logic_error("buddy-server: server model was not created");
  if (requestsStreaming(*call.request) &&
      (descriptor->capabilityMask & BUDDY_SERVER_CAPABILITY_STREAMING) == 0) {
    static constexpr char message[] =
        "streaming is not supported by the selected backend";
    BuddyServerErrorV1 error{BUDDY_SERVER_STATUS_NOT_SUPPORTED, message,
                             sizeof(message) - 1};
    checkError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED);
  }

  BuddyServerErrorV1 error{};
  const BuddyServerStatusCodeV1 status =
      descriptor->ops->invoke(model, &call, emit, userData, &error);
  checkError(error, status);
}

void ServerModelPluginHandle::acquireCall() const {
  std::unique_lock<std::mutex> lock(callMutex);
  if (closing) {
    BuddyServerErrorV1 error{};
    checkError(error, BUDDY_SERVER_STATUS_CANCELLED);
  }

  ++liveCalls;
  const uint32_t declaredLimit = descriptor->maxConcurrency;
  const uint32_t limit =
      (descriptor->capabilityMask & BUDDY_SERVER_CAPABILITY_THREAD_SAFE) != 0
          ? (declaredLimit == 0 ? 1u : declaredLimit)
          : 1u;
  callCondition.wait(lock, [&] { return closing || activeCalls < limit; });
  if (closing) {
    --liveCalls;
    lock.unlock();
    callCondition.notify_all();
    BuddyServerErrorV1 error{};
    checkError(error, BUDDY_SERVER_STATUS_CANCELLED);
  }
  ++activeCalls;
}

void ServerModelPluginHandle::releaseCall() const {
  {
    std::lock_guard<std::mutex> lock(callMutex);
    --activeCalls;
    --liveCalls;
  }
  callCondition.notify_all();
}

std::string
ServerModelPluginHandle::errorMessage(const BuddyServerErrorV1 &error) {
  if (!error.message)
    return {};
  if (error.messageSize == 0)
    return std::string(error.message);
  return std::string(error.message, static_cast<size_t>(error.messageSize));
}

void ServerModelPluginHandle::checkError(const BuddyServerErrorV1 &error,
                                         BuddyServerStatusCodeV1 status) const {
  if (status == BUDDY_SERVER_STATUS_OK)
    return;
  std::string message = errorMessage(error);
  if (message.empty()) {
    switch (status) {
    case BUDDY_SERVER_STATUS_INVALID_ARGUMENT:
      message = "invalid server model request";
      break;
    case BUDDY_SERVER_STATUS_NOT_SUPPORTED:
      message = "server model operation is not supported";
      break;
    case BUDDY_SERVER_STATUS_MODEL_NOT_READY:
      message = "server model is not ready";
      break;
    case BUDDY_SERVER_STATUS_RESOURCE_EXHAUSTED:
      message = "server model resources are exhausted";
      break;
    case BUDDY_SERVER_STATUS_CANCELLED:
      message = "server model request was cancelled";
      break;
    case BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED:
      message = "server model request deadline exceeded";
      break;
    case BUDDY_SERVER_STATUS_MODEL_ERROR:
      message = "server model operation failed";
      break;
    case BUDDY_SERVER_STATUS_INTERNAL_ERROR:
      message = "internal server model error";
      break;
    case BUDDY_SERVER_STATUS_OK:
      return;
    default:
      message = "server model operation failed with status " +
                std::to_string(static_cast<uint32_t>(status));
      break;
    }
  }
  throw ServerModelError(status, "buddy-server: " + message);
}

const char *serverServiceKindName(uint32_t kind) {
  switch (kind) {
  case BUDDY_SERVER_SERVICE_RESIDENT:
    return "resident";
  case BUDDY_SERVER_SERVICE_EMBEDDING:
    return "embedding";
  case BUDDY_SERVER_SERVICE_MASKED_LM:
    return "masked_lm";
  case BUDDY_SERVER_SERVICE_TRANSCRIPTION:
    return "transcription";
  default:
    return "invalid";
  }
}

} // namespace server
} // namespace buddy
