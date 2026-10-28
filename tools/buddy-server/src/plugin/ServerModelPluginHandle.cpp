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
    std::unique_lock<std::mutex> lock(invocationMutex);
    closing = true;
    invocationCondition.wait(lock, [&] { return activeInvocations == 0; });
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

BuddyServerStatusV1 ServerModelPluginHandle::status() const {
  if (!model)
    throw std::logic_error("buddy-server: server model was not created");
  BuddyServerStatusV1 result{};
  result.structSize = sizeof(result);
  BuddyServerErrorV1 error{};
  const BuddyServerStatusCodeV1 status =
      descriptor->ops->getStatus(model, &result, &error);
  checkError(error, status);
  return result;
}

BuddyServerStatusCodeV1
ServerModelPluginHandle::invoke(const BuddyServerCallV1 &call,
                                BuddyServerEmitFnV1 emit, void *userData) {
  if (!call.request)
    throw std::invalid_argument("buddy-server: server model request is null");
  if (!isKnownOperation(call.request->operation))
    throw std::invalid_argument("buddy-server: unknown server model operation");

  std::unique_lock<std::mutex> lock(invocationMutex);
  if (closing)
    return BUDDY_SERVER_STATUS_CANCELLED;
  if (!model)
    throw std::logic_error("buddy-server: server model was not created");
  const uint32_t declaredLimit = descriptor->maxConcurrency;
  const uint32_t limit =
      (descriptor->capabilityMask & BUDDY_SERVER_CAPABILITY_THREAD_SAFE) != 0
          ? (declaredLimit == 0 ? 1u : declaredLimit)
          : 1u;
  invocationCondition.wait(
      lock, [&] { return closing || activeInvocations < limit; });
  if (closing)
    return BUDDY_SERVER_STATUS_CANCELLED;
  ++activeInvocations;
  lock.unlock();
  struct InvocationGuard {
    ServerModelPluginHandle *owner;
    ~InvocationGuard() { owner->releaseInvocation(); }
  } guard{this};

  BuddyServerErrorV1 error{};
  const BuddyServerStatusCodeV1 status =
      descriptor->ops->invoke(model, &call, emit, userData, &error);
  if (status == BUDDY_SERVER_STATUS_OK)
    return status;
  if (status == BUDDY_SERVER_STATUS_CANCELLED ||
      status == BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED)
    return status;
  checkError(error, status);
  return status;
}

void ServerModelPluginHandle::releaseInvocation() {
  {
    std::lock_guard<std::mutex> lock(invocationMutex);
    if (activeInvocations > 0)
      --activeInvocations;
  }
  invocationCondition.notify_all();
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
  if (message.empty())
    message = "server model operation failed with status " +
              std::to_string(static_cast<uint32_t>(status));
  throw std::runtime_error("buddy-server: " + message);
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
