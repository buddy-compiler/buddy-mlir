//===- ModelPluginDiscovery.cpp - Unified server plugin discovery ---------===//
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

#include "ModelPluginDiscovery.h"

#include <dlfcn.h>

#include <stdexcept>

namespace buddy {
namespace server {
namespace {

bool isKnownServiceKind(buddy::runtime::ModelServiceKind kind) {
  using buddy::runtime::ModelServiceKind;
  switch (kind) {
  case ModelServiceKind::Resident:
  case ModelServiceKind::Embedding:
  case ModelServiceKind::MaskedLM:
  case ModelServiceKind::Transcription:
    return true;
  case ModelServiceKind::Invalid:
    return false;
  }
  return false;
}

class SharedLibrary {
public:
  explicit SharedLibrary(void *handle) : handle(handle) {}
  ~SharedLibrary() {
    if (handle)
      dlclose(handle);
  }
  SharedLibrary(const SharedLibrary &) = delete;
  SharedLibrary &operator=(const SharedLibrary &) = delete;

private:
  void *handle = nullptr;
};

} // namespace

ModelPluginDiscoveryResult discoverModelPlugin(const std::string &pluginPath) {
  if (pluginPath.empty())
    throw std::invalid_argument("buddy-server: model_library path is empty");

  dlerror();
  void *rawHandle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!rawHandle) {
    const char *error = dlerror();
    throw std::runtime_error("buddy-server: failed to open model plugin " +
                             pluginPath + ": " +
                             (error ? error : "unknown error"));
  }
  SharedLibrary library(rawHandle);

  dlerror();
  void *rawSymbol =
      dlsym(rawHandle, buddy::runtime::ModelPluginDescriptorSymbolV1);
  if (const char *error = dlerror())
    throw std::runtime_error(
        "buddy-server: model plugin " + pluginPath + " is missing " +
        buddy::runtime::ModelPluginDescriptorSymbolV1 + ": " + error);
  auto descriptorFn =
      reinterpret_cast<buddy::runtime::ModelPluginDescriptorFn>(rawSymbol);
  if (!descriptorFn)
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " has a null descriptor symbol");

  const auto *descriptor = descriptorFn();
  if (!descriptor)
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " returned a null descriptor");
  if (descriptor->structSize < sizeof(buddy::runtime::ModelPluginDescriptorV1))
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " descriptor structSize is too small");
  if (descriptor->abiVersion !=
      buddy::runtime::ModelPluginDescriptorAbiVersionV1)
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " descriptor ABI version is unsupported");
  if (descriptor->reserved != 0)
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " descriptor reserved field is non-zero");
  if (!isKnownServiceKind(descriptor->serviceKind))
    throw std::runtime_error("buddy-server: model plugin " + pluginPath +
                             " descriptor has an unknown service kind");

  return {pluginPath, descriptor->serviceKind};
}

const char *modelServiceKindName(buddy::runtime::ModelServiceKind kind) {
  using buddy::runtime::ModelServiceKind;
  switch (kind) {
  case ModelServiceKind::Resident:
    return "resident";
  case ModelServiceKind::Embedding:
    return "embedding";
  case ModelServiceKind::MaskedLM:
    return "masked_lm";
  case ModelServiceKind::Transcription:
    return "transcription";
  case ModelServiceKind::Invalid:
    return "invalid";
  }
  return "unknown";
}

} // namespace server
} // namespace buddy
