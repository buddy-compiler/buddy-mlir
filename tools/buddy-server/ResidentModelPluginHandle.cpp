//===- ResidentModelPluginHandle.cpp - Resident plugin loader -------------===//
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

#include "ResidentModelPluginHandle.h"

#include <dlfcn.h>
#include <stdexcept>

namespace buddy {
namespace server {

namespace {

template <typename Fn>
Fn lookupRequiredSymbol(void *handle, const std::string &pluginPath,
                        const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  const char *err = dlerror();
  if (err) {
    throw std::runtime_error("buddy-server: resident plugin " + pluginPath +
                             " missing " + symbol + ": " + err);
  }
  return reinterpret_cast<Fn>(raw);
}

template <typename Fn>
Fn lookupOptionalSymbol(void *handle, const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  const char *err = dlerror();
  if (err)
    return nullptr;
  return reinterpret_cast<Fn>(raw);
}

} // namespace

ResidentModelPluginHandle::ResidentModelPluginHandle(
    const std::string &pluginPath)
    : pluginPath(pluginPath) {
  if (pluginPath.empty())
    throw std::runtime_error("buddy-server: --serving-so path is empty");

  handle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    throw std::runtime_error("buddy-server: dlopen resident plugin failed: " +
                             pluginPath + ": " + dlerror());
  }

  try {
    create = lookupRequiredSymbol<buddy::runtime::CreateResidentModelFn>(
        handle, pluginPath, "buddy_create_resident_model_v1");
    destroy = lookupRequiredSymbol<buddy::runtime::DestroyResidentModelFn>(
        handle, pluginPath, "buddy_destroy_resident_model_v1");

    auto type = lookupOptionalSymbol<buddy::runtime::ResidentModelTypeFn>(
        handle, "buddy_resident_model_type_v1");
    if (type) {
      const char *name = type();
      if (name)
        pluginModelType = name;
    }
  } catch (...) {
    dlclose(handle);
    handle = nullptr;
    throw;
  }
}

ResidentModelPluginHandle::~ResidentModelPluginHandle() {
  if (handle)
    dlclose(handle);
}

ResidentModelPluginHandle::ModelPtr
ResidentModelPluginHandle::createModel() const {
  buddy::runtime::ResidentModel *model = create();
  if (!model) {
    throw std::runtime_error(
        "buddy-server: resident plugin returned null model: " + pluginPath);
  }

  return ModelPtr(model, [destroy = destroy](buddy::runtime::ResidentModel *m) {
    if (m)
      destroy(m);
  });
}

} // namespace server
} // namespace buddy
