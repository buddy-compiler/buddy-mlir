//===- MaskedLMModelPluginHandle.cpp - Masked-LM plugin loader -----------===//
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

#include "MaskedLMModelPluginHandle.h"
#include <dlfcn.h>
#include <stdexcept>

namespace buddy {
namespace server {
namespace {
template <typename Fn>
Fn required(void *h, const std::string &p, const char *s) {
  dlerror();
  void *raw = dlsym(h, s);
  if (const char *e = dlerror())
    throw std::runtime_error("buddy-server: masked-LM plugin " + p +
                             " missing " + s + ": " + e);
  return reinterpret_cast<Fn>(raw);
}
template <typename Fn> Fn optional(void *h, const char *s) {
  dlerror();
  void *raw = dlsym(h, s);
  if (dlerror())
    return nullptr;
  return reinterpret_cast<Fn>(raw);
}
} // namespace
MaskedLMModelPluginHandle::MaskedLMModelPluginHandle(const std::string &p)
    : pluginPath(p) {
  if (p.empty())
    throw std::runtime_error("buddy-server: --masked-lm-so path is empty");
  handle = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error(
        std::string("buddy-server: dlopen masked-LM plugin failed: ") +
        dlerror());
  lifetime = std::shared_ptr<void>(handle, [](void *library) {
    if (library)
      dlclose(library);
  });
  try {
    create = required<buddy::runtime::CreateMaskedLMModelFn>(
        handle, p, "buddy_create_masked_lm_model_v1");
    destroy = required<buddy::runtime::DestroyMaskedLMModelFn>(
        handle, p, "buddy_destroy_masked_lm_model_v1");
    if (auto type = optional<buddy::runtime::MaskedLMModelTypeFn>(
            handle, "buddy_masked_lm_model_type_v1"))
      if (const char *n = type())
        pluginModelType = n;
  } catch (...) {
    lifetime.reset();
    handle = nullptr;
    throw;
  }
}
MaskedLMModelPluginHandle::~MaskedLMModelPluginHandle() {
  lifetime.reset();
  handle = nullptr;
}
MaskedLMModelPluginHandle::ModelPtr
MaskedLMModelPluginHandle::createModel() const {
  auto *m = create();
  if (!m)
    throw std::runtime_error(
        "buddy-server: masked-LM plugin returned null model: " + pluginPath);
  auto keepAlive = lifetime;
  return ModelPtr(
      m, [destroy = destroy, keepAlive](buddy::runtime::MaskedLMModel *p) {
        if (p)
          destroy(p);
      });
}
} // namespace server
} // namespace buddy
