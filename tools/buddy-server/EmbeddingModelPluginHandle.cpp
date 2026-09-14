//===- EmbeddingModelPluginHandle.cpp - Embedding plugin loader -----------===//
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

#include "EmbeddingModelPluginHandle.h"

#include <dlfcn.h>
#include <stdexcept>

namespace buddy {
namespace server {

namespace {
template <typename Fn>
Fn lookupRequired(void *handle, const std::string &path, const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  if (const char *err = dlerror())
    throw std::runtime_error("buddy-server: embedding plugin " + path +
                             " missing " + symbol + ": " + err);
  return reinterpret_cast<Fn>(raw);
}

template <typename Fn> Fn lookupOptional(void *handle, const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  if (dlerror())
    return nullptr;
  return reinterpret_cast<Fn>(raw);
}
} // namespace

EmbeddingModelPluginHandle::EmbeddingModelPluginHandle(
    const std::string &pluginPath)
    : pluginPath(pluginPath) {
  if (pluginPath.empty())
    throw std::runtime_error("buddy-server: --embedding-so path is empty");
  handle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("buddy-server: dlopen embedding plugin failed: " +
                             pluginPath + ": " + dlerror());
  try {
    create = lookupRequired<buddy::runtime::CreateEmbeddingModelFn>(
        handle, pluginPath, "buddy_create_embedding_model_v1");
    destroy = lookupRequired<buddy::runtime::DestroyEmbeddingModelFn>(
        handle, pluginPath, "buddy_destroy_embedding_model_v1");
    auto type = lookupOptional<buddy::runtime::EmbeddingModelTypeFn>(
        handle, "buddy_embedding_model_type_v1");
    if (type) {
      if (const char *name = type())
        pluginModelType = name;
    }
  } catch (...) {
    dlclose(handle);
    handle = nullptr;
    throw;
  }
}

EmbeddingModelPluginHandle::~EmbeddingModelPluginHandle() {
  if (handle)
    dlclose(handle);
}

EmbeddingModelPluginHandle::ModelPtr
EmbeddingModelPluginHandle::createModel() const {
  buddy::runtime::EmbeddingModel *model = create();
  if (!model)
    throw std::runtime_error(
        "buddy-server: embedding plugin returned null model: " + pluginPath);
  return ModelPtr(model,
                  [destroy = destroy](buddy::runtime::EmbeddingModel *m) {
                    if (m)
                      destroy(m);
                  });
}

} // namespace server
} // namespace buddy
