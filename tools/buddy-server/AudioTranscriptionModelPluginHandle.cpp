//===- AudioTranscriptionModelPluginHandle.cpp - Plugin loader ----------===//
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

#include "AudioTranscriptionModelPluginHandle.h"

#include <dlfcn.h>
#include <stdexcept>

namespace buddy {
namespace server {
namespace {

template <typename Function>
Function requiredSymbol(void *handle, const std::string &path,
                        const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  if (const char *error = dlerror())
    throw std::runtime_error("buddy-server: transcription plugin " + path +
                             " missing " + symbol + ": " + error);
  return reinterpret_cast<Function>(raw);
}

template <typename Function>
Function optionalSymbol(void *handle, const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  if (dlerror())
    return nullptr;
  return reinterpret_cast<Function>(raw);
}

} // namespace

AudioTranscriptionModelPluginHandle::AudioTranscriptionModelPluginHandle(
    const std::string &path)
    : pluginPath(path) {
  if (path.empty())
    throw std::runtime_error("buddy-server: --transcription-so path is empty");

  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error(
        "buddy-server: dlopen transcription plugin failed: " + path + ": " +
        dlerror());
  lifetime = std::shared_ptr<void>(handle, [](void *library) {
    if (library)
      dlclose(library);
  });

  try {
    create = requiredSymbol<buddy::runtime::CreateAudioTranscriptionModelFn>(
        handle, path, "buddy_create_audio_transcription_model_v1");
    destroy = requiredSymbol<buddy::runtime::DestroyAudioTranscriptionModelFn>(
        handle, path, "buddy_destroy_audio_transcription_model_v1");
    if (auto type =
            optionalSymbol<buddy::runtime::AudioTranscriptionModelTypeFn>(
                handle, "buddy_audio_transcription_model_type_v1")) {
      if (const char *name = type())
        pluginModelType = name;
    }
  } catch (...) {
    lifetime.reset();
    throw;
  }
}

AudioTranscriptionModelPluginHandle::~AudioTranscriptionModelPluginHandle() =
    default;

AudioTranscriptionModelPluginHandle::ModelPtr
AudioTranscriptionModelPluginHandle::createModel() const {
  buddy::runtime::AudioTranscriptionModel *model = create();
  if (!model)
    throw std::runtime_error(
        "buddy-server: transcription plugin returned null model: " +
        pluginPath);

  std::shared_ptr<void> keepAlive = lifetime;
  return ModelPtr(model, [destroy = destroy, keepAlive](
                             buddy::runtime::AudioTranscriptionModel *value) {
    if (value)
      destroy(value);
  });
}

} // namespace server
} // namespace buddy
