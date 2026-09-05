//===- AudioTranscriptionModelPluginHandle.h - Plugin loader ------------===//
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

#ifndef BUDDY_TOOLS_BUDDY_SERVER_AUDIOTRANSCRIPTIONMODELPLUGINHANDLE_H
#define BUDDY_TOOLS_BUDDY_SERVER_AUDIOTRANSCRIPTIONMODELPLUGINHANDLE_H

#include "buddy/runtime/core/AudioTranscriptionModelPlugin.h"

#include <functional>
#include <memory>
#include <string>

namespace buddy {
namespace server {

class AudioTranscriptionModelPluginHandle {
public:
  using ModelPtr = std::unique_ptr<
      buddy::runtime::AudioTranscriptionModel,
      std::function<void(buddy::runtime::AudioTranscriptionModel *)>>;

  explicit AudioTranscriptionModelPluginHandle(const std::string &path);
  ~AudioTranscriptionModelPluginHandle();
  AudioTranscriptionModelPluginHandle(
      const AudioTranscriptionModelPluginHandle &) = delete;
  AudioTranscriptionModelPluginHandle &
  operator=(const AudioTranscriptionModelPluginHandle &) = delete;

  ModelPtr createModel() const;
  const std::string &modelType() const { return pluginModelType; }

private:
  std::string pluginPath;
  std::string pluginModelType;
  std::shared_ptr<void> lifetime;
  buddy::runtime::CreateAudioTranscriptionModelFn create = nullptr;
  buddy::runtime::DestroyAudioTranscriptionModelFn destroy = nullptr;
};

} // namespace server
} // namespace buddy

#endif // BUDDY_TOOLS_BUDDY_SERVER_AUDIOTRANSCRIPTIONMODELPLUGINHANDLE_H
