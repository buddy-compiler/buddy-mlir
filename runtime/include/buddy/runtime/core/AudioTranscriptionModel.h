//===- AudioTranscriptionModel.h - Audio transcription model interface ---===//
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

#ifndef BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONMODEL_H
#define BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONMODEL_H

#include "buddy/runtime/core/AudioTranscriptionTypes.h"

namespace buddy {
namespace runtime {

class AudioTranscriptionModel {
public:
  virtual ~AudioTranscriptionModel() = default;
  AudioTranscriptionModel(const AudioTranscriptionModel &) = delete;
  AudioTranscriptionModel &operator=(const AudioTranscriptionModel &) = delete;

  virtual void load(const AudioTranscriptionModelConfig &config) = 0;
  virtual ModelStatus status() const = 0;
  virtual AudioTranscriptionResult
  transcribe(const AudioTranscriptionRequest &request) = 0;

protected:
  AudioTranscriptionModel() = default;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONMODEL_H
