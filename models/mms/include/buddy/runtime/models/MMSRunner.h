//===- MMSRunner.h - MMS (VITS) vocoder runner ---------------------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_MMS_RUNNER_H
#define BUDDY_RUNTIME_MODELS_MMS_RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

#include <memory>
#include <vector>

namespace buddy {
namespace runtime {

/// MMS vocoder-stage runner.
///
/// The compiled kernel is the HiFi-GAN vocoder of facebook/mms-tts-eng
/// (spectrogram latent -> waveform).  See MMSRunner.cpp for the ABI.
class MMSRunner : public InferenceRunner {
public:
  MMSRunner();
  ~MMSRunner() override;

  /// Load once and reuse the same weights and compiled kernel for serving.
  void load(const RunConfig &cfg);
  std::vector<float> synthesize(const float *latents, size_t count);
  size_t latentCount() const;
  unsigned sampleRate() const;
  void run(const RunConfig &cfg) override;

private:
  struct State;
  std::unique_ptr<State> state;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_MMS_RUNNER_H
