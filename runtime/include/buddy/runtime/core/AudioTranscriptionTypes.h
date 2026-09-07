//===- AudioTranscriptionTypes.h - Audio transcription DTOs -------------===//
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

#ifndef BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONTYPES_H
#define BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONTYPES_H

#include "buddy/runtime/core/ServingTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

struct AudioTranscriptionModelConfig {
  std::string raxPath;
  std::string modelSoPath;
  std::vector<std::string> weightPaths;
  std::string vocabPath;
  std::string modelName;
};

/// One audio input. `bytes` is reserved for a future upload transport; the
/// initial server contract accepts only a local WAV path in `uri`.
struct AudioInput {
  std::string uri;
  std::string mimeType;
  std::vector<std::uint8_t> bytes;
};

struct AudioTranscriptionRequest {
  std::string model;
  AudioInput audio;
  int maxTokens = 64;
};

struct AudioTranscriptionTimings {
  double preprocessMs = 0.0;
  double inferenceMs = 0.0;
  double totalMs = 0.0;
};

struct AudioTranscriptionResult {
  std::string model;
  std::string text;
  std::size_t generatedTokens = 0;
  AudioTranscriptionTimings timings;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_AUDIOTRANSCRIPTIONTYPES_H
