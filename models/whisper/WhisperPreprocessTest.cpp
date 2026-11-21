//===- WhisperPreprocessTest.cpp - Preprocessing reuse tests
//---------------===//
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

#include "buddy/Core/Container.h"
#include "buddy/runtime/models/WhisperRuntime.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace detail {
int preprocessCalls = 0;
bool emptyFeatureOutputs = true;
} // namespace detail

// Replace the generated preprocessing entry point in this executable only.
// The separate whisper-runtime test still uses the real DAP implementation.
extern "C" void
_mlir_ciface_buddy_whisperPreprocess(MemRef<float, 3> *outputFeatures,
                                     MemRef<double, 1> *) {
  ++detail::preprocessCalls;
  detail::emptyFeatureOutputs &= outputFeatures->getData() == nullptr;
  *outputFeatures =
      MemRef<float, 3>(std::vector<std::size_t>{1, 80, 3000}, 0.0F);
}

// WhisperPreprocess.h also emits an unused RFFT wrapper. Satisfy its symbol
// without pulling in the DAP object that defines the real preprocessing entry.
extern "C" void _mlir_ciface_buddy_RFFT(MemRef<double, 1> *) {
  throw std::logic_error("Unexpected RFFT call in preprocessing reuse test");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Expected a test RAX and a WAV path\n";
    return 1;
  }

  buddy::runtime::AudioTranscriptionModelConfig config;
  config.raxPath = argv[1];
  buddy::runtime::WhisperRuntime runtime;
  runtime.load(config);

  buddy::runtime::AudioInput audio;
  audio.uri = argv[2];
  audio.mimeType = "audio/wav";
  for (int request = 0; request < 2; ++request) {
    std::vector<int> tokens;
    const auto result = runtime.transcribe(
        audio, 3, [&](const buddy::runtime::WhisperProgress &progress) {
          tokens.push_back(progress.tokenId);
        });
    if (detail::preprocessCalls != request + 1) {
      std::cerr << "Expected one preprocessing call per request, got "
                << detail::preprocessCalls << " calls after " << request + 1
                << " request(s)\n";
      return 1;
    }
    if (!detail::emptyFeatureOutputs || result.text != "hello" ||
        result.generatedTokens != 1 || tokens != std::vector<int>{1, 2}) {
      std::cerr << "Expected empty output descriptors and fresh features at "
                   "each decode step\n";
      return 1;
    }
  }

  std::cout << "Whisper preprocessing reuse tests passed\n";
  return 0;
}
