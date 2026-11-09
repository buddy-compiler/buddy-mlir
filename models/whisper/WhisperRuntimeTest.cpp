//===- WhisperRuntimeTest.cpp - Whisper runtime contract tests -----------===//
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

#include "buddy/runtime/models/WhisperRuntime.h"
#include "buddy/runtime/models/WhisperAudioTranscriptionModel.h"

#include <cassert>
#include <cstddef>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Exception, typename Callable>
void expectThrows(Callable &&callable) {
  bool threw = false;
  try {
    callable();
  } catch (const Exception &) {
    threw = true;
  }
  assert(threw);
}

using CallCountFn = std::size_t (*)();

/// Reads a call counter from the fake kernel the runtime has already loaded.
std::size_t fakeKernelCalls(const buddy::runtime::WhisperRuntime &runtime,
                            const char *counter) {
  void *handle = dlopen(runtime.modelSoPath().c_str(),
                        RTLD_NOW | RTLD_NOLOAD | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("fake kernel is not loaded");
  auto count = reinterpret_cast<CallCountFn>(dlsym(handle, counter));
  if (!count) {
    dlclose(handle);
    throw std::runtime_error("fake kernel has no call counter");
  }
  const std::size_t calls = count();
  dlclose(handle);
  return calls;
}

/// Fails in every build type, unlike assert().
void expectCalls(const char *what, std::size_t actual, std::size_t expected) {
  if (actual == expected)
    return;
  std::cerr << "FAIL: " << what << " calls: expected " << expected << ", got "
            << actual << "\n";
  std::exit(1);
}

buddy::runtime::AudioInput wavInput(const std::string &path) {
  buddy::runtime::AudioInput audio;
  audio.uri = path;
  audio.mimeType = "audio/wav";
  return audio;
}

} // namespace

int main(int argc, char **argv) {
  assert(argc == 4);
  const std::string raxPath = argv[1];
  const std::string audioPath = argv[2];
  const std::string shortVocabPath = argv[3];
  const auto audio = wavInput(audioPath);

  buddy::runtime::WhisperRuntime runtime;
  expectThrows<std::runtime_error>([&] { runtime.transcribe(audio, 1); });
  expectThrows<std::invalid_argument>(
      [&] { runtime.load(buddy::runtime::AudioTranscriptionModelConfig{}); });

  buddy::runtime::AudioTranscriptionModelConfig config;
  config.raxPath = raxPath;
  auto shortVocabConfig = config;
  shortVocabConfig.vocabPath = shortVocabPath;
  expectThrows<std::runtime_error>([&] { runtime.load(shortVocabConfig); });
  runtime.load(config);
  assert(runtime.isLoaded());
  assert(runtime.modelName() == "whisper_runtime_test");
  assert(runtime.maxTokenLength() == 4);

  for (std::size_t request = 0; request < 2; ++request) {
    std::vector<int> progressTokens;
    const auto result = runtime.transcribe(
        audio, 3, [&](const buddy::runtime::WhisperProgress &progress) {
          progressTokens.push_back(progress.tokenId);
        });
    assert(result.model == "whisper_runtime_test");
    assert(result.text == "hello");
    assert(result.generatedTokens == 1);
    assert((progressTokens == std::vector<int>{1, 2}));
    assert(result.timings.preprocessMs >= 0.0);
    assert(result.timings.inferenceMs >= 0.0);
    assert(result.timings.totalMs >= result.timings.inferenceMs);
    assert(result.timings.inferenceMs >=
           result.timings.encoderMs + result.timings.decoderMs - 1e-6);

    // Two decode steps per request, but only one encoder run.
    expectCalls("encoder",
                fakeKernelCalls(runtime, "whisper_fake_encoder_calls"),
                request + 1);
    expectCalls("decoder",
                fakeKernelCalls(runtime, "whisper_fake_decoder_calls"),
                (request + 1) * 2);
  }

  expectThrows<std::invalid_argument>([&] { runtime.transcribe(audio, 0); });
  expectThrows<std::invalid_argument>([&] { runtime.transcribe(audio, 4); });
  expectThrows<std::invalid_argument>([&] {
    auto invalid = audio;
    invalid.mimeType = "audio/mpeg";
    runtime.transcribe(invalid, 1);
  });
  expectThrows<std::invalid_argument>([&] {
    auto invalid = audio;
    invalid.bytes.push_back(0);
    runtime.transcribe(invalid, 1);
  });
  expectThrows<std::invalid_argument>(
      [&] { runtime.transcribe(wavInput(audioPath + ".missing.wav"), 1); });

  buddy::runtime::WhisperAudioTranscriptionModel model;
  assert(model.status().state == buddy::runtime::ModelLoadState::Unloaded);
  expectThrows<std::runtime_error>([&] {
    buddy::runtime::AudioTranscriptionRequest request;
    request.audio = audio;
    model.transcribe(request);
  });
  model.load(config);
  const auto status = model.status();
  assert(status.state == buddy::runtime::ModelLoadState::Ready);
  assert(status.modelName == "whisper_runtime_test");
  assert(status.contextLength == 4);

  buddy::runtime::AudioTranscriptionRequest request;
  request.model = "whisper_runtime_test";
  request.audio = audio;
  request.maxTokens = 3;
  assert(model.transcribe(request).text == "hello");

  request.model = "another_model";
  expectThrows<std::invalid_argument>([&] { model.transcribe(request); });
  request.model = "whisper_runtime_test";
  request.audio = {};
  expectThrows<std::invalid_argument>([&] { model.transcribe(request); });
  request.audio = audio;
  request.maxTokens = 4;
  expectThrows<std::invalid_argument>([&] { model.transcribe(request); });

  buddy::runtime::WhisperAudioTranscriptionModel failedModel;
  expectThrows<std::invalid_argument>([&] {
    failedModel.load(buddy::runtime::AudioTranscriptionModelConfig{});
  });
  assert(failedModel.status().state == buddy::runtime::ModelLoadState::Error);
  assert(!failedModel.status().message.empty());

  std::cout << "WhisperRuntime tests passed\n";
  return 0;
}
