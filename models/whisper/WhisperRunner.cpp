//===- WhisperRunner.cpp - Whisper buddy-cli inference adapter -----------===//
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

#include "buddy/runtime/models/WhisperRunner.h"
#include "buddy/runtime/models/WhisperRuntime.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace buddy {
namespace runtime {
namespace {

void printLog(const std::string &message, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << message << "\n";
}

} // namespace

void WhisperRunner::run(const RunConfig &config) {
  const bool suppress = config.suppressStats;
  if (!suppress)
    std::cerr
        << "\033[33;1mWhisper Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  AudioTranscriptionModelConfig runtimeConfig;
  runtimeConfig.raxPath = config.raxPath;
  runtimeConfig.modelSoPath = config.modelSoPath;
  if (!config.weightsPath.empty())
    runtimeConfig.weightPaths.push_back(config.weightsPath);
  runtimeConfig.vocabPath = config.vocabPath;

  WhisperRuntime runtime;
  if (!config.raxPath.empty())
    printLog("Manifest: " + config.raxPath, suppress);
  runtime.load(runtimeConfig);

  const std::filesystem::path baseDir =
      config.raxPath.empty()
          ? std::filesystem::path(runtime.modelSoPath()).parent_path()
          : std::filesystem::absolute(config.raxPath).parent_path();
  std::string audioPath = config.audioPath;
  if (audioPath.empty())
    audioPath = (baseDir / "audio.wav").string();

  printLog("Model .so : " + runtime.modelSoPath(), suppress);
  printLog("Weights   : " + runtime.weightsPath(), suppress);
  printLog("Vocab     : " + runtime.vocabPath(), suppress);
  printLog("Audio     : " + audioPath, suppress);
  printLog("Weights loaded in " + std::to_string(runtime.weightLoadSeconds()) +
               "s",
           suppress);

  AudioInput audio;
  audio.uri = std::move(audioPath);
  audio.mimeType = "audio/wav";
  const int maxTokens =
      config.maxNewTokens > 0 ? std::min(config.maxNewTokens, 447) : 447;
  AudioTranscriptionResult result = runtime.transcribe(
      audio, maxTokens, [&](const WhisperProgress &progress) {
        if (!suppress)
          std::cout << "\033[32;1m[Iteration " << progress.iteration
                    << "] \033[0mToken: " << progress.token
                    << " | Time: " << progress.inferenceSeconds << "s\n";
      });

  std::cout << "\033[33;1m[Output]\033[0m " << result.text << std::endl;
}

} // namespace runtime
} // namespace buddy
