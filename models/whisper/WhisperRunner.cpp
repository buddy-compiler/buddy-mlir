//===- WhisperRunner.cpp - Whisper full inference loop -------------------===//
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
//
// Implements WhisperRunner::run(). This file owns the full ASR inference loop:
//   - model .so loading (dlopen) and entrypoint resolution
//   - weight file loading
//   - audio preprocessing (DAP whisperPreprocess)
//   - greedy autoregressive decode over `_mlir_ciface_forward`
//   - detokenization (Text::revertWhisper)
//
// buddy-cli calls this through the generic InferenceRunner interface. Unlike
// the LLM runners, Whisper uses a single forward entrypoint (no prefill/decode
// KV cache) and takes an audio file rather than a text prompt.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/WhisperRunner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"
#include "buddy/DAP/DAP.h"
#include "buddy/LLM/TextContainer.h"

#include <algorithm>
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using buddy::Text;
using dap::Audio;

namespace buddy {
namespace runtime {

namespace {

// Compile-time Whisper-base constants (mirror models/whisper/specs/base.json
// and the shapes baked into the imported MLIR).
constexpr size_t kParamsSize = 72593920;
constexpr size_t kMaxVocabSize = 51865;
constexpr size_t kMaxTokenLength = 448;
constexpr size_t kEncSeq = 1500;
constexpr size_t kEncDim = 512;
constexpr size_t kMelBins = 80;
constexpr size_t kAudioFrames = 3000;
constexpr int kSotToken = 50258; // <|startoftranscript|>
constexpr int kEotToken = 50257; // <|endoftext|>

/// Whisper forward entrypoint exported by whisper_model.so.
///   resultContainer[0] = encoder output {1, 1500, 512}
///   resultContainer[1] = decoder logits  {1, 448, vocab}
using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<float, 3> *, MemRef<size_t, 2> *);

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

int findMaxIndex(const float *start, const float *end) {
  return static_cast<int>(std::distance(start, std::max_element(start, end)));
}

} // namespace

//===----------------------------------------------------------------------===//
// WhisperRunner::run
//===----------------------------------------------------------------------===//

void WhisperRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;
  const bool suppress = cfg.suppressStats;

  if (!suppress)
    std::cerr
        << "\033[33;1mWhisper Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  // ── Resolve model .so / weights / vocab ─────────────────────────────────
  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  std::string baseDir;

  if (!cfg.raxPath.empty()) {
    printLog("Manifest: " + cfg.raxPath, suppress);
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    if (manifest.weightPaths.empty())
      throw std::runtime_error("WhisperRunner: manifest has no weight file.");
    weightsPath = manifest.weightPaths.front();
    vocabPath = manifest.vocabPath;
    baseDir = std::filesystem::absolute(std::filesystem::path(cfg.raxPath))
                  .parent_path()
                  .string();
  } else {
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("WhisperRunner: Mode B requires --model-so.");
    if (cfg.weightsPath.empty())
      throw std::runtime_error("WhisperRunner: Mode B requires --weights.");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath;
    baseDir = std::filesystem::absolute(std::filesystem::path(cfg.modelSoPath))
                  .parent_path()
                  .string();
  }

  if (vocabPath.empty())
    vocabPath = (std::filesystem::path(baseDir) / "vocab.txt").string();

  // ── Resolve audio input (--audio, else audio.wav next to the model) ──────
  std::string audioPath = cfg.audioPath;
  if (audioPath.empty())
    audioPath = (std::filesystem::path(baseDir) / "audio.wav").string();
  if (!std::filesystem::exists(audioPath))
    throw std::runtime_error("WhisperRunner: audio file not found: " +
                             audioPath + " (pass --audio <path.wav>).");

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Vocab     : " + vocabPath, suppress);
  printLog("Audio     : " + audioPath, suppress);

  // ── Load model .so and resolve entrypoint ───────────────────────────────
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("WhisperRunner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error("WhisperRunner: missing _mlir_ciface_forward in " +
                             soPath + ": " + std::string(err));
  }

  // ── Containers ──────────────────────────────────────────────────────────
  Text<size_t, 2> outputContainer;
  outputContainer.loadVocab(vocabPath);

  MemRef<float, 1> paramsContainer({kParamsSize});
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ifstream paramFile(weightsPath, std::ios::in | std::ios::binary);
    if (!paramFile.is_open()) {
      dlclose(handle);
      throw std::runtime_error("WhisperRunner: failed to open weights: " +
                               weightsPath);
    }
    paramFile.read(reinterpret_cast<char *>(paramsContainer.getData()),
                   sizeof(float) * paramsContainer.getSize());
    if (paramFile.fail()) {
      dlclose(handle);
      throw std::runtime_error("WhisperRunner: error reading weights: " +
                               weightsPath);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    printLog(
        "Weights loaded in " +
            std::to_string(std::chrono::duration<double>(t1 - t0).count()) +
            "s",
        suppress);
  }

  // ── Decoder state ───────────────────────────────────────────────────────
  MemRef<size_t, 2> textContainer({1, kMaxTokenLength}, kSotToken);

  // Upper bound on decode steps (respect --max-tokens, capped to window).
  size_t maxSteps = kMaxTokenLength - 1;
  if (cfg.maxNewTokens > 0)
    maxSteps =
        std::min<size_t>(maxSteps, static_cast<size_t>(cfg.maxNewTokens));

  // ── Greedy autoregressive decode ────────────────────────────────────────
  // Two correctness requirements per step:
  //   1. The compiled encoder mutates its mel-feature input in place, so the
  //      audio features must be regenerated fresh before every forward call.
  //   2. Each forward writes freshly-allocated result MemRefs, so the result
  //      containers must also start from a clean struct each iteration.
  for (size_t i = 0; i < maxSteps; i++) {
    Audio<double, 1> rawAudioContainer(audioPath);
    MemRef<float, 3> audioInput({1, kMelBins, kAudioFrames});
    dap::whisperPreprocess(&rawAudioContainer, &audioInput);

    MemRef<float, 3> resultContainer[2] = {
        MemRef<float, 3>({1, kEncSeq, kEncDim}, false, 0),
        MemRef<float, 3>({1, kMaxTokenLength, kMaxVocabSize}, false, 0),
    };

    const auto t0 = std::chrono::high_resolution_clock::now();
    forward(resultContainer, &paramsContainer, &audioInput, &textContainer);
    const auto t1 = std::chrono::high_resolution_clock::now();

    const float *startPtr = resultContainer[1].getData() + i * kMaxVocabSize;
    const float *endPtr = startPtr + kMaxVocabSize;
    const int maxIndex = findMaxIndex(startPtr, endPtr);

    if (!suppress) {
      std::cout << "\033[32;1m[Iteration " << i << "] \033[0m"
                << "Token: " << outputContainer.getStr(maxIndex)
                << " | Time: " << std::chrono::duration<double>(t1 - t0).count()
                << "s\n";
    }

    if (maxIndex != kEotToken) {
      textContainer.getData()[i + 1] = maxIndex;
      outputContainer.appendTokenIdx(maxIndex);
    }

    free(resultContainer[0].release());
    free(resultContainer[1].release());

    if (maxIndex == kEotToken)
      break;
  }

  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertWhisper()
            << std::endl;

  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
