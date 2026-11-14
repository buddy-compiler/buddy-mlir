//===- WhisperRuntime.cpp - Reusable long-lived Whisper runtime ----------===//
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

#include "buddy/Core/Container.h"
#include "buddy/DAP/DAP.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace buddy {
namespace runtime {
namespace {

constexpr std::size_t kParamsSize = 72593920;
constexpr std::size_t kMaxVocabSize = 51865;
constexpr std::size_t kMaxTokenLength = 448;
constexpr std::size_t kEncSeq = 1500;
constexpr std::size_t kEncDim = 512;
constexpr std::size_t kMelBins = 80;
constexpr std::size_t kAudioFrames = 3000;
constexpr int kSotToken = 50258;
constexpr int kEotToken = 50257;

using Clock = std::chrono::steady_clock;
struct SharedLibraryCloser {
  void operator()(void *handle) const {
    if (handle)
      dlclose(handle);
  }
};

struct WhisperShape {
  std::size_t paramsSize = kParamsSize;
  std::size_t vocabSize = kMaxVocabSize;
  std::size_t maxTokenLength = kMaxTokenLength;
  std::size_t encSeq = kEncSeq;
  std::size_t encDim = kEncDim;
  std::size_t melBins = kMelBins;
  std::size_t audioFrames = kAudioFrames;
  int sotToken = kSotToken;
  int eotToken = kEotToken;
};

std::size_t manifestSize(const ModelManifest &manifest, const char *key,
                         std::size_t fallback) {
  auto value = manifest.moduleAttrs.find(key);
  if (value == manifest.moduleAttrs.end())
    return fallback;
  const std::size_t parsed =
      static_cast<std::size_t>(std::stoull(value->second));
  if (parsed == 0)
    throw std::invalid_argument(std::string("WhisperRuntime: invalid ") + key);
  return parsed;
}

int manifestToken(const ModelManifest &manifest, const char *key,
                  int fallback) {
  auto value = manifest.moduleAttrs.find(key);
  if (value == manifest.moduleAttrs.end())
    return fallback;
  const int parsed = std::stoi(value->second);
  if (parsed < 0)
    throw std::invalid_argument(std::string("WhisperRuntime: invalid ") + key);
  return parsed;
}

using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<float, 3> *, MemRef<std::size_t, 2> *);

double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

bool isWavMimeType(const std::string &mimeType) {
  return mimeType.empty() || mimeType == "audio/wav" ||
         mimeType == "audio/x-wav" || mimeType == "audio/wave";
}

bool hasWavExtension(const std::filesystem::path &path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return extension == ".wav" || extension == ".wave";
}

std::size_t countVocabEntries(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("WhisperRuntime: failed to open vocabulary: " +
                             path);
  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line))
    ++count;
  return count;
}

} // namespace

class WhisperRuntime::Impl {
public:
  ~Impl() { resetUnlocked(); }

  void load(const AudioTranscriptionModelConfig &config) {
    std::lock_guard<std::mutex> lock(mutex);
    resetUnlocked();

    std::string nextSoPath;
    std::string nextWeightsPath;
    std::string nextVocabPath;
    std::string nextModelName = config.modelName;
    WhisperShape nextShape;

    if (!config.raxPath.empty()) {
      ModelManifest manifest = ModelManifest::loadFromRax(config.raxPath);
      nextSoPath = manifest.soPath;
      if (manifest.weightPaths.empty())
        throw std::invalid_argument("WhisperRuntime: manifest has no weights");
      nextWeightsPath = manifest.weightPaths.front();
      nextVocabPath =
          config.vocabPath.empty() ? manifest.vocabPath : config.vocabPath;
      if (nextVocabPath.empty())
        nextVocabPath =
            (std::filesystem::path(config.raxPath).parent_path() / "vocab.txt")
                .string();
      if (nextModelName.empty())
        nextModelName = manifest.modelName;
      nextShape.paramsSize =
          manifestSize(manifest, "params_size", nextShape.paramsSize);
      nextShape.vocabSize =
          manifestSize(manifest, "vocab_size", nextShape.vocabSize);
      nextShape.maxTokenLength =
          manifestSize(manifest, "max_token_len", nextShape.maxTokenLength);
      nextShape.melBins = manifestSize(manifest, "mel_bins", nextShape.melBins);
      nextShape.audioFrames =
          manifestSize(manifest, "audio_frames", nextShape.audioFrames);
      nextShape.encSeq = manifestSize(manifest, "enc_seq", nextShape.encSeq);
      nextShape.encDim = manifestSize(manifest, "enc_dim", nextShape.encDim);
      nextShape.sotToken =
          manifestToken(manifest, "sot_token", nextShape.sotToken);
      nextShape.eotToken =
          manifestToken(manifest, "eot_token", nextShape.eotToken);
      if (nextShape.melBins != kMelBins ||
          nextShape.audioFrames != kAudioFrames)
        throw std::invalid_argument(
            "WhisperRuntime: manifest mel shape must be 80x3000");
      if (nextShape.maxTokenLength < 2)
        throw std::invalid_argument(
            "WhisperRuntime: max_token_len must be at least 2");
      if (nextShape.sotToken >= static_cast<int>(nextShape.vocabSize) ||
          nextShape.eotToken >= static_cast<int>(nextShape.vocabSize))
        throw std::invalid_argument(
            "WhisperRuntime: special token is outside vocab_size");
    } else {
      nextSoPath = config.modelSoPath;
      if (config.weightPaths.empty())
        throw std::invalid_argument("WhisperRuntime: no weight path provided");
      nextWeightsPath = config.weightPaths.front();
      nextVocabPath = config.vocabPath;
      if (nextVocabPath.empty())
        nextVocabPath =
            (std::filesystem::path(nextSoPath).parent_path() / "vocab.txt")
                .string();
    }

    if (nextSoPath.empty())
      throw std::invalid_argument(
          "WhisperRuntime: model shared library is empty");
    if (nextWeightsPath.empty())
      throw std::invalid_argument("WhisperRuntime: weights path is empty");
    if (nextVocabPath.empty())
      throw std::invalid_argument("WhisperRuntime: vocabulary path is empty");
    if (nextModelName.empty())
      nextModelName = "whisper_base";

    requireRegularFile(nextSoPath, "model shared library");
    requireRegularFile(nextWeightsPath, "weights");
    requireRegularFile(nextVocabPath, "vocabulary");
    if (countVocabEntries(nextVocabPath) < nextShape.vocabSize)
      throw std::runtime_error(
          "WhisperRuntime: vocabulary has fewer entries than vocab_size");

    const std::uintmax_t requiredWeightBytes =
        std::uintmax_t(nextShape.paramsSize) *
        static_cast<std::uintmax_t>(sizeof(float));
    if (std::filesystem::file_size(nextWeightsPath) < requiredWeightBytes)
      throw std::runtime_error("WhisperRuntime: weights file is smaller than " +
                               std::to_string(requiredWeightBytes) + " bytes");

    void *nextHandle = dlopen(nextSoPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!nextHandle)
      throw std::runtime_error("WhisperRuntime: dlopen failed: " + nextSoPath +
                               ": " + dlerror());
    std::unique_ptr<void, SharedLibraryCloser> handleGuard(nextHandle);

    dlerror();
    auto nextForward =
        reinterpret_cast<ForwardFn>(dlsym(nextHandle, "_mlir_ciface_forward"));
    if (const char *error = dlerror())
      throw std::runtime_error(
          "WhisperRuntime: missing _mlir_ciface_forward in " + nextSoPath +
          ": " + error);

    auto nextVocab = std::make_unique<buddy::Text<std::size_t, 2>>();
    nextVocab->loadVocab(nextVocabPath);

    const auto loadStart = Clock::now();
    auto nextParams = std::make_unique<MemRef<float, 1>>(
        std::vector<std::size_t>{nextShape.paramsSize});
    std::ifstream weights(nextWeightsPath, std::ios::binary);
    if (!weights)
      throw std::runtime_error("WhisperRuntime: failed to open weights: " +
                               nextWeightsPath);
    weights.read(reinterpret_cast<char *>(nextParams->getData()),
                 static_cast<std::streamsize>(requiredWeightBytes));
    if (weights.gcount() != static_cast<std::streamsize>(requiredWeightBytes))
      throw std::runtime_error("WhisperRuntime: short read from weights: " +
                               nextWeightsPath);
    const auto loadEnd = Clock::now();

    soPath = std::move(nextSoPath);
    weightsPath = std::move(nextWeightsPath);
    vocabPath = std::move(nextVocabPath);
    loadedModelName = std::move(nextModelName);
    forward = nextForward;
    vocabulary = std::move(nextVocab);
    params = std::move(nextParams);
    weightLoadSecs = std::chrono::duration<double>(loadEnd - loadStart).count();
    soHandle = handleGuard.release();
    shape = nextShape;
  }

  AudioTranscriptionResult transcribe(const AudioInput &audio, int maxTokens,
                                      const WhisperProgressCallback &progress) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    validateAudio(audio);
    if (maxTokens <= 0 || maxTokens >= static_cast<int>(shape.maxTokenLength))
      throw std::invalid_argument("max_tokens must be between 1 and " +
                                  std::to_string(shape.maxTokenLength - 1));

    const auto totalStart = Clock::now();
    AudioTranscriptionResult result;
    result.model = loadedModelName;
    buddy::Text<std::size_t, 2> outputTokens = *vocabulary;
    MemRef<std::size_t, 2> decoderTokens(
        std::vector<std::size_t>{1, shape.maxTokenLength},
        static_cast<std::size_t>(shape.sotToken));

    for (int step = 0; step < maxTokens; ++step) {
      const auto preprocessStart = Clock::now();
      dap::Audio<double, 1> rawAudio(audio.uri);
      MemRef<float, 3> audioInput(
          std::vector<std::size_t>{1, shape.melBins, shape.audioFrames});
      dap::whisperPreprocess(&rawAudio, &audioInput);
      const auto preprocessEnd = Clock::now();
      result.timings.preprocessMs +=
          milliseconds(preprocessEnd - preprocessStart);

      // The compiled function assigns result buffers. Non-owning clean
      // descriptors ensure each iteration starts empty; MemRef destructors
      // release any buffers assigned by the compiled function.
      MemRef<float, 3> outputs[2] = {
          MemRef<float, 3>(
              std::vector<std::size_t>{1, shape.encSeq, shape.encDim}, false,
              0),
          MemRef<float, 3>(std::vector<std::size_t>{1, shape.maxTokenLength,
                                                    shape.vocabSize},
                           false, 0),
      };

      const auto inferenceStart = Clock::now();
      forward(outputs, params.get(), &audioInput, &decoderTokens);
      const auto inferenceEnd = Clock::now();
      const double inferenceMs = milliseconds(inferenceEnd - inferenceStart);
      result.timings.inferenceMs += inferenceMs;

      const float *row = outputs[1].getData() +
                         static_cast<std::size_t>(step) * shape.vocabSize;
      const int tokenId = static_cast<int>(
          std::distance(row, std::max_element(row, row + shape.vocabSize)));

      if (progress)
        progress(WhisperProgress{static_cast<std::size_t>(step), tokenId,
                                 outputTokens.getStr(tokenId),
                                 inferenceMs / 1000.0});

      if (tokenId == shape.eotToken)
        break;
      decoderTokens.getData()[step + 1] = static_cast<std::size_t>(tokenId);
      outputTokens.appendTokenIdx(static_cast<std::size_t>(tokenId));
      ++result.generatedTokens;
    }

    result.text = result.generatedTokens == 0 ? std::string()
                                              : outputTokens.revertWhisper();
    result.timings.totalMs = milliseconds(Clock::now() - totalStart);
    return result;
  }

  bool isLoaded() const {
    std::lock_guard<std::mutex> lock(mutex);
    return soHandle && forward && params && vocabulary;
  }

  std::string getModelName() const {
    std::lock_guard<std::mutex> lock(mutex);
    return loadedModelName;
  }
  std::string getSoPath() const {
    std::lock_guard<std::mutex> lock(mutex);
    return soPath;
  }
  std::string getWeightsPath() const {
    std::lock_guard<std::mutex> lock(mutex);
    return weightsPath;
  }
  std::string getVocabPath() const {
    std::lock_guard<std::mutex> lock(mutex);
    return vocabPath;
  }
  double getWeightLoadSeconds() const {
    std::lock_guard<std::mutex> lock(mutex);
    return weightLoadSecs;
  }
  std::size_t getMaxTokenLength() const {
    std::lock_guard<std::mutex> lock(mutex);
    return shape.maxTokenLength;
  }

private:
  static void requireRegularFile(const std::string &path, const char *kind) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
      throw std::invalid_argument("WhisperRuntime: " + std::string(kind) +
                                  " not found: " + path);
  }

  static void validateAudio(const AudioInput &audio) {
    if (!audio.bytes.empty())
      throw std::invalid_argument(
          "in-memory audio bytes are not supported; provide a local WAV path");
    if (audio.uri.empty())
      throw std::invalid_argument("audio path must not be empty");
    if (audio.uri.find('\0') != std::string::npos)
      throw std::invalid_argument("audio path contains NUL");
    if (!isWavMimeType(audio.mimeType))
      throw std::invalid_argument("only WAV audio is supported");
    if (!hasWavExtension(audio.uri))
      throw std::invalid_argument("audio path must have a .wav extension");
    requireRegularFile(audio.uri, "audio file");
  }

  void ensureLoadedUnlocked() const {
    if (!soHandle || !forward || !params || !vocabulary)
      throw std::runtime_error("WhisperRuntime: model is not loaded");
  }

  void resetUnlocked() {
    vocabulary.reset();
    params.reset();
    forward = nullptr;
    if (soHandle) {
      dlclose(soHandle);
      soHandle = nullptr;
    }
    soPath.clear();
    weightsPath.clear();
    vocabPath.clear();
    loadedModelName.clear();
    shape = {};
    weightLoadSecs = 0.0;
  }

  mutable std::mutex mutex;
  void *soHandle = nullptr;
  ForwardFn forward = nullptr;
  std::unique_ptr<MemRef<float, 1>> params;
  std::unique_ptr<buddy::Text<std::size_t, 2>> vocabulary;
  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  std::string loadedModelName;
  WhisperShape shape;
  double weightLoadSecs = 0.0;
};

WhisperRuntime::WhisperRuntime() : impl(std::make_unique<Impl>()) {}
WhisperRuntime::~WhisperRuntime() = default;

void WhisperRuntime::load(const AudioTranscriptionModelConfig &config) {
  impl->load(config);
}

AudioTranscriptionResult
WhisperRuntime::transcribe(const AudioInput &audio, int maxTokens,
                           const WhisperProgressCallback &progress) {
  return impl->transcribe(audio, maxTokens, progress);
}

bool WhisperRuntime::isLoaded() const { return impl->isLoaded(); }
std::string WhisperRuntime::modelName() const { return impl->getModelName(); }
std::string WhisperRuntime::modelSoPath() const { return impl->getSoPath(); }
std::string WhisperRuntime::weightsPath() const {
  return impl->getWeightsPath();
}
std::string WhisperRuntime::vocabPath() const { return impl->getVocabPath(); }
double WhisperRuntime::weightLoadSeconds() const {
  return impl->getWeightLoadSeconds();
}
std::size_t WhisperRuntime::maxTokenLength() const {
  return impl->getMaxTokenLength();
}

} // namespace runtime
} // namespace buddy
