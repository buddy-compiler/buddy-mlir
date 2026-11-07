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
constexpr std::size_t kEncoderParamsSize = 20590592;
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
  std::size_t encoderParamsSize = kEncoderParamsSize;
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

// The compiled functions write their single result through the first argument.
using EncoderFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<float, 3> *);
using DecoderFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<std::size_t, 2> *, MemRef<float, 3> *);

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
      nextShape.encoderParamsSize = manifestSize(
          manifest, "encoder_params_size", nextShape.encoderParamsSize);
      if (nextShape.encoderParamsSize >= nextShape.paramsSize)
        throw std::invalid_argument(
            "WhisperRuntime: encoder_params_size must be below params_size");
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
    auto nextEncoder = reinterpret_cast<EncoderFn>(
        dlsym(nextHandle, "_mlir_ciface_forward_encoder"));
    if (const char *error = dlerror())
      throw std::runtime_error(
          "WhisperRuntime: missing _mlir_ciface_forward_encoder in " +
          nextSoPath + ": " + error);
    auto nextDecoder = reinterpret_cast<DecoderFn>(
        dlsym(nextHandle, "_mlir_ciface_forward_decoder"));
    if (const char *error = dlerror())
      throw std::runtime_error(
          "WhisperRuntime: missing _mlir_ciface_forward_decoder in " +
          nextSoPath + ": " + error);

    auto nextVocab = std::make_unique<buddy::Text<std::size_t, 2>>();
    nextVocab->loadVocab(nextVocabPath);

    const auto loadStart = Clock::now();
    std::ifstream weights(nextWeightsPath, std::ios::binary);
    if (!weights)
      throw std::runtime_error("WhisperRuntime: failed to open weights: " +
                               nextWeightsPath);
    // The weights file holds the encoder parameters, then the decoder's.
    const auto readParams = [&](std::size_t size) {
      auto next =
          std::make_unique<MemRef<float, 1>>(std::vector<std::size_t>{size});
      const auto bytes = static_cast<std::streamsize>(size * sizeof(float));
      weights.read(reinterpret_cast<char *>(next->getData()), bytes);
      if (weights.gcount() != bytes)
        throw std::runtime_error("WhisperRuntime: short read from weights: " +
                                 nextWeightsPath);
      return next;
    };
    auto nextEncoderParams = readParams(nextShape.encoderParamsSize);
    auto nextDecoderParams =
        readParams(nextShape.paramsSize - nextShape.encoderParamsSize);
    const auto loadEnd = Clock::now();

    soPath = std::move(nextSoPath);
    weightsPath = std::move(nextWeightsPath);
    vocabPath = std::move(nextVocabPath);
    loadedModelName = std::move(nextModelName);
    encoder = nextEncoder;
    decoder = nextDecoder;
    vocabulary = std::move(nextVocab);
    encoderParams = std::move(nextEncoderParams);
    decoderParams = std::move(nextDecoderParams);
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

    // The compiled functions assign result buffers. Non-owning clean
    // descriptors ensure each result starts empty; MemRef destructors release
    // any buffers assigned by the compiled function.
    const auto preprocessStart = Clock::now();
    dap::Audio<double, 1> rawAudio(audio.uri);
    MemRef<float, 3> audioInput(
        std::vector<std::size_t>{1, shape.melBins, shape.audioFrames}, false,
        0);
    dap::whisperPreprocess(&rawAudio, &audioInput);
    result.timings.preprocessMs = milliseconds(Clock::now() - preprocessStart);

    MemRef<float, 3> encoderStates(
        std::vector<std::size_t>{1, shape.encSeq, shape.encDim}, false, 0);
    const auto encoderStart = Clock::now();
    encoder(&encoderStates, encoderParams.get(), &audioInput);
    result.timings.encoderMs = milliseconds(Clock::now() - encoderStart);

    for (int step = 0; step < maxTokens; ++step) {
      // Compiled kernels may overwrite their inputs, so each step decodes from
      // its own copy of the encoder states.
      MemRef<float, 3> decoderStates(encoderStates);
      MemRef<float, 3> logits(
          std::vector<std::size_t>{1, shape.maxTokenLength, shape.vocabSize},
          false, 0);

      const auto decoderStart = Clock::now();
      decoder(&logits, decoderParams.get(), &decoderTokens, &decoderStates);
      const double decoderMs = milliseconds(Clock::now() - decoderStart);
      result.timings.decoderMs += decoderMs;

      const float *row =
          logits.getData() + static_cast<std::size_t>(step) * shape.vocabSize;
      const int tokenId = static_cast<int>(
          std::distance(row, std::max_element(row, row + shape.vocabSize)));

      if (progress)
        progress(WhisperProgress{static_cast<std::size_t>(step), tokenId,
                                 outputTokens.getStr(tokenId),
                                 decoderMs / 1000.0});

      if (tokenId == shape.eotToken)
        break;
      decoderTokens.getData()[step + 1] = static_cast<std::size_t>(tokenId);
      outputTokens.appendTokenIdx(static_cast<std::size_t>(tokenId));
      ++result.generatedTokens;
    }

    result.text = result.generatedTokens == 0 ? std::string()
                                              : outputTokens.revertWhisper();
    result.timings.inferenceMs =
        result.timings.encoderMs + result.timings.decoderMs;
    result.timings.totalMs = milliseconds(Clock::now() - totalStart);
    return result;
  }

  bool isLoaded() const {
    std::lock_guard<std::mutex> lock(mutex);
    return soHandle && encoder && decoder && encoderParams && decoderParams &&
           vocabulary;
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
    if (!soHandle || !encoder || !decoder || !encoderParams || !decoderParams ||
        !vocabulary)
      throw std::runtime_error("WhisperRuntime: model is not loaded");
  }

  void resetUnlocked() {
    vocabulary.reset();
    encoderParams.reset();
    decoderParams.reset();
    encoder = nullptr;
    decoder = nullptr;
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
  EncoderFn encoder = nullptr;
  DecoderFn decoder = nullptr;
  std::unique_ptr<MemRef<float, 1>> encoderParams;
  std::unique_ptr<MemRef<float, 1>> decoderParams;
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
