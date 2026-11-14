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
#include "WhisperKvCacheABI.h"

#include "buddy/Core/Container.h"
#include "buddy/DAP/DAP.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
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
  bool f16Weights = false;
  bool encoderCache = false;
  bool lastLogitsOnly = false;
  bool kvCache = false;
  std::size_t encoderParamsSize = 0;
  std::size_t decoderInitParamsSize = 0;
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
using ForwardF16Fn = void (*)(MemRef<float, 3> *, MemRef<_Float16, 1> *,
                              MemRef<float, 3> *, MemRef<std::size_t, 2> *);
using EncoderF16Fn = void (*)(MemRef<float, 3> *, MemRef<_Float16, 1> *,
                              MemRef<float, 3> *);

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
      auto precision = manifest.moduleAttrs.find("precision");
      if (precision != manifest.moduleAttrs.end()) {
        if (precision->second != "f16" && precision->second != "f32")
          throw std::invalid_argument("WhisperRuntime: unsupported precision");
        nextShape.f16Weights = precision->second == "f16";
      }
      auto cache = manifest.moduleAttrs.find("encoder_cache");
      nextShape.encoderCache =
          cache != manifest.moduleAttrs.end() && cache->second == "true";
      auto lastLogits = manifest.moduleAttrs.find("last_logits_only");
      nextShape.lastLogitsOnly = lastLogits != manifest.moduleAttrs.end() &&
                                 lastLogits->second == "true";
      auto kvCache = manifest.moduleAttrs.find("kv_cache");
      nextShape.kvCache =
          kvCache != manifest.moduleAttrs.end() && kvCache->second == "true";
      if (nextShape.kvCache) {
        if (!nextShape.encoderCache || !nextShape.lastLogitsOnly)
          throw std::invalid_argument("WhisperRuntime: KV cache requires "
                                      "encoder cache and single logits");
        nextShape.decoderInitParamsSize =
            manifestSize(manifest, "decoder_init_params_size", 0);
      }
      if (nextShape.lastLogitsOnly && !nextShape.encoderCache)
        throw std::invalid_argument(
            "WhisperRuntime: last-position logits require encoder cache");
      if (nextShape.lastLogitsOnly && !nextShape.kvCache)
        throw std::invalid_argument(
            "WhisperRuntime: single logits require the KV decoder graph");
      if (nextShape.encoderCache) {
        if (!nextShape.f16Weights)
          throw std::invalid_argument(
              "WhisperRuntime: encoder cache requires FP16 weights");
        nextShape.encoderParamsSize =
            manifestSize(manifest, "encoder_params_size", 0);
        if (nextShape.encoderParamsSize >= nextShape.paramsSize)
          throw std::invalid_argument(
              "WhisperRuntime: invalid encoder parameter count");
      }
      if (nextShape.kvCache &&
          nextShape.encoderParamsSize + nextShape.decoderInitParamsSize >=
              nextShape.paramsSize)
        throw std::invalid_argument(
            "WhisperRuntime: invalid KV parameter counts");
      if (nextShape.kvCache &&
          (nextShape.encSeq != 1500 || nextShape.encDim != 512))
        throw std::invalid_argument(
            "WhisperRuntime: KV ABI requires Whisper Base encoder shape");
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
        (nextShape.f16Weights ? sizeof(_Float16) : sizeof(float));
    if (std::filesystem::file_size(nextWeightsPath) < requiredWeightBytes)
      throw std::runtime_error("WhisperRuntime: weights file is smaller than " +
                               std::to_string(requiredWeightBytes) + " bytes");

    void *nextHandle = dlopen(nextSoPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!nextHandle)
      throw std::runtime_error("WhisperRuntime: dlopen failed: " + nextSoPath +
                               ": " + dlerror());
    std::unique_ptr<void, SharedLibraryCloser> handleGuard(nextHandle);

    const char *forwardSymbol =
        nextShape.kvCache
            ? "_mlir_ciface_forward_kv_init"
            : (nextShape.encoderCache ? "_mlir_ciface_forward_decoder"
                                      : "_mlir_ciface_forward");
    dlerror();
    void *nextForward = dlsym(nextHandle, forwardSymbol);
    if (const char *error = dlerror())
      throw std::runtime_error("WhisperRuntime: missing " +
                               std::string(forwardSymbol) + " in " +
                               nextSoPath + ": " + error);
    void *nextEncoder = nullptr;
    if (nextShape.encoderCache) {
      dlerror();
      nextEncoder = dlsym(nextHandle, "_mlir_ciface_forward_encoder");
      if (const char *error = dlerror())
        throw std::runtime_error(
            "WhisperRuntime: missing encoder entrypoint: " +
            std::string(error));
    }
    void *nextKvStep = nullptr;
    if (nextShape.kvCache) {
      dlerror();
      nextKvStep = dlsym(nextHandle, "_mlir_ciface_forward_kv_step");
      if (const char *error = dlerror())
        throw std::runtime_error("WhisperRuntime: missing KV step: " +
                                 std::string(error));
    }

    auto nextVocab = std::make_unique<buddy::Text<std::size_t, 2>>();
    nextVocab->loadVocab(nextVocabPath);

    const auto loadStart = Clock::now();
    std::unique_ptr<MemRef<float, 1>> nextParams;
    std::unique_ptr<MemRef<_Float16, 1>> nextParamsF16;
    void *weightData;
    if (nextShape.f16Weights) {
      nextParamsF16 = std::make_unique<MemRef<_Float16, 1>>(
          std::vector<std::size_t>{nextShape.paramsSize});
      weightData = nextParamsF16->getData();
    } else {
      nextParams = std::make_unique<MemRef<float, 1>>(
          std::vector<std::size_t>{nextShape.paramsSize});
      weightData = nextParams->getData();
    }
    std::ifstream weights(nextWeightsPath, std::ios::binary);
    if (!weights)
      throw std::runtime_error("WhisperRuntime: failed to open weights: " +
                               nextWeightsPath);
    weights.read(reinterpret_cast<char *>(weightData),
                 static_cast<std::streamsize>(requiredWeightBytes));
    if (weights.gcount() != static_cast<std::streamsize>(requiredWeightBytes))
      throw std::runtime_error("WhisperRuntime: short read from weights: " +
                               nextWeightsPath);
    std::unique_ptr<MemRef<_Float16, 1>> nextEncoderParamsF16;
    std::unique_ptr<MemRef<_Float16, 1>> nextDecoderParamsF16;
    std::unique_ptr<MemRef<_Float16, 1>> nextKvInitParamsF16;
    std::unique_ptr<MemRef<_Float16, 1>> nextKvStepParamsF16;
    if (nextShape.encoderCache) {
      const auto encoderSize = nextShape.encoderParamsSize;
      nextEncoderParamsF16 = std::make_unique<MemRef<_Float16, 1>>(
          std::vector<std::size_t>{encoderSize});
      std::memcpy(nextEncoderParamsF16->getData(), nextParamsF16->getData(),
                  encoderSize * sizeof(_Float16));
      if (nextShape.kvCache) {
        const auto initSize = nextShape.decoderInitParamsSize;
        const auto stepSize = nextShape.paramsSize - encoderSize - initSize;
        nextKvInitParamsF16 = std::make_unique<MemRef<_Float16, 1>>(
            std::vector<std::size_t>{initSize});
        nextKvStepParamsF16 = std::make_unique<MemRef<_Float16, 1>>(
            std::vector<std::size_t>{stepSize});
        std::memcpy(nextKvInitParamsF16->getData(),
                    nextParamsF16->getData() + encoderSize,
                    initSize * sizeof(_Float16));
        std::memcpy(nextKvStepParamsF16->getData(),
                    nextParamsF16->getData() + encoderSize + initSize,
                    stepSize * sizeof(_Float16));
      } else {
        const auto decoderSize = nextShape.paramsSize - encoderSize;
        nextDecoderParamsF16 = std::make_unique<MemRef<_Float16, 1>>(
            std::vector<std::size_t>{decoderSize});
        std::memcpy(nextDecoderParamsF16->getData(),
                    nextParamsF16->getData() + encoderSize,
                    decoderSize * sizeof(_Float16));
      }
      nextParamsF16.reset();
    }
    const auto loadEnd = Clock::now();

    soPath = std::move(nextSoPath);
    weightsPath = std::move(nextWeightsPath);
    vocabPath = std::move(nextVocabPath);
    loadedModelName = std::move(nextModelName);
    forward = nextShape.f16Weights ? nullptr
                                   : reinterpret_cast<ForwardFn>(nextForward);
    forwardF16 = nextShape.f16Weights && !nextShape.kvCache
                     ? reinterpret_cast<ForwardF16Fn>(nextForward)
                     : nullptr;
    kvInitF16 = nextShape.kvCache
                    ? reinterpret_cast<detail::InitFn>(nextForward)
                    : nullptr;
    kvStepF16 = reinterpret_cast<detail::StepFn>(nextKvStep);
    encoderF16 = reinterpret_cast<EncoderF16Fn>(nextEncoder);
    vocabulary = std::move(nextVocab);
    params = std::move(nextParams);
    paramsF16 = std::move(nextParamsF16);
    encoderParamsF16 = std::move(nextEncoderParamsF16);
    decoderParamsF16 = std::move(nextDecoderParamsF16);
    kvInitParamsF16 = std::move(nextKvInitParamsF16);
    kvStepParamsF16 = std::move(nextKvStepParamsF16);
    weightLoadSecs = std::chrono::duration<double>(loadEnd - loadStart).count();
    soHandle = handleGuard.release();
    shape = nextShape;
  }

  AudioTranscriptionResult transcribe(const AudioInput &audio, int maxTokens,
                                      const WhisperProgressCallback &progress) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureLoadedUnlocked();
    lastEncoderMs = 0.0;
    lastDecoderMs = 0.0;
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

    const auto preprocessStart = Clock::now();
    dap::Audio<double, 1> rawAudio(audio.uri);
    MemRef<float, 3> audioInput(
        std::vector<std::size_t>{1, shape.melBins, shape.audioFrames});
    dap::whisperPreprocess(&rawAudio, &audioInput);
    result.timings.preprocessMs = milliseconds(Clock::now() - preprocessStart);

    MemRef<float, 3> encoderState(
        std::vector<std::size_t>{1, shape.encSeq, shape.encDim}, false, 0);
    if (shape.encoderCache) {
      const auto encoderStart = Clock::now();
      encoderF16(&encoderState, encoderParamsF16.get(), &audioInput);
      lastEncoderMs = milliseconds(Clock::now() - encoderStart);
      result.timings.inferenceMs += lastEncoderMs;
    }
    std::unique_ptr<detail::CacheState> kvState;
    if (shape.kvCache)
      kvState = std::make_unique<detail::CacheState>(shape.maxTokenLength);

    for (int step = 0; step < maxTokens; ++step) {
      // The compiled function assigns result buffers. Non-owning clean
      // descriptors ensure each iteration starts empty; MemRef destructors
      // release any buffers assigned by the compiled function.
      MemRef<float, 3> outputs[2] = {
          MemRef<float, 3>(
              std::vector<std::size_t>{1, shape.encSeq, shape.encDim}, false,
              0),
          MemRef<float, 3>(
              std::vector<std::size_t>{
                  1, shape.kvCache ? 1 : shape.maxTokenLength, shape.vocabSize},
              false, 0),
      };

      std::unique_ptr<detail::InitResult> kvInitResult;
      std::unique_ptr<detail::StepResult> kvStepResult;
      const auto inferenceStart = Clock::now();
      if (shape.kvCache) {
        MemRef<std::size_t, 2> currentToken(std::vector<std::size_t>{1, 1},
                                            decoderTokens.getData()[step]);
        if (step == 0) {
          kvInitResult = std::make_unique<detail::InitResult>(
              shape.maxTokenLength, shape.vocabSize);
          detail::callInit(kvInitF16, *kvInitResult, *kvInitParamsF16,
                           encoderState, currentToken, *kvState);
          detail::updateState(*kvState, *kvInitResult);
        } else {
          kvStepResult = std::make_unique<detail::StepResult>(
              shape.maxTokenLength, shape.vocabSize);
          detail::callStep(kvStepF16, *kvStepResult, *kvStepParamsF16,
                           encoderState, currentToken, *kvState);
          detail::updateState(*kvState, *kvStepResult);
        }
      } else if (shape.encoderCache)
        forwardF16(&outputs[1], decoderParamsF16.get(), &encoderState,
                   &decoderTokens);
      else if (shape.f16Weights)
        forwardF16(outputs, paramsF16.get(), &audioInput, &decoderTokens);
      else
        forward(outputs, params.get(), &audioInput, &decoderTokens);
      const auto inferenceEnd = Clock::now();
      const double inferenceMs = milliseconds(inferenceEnd - inferenceStart);
      result.timings.inferenceMs += inferenceMs;
      if (shape.encoderCache)
        lastDecoderMs += inferenceMs;

      const float *row =
          shape.kvCache ? (step == 0 ? kvInitResult->logits.getData()
                                     : kvStepResult->logits.getData())
                        : outputs[1].getData() +
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
    return hasModelUnlocked();
  }

  std::pair<double, double> stageTimingsMs() const {
    std::lock_guard<std::mutex> lock(mutex);
    return {lastEncoderMs, lastDecoderMs};
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

  bool hasModelUnlocked() const {
    if (!soHandle || !vocabulary)
      return false;
    if (shape.kvCache)
      return encoderF16 && encoderParamsF16 && kvInitF16 && kvStepF16 &&
             kvInitParamsF16 && kvStepParamsF16;
    if (shape.encoderCache)
      return encoderF16 && encoderParamsF16 && decoderParamsF16 && forwardF16;
    return shape.f16Weights ? forwardF16 && paramsF16 : forward && params;
  }

  void ensureLoadedUnlocked() const {
    if (!hasModelUnlocked())
      throw std::runtime_error("WhisperRuntime: model is not loaded");
  }

  void resetUnlocked() {
    vocabulary.reset();
    params.reset();
    paramsF16.reset();
    encoderParamsF16.reset();
    decoderParamsF16.reset();
    kvInitParamsF16.reset();
    kvStepParamsF16.reset();
    forward = nullptr;
    forwardF16 = nullptr;
    encoderF16 = nullptr;
    kvInitF16 = nullptr;
    kvStepF16 = nullptr;
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
    lastEncoderMs = 0.0;
    lastDecoderMs = 0.0;
  }

  mutable std::mutex mutex;
  void *soHandle = nullptr;
  ForwardFn forward = nullptr;
  ForwardF16Fn forwardF16 = nullptr;
  EncoderF16Fn encoderF16 = nullptr;
  detail::InitFn kvInitF16 = nullptr;
  detail::StepFn kvStepF16 = nullptr;
  std::unique_ptr<MemRef<float, 1>> params;
  std::unique_ptr<MemRef<_Float16, 1>> paramsF16;
  std::unique_ptr<MemRef<_Float16, 1>> encoderParamsF16;
  std::unique_ptr<MemRef<_Float16, 1>> decoderParamsF16;
  std::unique_ptr<MemRef<_Float16, 1>> kvInitParamsF16;
  std::unique_ptr<MemRef<_Float16, 1>> kvStepParamsF16;
  std::unique_ptr<buddy::Text<std::size_t, 2>> vocabulary;
  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  std::string loadedModelName;
  WhisperShape shape;
  double weightLoadSecs = 0.0;
  double lastEncoderMs = 0.0;
  double lastDecoderMs = 0.0;
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
std::pair<double, double> WhisperRuntime::stageTimingsMs() const {
  return impl->stageTimingsMs();
}
std::size_t WhisperRuntime::maxTokenLength() const {
  return impl->getMaxTokenLength();
}

} // namespace runtime
} // namespace buddy
