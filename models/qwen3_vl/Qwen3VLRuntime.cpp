//===- Qwen3VLRuntime.cpp - Resident Qwen3-VL runtime ---------------------===//
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

#include "Qwen3VLRuntime.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ImagePreprocess.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/ServingGeneration.h"
#include "buddy/runtime/models/Qwen3VLRawCodec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace fs = std::filesystem;

namespace buddy {
namespace runtime {
namespace {

using VisionFn = void (*)(const float *, long, const float *, float *, float *,
                          float *, float *);
using DecoderFn = void (*)(const float *, long, const float *, const float *,
                           const float *, const float *, const float *,
                           const float *, const float *, float *, long, long,
                           long, long);

struct MappedFloats {
  const float *data = nullptr;
  std::size_t count = 0;
  void *base = nullptr;
  std::size_t bytes = 0;
  ~MappedFloats() {
    if (base && base != MAP_FAILED)
      munmap(base, bytes);
  }
};

struct MappedI64 {
  const int64_t *data = nullptr;
  std::size_t count = 0;
  void *base = nullptr;
  std::size_t bytes = 0;

  ~MappedI64() {
    if (base && base != MAP_FAILED)
      munmap(base, bytes);
  }
};
void mapFloats(const std::string &path, MappedFloats &out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("qwen3_vl: cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("qwen3_vl: fstat failed " + path);
  }
  out.bytes = static_cast<std::size_t>(st.st_size);
  if (out.bytes == 0 || out.bytes % sizeof(float) != 0) {
    close(fd);
    throw std::runtime_error("qwen3_vl: invalid float file size: " + path);
  }
  out.base = mmap(nullptr, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (out.base == MAP_FAILED) {
    out.base = nullptr;
    throw std::runtime_error("qwen3_vl: mmap failed " + path);
  }
  out.data = reinterpret_cast<const float *>(out.base);
  out.count = out.bytes / sizeof(float);
}

void mapI64(const std::string &path, MappedI64 &out) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("qwen3_vl: cannot open " + path);
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("qwen3_vl: fstat failed " + path);
  }
  out.bytes = static_cast<std::size_t>(st.st_size);
  if (out.bytes == 0 || out.bytes % sizeof(int64_t) != 0) {
    close(fd);
    throw std::runtime_error("qwen3_vl: invalid int64 file size: " + path);
  }
  out.base = mmap(nullptr, out.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (out.base == MAP_FAILED) {
    out.base = nullptr;
    throw std::runtime_error("qwen3_vl: mmap failed " + path);
  }

  out.data = reinterpret_cast<const int64_t *>(out.base);
  out.count = out.bytes / sizeof(int64_t);
}
std::vector<float> readFloats(const std::string &path, std::size_t expected) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("qwen3_vl: cannot open " + path);
  const std::streamoff end = f.tellg();
  if (end < 0 || static_cast<std::size_t>(end) % sizeof(float) != 0)
    throw std::runtime_error("qwen3_vl: invalid float file size: " + path);
  const std::size_t count = static_cast<std::size_t>(end) / sizeof(float);
  if (expected != 0 && count != expected)
    throw std::runtime_error("qwen3_vl: unexpected float count in " + path);
  f.seekg(0);
  std::vector<float> values(count);
  f.read(reinterpret_cast<char *>(values.data()),
         static_cast<std::streamsize>(count * sizeof(float)));
  if (!f)
    throw std::runtime_error("qwen3_vl: short read from " + path);
  return values;
}

std::size_t checkedMul(std::size_t a, std::size_t b, const char *what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::runtime_error(std::string("qwen3_vl: size overflow: ") + what);
  return a * b;
}

void *openLocal(const std::string &path) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("qwen3_vl: dlopen " + path + ": " + dlerror());
  return handle;
}

template <typename Fn>
Fn lookup(void *handle, const std::string &path, const char *symbol) {
  dlerror();
  void *raw = dlsym(handle, symbol);
  const char *error = dlerror();
  if (error)
    throw std::runtime_error("qwen3_vl: missing " + std::string(symbol) +
                             " in " + path + ": " + error);
  return reinterpret_cast<Fn>(raw);
}

std::string nextId() {
  static std::atomic<unsigned long long> id{1};
  return "qwen-cmpl-" + std::to_string(id.fetch_add(1));
}

std::string resourcePath(const ModelManifest &manifest, const char *name) {
  for (const auto &constant : manifest.constants)
    if (constant.name == name)
      return constant.path;
  throw std::runtime_error("qwen3_vl: manifest missing constant " +
                           std::string(name));
}

std::string codePath(const ModelManifest &manifest, const char *name) {
  for (const auto &code : manifest.codeObjects)
    if (code.name == name)
      return code.path;
  throw std::runtime_error("qwen3_vl: manifest missing code object " +
                           std::string(name));
}

void validateGreedyServing(const SamplingParams &sampling) {
  const auto &provided = sampling.explicitParameters;
  const auto &config = sampling.samplerConfig;
  for (const auto &field :
       {std::make_pair(provided.temperature || config.temperature != 0,
                       "temperature"),
        std::make_pair(provided.topK || config.topK != 0, "top_k"),
        std::make_pair(provided.topP || config.topP != 1, "top_p"),
        std::make_pair(provided.minP || config.minP != 0, "min_p"),
        std::make_pair(provided.repeatPenalty || config.repeatPenalty != 1,
                       "repeat_penalty"),
        std::make_pair(provided.repeatLastN || config.repeatLastN != 64,
                       "repeat_last_n"),
        std::make_pair(provided.seed || config.seed != 0 ||
                           sampling.seedPolicy == SamplingSeedPolicy::Fixed ||
                           sampling.fixedSeed != 0,
                       "seed")})
    if (field.first)
      throw std::invalid_argument(std::string(field.second) +
                                  ": unsupported by qwen3_vl greedy serving");
}

std::string servingImagePath(const std::vector<ImageInput> &images) {
  if (images.empty())
    return "bundled";
  if (images.size() != 1)
    throw std::invalid_argument("qwen3_vl: only one image is supported");
  const auto &image = images.front();
  if (!image.bytes.empty())
    throw std::invalid_argument(
        "qwen3_vl: in-memory image bytes are not supported");
  if (image.uri.empty() || image.uri.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "qwen3_vl: image path is empty or contains NUL");
  if (image.uri == "bundled")
    return image.uri;

  std::string path = image.uri;
  if (path.rfind("file:", 0) == 0) {
    path.erase(0, 5);
    if (path.rfind("//", 0) == 0) {
      if (path.size() < 3 || path[2] != '/')
        throw std::invalid_argument(
            "qwen3_vl: image file URI must not have a host");
      path.erase(0, 2);
    }
  }
  const auto colon = path.find(':');
  const auto slash = path.find('/');
  if (path.empty() || (colon != std::string::npos &&
                       (slash == std::string::npos || colon < slash)))
    throw std::invalid_argument(
        "qwen3_vl: only local image paths and file: URIs are supported");
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error)
    throw std::invalid_argument("qwen3_vl: image path is not a regular file");
  const auto bytes = fs::file_size(path, error);
  if (error || bytes > 64u * 1024u * 1024u)
    throw std::invalid_argument(
        "qwen3_vl: image file is unreadable or too large");
  const auto absolute = fs::absolute(path, error);
  if (error)
    throw std::invalid_argument("qwen3_vl: cannot resolve local image path");
  return absolute.string();
}

} // namespace

class Qwen3VLRuntime::Impl {
public:
  ~Impl() {
    if (visionHandle)
      dlclose(visionHandle);
    if (decoderHandle)
      dlclose(decoderHandle);
  }

  TokenizeResult tokenize(const std::string &prompt, bool countOnly) const;
  CompletionResult generate(const std::string &prompt,
                            const std::vector<ImageInput> &images,
                            const SamplingParams &sampling,
                            const CompletionStreamCallback &callback);
  CompletionResult generateServing(const std::string &prompt,
                                   const std::vector<ImageInput> &images,
                                   const SamplingParams &sampling,
                                   const CompletionStreamCallback &callback);

  void load(const std::string &raxPath) {
    if (raxPath.empty())
      throw std::runtime_error("qwen3_vl: raxPath is required");
    if (loadAttempted)
      throw std::runtime_error("qwen3_vl: runtime load was already attempted");
    loadAttempted = true;
    if (visionHandle || decoderHandle)
      throw std::runtime_error("qwen3_vl: runtime is already loaded");

    manifest = ModelManifest::loadFromRax(raxPath);
    modelName = manifest.modelName.empty() ? "qwen3_vl" : manifest.modelName;
    vocabPath = manifest.vocabPath;
    if (vocabPath.empty())
      throw std::runtime_error("qwen3_vl: manifest has no vocab_uri");

    mapFloats(resourcePath(manifest, "vision_weights"), visionWeights);
    mapFloats(resourcePath(manifest, "decoder_weights"), decoderWeights);
    mapFloats(resourcePath(manifest, "embed_table"), embedTable);
    mapI64(resourcePath(manifest, "img_pos"), imagePositions);
    cos = readFloats(resourcePath(manifest, "cos"), 0);
    sin = readFloats(resourcePath(manifest, "sin"), 0);
    causalMask = readFloats(resourcePath(manifest, "cmask"), 0);

    std::ifstream meta(resourcePath(manifest, "meta"));
    if (!meta)
      throw std::runtime_error("qwen3_vl: cannot read meta resource");
    std::size_t ignoredPrompt = 0;
    if (!(meta >> ignoredPrompt >> N >> NIMG >> HID >> VOCAB))
      throw std::runtime_error("qwen3_vl: invalid meta resource");
    packagedPromptLength = ignoredPrompt;
    if (N == 0 ||
        N > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        NIMG != 98 || HID != 2048 || VOCAB == 0 ||
        VOCAB > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("qwen3_vl: unsupported dimensions in meta");
    if (packagedPromptLength < NIMG + 2 || packagedPromptLength > N)
      throw std::runtime_error("qwen3_vl: invalid packaged prompt length");
    for (std::size_t elements :
         {checkedMul(N, HID, "inputs_embeds"),
          checkedMul(NIMG, HID, "vision outputs"),
          checkedMul(N, VOCAB, "logits"), checkedMul(N, N, "causal mask")})
      (void)checkedMul(elements, sizeof(float), "float buffer bytes");
    if (visionWeights.count >
            static_cast<std::size_t>(std::numeric_limits<long>::max()) ||
        decoderWeights.count >
            static_cast<std::size_t>(std::numeric_limits<long>::max()))
      throw std::runtime_error("qwen3_vl: weight count exceeds kernel ABI");
    if (imagePositions.count != NIMG)
      throw std::runtime_error("qwen3_vl: img_pos count does not match meta");
    if (cos.size() != checkedMul(N, kHeadDim, "cos") ||
        sin.size() != checkedMul(N, kHeadDim, "sin") ||
        causalMask.size() != checkedMul(N, N, "cmask"))
      throw std::runtime_error("qwen3_vl: positional resource shape mismatch");
    if (embedTable.count < checkedMul(VOCAB, HID, "embed_table"))
      throw std::runtime_error("qwen3_vl: embed table is too small");
    for (std::size_t i = 0; i < imagePositions.count; ++i)
      if (imagePositions.data[i] < 0 ||
          static_cast<std::size_t>(imagePositions.data[i]) >=
              packagedPromptLength ||
          (i != 0 && imagePositions.data[i] <= imagePositions.data[i - 1]))
        throw std::runtime_error("qwen3_vl: image position is out of range");

    rawCodec = std::make_unique<Qwen3VLRawCodec>(vocabPath);
    if (rawCodec->vocabSize() <= 0 ||
        static_cast<std::size_t>(rawCodec->vocabSize()) > VOCAB)
      throw std::runtime_error("qwen3_vl: vocabulary exceeds decoder logits");

    const std::string visionPath = codePath(manifest, "vision_kernels");
    const std::string decoderPath = codePath(manifest, "decoder_kernels");
    try {
      visionHandle = openLocal(visionPath);
      decoderHandle = openLocal(decoderPath);
      vision = lookup<VisionFn>(visionHandle, visionPath, "qwen3vl_vision");
      decoder =
          lookup<DecoderFn>(decoderHandle, decoderPath, "qwen3vl_decoder");
    } catch (...) {
      if (decoderHandle) {
        dlclose(decoderHandle);
        decoderHandle = nullptr;
      }
      if (visionHandle) {
        dlclose(visionHandle);
        visionHandle = nullptr;
      }
      throw;
    }
    loaded = true;
  }

  static constexpr std::size_t kHeadDim = 128;
  ModelManifest manifest;
  std::string modelName;
  std::string vocabPath;
  std::unique_ptr<Qwen3VLRawCodec> rawCodec;
  std::size_t packagedPromptLength = 0;
  std::size_t N = 0, NIMG = 0, HID = 0, VOCAB = 0;
  MappedFloats visionWeights, decoderWeights, embedTable;
  MappedI64 imagePositions;
  std::vector<float> cos, sin, causalMask;
  void *visionHandle = nullptr;
  void *decoderHandle = nullptr;
  VisionFn vision = nullptr;
  DecoderFn decoder = nullptr;
  mutable std::mutex kernelMutex;
  bool loadAttempted = false;
  bool loaded = false;
};

TokenizeResult Qwen3VLRuntime::Impl::tokenize(const std::string &prompt,
                                              bool countOnly) const {
  if (!loaded)
    throw std::runtime_error("qwen3_vl: runtime is not loaded");
  Text<size_t, 2> tokens(prompt);
  tokens.tokenizeQwen3VL(vocabPath, N, NIMG);
  TokenizeResult result;
  result.count = tokens.getTokenCnt();
  if (!countOnly) {
    result.tokens.reserve(result.count);
    for (std::size_t i = 0; i < result.count; ++i)
      result.tokens.push_back(static_cast<int>(tokens.getData()[i]));
  }
  return result;
}

CompletionResult Qwen3VLRuntime::Impl::generate(
    const std::string &prompt, const std::vector<ImageInput> &images,
    const SamplingParams &sampling, const CompletionStreamCallback &callback) {
  if (!loaded)
    throw std::runtime_error("qwen3_vl: runtime is not loaded");
  if (images.size() > 1)
    throw std::runtime_error("qwen3_vl: only one image is supported");
  if (!images.empty() && !images.front().bytes.empty())
    throw std::runtime_error(
        "qwen3_vl: in-memory image bytes are not supported");

  CompletionResult result;
  result.id = nextId();
  result.model = modelName;

  Text<size_t, 2> promptTokens(prompt);
  promptTokens.tokenizeQwen3VL(vocabPath, N, NIMG);
  const std::size_t promptCount = promptTokens.getTokenCnt();
  if (promptCount == 0 || promptCount > N)
    throw std::runtime_error("qwen3_vl: invalid tokenized prompt length");
  if (packagedPromptLength != 0 && promptCount != packagedPromptLength)
    throw std::runtime_error("qwen3_vl: prompt token length " +
                             std::to_string(promptCount) +
                             " differs from packaged static RoPE length " +
                             std::to_string(packagedPromptLength) +
                             "; rebuild positional constants for this prompt");

  std::vector<float> pixel;
  const bool bundled = images.empty() || images.front().uri.empty() ||
                       images.front().uri == "bundled";
  const auto prefillStart = std::chrono::high_resolution_clock::now();
  if (bundled) {
    pixel = readFloats(resourcePath(manifest, "pixel_values"), 392u * 1536u);
  } else {
    std::error_code fileError;
    const auto fileBytes = fs::file_size(images.front().uri, fileError);
    if (!fileError && fileBytes > 64u * 1024u * 1024u)
      throw std::runtime_error("qwen3_vl: image file is too large");
    qwen3vl_image::preprocessImage(images.front().uri, pixel);
  }
  if (pixel.size() != 392u * 1536u)
    throw std::runtime_error("qwen3_vl: pixel_values has unexpected size");

  std::vector<float> pooled(NIMG * HID);
  std::vector<float> ds0(NIMG * HID), ds1(NIMG * HID), ds2(NIMG * HID);
  {
    std::lock_guard<std::mutex> lock(kernelMutex);
    vision(visionWeights.data, static_cast<long>(visionWeights.count),
           pixel.data(), ds0.data(), ds1.data(), ds2.data(), pooled.data());
  }
  result.timings.prefillMs =
      std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - prefillStart)
          .count();

  std::vector<int64_t> inputIds(promptCount);
  for (std::size_t i = 0; i < promptCount; ++i)
    inputIds[i] = static_cast<int64_t>(promptTokens.getData()[i]);
  auto embedRow = [&](int64_t token) {
    if (token < 0 || static_cast<std::size_t>(token) >= VOCAB)
      throw std::runtime_error("qwen3_vl: prompt token is out of range");
    return embedTable.data + static_cast<std::size_t>(token) * HID;
  };
  std::vector<float> inputs(N * HID, 0.0f);
  for (std::size_t i = 0; i < promptCount; ++i)
    std::copy(embedRow(inputIds[i]), embedRow(inputIds[i]) + HID,
              inputs.begin() + i * HID);
  for (std::size_t i = 0; i < imagePositions.count; ++i) {
    const std::size_t off =
        static_cast<std::size_t>(imagePositions.data[i]) * HID;
    std::copy(pooled.begin() + i * HID, pooled.begin() + (i + 1) * HID,
              inputs.begin() + off);
  }
  std::vector<float> full0(N * HID, 0.0f), full1(N * HID, 0.0f),
      full2(N * HID, 0.0f);
  for (std::size_t i = 0; i < imagePositions.count; ++i) {
    const std::size_t off =
        static_cast<std::size_t>(imagePositions.data[i]) * HID;
    std::copy(ds0.begin() + i * HID, ds0.begin() + (i + 1) * HID,
              full0.begin() + off);
    std::copy(ds1.begin() + i * HID, ds1.begin() + (i + 1) * HID,
              full1.begin() + off);
    std::copy(ds2.begin() + i * HID, ds2.begin() + (i + 1) * HID,
              full2.begin() + off);
  }

  Text<size_t, 2> output;
  output.loadVocab(vocabPath);
  std::vector<float> logits(N * VOCAB);
  const std::size_t contextNew = N - promptCount;
  const std::size_t maxNew =
      sampling.maxTokens <= 0
          ? contextNew
          : (sampling.maxTokens > static_cast<int>(promptCount)
                 ? std::min<std::size_t>(contextNew, static_cast<std::size_t>(
                                                         sampling.maxTokens) -
                                                         promptCount)
                 : 0);
  bool stopped = false;
  bool cancelled = false;
  std::size_t generated = 0;
  double decodeMs = 0.0;
  std::string streamedText;
  for (std::size_t t = promptCount - 1; t + 1 < N && generated < maxNew; ++t) {
    const auto stepStart = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard<std::mutex> lock(kernelMutex);
      decoder(decoderWeights.data, static_cast<long>(decoderWeights.count),
              inputs.data(), cos.data(), sin.data(), causalMask.data(),
              full0.data(), full1.data(), full2.data(), logits.data(),
              static_cast<long>(N), static_cast<long>(VOCAB),
              static_cast<long>(HID), static_cast<long>(kHeadDim));
    }
    decodeMs += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - stepStart)
                    .count();
    const float *row = logits.data() + t * VOCAB;
    int token = 0;
    for (std::size_t j = 1; j < VOCAB; ++j)
      if (row[j] > row[token])
        token = static_cast<int>(j);
    if (token == 151645 || token == 151643 ||
        std::find(sampling.stopTokenIds.begin(), sampling.stopTokenIds.end(),
                  static_cast<long long>(token)) !=
            sampling.stopTokenIds.end()) {
      stopped = true;
      break;
    }
    output.appendTokenIdx(static_cast<std::size_t>(token));
    ++generated;
    const std::string current = output.revertQwen3();
    std::string delta;
    if (current.size() > streamedText.size())
      delta.assign(current.data() + streamedText.size(),
                   current.size() - streamedText.size());
    streamedText = current;
    if (callback) {
      CompletionChunk chunk;
      chunk.id = result.id;
      chunk.model = result.model;
      chunk.delta = std::move(delta);
      chunk.tokenId = token;
      if (!callback(chunk)) {
        cancelled = true;
        break;
      }
    }
    std::copy(embedRow(token), embedRow(token) + HID,
              inputs.begin() + (t + 1) * HID);
  }
  result.content = output.revertQwen3();
  result.usage.promptTokens = static_cast<int>(promptCount);
  result.usage.completionTokens = static_cast<int>(generated);
  result.usage.totalTokens =
      result.usage.promptTokens + result.usage.completionTokens;
  result.timings.decodeMs = decodeMs;
  result.timings.tokensPerSecond =
      decodeMs > 0.0 ? static_cast<double>(generated) / (decodeMs / 1000.0)
                     : 0.0;
  if (cancelled)
    result.finishReason = FinishReason::Cancelled;
  else if (stopped)
    result.finishReason = FinishReason::Stop;
  else
    result.finishReason = FinishReason::Length;
  return result;
}

CompletionResult Qwen3VLRuntime::Impl::generateServing(
    const std::string &prompt, const std::vector<ImageInput> &images,
    const SamplingParams &sampling, const CompletionStreamCallback &callback) {
  std::lock_guard<std::mutex> requestLock(kernelMutex);
  if (!loaded || !rawCodec)
    throw std::runtime_error("qwen3_vl: runtime is not loaded");
  validateGreedyServing(sampling);
  const std::string imagePath = servingImagePath(images);
  const auto imagePad = rawCodec->tokenId("<|image_pad|>");
  const auto visionStart = rawCodec->tokenId("<|vision_start|>");
  const auto visionEnd = rawCodec->tokenId("<|vision_end|>");
  const auto visionPad = rawCodec->tokenId("<|vision_pad|>");
  const auto videoPad = rawCodec->tokenId("<|video_pad|>");
  if (!imagePad || !visionStart || !visionEnd)
    throw std::invalid_argument(
        "qwen3_vl: vocabulary lacks the required image markers");
  if (prompt.find("<|video_pad|>") != std::string::npos ||
      prompt.find("<|vision_pad|>") != std::string::npos)
    throw std::invalid_argument(
        "qwen3_vl: video and generic vision markers are not supported");

  SamplingParams params = sampling;
  for (const char *spelling : {"<|im_end|>", "<|endoftext|>"})
    if (auto token = rawCodec->tokenId(spelling))
      params.stopTokenIds.push_back(*token);

  std::vector<float> pixel, pooled, ds0, ds1, ds2;
  std::vector<float> inputs, full0, full1, full2, logits;
  std::size_t outputPosition = 0;
  bool prefilled = false;
  const auto resetRequest = [&] {
    pixel.clear();
    pooled.clear();
    ds0.clear();
    ds1.clear();
    ds2.clear();
    inputs.clear();
    full0.clear();
    full1.clear();
    full2.clear();
    logits.clear();
    outputPosition = 0;
    prefilled = false;
  };
  const auto embedRow = [&](int token) {
    if (token < 0 || static_cast<std::size_t>(token) >= VOCAB)
      throw std::runtime_error("qwen3_vl: token embedding is out of range");
    return embedTable.data + static_cast<std::size_t>(token) * HID;
  };
  const auto runDecoder = [&] {
    decoder(decoderWeights.data, static_cast<long>(decoderWeights.count),
            inputs.data(), cos.data(), sin.data(), causalMask.data(),
            full0.data(), full1.data(), full2.data(), logits.data(),
            static_cast<long>(N), static_cast<long>(VOCAB),
            static_cast<long>(HID), static_cast<long>(kHeadDim));
  };

  ServingTextCodec codec;
  codec.contextLength = static_cast<int>(N);
  codec.vocabSize = rawCodec->vocabSize();
  codec.encode = [&](const std::string &text) {
    return rawCodec->encode(text, false, true);
  };
  codec.tokenBytes = [&](int token) { return rawCodec->tokenBytes(token); };

  ServingTokenSource source;
  source.reset = resetRequest;
  source.validToken = [&](int token) {
    if (token < 0 || token >= codec.vocabSize)
      return false;
    try {
      (void)rawCodec->tokenBytes(token);
      return true;
    } catch (const std::invalid_argument &) {
      return false;
    }
  };
  source.prefill = [&](const std::vector<int> &tokens) {
    if (tokens.size() != packagedPromptLength)
      throw std::invalid_argument(
          "qwen3_vl: prompt token length differs from packaged static RoPE "
          "length; rebuild positional constants for this prompt");
    std::vector<std::size_t> actualImagePositions;
    std::size_t starts = 0, ends = 0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const int token = tokens[i];
      if ((visionPad && token == *visionPad) ||
          (videoPad && token == *videoPad))
        throw std::invalid_argument(
            "qwen3_vl: video and generic vision markers are not supported");
      starts += token == *visionStart;
      ends += token == *visionEnd;
      if (token == *imagePad)
        actualImagePositions.push_back(i);
    }
    if (starts != 1 || ends != 1 || actualImagePositions.size() != NIMG)
      throw std::invalid_argument(
          "qwen3_vl: prompt requires exactly one complete fixed-grid image "
          "block");
    const std::size_t firstImage = actualImagePositions.front();
    const std::size_t lastImage = actualImagePositions.back();
    if (firstImage == 0 || lastImage + 1 >= tokens.size() ||
        tokens[firstImage - 1] != *visionStart ||
        tokens[lastImage + 1] != *visionEnd)
      throw std::invalid_argument(
          "qwen3_vl: image block must be enclosed by vision_start/vision_end");
    for (std::size_t i = 0; i < NIMG; ++i)
      if (actualImagePositions[i] != firstImage + i ||
          actualImagePositions[i] !=
              static_cast<std::size_t>(imagePositions.data[i]))
        throw std::invalid_argument(
            "qwen3_vl: image block differs from packaged image positions");

    if (imagePath == "bundled")
      pixel = readFloats(resourcePath(manifest, "pixel_values"), 392u * 1536u);
    else {
      try {
        qwen3vl_image::preprocessImage(imagePath, pixel);
      } catch (const std::runtime_error &error) {
        throw std::invalid_argument(error.what());
      }
    }
    if (pixel.size() != 392u * 1536u)
      throw std::runtime_error("qwen3_vl: pixel_values has unexpected size");
    const std::size_t visionElements = checkedMul(NIMG, HID, "vision outputs");
    pooled.resize(visionElements);
    ds0.resize(visionElements);
    ds1.resize(visionElements);
    ds2.resize(visionElements);
    vision(visionWeights.data, static_cast<long>(visionWeights.count),
           pixel.data(), ds0.data(), ds1.data(), ds2.data(), pooled.data());

    const std::size_t embeddingElements = checkedMul(N, HID, "inputs_embeds");
    inputs.assign(embeddingElements, 0.0f);
    full0.assign(embeddingElements, 0.0f);
    full1.assign(embeddingElements, 0.0f);
    full2.assign(embeddingElements, 0.0f);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const float *row = embedRow(tokens[i]);
      std::copy(row, row + HID, inputs.begin() + i * HID);
    }
    for (std::size_t i = 0; i < NIMG; ++i) {
      const std::size_t offset = actualImagePositions[i] * HID;
      std::copy(pooled.begin() + i * HID, pooled.begin() + (i + 1) * HID,
                inputs.begin() + offset);
      std::copy(ds0.begin() + i * HID, ds0.begin() + (i + 1) * HID,
                full0.begin() + offset);
      std::copy(ds1.begin() + i * HID, ds1.begin() + (i + 1) * HID,
                full1.begin() + offset);
      std::copy(ds2.begin() + i * HID, ds2.begin() + (i + 1) * HID,
                full2.begin() + offset);
    }
    logits.resize(checkedMul(N, VOCAB, "logits"));
    outputPosition = tokens.size() - 1;
    // The compiled decoder has no KV ABI: preserve its full-sequence
    // computation, including for prefill-only and full-context requests.
    runDecoder();
    prefilled = true;
  };
  source.decode = [&](int token) {
    if (!prefilled || outputPosition + 1 >= N)
      throw std::runtime_error("qwen3_vl: decode exceeds the context");
    ++outputPosition;
    const float *row = embedRow(token);
    std::copy(row, row + HID, inputs.begin() + outputPosition * HID);
    runDecoder();
  };
  source.selectToken = [&](const std::vector<int> &) {
    if (!prefilled || outputPosition >= N)
      throw std::runtime_error("qwen3_vl: no decoder output to select a token");
    const float *row = logits.data() + outputPosition * VOCAB;
    int token = 0;
    // The compiled LM head is padded beyond the packaged vocabulary. Keep
    // the legacy greedy tie order while excluding non-decodable padding.
    for (int i = 1; i < codec.vocabSize; ++i)
      if (row[i] > row[token])
        token = i;
    return token;
  };

  const std::string id = nextId();
  CompletionStreamCallback wrapped;
  if (callback)
    wrapped = [&](const CompletionChunk &chunk) {
      CompletionChunk event = chunk;
      event.id = id;
      event.model = modelName;
      return callback(event);
    };
  CompletionResult result =
      runServingGeneration(prompt, source, params, codec, wrapped);
  resetRequest();
  result.id = id;
  result.model = modelName;
  return result;
}

Qwen3VLRuntime::Qwen3VLRuntime() : impl(std::make_unique<Impl>()) {}
Qwen3VLRuntime::~Qwen3VLRuntime() = default;

void Qwen3VLRuntime::load(const std::string &raxPath) { impl->load(raxPath); }
bool Qwen3VLRuntime::isLoaded() const { return impl->loaded; }
int Qwen3VLRuntime::contextLength() const { return static_cast<int>(impl->N); }
std::size_t Qwen3VLRuntime::imageTokenCount() const { return impl->NIMG; }
const std::string &Qwen3VLRuntime::modelName() const { return impl->modelName; }

ResidentModelCapabilities Qwen3VLRuntime::capabilities() const {
  ResidentModelCapabilities result;
  if (!impl->loaded || !impl->rawCodec)
    return result;
  result.rawTokenize = result.rawDetokenize = true;
  result.tokenizeAddSpecial = result.tokenizeParseSpecial = true;
  result.tokenizeWithPieces = true;
  if (!impl->rawCodec->tokenId("<|image_pad|>") ||
      !impl->rawCodec->tokenId("<|vision_start|>") ||
      !impl->rawCodec->tokenId("<|vision_end|>"))
    return result;
  result.chatRoles = {"user"};
  result.maxImages = 1;
  result.imagePaths = true;
  result.streaming = result.maxNewTokens = true;
  result.stop = result.stopTokenIds = true;
  return result;
}

TokenizeResult Qwen3VLRuntime::tokenize(const std::string &prompt,
                                        bool countOnly) const {
  return impl->tokenize(prompt, countOnly);
}

TokenizeResult Qwen3VLRuntime::tokenize(const TokenizeRequest &request) const {
  if (!impl->loaded || !impl->rawCodec)
    throw std::runtime_error("qwen3_vl: runtime is not loaded");
  return impl->rawCodec->tokenize(request);
}

DetokenizeResult
Qwen3VLRuntime::detokenize(const DetokenizeRequest &request) const {
  if (!impl->loaded || !impl->rawCodec)
    throw std::runtime_error("qwen3_vl: runtime is not loaded");
  return impl->rawCodec->detokenize(request);
}

CompletionResult Qwen3VLRuntime::generate(
    const std::string &prompt, const std::vector<ImageInput> &images,
    const SamplingParams &sampling, const CompletionStreamCallback &callback) {
  return impl->generate(prompt, images, sampling, callback);
}

CompletionResult Qwen3VLRuntime::generateServing(
    const std::string &prompt, const std::vector<ImageInput> &images,
    const SamplingParams &sampling, const CompletionStreamCallback &callback) {
  return impl->generateServing(prompt, images, sampling, callback);
}

} // namespace runtime
} // namespace buddy
