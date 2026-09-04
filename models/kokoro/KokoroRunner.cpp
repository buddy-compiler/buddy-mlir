//===- KokoroRunner.cpp - Kokoro-82M text-to-speech runner --------------===//
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
// Kokoro-82M text-to-speech runner.
//
// Loads the compiled `kokoro_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), then invokes the AOT forward graph.
//
// ---------------------------------------------------------------------------
// FORWARD ABI (single result, then input memrefs in declaration order)
// ---------------------------------------------------------------------------
// The C ABI wrapper `_mlir_ciface_forward` exported by kokoro_model.so is
//
//   void _mlir_ciface_forward(
//       MemRef<float, 2> *waveform,   // result  1 x audio_buffer_size x f32
//       MemRef<float, 1> *weights,    // input   1 x params_size      x f32
//       MemRef<int64_t, 2> *inputIds, // input   1 x max_seq_len      x i64
//       MemRef<float, 2> *refS,       // input   1 x 256              x f32
//       MemRef<float, 1> *speed);     // input   1 x 1                x f32
//
// One pointer is passed per result memref first (here a single waveform
// result), then one pointer per input memref in declaration order. The
// flattened weights (`arg0.data`) are the first input memref.
//
// MemRef<T, N> layout (buddy/Core/Container.h):
//   1-D: {allocated, aligned, offset, size0, stride0}
//   2-D: {allocated, aligned, offset, size0, stride0, size1, stride1}
// Outputs are built with needMalloc=false (the kernel allocates and fills the
// buffer, which the caller later frees via release()).
//
// ---------------------------------------------------------------------------
//
// Input construction:
//   * input_ids: in the real pipeline the prompt is phonemized (espeak-ng +
//     misaki G2P) then packed as [0, <phonemes>, 0] with the 178-token phoneme
//     vocab from config.json. Phonemization is not implemented in this C++
//     runner, so a fixed deterministic input_ids buffer is used instead (and
//     documented as such).
//   * ref_s: a fixed 256-dim reference speaker embedding (real value would come
//     from a `voices/*.pt` style vector); a deterministic placeholder is used.
//   * speed: 1.0 (the trace is baked for speed=1.0).
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/KokoroRunner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 30;    // traced input_ids length
constexpr size_t kDefaultStyleDim = 128;    // ref_s = 2 * style_dim = 256
constexpr size_t kDefaultMaxDur = 50;       // max per-token phoneme duration
constexpr size_t kDefaultUpsampleFactor = 300; // prod(upsample_rates)*istft_hop
constexpr size_t kDefaultAudioBufferSize = 450000; // 30 * 50 * 300
constexpr size_t kDefaultParamsSize = 81810022;    // total Kokoro-82M params

// Single result, then one input memref pointer per AOT input, in declaration
// order: weights, input_ids, ref_s, speed.
using ForwardFn = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<float, 2> *,
                           MemRef<float, 1> *);

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

size_t parseSizeAttr(const ModelManifest &manifest, const char *key,
                     size_t fallback) {
  auto it = manifest.moduleAttrs.find(key);
  if (it == manifest.moduleAttrs.end() || it->second.empty())
    return fallback;
  return static_cast<size_t>(std::stoull(it->second));
}

std::string findConstantPath(const ModelManifest &manifest,
                             const std::string &name) {
  for (const auto &constant : manifest.constants) {
    if (constant.name == name)
      return constant.path;
  }
  return "";
}

void loadWeights(const std::string &weightsPath, MemRef<float, 1> &params) {
  std::ifstream paramFile(weightsPath, std::ios::in | std::ios::binary);
  if (!paramFile)
    throw std::runtime_error("KokoroRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("KokoroRunner: error reading weights: " +
                             weightsPath);
}

/// Fill `ids` (already sized maxSeqLen) with a fixed deterministic phoneme-id
/// sequence. The real pipeline would phonemize the prompt with espeak-ng /
/// misaki G2P and wrap it as [0, <phonemes>, 0]; that is not implemented here,
/// so a reproducible placeholder in the 178-token vocab range is used.
void fillDeterministicInputIds(int64_t *ids, size_t maxSeqLen) {
  if (maxSeqLen == 0)
    return;
  for (size_t i = 0; i < maxSeqLen; ++i)
    ids[i] = 16 + static_cast<int64_t>((i * 17) % 160); // vocab range [16, 175]
  ids[0] = 0;               // BOS token
  ids[maxSeqLen - 1] = 0;   // EOS token
}

/// Fill `refS` (2 * styleDim floats, i.e. a (1, 256) row) with a fixed
/// deterministic reference speaker embedding. A real value would come from a
/// `voices/<voice>.pt` style vector; the placeholder is reproducible so that
/// repeated runs are bit-identical.
void fillDeterministicRefS(float *refS, size_t styleDim) {
  const size_t n = 2 * styleDim;
  for (size_t i = 0; i < n; ++i) {
    // Deterministic small-amplitude pseudo-random values.
    const uint32_t x =
        static_cast<uint32_t>(i * 2654435761u) ^ 0x9E3779B9u;
    const float v = static_cast<float>(x & 0xFFFFu) / 32768.0f - 1.0f;
    refS[i] = 0.05f * v;
  }
}

} // namespace

void KokoroRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t styleDim = kDefaultStyleDim;
  size_t audioBufferSize = kDefaultAudioBufferSize;
  size_t paramsSize = kDefaultParamsSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("KokoroRunner: manifest has no weight file");
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    styleDim = parseSizeAttr(manifest, "style_dim", styleDim);
    audioBufferSize =
        parseSizeAttr(manifest, "audio_buffer_size", audioBufferSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("KokoroRunner: legacy mode requires "
                               "--model-so and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("seq_len   : " + std::to_string(maxSeqLen), suppress);
  printLog("ref_s dim : " + std::to_string(2 * styleDim), suppress);
  printLog("audio buf : " + std::to_string(audioBufferSize), suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("KokoroRunner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "KokoroRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("KokoroRunner: weight file is not f32-aligned");
  }
  // Sanity-check the staged arg0.data size against the spec.
  paramsSize = weightBytes / sizeof(float);
  printLog("Loading weights (" + std::to_string(paramsSize) + " floats)",
           suppress);
  MemRef<float, 1> weightsContainer({paramsSize});
  loadWeights(weightsPath, weightsContainer);
  printLog("Weights loaded", suppress);

  // input_ids: fixed deterministic 30-token placeholder (see header comment).
  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  fillDeterministicInputIds(inputIds.getData(), maxSeqLen);

  // ref_s: fixed deterministic 256-dim reference speaker embedding.
  MemRef<float, 2> refS({1, 2 * styleDim});
  fillDeterministicRefS(refS.getData(), styleDim);

  // speed: 1.0 (the trace is baked for speed=1.0).
  MemRef<float, 1> speed({1});
  speed.getData()[0] = 1.0f;

  // Output waveform buffer: the kernel allocates and fills this (needMalloc=false).
  MemRef<float, 2> waveform({1, audioBufferSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&waveform, &weightsContainer, &inputIds, &refS, &speed);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[35;1mKokoro-82M TTS Inference\033[0m\n";
    std::cerr << "  time      : " << seconds << "s\n";
    std::cerr << "  samples   : " << audioBufferSize << "\n";
  }

  // Emit the generated waveform samples in a simple, deterministic form:
  // a summary plus the first 8 samples. The compiled kernel fills the fixed
  // audio_buffer_size buffer; the true voiced length is data-dependent and is
  // not recovered here.
  const float *data = waveform.getData();
  std::cout << "{\n";
  std::cout << "  \"sample_count\": " << audioBufferSize << ",\n";
  std::cout << "  \"first_samples\": [";
  for (size_t i = 0; i < std::min<size_t>(8, audioBufferSize); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "],\n";
  if (audioBufferSize > 0) {
    double sum = 0.0;
    float mn = data[0], mx = data[0];
    for (size_t i = 0; i < audioBufferSize; ++i) {
      const float v = data[i];
      sum += v;
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    const double mean = sum / static_cast<double>(audioBufferSize);
    double sq = 0.0;
    for (size_t i = 0; i < audioBufferSize; ++i) {
      const double d = static_cast<double>(data[i]) - mean;
      sq += d * d;
    }
    const double rms = std::sqrt(sq / static_cast<double>(audioBufferSize));
    std::cout << "  \"min\": " << mn << ",\n";
    std::cout << "  \"max\": " << mx << ",\n";
    std::cout << "  \"mean\": " << mean << ",\n";
    std::cout << "  \"rms\": " << rms << "\n";
  } else {
    std::cout << "  \"min\": 0.0,\n  \"max\": 0.0,\n  \"mean\": 0.0,\n"
                 "  \"rms\": 0.0\n";
  }
  std::cout << "}\n";

  free(waveform.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
