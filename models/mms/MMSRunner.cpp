//===- MMSRunner.cpp - MMS (VITS) vocoder runner --------------------------===//
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
// MMS (facebook/mms-tts-eng) runner.
//
// The compiled kernel is the HiFi-GAN vocoder stage of VITS:
//
//   forward(weights: memref<params_size x f32>,
//           latents:  memref<1 x flow_size x max_seq_len x f32>)
//     -> waveform:   memref<1 x audio_buffer_size x f32>
//
// Text encoding, data-dependent duration alignment and flow decoding remain
// outside this compiled stage. The runner takes the resulting latent
// spectrogram. See README.md for input generation and validation.
//
// The latent input is a raw f32 blob (1 x flow_size x max_seq_len) read from
// --prompt; when --prompt is empty, a deterministic all-zero latent is used.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/MMSRunner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 30;
constexpr size_t kDefaultFlowSize = 192;
constexpr size_t kDefaultAudioRate = 256; // prod(upsample_rates) = 8*8*2*2
constexpr size_t kDefaultParamsSize = 14327424;

using ForwardFn = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                           MemRef<float, 3> *);

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

size_t parseSizeAttr(const ModelManifest &manifest, const char *key) {
  auto it = manifest.moduleAttrs.find(key);
  if (it == manifest.moduleAttrs.end() || it->second.empty())
    throw std::runtime_error(
        std::string("MMSRunner: missing manifest attribute: ") + key);
  size_t value = 0;
  const auto &text = it->second;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size() ||
      value == 0 ||
      value > static_cast<size_t>(std::numeric_limits<intptr_t>::max()))
    throw std::runtime_error(
        std::string("MMSRunner: invalid manifest attribute: ") + key);
  return value;
}

size_t checkedProduct(size_t lhs, size_t rhs) {
  const auto maxElements =
      static_cast<size_t>(std::numeric_limits<intptr_t>::max()) / sizeof(float);
  if (lhs > maxElements / rhs)
    throw std::runtime_error(
        "MMSRunner: tensor size exceeds addressable memory");
  return lhs * rhs;
}

std::string findConstantPath(const ModelManifest &manifest,
                             const std::string &name) {
  for (const auto &constant : manifest.constants)
    if (constant.name == name)
      return constant.path;
  return "";
}

void loadWeights(const std::string &weightsPath, MemRef<float, 1> &params) {
  std::ifstream paramFile(weightsPath, std::ios::in | std::ios::binary);
  if (!paramFile)
    throw std::runtime_error("MMSRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("MMSRunner: error reading weights: " +
                             weightsPath);
}

/// Load a raw f32 latent spectrogram blob (1 x flow_size x max_seq_len).
void loadLatents(const std::string &path, MemRef<float, 3> &latents) {
  const size_t expected = sizeof(float) * latents.getSize();
  if (std::filesystem::file_size(path) != expected)
    throw std::runtime_error(
        "MMSRunner: latent file size does not match 1 x flow_size x "
        "max_seq_len: " +
        path);
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in)
    throw std::runtime_error("MMSRunner: failed to open latent file: " + path);
  in.read(reinterpret_cast<char *>(latents.getData()), expected);
  if (!in)
    throw std::runtime_error("MMSRunner: error reading latent file: " + path);
  for (size_t i = 0; i < latents.getSize(); ++i)
    if (!std::isfinite(latents.getData()[i]))
      throw std::runtime_error(
          "MMSRunner: latent input contains non-finite values");
}

} // namespace

void MMSRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t flowSize = kDefaultFlowSize;
  size_t audioRate = kDefaultAudioRate;
  size_t paramsSize = kDefaultParamsSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("MMSRunner: manifest has no weight file");
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len");
    flowSize = parseSizeAttr(manifest, "flow_size");
    audioRate = parseSizeAttr(manifest, "audio_rate");
    paramsSize = parseSizeAttr(manifest, "params_size");
    if (parseSizeAttr(manifest, "audio_buffer_size") !=
        checkedProduct(maxSeqLen, audioRate))
      throw std::runtime_error(
          "MMSRunner: inconsistent waveform shape in manifest");
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error(
          "MMSRunner: legacy mode requires --model-so and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  const size_t audioBufferSize = checkedProduct(maxSeqLen, audioRate);
  checkedProduct(flowSize, maxSeqLen);
  checkedProduct(paramsSize, 1);
  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes != paramsSize * sizeof(float))
    throw std::runtime_error(
        "MMSRunner: weight file size does not match params_size");

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);

  printLog("Loading model shared library", suppress);
  std::unique_ptr<void, decltype(&dlclose)> handle(
      dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL), &dlclose);
  if (!handle)
    throw std::runtime_error("MMSRunner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle.get(), "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    throw std::runtime_error("MMSRunner: missing _mlir_ciface_forward in " +
                             soPath + ": " + std::string(err));
  }

  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  // Latent spectrogram input (1 x flow_size x max_seq_len).
  MemRef<float, 3> latents({1, flowSize, maxSeqLen});
  if (!cfg.prompt.empty()) {
    printLog("Loading latent spectrogram from --prompt", suppress);
    loadLatents(cfg.prompt, latents);
  } else {
    printLog("No latent provided; using all-zero latent", suppress);
    std::fill(latents.getData(), latents.getData() + latents.getSize(), 0.0f);
  }

  MemRef<float, 2> waveform({1, audioBufferSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&waveform, &paramsContainer, &latents);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  const float *data = waveform.getData();
  if (!data || waveform.getSizes()[0] != 1 ||
      waveform.getSizes()[1] != static_cast<intptr_t>(audioBufferSize))
    throw std::runtime_error(
        "MMSRunner: compiled waveform has an unexpected shape");

  if (!suppress) {
    std::cerr << "\033[33;1mMMS Vocoder Output\033[0m\n";
    std::cerr << "  seq_len        : " << maxSeqLen << "\n";
    std::cerr << "  flow_size      : " << flowSize << "\n";
    std::cerr << "  audio_buffer   : " << audioBufferSize << " samples\n";
    std::cerr << "  time           : " << seconds << "s\n";
  }

  // Emit a compact JSON summary of the generated waveform.
  double min = 0.0, max = 0.0, sum = 0.0, sumSq = 0.0;
  size_t n = waveform.getSize();
  if (n > 0) {
    min = max = data[0];
    for (size_t i = 0; i < n; ++i) {
      const float v = data[i];
      if (!std::isfinite(v))
        throw std::runtime_error(
            "MMSRunner: compiled waveform contains non-finite values");
      min = std::min(min, static_cast<double>(v));
      max = std::max(max, static_cast<double>(v));
      sum += v;
      sumSq += static_cast<double>(v) * v;
    }
  }
  const double mean = n ? sum / n : 0.0;
  const double rms = n ? std::sqrt(sumSq / n) : 0.0;

  std::cout << "{ \"n_samples\": " << n << ", \"min\": " << min
            << ", \"max\": " << max << ", \"mean\": " << mean
            << ", \"rms\": " << rms << ", \"first\": [";
  const size_t kShow = std::min<size_t>(8, n);
  for (size_t i = 0; i < kShow; ++i) {
    if (i)
      std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "] }\n";
}

} // namespace runtime
} // namespace buddy
