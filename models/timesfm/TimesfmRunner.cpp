//===- TimesfmRunner.cpp - TimesFM 2.5 time-series runner -----------------===//
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
// TimesFM 2.5 (200M) time-series foundation model runner.
//
// Loads the compiled `timesfm_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), builds a fixed-length context window
// (num_patches x patch_length = 16 x 32 = 512 time points), then invokes the
// AOT forward graph:
//
//   forward(weights: memref<params_size x f32>,
//           inputs:   memref<1 x num_patches x patch_length x f32>,
//           masks:    memref<1 x num_patches x patch_length x f32>)
//     -> (point_forecast: memref<1 x num_patches x (output_patch_len*quantile_len) x f32>)
//
// The forward has a SINGLE result, so the C ABI wrapper `_mlir_ciface_forward`
// takes one pointer for the result memref first, then one pointer per input
// memref in declaration order:
//
//   void _mlir_ciface_forward(MemRef<float,3>* point_forecast,
//                             MemRef<float,1>* weights,
//                             MemRef<float,3>* inputs,
//                             MemRef<float,3>* masks)
//
// MemRef descriptor layout (buddy/Core/Container.h):
//   1-D: {allocated, aligned, offset, size0, stride0}
//   3-D: {allocated, aligned, offset, size0, stride0, size1, stride1,
//         size2, stride2}
//
// Output interpretation: the result is the raw point-projection output of the
// TimesFM decoder, shaped (1, num_patches, o*q) with o=output_patch_len=128
// forecast points per patch and q=quantile_len=10 (9 quantiles + point).  The
// LAST patch (index num_patches-1) holds the forecast for the next o=128 time
// points; per the official TimesFM decode(), column `decode_index` (5) is the
// point-forecast column.  The official pipeline additionally applies reversible
// instance normalization (revin) over the input running stats; the AOT graph
// returns the raw point projection, so this runner emits that raw tensor.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/TimesfmRunner.h"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultNumPatches = 16;
constexpr size_t kDefaultPatchLength = 32;
constexpr size_t kDefaultOutputPatchLen = 128;
constexpr size_t kDefaultQuantileLen = 10;
constexpr size_t kDefaultDecodeIndex = 5;

/// Forward ABI (see file header):
///   void _mlir_ciface_forward(MemRef<float,3>* point_forecast,
///                             MemRef<float,1>* weights,
///                             MemRef<float,3>* inputs,
///                             MemRef<float,3>* masks)
using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<float, 3> *, MemRef<float, 3> *);

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
    throw std::runtime_error("TimesfmRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("TimesfmRunner: error reading weights: " +
                             weightsPath);
}

/// Parse a context time series from `text` (comma / whitespace separated
/// floats).  Values are aligned to the END of the fixed window (most recent
/// observation last); a short series is left-padded with zeros and a long
/// series keeps its last `windowLen` values.
void fillContextWindow(const std::string &text, size_t windowLen,
                       std::vector<float> &series) {
  std::vector<float> values;
  std::string cleaned;
  for (char c : text)
    cleaned += (c == ',') ? ' ' : c;
  std::istringstream in(cleaned);
  double v;
  while (in >> v)
    values.push_back(static_cast<float>(v));

  series.assign(windowLen, 0.0f);
  const size_t copyLen = std::min(values.size(), windowLen);
  for (size_t i = 0; i < copyLen; ++i)
    series[windowLen - copyLen + i] = values[values.size() - copyLen + i];
}

/// Deterministic default context series (sum of two sines) used when no
/// `--prompt` series is supplied.
void fillDefaultContextWindow(size_t windowLen, std::vector<float> &series) {
  series.resize(windowLen);
  for (size_t i = 0; i < windowLen; ++i)
    series[i] = static_cast<float>(std::sin(0.05 * static_cast<double>(i)) +
                                   0.5 * std::sin(0.3 * static_cast<double>(i)));
}

} // namespace

void TimesfmRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "TimesfmRunner: only single-series inference is implemented");

  std::string soPath;
  std::string weightsPath;
  size_t numPatches = kDefaultNumPatches;
  size_t patchLength = kDefaultPatchLength;
  size_t paramsSize = 0;
  size_t outputPatchLen = kDefaultOutputPatchLen;
  size_t quantileLen = kDefaultQuantileLen;
  size_t decodeIndex = kDefaultDecodeIndex;
  size_t numThreads = 0;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("TimesfmRunner: manifest has no weight file");
    numPatches = parseSizeAttr(manifest, "num_patches", numPatches);
    patchLength = parseSizeAttr(manifest, "patch_length", patchLength);
    paramsSize = parseSizeAttr(manifest, "params_size", paramsSize);
    outputPatchLen = parseSizeAttr(manifest, "output_patch_len", outputPatchLen);
    quantileLen = parseSizeAttr(manifest, "quantile_len", quantileLen);
    decodeIndex = parseSizeAttr(manifest, "decode_index", decodeIndex);
    numThreads = parseSizeAttr(manifest, "num_threads", numThreads);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("TimesfmRunner: legacy mode requires "
                               "--model-so and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  const size_t windowLen = numPatches * patchLength;
  const size_t forecastFeatures = outputPatchLen * quantileLen;
  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so  : " + soPath, suppress);
  printLog("Weights    : " + weightsPath, suppress);
  printLog("Context    : " + std::to_string(windowLen) + " points (" +
               std::to_string(numPatches) + " x " + std::to_string(patchLength) +
               ")",
           suppress);

  std::vector<float> series;
  if (!prompt.empty())
    fillContextWindow(prompt, windowLen, series);
  else
    fillDefaultContextWindow(windowLen, series);

  // Pin the OpenMP thread count (from the manifest/spec) before the compiled
  // kernels hit their first parallel region, keeping the result reproducible.
  if (numThreads > 0)
    setenv("OMP_NUM_THREADS", std::to_string(numThreads).c_str(), 1);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("TimesfmRunner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "TimesfmRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  if (paramsSize == 0) {
    const auto weightBytes = fs::file_size(weightsPath);
    if (weightBytes % sizeof(float) != 0) {
      dlclose(handle);
      throw std::runtime_error(
          "TimesfmRunner: weight file is not f32-aligned");
    }
    paramsSize = weightBytes / sizeof(float);
  }

  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({paramsSize});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  MemRef<float, 3> inputs({1, numPatches, patchLength});
  MemRef<float, 3> masks({1, numPatches, patchLength});
  std::copy(series.begin(), series.end(), inputs.getData());
  std::fill(masks.getData(), masks.getData() + masks.getSize(), 1.0f);

  // Single result: the compiled kernel allocates the output buffer and fills
  // the descriptor's data pointers during the call.
  MemRef<float, 3> pointForecast({1, numPatches, forecastFeatures}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&pointForecast, &paramsContainer, &inputs, &masks);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mTimesFM 2.5 Point Forecast\033[0m\n";
    std::cerr << "  context   : " << windowLen << " points\n";
    std::cerr << "  patches   : " << numPatches << "\n";
    std::cerr << "  forecast  : " << outputPatchLen << " points x "
              << quantileLen << " columns per patch\n";
    std::cerr << "  time      : " << seconds << "s\n";
  }

  // Emit the raw point-forecast tensor as nested JSON arrays:
  //   [[patch0: 1280 values], ..., [patch15: 1280 values]]
  const float *data = pointForecast.getData();
  std::cout << "[";
  for (size_t p = 0; p < numPatches; ++p) {
    if (p)
      std::cout << ", ";
    std::cout << "[";
    for (size_t j = 0; j < forecastFeatures; ++j) {
      if (j)
        std::cout << ", ";
      std::cout << data[p * forecastFeatures + j];
    }
    std::cout << "]";
  }
  std::cout << "]\n";

  if (!suppress) {
    // Concise point-forecast summary: last patch, column decode_index.
    const float *lastPatch = data + (numPatches - 1) * forecastFeatures;
    std::cerr << "\033[33;1mPoint forecast (last patch, column "
              << decodeIndex << ")\033[0m\n  [";
    for (size_t j = 0; j < outputPatchLen; ++j) {
      if (j)
        std::cerr << ", ";
      std::cerr << lastPatch[j * quantileLen + decodeIndex];
    }
    std::cerr << "]\n";
  }

  free(pointForecast.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
