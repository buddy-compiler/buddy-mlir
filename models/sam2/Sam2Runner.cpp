//===- Sam2Runner.cpp - SAM2-hiera-tiny vision encoder runner -------------===//
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
// SAM2-hiera-tiny image-segmentation vision encoder runner.
//
// Loads the compiled `sam2_model.so` and `arg0.data` weights through the `.rax`
// manifest (or explicit legacy paths), feeds a fixed-shape `1 x 3 x 256 x 256`
// f32 image tensor into the AOT forward graph, and emits the image feature
// maps. The runner owns weight loading, kernel invocation and output emission.
//
// ── Forward ABI ──────────────────────────────────────────────────────────
// The traced graph is the SAM2 vision encoder (Sam2VisionModel), so the
// compiled `_mlir_ciface_forward` wrapper has the signature
//
//   forward(weights:      memref<27219136 x f32>,        // flattened arg0.data
//           pixel_values: memref<1 x 3 x 256 x 256 x f32>)
//     -> (last_hidden_state: memref<1 x 8 x 8 x 768 x f32>,
//         fpn_0:             memref<1 x 256 x 16 x 16 x f32>,
//         fpn_1:             memref<1 x 256 x 16 x 16 x f32>,
//         fpn_2:             memref<1 x 256 x 32 x 32 x f32>,
//         fpn_3:             memref<1 x 256 x 32 x 32 x f32>,
//         fpn_4:             memref<1 x 256 x 64 x 64 x f32>,
//         fpn_5:             memref<1 x 256 x 64 x 64 x f32>)
//
// The 7 results are packed in a single C struct, so the C ABI wrapper
// `_mlir_ciface_forward` takes one pointer per result memref followed by one
// pointer per input memref, in declaration order:
//
//   void _mlir_ciface_forward(ForwardResults *,
//                             MemRef<float, 1> *weights,   // arg0.data
//                             MemRef<float, 4> *pixelValues);
//
// MemRef<T,N> (buddy/Core/Container.h) is a standard-layout descriptor
// {allocated, aligned, offset, size[0..N-1], stride[0..N-1]}, which matches
// the MLIR C memref descriptor layout, so descriptors are passed by address.
// Output descriptors are created with needMalloc=false (the kernel allocates
// the buffers); the runner frees them with free() after the call.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Sam2Runner.h"
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

constexpr size_t kDefaultImageSize = 256; // input 1 x 3 x 256 x 256
constexpr size_t kDefaultHiddenSize = 768;
constexpr size_t kDefaultOutputHeight = 8;  // last_hidden_state spatial dims
constexpr size_t kDefaultOutputWidth = 8;
constexpr size_t kDefaultFpnChannels = 256;

/// Packed output descriptors. `@forward` returns all seven memrefs in one
/// struct, so the C ABI wrapper `_mlir_ciface_forward` takes a single pointer
/// to this struct (results first, then the two input descriptors).
struct ForwardResults {
  MemRef<float, 4> lastHiddenState; // 1 x 8 x 8 x 768
  MemRef<float, 4> fpn0;            // 1 x 256 x 16 x 16
  MemRef<float, 4> fpn1;            // 1 x 256 x 16 x 16
  MemRef<float, 4> fpn2;            // 1 x 256 x 32 x 32
  MemRef<float, 4> fpn3;            // 1 x 256 x 32 x 32
  MemRef<float, 4> fpn4;            // 1 x 256 x 64 x 64
  MemRef<float, 4> fpn5;            // 1 x 256 x 64 x 64
};

using ForwardFn = void (*)(ForwardResults *, MemRef<float, 1> *,
                           MemRef<float, 4> *);

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
    throw std::runtime_error("Sam2Runner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("Sam2Runner: error reading weights: " +
                             weightsPath);
}

} // namespace

void Sam2Runner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (!cfg.imagePath.empty())
    throw std::runtime_error(
        "Sam2Runner: image input loading is not implemented; the encoder is "
        "run on a deterministic black (all-zeros) image to validate the "
        "pipeline.");

  std::string soPath;
  std::string weightsPath;
  size_t imageSize = kDefaultImageSize;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t outH = kDefaultOutputHeight;
  size_t outW = kDefaultOutputWidth;
  size_t fpnChannels = kDefaultFpnChannels;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("Sam2Runner: manifest has no weight file");
    imageSize = parseSizeAttr(manifest, "image_size", imageSize);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    outH = parseSizeAttr(manifest, "output_height", outH);
    outW = parseSizeAttr(manifest, "output_width", outW);
    fpnChannels = parseSizeAttr(manifest, "fpn_channels", fpnChannels);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("Sam2Runner: legacy mode requires --model-so "
                               "and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("Sam2Runner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "Sam2Runner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("Sam2Runner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  // Deterministic input: an all-zeros 1 x 3 x imageSize x imageSize image
  // (matches the original PR's validate_accuracy reference input).
  printLog("Preparing pixel_values (black image)", suppress);
  MemRef<float, 4> pixelValues({1, 3, imageSize, imageSize});
  std::fill(pixelValues.getData(),
            pixelValues.getData() + pixelValues.getSize(), 0.0f);

  ForwardResults results = {
      MemRef<float, 4>({1, outH, outW, hiddenSize}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 16, 16}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 16, 16}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 32, 32}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 32, 32}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 64, 64}, false, 0),
      MemRef<float, 4>({1, fpnChannels, 64, 64}, false, 0)};

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&results, &paramsContainer, &pixelValues);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mSAM2 Vision Encoder Feature Map\033[0m\n";
    std::cerr << "  input  : 1 x 3 x " << imageSize << " x " << imageSize
              << "\n";
    std::cerr << "  time   : " << seconds << "s\n";
  }

  // Deterministic output: dims of all outputs, then the flattened
  // last_hidden_state image feature map.
  std::cout << "{\n";
  std::cout << "  \"outputs\": [\n";
  std::cout << "    {\"name\": \"last_hidden_state\", \"shape\": [1, " << outH
            << ", " << outW << ", " << hiddenSize << "]},\n";
  std::cout << "    {\"name\": \"fpn_0\", \"shape\": [1, " << fpnChannels
            << ", 16, 16]},\n";
  std::cout << "    {\"name\": \"fpn_1\", \"shape\": [1, " << fpnChannels
            << ", 16, 16]},\n";
  std::cout << "    {\"name\": \"fpn_2\", \"shape\": [1, " << fpnChannels
            << ", 32, 32]},\n";
  std::cout << "    {\"name\": \"fpn_3\", \"shape\": [1, " << fpnChannels
            << ", 32, 32]},\n";
  std::cout << "    {\"name\": \"fpn_4\", \"shape\": [1, " << fpnChannels
            << ", 64, 64]},\n";
  std::cout << "    {\"name\": \"fpn_5\", \"shape\": [1, " << fpnChannels
            << ", 64, 64]}\n";
  std::cout << "  ],\n";
  std::cout << "  \"last_hidden_state\": [";
  const float *data = results.lastHiddenState.getData();
  const size_t numel = results.lastHiddenState.getSize();
  for (size_t i = 0; i < numel; ++i) {
    if (i)
      std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "]\n";
  std::cout << "}\n";

  free(results.lastHiddenState.release());
  free(results.fpn0.release());
  free(results.fpn1.release());
  free(results.fpn2.release());
  free(results.fpn3.release());
  free(results.fpn4.release());
  free(results.fpn5.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
