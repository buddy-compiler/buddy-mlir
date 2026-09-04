//===- PaligemmaRunner.cpp - PaliGemma-3B-224 VLM runner -----------------===//
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
// PaliGemma-3B-224 (google/paligemma-3b-mix-224) vision-language runner.
//
// Loads the compiled `paligemma_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), builds a fixed, deterministic input
// batch, and invokes the AOT forward graph:
//
//   forward(weights       : memref<params_size x f32>,      // arg0.data
//           position_ids  : memref<num_image_patches x i64>, // 0..255
//           input_ids     : memref<1 x max_seq_len x i64>,
//           pixel_values  : memref<1 x 3 x image_size x image_size x f32>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (image_features : memref<1 x num_image_tokens x hidden_size x f32>,
//         logits         : memref<1 x max_seq_len x vocab_size x f32>)
//
// The two results are packed into one struct (ForwardResults), so the C ABI
// wrapper `_mlir_ciface_forward` takes a single pointer to that struct
// (results first) followed by one pointer per input memref in declaration
// order:
//
//   void _mlir_ciface_forward(ForwardResults *, MemRef<float,1> *weights,
//                             MemRef<int64_t,1> *position_ids,
//                             MemRef<int64_t,2> *input_ids,
//                             MemRef<float,4> *pixel_values,
//                             MemRef<int64_t,2> *attention_mask);
//
// Input construction is deliberately simple and deterministic (see
// buildInputs): a zero pixel_values image and fixed token ids (256 <image>
// tokens followed by 24 text tokens). The kernel is shape-specialized, not
// value-specialized, so any tokens within the fixed shapes are valid.
//
// Defaults (from models/paligemma/specs/f32.json):
//   max_seq_len = 280, num_image_tokens = 256, image_size = 224,
//   hidden_size = 2048, vocab_size = 257216, image_token_id = 257152.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/PaligemmaRunner.h"
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 280;
constexpr size_t kDefaultHiddenSize = 2048;
constexpr size_t kDefaultVocabSize = 257216;
constexpr size_t kDefaultNumImageTokens = 256;
constexpr size_t kDefaultNumImagePatches = 256;
constexpr size_t kDefaultImageSize = 224;
constexpr int64_t kDefaultImageTokenId = 257152;

/// Packed output descriptors.  `@forward` returns both memrefs in one struct,
/// so the C ABI wrapper `_mlir_ciface_forward` takes a single pointer to this
/// struct (results first, then the five input descriptors).
struct ForwardResults {
  MemRef<float, 3> imageFeatures; // 1 x num_image_tokens x hidden_size
  MemRef<float, 3> logits;        // 1 x max_seq_len x vocab_size
};

using ForwardFn = void (*)(ForwardResults *, MemRef<float, 1> *,
                           MemRef<int64_t, 1> *, MemRef<int64_t, 2> *,
                           MemRef<float, 4> *, MemRef<int64_t, 2> *);

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

int64_t parseIntAttr(const ModelManifest &manifest, const char *key,
                     int64_t fallback) {
  auto it = manifest.moduleAttrs.find(key);
  if (it == manifest.moduleAttrs.end() || it->second.empty())
    return fallback;
  return static_cast<int64_t>(std::stoll(it->second));
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
    throw std::runtime_error("PaligemmaRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("PaligemmaRunner: error reading weights: " +
                             weightsPath);
}

/// Deterministic input batch for the traced VLM forward.
///
/// pixel_values  : zeros, shape [1, 3, imageSize, imageSize].
/// position_ids  : 0..numImagePatches-1, shape [numImagePatches].
/// input_ids     : numImageTokens copies of imageTokenId followed by
///                 (maxSeqLen - numImageTokens) copies of 1 (a fixed text
///                 token placeholder), shape [1, maxSeqLen].
/// attention_mask: all ones, shape [1, maxSeqLen].
///
/// Note: cfg.imagePath is intentionally ignored; a real image would require a
/// runtime image pipeline (PIL/torchvision preprocessing + SigLIP resize), so
/// the runner feeds a zero image for deterministic, dependency-free inference.
/// The kernel is shape-specialized, not value-specialized, so a zero image is
/// a valid input; logits are for the placeholder text tokens.
struct Inputs {
  MemRef<int64_t, 1> positionIds;
  MemRef<int64_t, 2> inputIds;
  MemRef<float, 4> pixelValues;
  MemRef<int64_t, 2> attentionMask;
};

Inputs buildInputs(size_t maxSeqLen, size_t numImageTokens,
                   size_t numImagePatches, size_t imageSize,
                   int64_t imageTokenId) {
  // Aggregate-init members directly (MemRef's default ctor is protected).
  Inputs in{MemRef<int64_t, 1>({numImagePatches}),
            MemRef<int64_t, 2>({1, maxSeqLen}),
            MemRef<float, 4>({1, 3, imageSize, imageSize}),
            MemRef<int64_t, 2>({1, maxSeqLen})};
  for (size_t i = 0; i < numImagePatches; ++i)
    in.positionIds.getData()[i] = static_cast<int64_t>(i);
  for (size_t i = 0; i < maxSeqLen; ++i)
    in.inputIds.getData()[i] = (i < numImageTokens) ? imageTokenId : 1;
  std::fill_n(in.pixelValues.getData(), 3 * imageSize * imageSize, 0.0f);
  std::fill_n(in.attentionMask.getData(), maxSeqLen, 1);
  return in;
}

} // namespace

void PaligemmaRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t vocabSize = kDefaultVocabSize;
  size_t numImageTokens = kDefaultNumImageTokens;
  size_t numImagePatches = kDefaultNumImagePatches;
  size_t imageSize = kDefaultImageSize;
  int64_t imageTokenId = kDefaultImageTokenId;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("PaligemmaRunner: manifest has no weight file");
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    vocabSize = parseSizeAttr(manifest, "vocab_size", vocabSize);
    numImageTokens = parseSizeAttr(manifest, "num_image_tokens",
                                   numImageTokens);
    numImagePatches = parseSizeAttr(manifest, "num_image_patches",
                                    numImagePatches);
    imageSize = parseSizeAttr(manifest, "image_size", imageSize);
    imageTokenId = parseIntAttr(manifest, "image_token_id", imageTokenId);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("PaligemmaRunner: legacy mode requires "
                               "--model-so and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("PaligemmaRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "PaligemmaRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("PaligemmaRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  // arg0.data holds exactly the f32 params of the forward (the i64 position_ids
  // buffer is packed into its own memref argument). Allocating by file size
  // (like ColBERTv2) yields the exact ABI memref<2923466608xf32>.
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  Inputs inputs = buildInputs(maxSeqLen, numImageTokens, numImagePatches,
                              imageSize, imageTokenId);

  ForwardResults results = {
      MemRef<float, 3>({1, numImageTokens, hiddenSize}, false, 0),
      MemRef<float, 3>({1, maxSeqLen, vocabSize}, false, 0)};

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&results, &paramsContainer, &inputs.positionIds, &inputs.inputIds,
          &inputs.pixelValues, &inputs.attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mPaliGemma-3B-224 Vision-Language Inference\033[0m\n";
    std::cerr << "  seq_len    : " << maxSeqLen << "\n";
    std::cerr << "  image_size : " << imageSize << "\n";
    std::cerr << "  time       : " << seconds << "s\n";
  }

  // Emit the language-model logits for the last sequence position (top-5).
  const float *logitsData = results.logits.getData();
  const size_t lastPos = maxSeqLen - 1;
  const float *lastLogits = logitsData + lastPos * vocabSize;
  std::vector<size_t> idx(vocabSize);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(
      idx.begin(), idx.begin() + 5, idx.end(),
      [&](size_t a, size_t b) { return lastLogits[a] > lastLogits[b]; });

  std::cout << "{\"logits_shape\":[" << results.logits.getSizes()[0] << ","
            << results.logits.getSizes()[1] << "," << results.logits.getSizes()[2]
            << "],\"last_token_top5\":[";
  for (size_t k = 0; k < 5; ++k) {
    if (k)
      std::cout << ",";
    std::cout << "{\"token\":" << idx[k] << ",\"logit\":" << lastLogits[idx[k]]
              << "}";
  }
  std::cout << "]}\n";

  if (!suppress) {
    const float *feat = results.imageFeatures.getData();
    std::cerr << "\033[33;1mImage features (first 8)\033[0m\n[";
    for (size_t i = 0; i < 8; ++i) {
      if (i)
        std::cerr << ", ";
      std::cerr << feat[i];
    }
    std::cerr << "]\n";
  }

  free(results.imageFeatures.release());
  free(results.logits.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
