//===- PaddleocrRunner.cpp - PaddleOCR-VL single-shot OCR runner ----------===//
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
// PaddleOCR-VL-0.9B OCR vision-language runner (single_forward).
//
// Loads the compiled `paddleocr_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), builds the fixed-shape OCR input
// (972 image tokens + 10 text tokens, see below), invokes the AOT forward
// graph, and emits the last-token logits.
//
// ─────────────────────────────────────────────────────────────────────────────
// FORWARD ABI (from codegen/import-paddleocr.py, verified against forward.mlir)
// ─────────────────────────────────────────────────────────────────────────────
//   forward(weights:      memref<905601730 x f32>,
//           input_ids:     memref<1 x 982 x i64>,
//           pixel_values:  memref<3888 x 3 x 14 x 14 x f32>,
//           attention_mask: memref<1 x 982 x i64>,
//           position_ids:  memref<3 x 1 x 982 x i64>)
//     -> (logits:         memref<1 x 982 x 103424 x f32>)
//
// The C wrapper `_mlir_ciface_forward` takes ONE pointer per result memref
// (single result -> a direct MemRef pointer), then ONE pointer per input
// memref in declaration order: weights, input_ids, pixel_values,
// attention_mask, position_ids.
//
// Input layout (matches the reference HF trace / validate_accuracy.py):
//   - positions [0:972]   = image tokens  (id = image_token_id = 100295)
//   - positions [972:982] = text tokens   (10 slots)
//   - pixel_values        = fixed-size zeros (3888 x 3 x 14 x 14) — the
//     runner does NOT decode images; it is deterministic by design.
//   - attention_mask      = all ones
//   - position_ids        = all zeros (3 x 1 x 982; 3D mRoPE section)
// The 10 text slots are filled with a simplified, deterministic byte->token
// mapping of cfg.prompt (or pad id 1 when no prompt is given). This is NOT a
// full Qwen byte-level-BPE tokenizer; the model was traced with these fixed
// text positions, and exact prompt tokenization is out of scope for the
// single_forward integration (see README.md Notes).
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/PaddleocrRunner.h"
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
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

// Defaults from models/paddleocr/specs/f32.json (overridden by manifest attrs).
constexpr size_t kDefaultParamsSize = 905601730;
constexpr size_t kDefaultMaxSeqLen = 982;
constexpr size_t kDefaultHiddenSize = 1024;
constexpr size_t kDefaultVocabSize = 103424;
constexpr size_t kDefaultNumImagePatches = 3888;
constexpr size_t kDefaultPatchSize = 14;
constexpr size_t kNumImageTokens = 972; // (54*72) / (2*2) = 3888/4
constexpr int64_t kImageTokenId = 100295; // config.image_token_id
constexpr int64_t kPadTokenId = 1;        // traced text-token id

/// Forward function pointer for the single-result ABI:
///   void forward(logits*, weights*, input_ids*, pixel_values*,
///                attention_mask*, position_ids*)
using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<float, 4> *,
                           MemRef<int64_t, 2> *, MemRef<int64_t, 3> *);

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
    throw std::runtime_error("PaddleocrRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("PaddleocrRunner: error reading weights: " +
                             weightsPath);
}

/// Simplified deterministic text encoder for the 10 text-token slots.
/// Each UTF-8 byte maps to token id in [1, 199]; remaining slots use the pad
/// id 1 (matching the traced HF reference input). Not a real BPE tokenizer.
void encodeText(const std::string &prompt, size_t textSlots,
                int64_t *ids) {
  size_t n = std::min(prompt.size(), textSlots);
  for (size_t i = 0; i < textSlots; ++i) {
    if (i < n)
      ids[i] = 1 + static_cast<int64_t>(static_cast<unsigned char>(prompt[i]) %
                                        199);
    else
      ids[i] = kPadTokenId;
  }
}

} // namespace

void PaddleocrRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  size_t paramsSize = kDefaultParamsSize;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t vocabSize = kDefaultVocabSize;
  size_t numImagePatches = kDefaultNumImagePatches;
  size_t patchSize = kDefaultPatchSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error(
          "PaddleocrRunner: manifest has no weight file");
    paramsSize = parseSizeAttr(manifest, "params_size", paramsSize);
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    vocabSize = parseSizeAttr(manifest, "vocab_size", vocabSize);
    numImagePatches = parseSizeAttr(manifest, "num_image_patches",
                                    numImagePatches);
    patchSize = parseSizeAttr(manifest, "patch_size", patchSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error(
          "PaddleocrRunner: legacy mode requires --model-so and --weights");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
  }

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;
  const size_t textSlots =
      (maxSeqLen > kNumImageTokens) ? (maxSeqLen - kNumImageTokens) : 0;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Seq len   : " + std::to_string(maxSeqLen) +
               " (image=" + std::to_string(kNumImageTokens) +
               ", text=" + std::to_string(textSlots) + ")",
           suppress);

  // ── Load the compiled kernel ───────────────────────────────────────────
  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("PaddleocrRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(
      dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "PaddleocrRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  // ── Weights (flattened param buffer; forward input #1) ─────────────────
  if (fs::file_size(weightsPath) != paramsSize * sizeof(float))
    printLog("WARNING: weight file size does not match params_size (" +
                 std::to_string(fs::file_size(weightsPath)) + " vs " +
                 std::to_string(paramsSize * sizeof(float)) + " bytes)",
             suppress);
  printLog("Loading weights", suppress);
  MemRef<float, 1> params({paramsSize});
  loadWeights(weightsPath, params);
  printLog("Weights loaded", suppress);

  // ── Fixed-shape inputs ─────────────────────────────────────────────────
  // input_ids: 972 image tokens + text prompt in the remaining slots.
  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  for (size_t i = 0; i < maxSeqLen; ++i) {
    if (i < kNumImageTokens)
      inputIds.getData()[i] = kImageTokenId;
    else
      inputIds.getData()[i] = kPadTokenId;
  }
  if (textSlots > 0)
    encodeText(prompt, textSlots, inputIds.getData() + kNumImageTokens);

  // pixel_values: deterministic zeros (no real image decoding).
  MemRef<float, 4> pixelValues(
      {numImagePatches, 3, patchSize, patchSize}, 0.0f);

  // attention_mask: all ones.
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen}, 1);

  // position_ids: 3D mRoPE section ids, all zeros (matches the HF trace).
  MemRef<int64_t, 3> positionIds({3, 1, maxSeqLen}, 0);

  // ── Output ─────────────────────────────────────────────────────────────
  // Single result memref; the compiled kernel allocates its storage.
  MemRef<float, 3> logits({1, maxSeqLen, vocabSize}, /*needMalloc=*/false, 0);

  printLog("Calling _mlir_ciface_forward", suppress);
  const auto t0 = std::chrono::high_resolution_clock::now();
  forward(&logits, &params, &inputIds, &pixelValues, &attentionMask,
          &positionIds);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  // ── Emit last-token logits (position maxSeqLen - 1) ────────────────────
  const size_t lastPos = maxSeqLen - 1;
  const float *logitsData = logits.getData();
  const float *lastLogits = logitsData + lastPos * vocabSize;

  double sum = 0.0, sumSq = 0.0;
  float maxV = lastLogits[0];
  for (size_t v = 0; v < vocabSize; ++v) {
    sum += lastLogits[v];
    sumSq += static_cast<double>(lastLogits[v]) * lastLogits[v];
    maxV = std::max(maxV, lastLogits[v]);
  }
  const double mean = sum / static_cast<double>(vocabSize);

  std::vector<std::pair<float, int64_t>> topK;
  topK.reserve(vocabSize);
  for (size_t v = 0; v < vocabSize; ++v)
    topK.emplace_back(lastLogits[v], static_cast<int64_t>(v));
  std::partial_sort(
      topK.begin(), topK.begin() + 5, topK.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });
  topK.resize(5);

  if (!suppress) {
    const double seconds =
        std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mPaddleOCR-VL Last-Token Logits\033[0m\n";
    std::cerr << "  logits shape : 1 x " << maxSeqLen << " x " << vocabSize
              << "\n";
    std::cerr << "  time         : " << seconds << "s\n";
  }
  std::cout << "{\"logits_shape\": [1, " << maxSeqLen << ", " << vocabSize
            << "],\n";
  std::cout << " \"last_logits_sum\": " << sum << ",\n";
  std::cout << " \"last_logits_mean\": " << mean << ",\n";
  std::cout << " \"last_logits_max\": " << maxV << ",\n";
  std::cout << " \"last_logits_rms\": " << std::sqrt(sumSq / vocabSize)
            << ",\n";
  std::cout << " \"top5\": [";
  for (size_t i = 0; i < topK.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << "{\"token\": " << topK[i].second
              << ", \"logit\": " << topK[i].first << "}";
  }
  std::cout << "]}\n";

  // ── Cleanup ────────────────────────────────────────────────────────────
  free(logits.release()); // storage allocated by the compiled kernel
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
