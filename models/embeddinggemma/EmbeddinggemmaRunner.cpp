//===- EmbeddinggemmaRunner.cpp - embeddinggemma embedding runner ---------===//
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
// google/embeddinggemma-300m sentence embedding runner.
//
// Loads the compiled `embeddinggemma_model.so` and `arg0.data` weights through
// the `.rax` manifest (or explicit paths), tokenizes the input text with the
// Gemma byte-level BPE tokenizer (EmbeddinggemmaTokenizer.h, a pure-C++
// reimplementation of tokenizer.json), then invokes the AOT forward graph:
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (embedding: memref<1 x hidden_size x f32>)
//
// The L2-normalized 768-dim embedding (the SentenceTransformer output after
// Gemma3TextModel -> mean pooling -> two dense layers -> L2 normalize) is
// printed together with a short summary.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/EmbeddinggemmaRunner.h"
#include "buddy/runtime/models/EmbeddinggemmaTokenizer.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

constexpr size_t kDefaultMaxSeqLen = 128;
constexpr size_t kDefaultHiddenSize = 768;

/// Single-result ABI: the wrapper takes one pointer to the output embedding
/// descriptor first, then one pointer per input descriptor.
using ForwardFn = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<int64_t, 2> *);

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
    throw std::runtime_error("EmbeddinggemmaRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("EmbeddinggemmaRunner: error reading weights: " +
                             weightsPath);
}

} // namespace

void EmbeddinggemmaRunner::run(const RunConfig &cfg) {
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  std::string tokenizerPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error(
          "EmbeddinggemmaRunner: manifest has no weight file");
    tokenizerPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("EmbeddinggemmaRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
  }

  if (tokenizerPath.empty())
    throw std::runtime_error("EmbeddinggemmaRunner: tokenizer path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so  : " + soPath, suppress);
  printLog("Weights    : " + weightsPath, suppress);
  printLog("Tokenizer  : " + tokenizerPath, suppress);

  EmbeddinggemmaTokenizer tokenizer =
      EmbeddinggemmaTokenizer::loadFromFile(tokenizerPath);
  std::vector<int64_t> inputIdVec, attentionMaskVec;
  tokenizer.encode(prompt, maxSeqLen, inputIdVec, attentionMaskVec);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("EmbeddinggemmaRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "EmbeddinggemmaRunner: missing _mlir_ciface_forward in " + soPath +
        ": " + std::string(err));
  }

  const auto weightBytes = std::filesystem::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error(
        "EmbeddinggemmaRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  std::copy(inputIdVec.begin(), inputIdVec.end(), inputIds.getData());
  std::copy(attentionMaskVec.begin(), attentionMaskVec.end(),
            attentionMask.getData());

  MemRef<float, 2> embedding({1, hiddenSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&embedding, &paramsContainer, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  const float *data = embedding.getData();
  double norm2 = 0.0;
  for (size_t d = 0; d < hiddenSize; ++d)
    norm2 += static_cast<double>(data[d]) * static_cast<double>(data[d]);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mEmbeddingGemma Sentence Embedding\033[0m\n";
    std::cerr << "  seq_len : " << maxSeqLen << "\n";
    std::cerr << "  dim     : " << hiddenSize << "\n";
    std::cerr << "  time    : " << seconds << "s\n";
    std::cerr << "  |emb|   : " << std::sqrt(norm2) << " (L2-normalized)\n";
  }

  std::cout << "[";
  for (size_t d = 0; d < hiddenSize; ++d) {
    if (d)
      std::cout << ", ";
    std::cout << data[d];
  }
  std::cout << "]\n";

  free(embedding.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
