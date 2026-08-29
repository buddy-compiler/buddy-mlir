//===- BgeRerankerRunner.cpp - BGE-Reranker-v2-M3 cross-encoder runner ---===//
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
// BGE-Reranker-v2-M3 (XLMRobertaForSequenceClassification) cross-encoder
// reranker runner.
//
// Loads the compiled `bge_reranker_model.so` and `arg0.data` weights through
// the `.rax` manifest (or explicit paths), tokenizes the query/document pair
// with a pure-C++ SentencePiece-Unigram tokenizer over the staged
// `tokenizer.json`, then invokes the AOT forward graph:
//
//   forward(weights: memref<params_size x f32>,
//           position_ids: memref<max_position_embeddings x i64>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (logits: memref<1 x 1 x f32>)
//
// Single result, so the C ABI wrapper `_mlir_ciface_forward` takes one pointer
// per result memref first, then one per input memref in declaration order:
//
//   void (*)(MemRef<float, 2> *logits,
//            MemRef<float, 1> *weights,
//            MemRef<int64_t, 1> *positionIds,
//            MemRef<int64_t, 2> *inputIds,
//            MemRef<int64_t, 2> *attentionMask);
//
// position_ids is a full `max_position_embeddings`-element lookup table: the
// fused graph gathers row 0..max_seq_len from it to index the learned position
// embeddings, so the runner fills it with arange(0, max_position_embeddings).
// The relevance score is logits[0, 0].
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Bge_rerankerRunner.h"
#include "buddy/runtime/models/BgeRerankerTokenizer.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
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

constexpr size_t kDefaultMaxSeqLen = 512;
constexpr size_t kDefaultMaxPositionEmbeddings = 8194;
constexpr size_t kDefaultHiddenSize = 1024;
constexpr size_t kDefaultNumLabels = 1;

/// Single-result forward: `_mlir_ciface_forward(logits*, weights*,
/// position_ids*, input_ids*, attention_mask*)`.
using ForwardFn = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                           MemRef<int64_t, 1> *, MemRef<int64_t, 2> *,
                           MemRef<int64_t, 2> *);

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
    throw std::runtime_error("Bge_rerankerRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("Bge_rerankerRunner: error reading weights: " +
                             weightsPath);
}

} // namespace

void Bge_rerankerRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "Bge_rerankerRunner: pass --prompt \"query <sep> document\" or "
        "--prompt-file <query\\ndocument>");

  // The reranker needs exactly one query/document pair. When the input is a
  // two-line prompt file, the first line is the query and the second is the
  // document; otherwise a single --prompt of the form
  // "query <sep> document" is split on the last " <sep> ".
  std::string query;
  std::string document;
  if (cfg.prompts.size() >= 2) {
    query = cfg.prompts[0];
    document = cfg.prompts[1];
    if (cfg.prompts.size() > 2)
      throw std::runtime_error("Bge_rerankerRunner: --prompt-file must contain "
                               "exactly two lines (query, document)");
  } else if (!cfg.prompts.empty()) {
    const std::string sep = " <sep> ";
    const std::string &text = cfg.prompts[0];
    const size_t pos = text.rfind(sep);
    if (pos == std::string::npos)
      throw std::runtime_error(
          "Bge_rerankerRunner: --prompt must contain \" <sep> \" between "
          "query and document");
    query = text.substr(0, pos);
    document = text.substr(pos + sep.size());
  } else {
    const std::string sep = " <sep> ";
    const std::string &text = cfg.prompt;
    const size_t pos = text.rfind(sep);
    if (pos == std::string::npos)
      throw std::runtime_error(
          "Bge_rerankerRunner: --prompt must contain \" <sep> \" between "
          "query and document");
    query = text.substr(0, pos);
    document = text.substr(pos + sep.size());
  }

  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t maxPositionEmbeddings = kDefaultMaxPositionEmbeddings;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t numLabels = kDefaultNumLabels;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("Bge_rerankerRunner: manifest has no weight file");
    vocabPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    maxPositionEmbeddings = parseSizeAttr(manifest, "max_position_embeddings",
                                          maxPositionEmbeddings);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    numLabels = parseSizeAttr(manifest, "num_labels", numLabels);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("Bge_rerankerRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath;
  }

  if (vocabPath.empty())
    throw std::runtime_error("Bge_rerankerRunner: vocab path is empty");

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Vocab     : " + vocabPath, suppress);
  printLog("Query     : " + query, suppress);
  printLog("Document  : " + document, suppress);

  BgeRerankerTokenizer tokenizer = BgeRerankerTokenizer::loadFromFile(vocabPath);
  std::vector<int64_t> inputIdVec, attentionMaskVec;
  tokenizer.encodePair(query, document, maxSeqLen, inputIdVec, attentionMaskVec);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("Bge_rerankerRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "Bge_rerankerRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error(
        "Bge_rerankerRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  // position_ids is a full max_position_embeddings lookup table that the graph
  // gathers rows 0..max_seq_len from; it must be arange(0, N) (not zeros).
  MemRef<int64_t, 1> positionIds({maxPositionEmbeddings});
  for (size_t i = 0; i < maxPositionEmbeddings; ++i)
    positionIds.getData()[i] = static_cast<int64_t>(i);

  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  std::copy(inputIdVec.begin(), inputIdVec.end(), inputIds.getData());
  std::copy(attentionMaskVec.begin(), attentionMaskVec.end(),
            attentionMask.getData());

  // Single result: allocate the output memref on the stack; the callee writes
  // through the descriptor's data pointer.
  MemRef<float, 2> logits({1, numLabels}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&logits, &paramsContainer, &positionIds, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mBGE-Reranker Relevance Score\033[0m\n";
    std::cerr << "  seq_len : " << maxSeqLen << "\n";
    std::cerr << "  time    : " << seconds << "s\n";
  }

  std::cout << logits.getData()[0] << "\n";

  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
