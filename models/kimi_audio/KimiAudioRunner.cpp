//===- KimiAudioRunner.cpp - Kimi-Audio single-forward runner ------------===//
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
// Kimi-Audio-7B-Instruct single-forward runner.
//
// Loads the compiled `kimi_audio_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), tokenizes the input text with the Qwen
// byte-level BPE tokenizer over the staged `vocab.txt`, then invokes the AOT
// forward graph (text-only path, whisper features disabled):
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           position_ids: memref<1 x max_seq_len x i64>)
//     -> (audio_logits: memref<1 x max_seq_len x vocab_size x f32>,
//         text_logits : memref<1 x max_seq_len x vocab_size x f32>)
//
// The per-position argmax token ids of text_logits are printed as a JSON list.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/KimiAudioRunner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 1024;
constexpr size_t kDefaultHiddenSize = 3584;
constexpr size_t kDefaultVocabSize = 168448;

/// Packed output descriptors.  `@forward` returns both logit tensors in one
/// struct, so the C ABI wrapper `_mlir_ciface_forward` takes a single pointer
/// to this struct (results first, then the input descriptors).
struct ForwardResults {
  MemRef<float, 3> audioLogits; // 1 x max_seq_len x vocab_size
  MemRef<float, 3> textLogits;  // 1 x max_seq_len x vocab_size
};

using ForwardFn = void (*)(ForwardResults *, MemRef<float, 1> *,
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
    throw std::runtime_error("KimiAudioRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("KimiAudioRunner: error reading weights: " +
                             weightsPath);
}

/// Emit a JSON array of the argmax token id per sequence position.
void printArgmaxTokens(const char *label, const float *logits,
                       size_t maxSeqLen, size_t vocabSize, bool suppress) {
  if (!suppress)
    std::cerr << "\033[33;1m" << label << " argmax token ids\033[0m\n";
  std::cout << "[";
  for (size_t tok = 0; tok < maxSeqLen; ++tok) {
    if (tok)
      std::cout << ", ";
    const float *row = logits + tok * vocabSize;
    size_t best = 0;
    float bestVal = -std::numeric_limits<float>::infinity();
    for (size_t d = 0; d < vocabSize; ++d) {
      if (row[d] > bestVal) {
        bestVal = row[d];
        best = d;
      }
    }
    std::cout << best;
  }
  std::cout << "]\n";
}

} // namespace

void KimiAudioRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "KimiAudioRunner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "KimiAudioRunner: only single-prompt inference is implemented");

  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t vocabSize = kDefaultVocabSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("KimiAudioRunner: manifest has no weight file");
    vocabPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    vocabSize = parseSizeAttr(manifest, "vocab_size", vocabSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("KimiAudioRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath;
  }

  if (vocabPath.empty())
    throw std::runtime_error("KimiAudioRunner: vocab path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Vocab     : " + vocabPath, suppress);

  // Qwen byte-level BPE tokenization (Kimi-Audio uses a Qwen2.5-style
  // tokenizer).  `text` is a MemRef<int64_t,2> shaped {1, maxSeqLen} filled
  // with the tokenized + padded input ids.
  Text<int64_t, 2> text(prompt);
  text.tokenizeQwen3(vocabPath, maxSeqLen);
  printLog("Tokenization complete", suppress);

  // Position ids: 0..maxSeqLen-1 (matches the traced explicit position_ids).
  MemRef<int64_t, 2> positionIds({1, maxSeqLen});
  for (size_t i = 0; i < maxSeqLen; ++i)
    positionIds.getData()[i] = static_cast<int64_t>(i);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("KimiAudioRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "KimiAudioRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("KimiAudioRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights (this is a 7B model, expect tens of GB of RAM)",
           suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  ForwardResults results = {
      MemRef<float, 3>({1, maxSeqLen, vocabSize}, false, 0),
      MemRef<float, 3>({1, maxSeqLen, vocabSize}, false, 0)};

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&results, &paramsContainer, &text, &positionIds);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mKimi-Audio Forward\033[0m\n";
    std::cerr << "  seq_len  : " << maxSeqLen << "\n";
    std::cerr << "  vocab    : " << vocabSize << "\n";
    std::cerr << "  time     : " << seconds << "s\n";
  }

  printArgmaxTokens("audio_logits", results.audioLogits.getData(), maxSeqLen,
                    vocabSize, suppress);
  printArgmaxTokens("text_logits", results.textLogits.getData(), maxSeqLen,
                    vocabSize, suppress);

  free(results.audioLogits.release());
  free(results.textLogits.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
