//===- E5MistralRunner.cpp - E5-Mistral sentence-embedding runner ---------===//
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
// E5-Mistral-7B-Instruct sentence-embedding runner (Mistral encoder).
//
// Loads the compiled `e5_mistral_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), tokenizes the input text with a pure
// C++ Llama BPE tokenizer over the staged `tokenizer.json`, then invokes the
// AOT forward graph:
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (last_hidden_state: memref<1 x max_seq_len x hidden_size x f32>)
//
// EXACT forward ABI (confirmed from the generated forward.mlir):
//   `func.func @forward(%arg0: memref<7110660160xf32>,
//                       %arg1: memref<1x128xi64>,
//                       %arg2: memref<1x128xi64>)
//                       -> memref<1x128x4096xf32>`
//
// The model has a SINGLE result, so the C wrapper `_mlir_ciface_forward`
// (from `-llvm-request-c-wrappers`) takes ONE pointer to the result memref
// first, then one pointer per input memref in declaration order:
//
//   void _mlir_ciface_forward(MemRef<float,3> *last_hidden_state,   // result
//                             MemRef<float,1> *weights,             // arg0.data
//                             MemRef<int64_t,2> *input_ids,
//                             MemRef<int64_t,2> *attention_mask);
//
// The tokenizer reproduces `AutoTokenizer(text, padding="max_length",
// truncation=True, max_length=max_seq_len)`, which for this model LEFT-pads:
// [</s>] * pad + [<s>] + bpe(text) + [</s>]. e5-mistral's tokenizer_config
// only sets add_eos_token=True; add_bos_token is unset so the Llama
// tokenizer's default (prepend "<s>") applies -- the "<s>" IS in the output.
// Because the content is right-aligned, the last token position is always the
// "</s>" end-of-sequence token, so the sentence embedding is
// `last_hidden_state[0, max_seq_len-1, :]` (the same pooling the original PR's
// validate_accuracy.py used).
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/E5MistralRunner.h"
#include "buddy/runtime/models/E5MistralTokenizer.h"
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

constexpr size_t kDefaultMaxSeqLen = 128;
constexpr size_t kDefaultHiddenSize = 4096;

using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
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
    throw std::runtime_error("E5MistralRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("E5MistralRunner: error reading weights: " +
                             weightsPath);
}

} // namespace

void E5MistralRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "E5MistralRunner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "E5MistralRunner: only single-prompt inference is implemented");

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
      throw std::runtime_error("E5MistralRunner: manifest has no weight file");
    // manifest.vocabPath is payload-resolved (extracted from the embedded
    // .rax payload if needed), unlike the raw "vocab_uri" module attr which
    // rax-pack never rewrites to payload:* and would break once the .rax
    // ships without the build tree next to it.
    tokenizerPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("E5MistralRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
  }

  if (tokenizerPath.empty())
    throw std::runtime_error("E5MistralRunner: tokenizer path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Tokenizer : " + tokenizerPath, suppress);

  E5MistralTokenizer tokenizer = E5MistralTokenizer::loadFromFile(tokenizerPath);
  std::vector<int64_t> inputIdVec, attentionMaskVec;
  tokenizer.encode(prompt, maxSeqLen, inputIdVec, attentionMaskVec);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("E5MistralRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "E5MistralRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("E5MistralRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights (7B, ~28 GB)", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  std::copy(inputIdVec.begin(), inputIdVec.end(), inputIds.getData());
  std::copy(attentionMaskVec.begin(), attentionMaskVec.end(),
            attentionMask.getData());

  MemRef<float, 3> lastHiddenState({1, maxSeqLen, hiddenSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&lastHiddenState, &paramsContainer, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  // Sentence embedding = last token (always "</s>", right-aligned by the
  // left-padding above), matching last_hidden_state[0, -1, :] in PyTorch.
  const float *embedding = lastHiddenState.getData() + (maxSeqLen - 1) * hiddenSize;

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mE5-Mistral Sentence Embedding\033[0m\n";
    std::cerr << "  dim  : " << hiddenSize << "\n";
    std::cerr << "  time : " << seconds << "s\n";
  }

  std::cout << "[";
  for (size_t d = 0; d < hiddenSize; ++d) {
    if (d)
      std::cout << ", ";
    std::cout << embedding[d];
  }
  std::cout << "]\n";

  free(lastHiddenState.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
