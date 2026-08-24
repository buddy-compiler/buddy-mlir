//===- ChembertaRunner.cpp - ChemBERTa MLM encoder runner ----------------===//
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
// ChemBERTa (DeepChem/ChemBERTa-77M-MLM, a RoBERTa-style masked-LM chemistry
// encoder) runner.
//
// Loads the compiled `chemberta_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), tokenizes the input text with a small
// C++ WordPiece tokenizer over the staged `vocab.txt`, then invokes the AOT
// forward graph.
//
// ── EXACT forward ABI ─────────────────────────────────────────────────────
// The compiled `_mlir_ciface_forward` wrapper (produced by
// `-llvm-request-c-wrappers`) takes ONE pointer per result memref followed by
// ONE pointer per input memref, in declaration order. The MLIR `@forward`
// (see codegen/import-chemberta.py and forward.mlir) is:
//
//   func.func @forward(
//       %arg0: memref<params_size x f32>,      // flattened weights (arg0.data)
//       %arg1: memref<position_buffer_size x i64>, // position_ids gather buf
//       %arg2: memref<1 x max_seq_len x i64>,  // input_ids
//       %arg3: memref<1 x max_seq_len x i64>)  // attention_mask
//     -> memref<1 x max_seq_len x vocab_size x f32>   // logits
//
// The forward returns a SINGLE result (logits), so no result-struct packing is
// needed and the C ABI is:
//
//   void _mlir_ciface_forward(
//       MemRef<float, 3> *logits,        // 1 x max_seq_len x vocab_size
//       MemRef<float, 1> *weights,       // params_size
//       MemRef<int64_t, 1> *position,    // position_buffer_size
//       MemRef<int64_t, 2> *inputIds,    // 1 x max_seq_len
//       MemRef<int64_t, 2> *attentionMask); // 1 x max_seq_len
//
// `position` feeds the token-type gather inside subgraph0
// (token_type_embeddings[position[0:seq_len]]). The traced graph captures the
// (all-zero) `token_type_ids` as this buffer input instead of as a constant, so
// the runner fills it with zeros: gather index 0 always reads the single
// captured row 0, reproducing token_type_embeddings[zeros_like(input_ids)].
// (The actual position_ids used for the position-embedding gather are computed
// at runtime inside subgraph0 as cumsum(input_ids != pad) + pad, not from this
// buffer -- verified: with the buffer zeroed the logits match the PyTorch
// reference exactly.)
//
// MemRef descriptor layout (buddy/Core/Container.h):
//   1-D {allocated, aligned, offset, size0, stride0}
//   2-D adds {size1, stride1}; 3-D adds {size2, stride2}.
// Outputs use MemRef<T,N>({dims...}, /*needMalloc=*/false, 0).
//
// ===----------------------------------------------------------------------===//

#include "buddy/runtime/models/ChembertaRunner.h"
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
#include <unordered_map>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 128;
constexpr size_t kDefaultPositionBufferSize = 515;
constexpr size_t kDefaultHiddenSize = 384;
constexpr size_t kDefaultVocabSize = 600;

/// Single forward output: the MLM logits tensor `1 x max_seq_len x vocab_size`.
/// `_mlir_ciface_forward` takes one pointer per result memref (here just one),
/// then one pointer per input memref, in declaration order.
using ForwardFn = void (*)(MemRef<float, 3> *logits, MemRef<float, 1> *,
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
    throw std::runtime_error("ChembertaRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("ChembertaRunner: error reading weights: " +
                             weightsPath);
}

/// Byte-level tokenizer for DeepChem/ChemBERTa-77M-MLM over a staged vocab.txt.
///
/// The model's official tokenizer is a HuggingFace `tokenizers` BPE with an
/// EMPTY merges table and a ByteLevel pre-tokenizer. With zero merges the BPE
/// degenerates to: pre-tokenize into UTF-8 bytes, keep every byte whose
/// character is present in the vocabulary, drop all others (there is no
/// byte_fallback and no `<unk>` in the model's `tokenizer.json`). Special
/// tokens are the BERT-style set from the vocab: `[CLS]`=12, `[SEP]`=13,
/// `[PAD]`=0, `[UNK]`=11. This reproduces `AutoTokenizer(smiles)` token-for-
/// token (verified against transformers 4.57.1 on SMILES strings).
class ByteLevelTokenizer {
public:
  explicit ByteLevelTokenizer(const std::string &vocabPath) {
    std::ifstream in(vocabPath);
    if (!in)
      throw std::runtime_error(
          "ChembertaRunner: failed to open vocab: " + vocabPath);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (vocab_.count(line) == 0)
        vocab_[line] = static_cast<int>(vocab_.size());
    }
    if (vocab_.empty())
      throw std::runtime_error("ChembertaRunner: empty vocab: " + vocabPath);
    clsId_ = lookup("[CLS]", 12);
    sepId_ = lookup("[SEP]", 13);
    padId_ = lookup("[PAD]", 0);
    unkId_ = lookup("[UNK]", 11);
  }

  void encode(const std::string &text, size_t maxLen,
              std::vector<int64_t> &inputIds,
              std::vector<int64_t> &attentionMask) {
    std::vector<int64_t> content;
    content.reserve(text.size());
    // ByteLevel pre-tokenization: the whole input is split into its UTF-8
    // bytes. Each byte whose ASCII character is a vocab entry becomes a token;
    // every other byte (multi-byte UTF-8, or a character absent from the
    // 591-entry word-level vocabulary) is dropped. No byte merges exist.
    for (unsigned char c : text) {
      if (c < 0x80) {
        std::string sym(1, static_cast<char>(c));
        auto it = vocab_.find(sym);
        if (it != vocab_.end())
          content.push_back(it->second);
      }
    }
    // Wrap as [CLS] content... [SEP], truncating content so the wrapped
    // sequence fits maxLen, then right-pad with [PAD] / mask 0.
    std::vector<int64_t> ids;
    ids.reserve(maxLen);
    ids.push_back(clsId_);
    for (int64_t id : content) {
      if (ids.size() >= maxLen - 1)
        break;
      ids.push_back(id);
    }
    ids.push_back(sepId_);
    if (ids.size() > maxLen) {
      ids.resize(maxLen);
      ids.back() = sepId_;
    }
    inputIds = ids;
    attentionMask.assign(ids.size(), 1);
    inputIds.resize(maxLen, padId_);
    attentionMask.resize(maxLen, 0);
  }

private:
  int lookup(const std::string &tok, int fallback) {
    auto it = vocab_.find(tok);
    return it != vocab_.end() ? it->second : fallback;
  }

  std::unordered_map<std::string, int> vocab_;
  int clsId_ = 12;
  int sepId_ = 13;
  int padId_ = 0;
  int unkId_ = 11;
};

} // namespace

void ChembertaRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "ChembertaRunner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "ChembertaRunner: only single-prompt inference is implemented");

  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t positionBufferSize = kDefaultPositionBufferSize;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t vocabSize = kDefaultVocabSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("ChembertaRunner: manifest has no weight file");
    vocabPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    positionBufferSize =
        parseSizeAttr(manifest, "position_buffer_size", positionBufferSize);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    vocabSize = parseSizeAttr(manifest, "vocab_size", vocabSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("ChembertaRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath;
  }

  if (vocabPath.empty())
    throw std::runtime_error("ChembertaRunner: vocab path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Vocab     : " + vocabPath, suppress);

  ByteLevelTokenizer tokenizer(vocabPath);
  std::vector<int64_t> inputIdVec, attentionMaskVec;
  tokenizer.encode(prompt, maxSeqLen, inputIdVec, attentionMaskVec);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("ChembertaRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "ChembertaRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("ChembertaRunner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  // `position` feeds the token-type gather; all-zero token_type_ids must be
  // reproduced, so the whole buffer is zeroed (gather index 0 -> row 0).
  MemRef<int64_t, 1> positionIds({positionBufferSize});
  for (size_t i = 0; i < positionBufferSize; ++i)
    positionIds.getData()[i] = 0;

  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  std::copy(inputIdVec.begin(), inputIdVec.end(), inputIds.getData());
  std::copy(attentionMaskVec.begin(), attentionMaskVec.end(),
            attentionMask.getData());

  // Single result: one pointer to the logits memref, then the three inputs.
  MemRef<float, 3> logits({1, maxSeqLen, vocabSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&logits, &paramsContainer, &positionIds, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mChemBERTa Masked-LM Logits\033[0m\n";
    std::cerr << "  seq_len : " << maxSeqLen << "\n";
    std::cerr << "  vocab   : " << vocabSize << "\n";
    std::cerr << "  time    : " << seconds << "s\n";
  }

  // Emit the full logits tensor (1 x max_seq_len x vocab_size) as nested arrays.
  const float *data = logits.getData();
  std::cout << "[";
  for (size_t tok = 0; tok < maxSeqLen; ++tok) {
    if (tok)
      std::cout << ", ";
    std::cout << "[";
    for (size_t v = 0; v < vocabSize; ++v) {
      if (v)
        std::cout << ", ";
      std::cout << data[tok * vocabSize + v];
    }
    std::cout << "]";
  }
  std::cout << "]\n";

  free(logits.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
