//===- MoLFormerRunner.cpp - MoLFormer molecular embedding runner -------===//
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
// MoLFormer (ibm/MoLFormer-XL-both-10pct) molecular embedding runner.
//
// Loads the compiled `molformer_model.so` and `arg0.data` weights through the
// `.rax` manifest (or explicit paths), tokenizes an input SMILES string with a
// C++ reimplementation of the checkpoint's WordLevel SMILES tokenizer, then
// invokes the AOT forward graph (the full 12-layer linear-attention encoder
// traced as a single graph):
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (last_hidden_state: memref<1 x max_seq_len x hidden_size x f32>,
//         pooled: memref<1 x hidden_size x f32>)
//
// The pooled (masked-average) embedding is the molecular representation and
// is printed together with a summary of the per-token hidden states.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/MoLFormerRunner.h"
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
constexpr size_t kDefaultHiddenSize = 768;
constexpr int64_t kBosId = 0;
constexpr int64_t kEosId = 1;
constexpr int64_t kPadId = 2;

/// Packed output descriptors.  `@forward` returns both memrefs in one struct,
/// so the C ABI wrapper `_mlir_ciface_forward` takes a single pointer to this
/// struct (results first, then the input descriptors).
struct ForwardResults {
  MemRef<float, 3> lastHiddenState; // 1 x max_seq_len x hidden_size
  MemRef<float, 2> pooled;          // 1 x hidden_size
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
    throw std::runtime_error("MoLFormerRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("MoLFormerRunner: error reading weights: " +
                             weightsPath);
}

namespace fs = std::filesystem;

bool isAsciiDigit(char c) { return c >= '0' && c <= '9'; }

/// MoLFormer SMILES WordLevel tokenizer.
///
/// The checkpoint ships a WordLevel tokenizer whose pre-tokenizer applies the
/// SMILES token regex `(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])`
/// and whose post-processor wraps the sequence in `<bos>` ... `<eos>`.  We
/// reproduce the scanner (leftmost, greedy by alternation order) and the
/// static WordLevel vocabulary; tokens are padded to max_seq_len with `<pad>`.
class SmilesTokenizer {
public:
  explicit SmilesTokenizer(const std::string &vocabPath)
      : vocab_(loadVocab(vocabPath)), unkId_(lookup("<unk>", 2361)) {}

  void encode(const std::string &smiles, size_t maxLen,
              std::vector<int64_t> &inputIds,
              std::vector<int64_t> &attentionMask) {
    inputIds.clear();
    attentionMask.clear();
    std::vector<int64_t> ids;
    ids.push_back(kBosId);
    size_t pos = 0;
    while (pos < smiles.size()) {
      std::string tok;
      nextToken(smiles, pos, tok);
      if (tok.empty())
        continue;
      if (ids.size() >= maxLen - 1)
        break;
      ids.push_back(vocabToken(tok));
    }
    ids.push_back(kEosId);
    if (ids.size() > maxLen) {
      ids.resize(maxLen);
      ids.back() = kEosId;
    }
    inputIds = ids;
    attentionMask.assign(ids.size(), 1);
    inputIds.resize(maxLen, kPadId);
    attentionMask.resize(maxLen, 0);
  }

private:
  static std::unordered_map<std::string, int64_t> loadVocab(
      const std::string &vocabPath) {
    std::ifstream in(vocabPath);
    if (!in)
      throw std::runtime_error(
          "MoLFormerRunner: failed to open tokenizer vocab: " + vocabPath);
    std::unordered_map<std::string, int64_t> vocab;
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (!line.empty() && vocab.count(line) == 0)
        vocab[line] = static_cast<int64_t>(vocab.size());
    }
    if (vocab.empty())
      throw std::runtime_error("MoLFormerRunner: empty vocab: " + vocabPath);
    return vocab;
  }

  int64_t lookup(const std::string &tok, int64_t fallback) const {
    auto it = vocab_.find(tok);
    return it != vocab_.end() ? it->second : fallback;
  }

  int64_t vocabToken(const std::string &tok) const { return lookup(tok, unkId_); }

  /// Extract the next SMILES token at `pos` (regex alternation order).
  static void nextToken(const std::string &s, size_t &pos, std::string &tok) {
    const size_t n = s.size();
    tok.clear();
    while (pos < n) {
      const char c = s[pos];
      // [C@H], [nH], [N+], ... (bracketed atoms).
      if (c == '[') {
        const size_t close = s.find(']', pos + 1);
        if (close != std::string::npos && close > pos + 1) {
          tok = s.substr(pos, close - pos + 1);
          pos = close + 1;
          return;
        }
        tok.assign(1, c);
        ++pos;
        return;
      }
      // Br? / Cl?  (greedy: "Br"/"Cl" preferred over bare B/C).
      if (c == 'B') {
        if (pos + 1 < n && s[pos + 1] == 'r') {
          tok = "Br";
          pos += 2;
        } else {
          tok.assign(1, c);
          ++pos;
        }
        return;
      }
      if (c == 'C') {
        if (pos + 1 < n && s[pos + 1] == 'l') {
          tok = "Cl";
          pos += 2;
        } else {
          tok.assign(1, c);
          ++pos;
        }
        return;
      }
      // Single-letter organic subset: N O S P F I b c n o s p.
      if (c == 'N' || c == 'O' || c == 'S' || c == 'P' || c == 'F' ||
          c == 'I' || c == 'b' || c == 'c' || c == 'n' || c == 'o' ||
          c == 's' || c == 'p') {
        tok.assign(1, c);
        ++pos;
        return;
      }
      // Bond/branch/ring punctuation.
      if (std::strchr("().=#-+\\/:~@?>*$", c) != nullptr) {
        tok.assign(1, c);
        ++pos;
        return;
      }
      // %NN ring-closure index.
      if (c == '%' && pos + 2 < n && isAsciiDigit(s[pos + 1]) &&
          isAsciiDigit(s[pos + 2])) {
        tok = s.substr(pos, 3);
        pos += 3;
        return;
      }
      // Single digit.
      if (isAsciiDigit(c)) {
        tok.assign(1, c);
        ++pos;
        return;
      }
      // Drop any unexpected character.
      ++pos;
    }
  }

  std::unordered_map<std::string, int64_t> vocab_;
  int64_t unkId_;
};

} // namespace

void MoLFormerRunner::run(const RunConfig &cfg) {
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  std::string vocabPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("MoLFormerRunner: manifest has no weight file");
    vocabPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("MoLFormerRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath;
  }

  if (vocabPath.empty())
    throw std::runtime_error("MoLFormerRunner: vocab path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Vocab     : " + vocabPath, suppress);

  SmilesTokenizer tokenizer(vocabPath);
  std::vector<int64_t> inputIdVec, attentionMaskVec;
  tokenizer.encode(prompt, maxSeqLen, inputIdVec, attentionMaskVec);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("MoLFormerRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "MoLFormerRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("MoLFormerRunner: weight file is not f32-aligned");
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

  ForwardResults results = {
      MemRef<float, 3>({1, maxSeqLen, hiddenSize}, false, 0),
      MemRef<float, 2>({1, hiddenSize}, false, 0)};

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&results, &paramsContainer, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  // Pooled molecular embedding (masked average over the real tokens).
  const float *pooled = results.pooled.getData();
  double norm2 = 0.0;
  for (size_t d = 0; d < hiddenSize; ++d)
    norm2 += static_cast<double>(pooled[d]) * static_cast<double>(pooled[d]);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mMoLFormer Molecular Embedding\033[0m\n";
    std::cerr << "  seq_len : " << maxSeqLen << "\n";
    std::cerr << "  dim     : " << hiddenSize << "\n";
    std::cerr << "  time    : " << seconds << "s\n";
    std::cerr << "  |pooled| : " << std::sqrt(norm2) << "\n";
  }

  // Print the pooled embedding (the molecular representation).
  std::cout << "[";
  for (size_t d = 0; d < hiddenSize; ++d) {
    if (d)
      std::cout << ", ";
    std::cout << pooled[d];
  }
  std::cout << "]\n";

  free(results.lastHiddenState.release());
  free(results.pooled.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
