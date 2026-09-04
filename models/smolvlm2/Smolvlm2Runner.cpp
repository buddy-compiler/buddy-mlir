//===- Smolvlm2Runner.cpp - SmolVLM2 text-LM runner -----------------------===//
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
// SmolVLM2 text-only (Llama3 decoder) runner.
//
// The AOT importer (codegen/import-smolvlm2.py) traces SmolVLMModel.text_model
// (a 32-layer Llama3 decoder, hidden=960) for a fixed 64-token input. The
// resulting forward ABI -- this is the EXACT ABI encoded below -- is:
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<1 x max_seq_len x i64>,
//           attention_mask: memref<1 x max_seq_len x i64>)
//     -> (last_hidden_state: memref<1 x max_seq_len x hidden_size x f32>)
//
// `_mlir_ciface_forward` follows the buddy C ABI rule: ONE pointer per result
// memref, then ONE pointer per input memref, in declaration order.  Because the
// forward has a single result, the function pointer is:
//
//   void (*)(MemRef<float,3>  *last_hidden_state,
//            MemRef<float,1>  *weights,          // flattened arg0.data
//            MemRef<int64_t,2> *input_ids,       // 1 x max_seq_len
//            MemRef<int64_t,2> *attention_mask)  // 1 x max_seq_len
//
// The runner loads the `.rax` manifest (or explicit model-so/weights/vocab
// paths), byte-level-BPE-tokenizes the prompt against the staged
// `tokenizer.json`, invokes the compiled kernel, and emits the final-token
// 960-dim last_hidden_state vector.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/Smolvlm2Runner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 64;
constexpr size_t kDefaultHiddenSize = 960;
constexpr size_t kDefaultParamsSize = 361944032;

/// Single result, so the C ABI wrapper takes a pointer to this memref first,
/// then the three input descriptors.
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
    throw std::runtime_error("Smolvlm2Runner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("Smolvlm2Runner: error reading weights: " +
                             weightsPath);
}

//===----------------------------------------------------------------------===//
// Minimal UTF-8 helpers
//===----------------------------------------------------------------------===//

inline void utf8Encode(unsigned int cp, std::string &out) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

/// GPT-2 byte-level encoder: byte -> UTF-8 of the mapped code point.  Byte
/// 0x20 (space) maps to U+0120 "Ġ"; printable ASCII maps to itself.
struct ByteEncoder {
  std::string byteToChar[256];
  ByteEncoder() {
    std::vector<int> bs, cs;
    for (int b = 33; b <= 126; ++b)
      bs.push_back(b);
    for (int b = 161; b <= 172; ++b)
      bs.push_back(b);
    for (int b = 174; b <= 255; ++b)
      bs.push_back(b);
    cs = bs;
    std::array<bool, 256> inSet;
    inSet.fill(false);
    for (int b : bs)
      inSet[static_cast<size_t>(b)] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (!inSet[static_cast<size_t>(b)]) {
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }
    for (size_t i = 0; i < bs.size(); ++i)
      utf8Encode(static_cast<unsigned int>(cs[i]),
                 byteToChar[static_cast<size_t>(bs[i])]);
  }
};

/// Minimal GPT-2-style byte-level BPE tokenizer reading a HuggingFace
/// `tokenizer.json`.  This is a faithful byte-level BPE (special tokens,
/// byte-encoding, and the model's merge table); it does not apply the exact
/// HF regex pre-tokenizer, but is fully deterministic and self-contained.
class ByteLevelBPETokenizer {
public:
  explicit ByteLevelBPETokenizer(const std::string &tokenizerJson) {
    load(tokenizerJson);
  }

  /// Encode `text` into token ids.  Never exceeds `maxLen`.
  std::vector<int64_t> encode(const std::string &text, size_t maxLen) const {
    std::vector<int64_t> ids;
    encodeImpl(text, ids);
    if (ids.size() > maxLen)
      ids.resize(maxLen);
    return ids;
  }

  int padId() const { return padId_; }

private:
  void load(const std::string &tokenizerJson) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(tokenizerJson);
    if (!bufOrErr)
      throw std::runtime_error("Smolvlm2Runner: cannot read tokenizer: " +
                               tokenizerJson);
    auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
    if (!parsed)
      throw std::runtime_error("Smolvlm2Runner: cannot parse tokenizer.json: " +
                               llvm::toString(parsed.takeError()));

    const llvm::json::Object *root = parsed->getAsObject();
    const llvm::json::Object *model =
        root ? root->getObject("model") : nullptr;
    const llvm::json::Object *vocab =
        model ? model->getObject("vocab") : nullptr;
    if (!vocab)
      throw std::runtime_error(
          "Smolvlm2Runner: tokenizer.json missing model.vocab");

    vocab_.reserve(vocab->size());
    for (const auto &kv : *vocab) {
      auto id = kv.second.getAsInteger();
      if (id)
        vocab_[kv.first.str()] = static_cast<int>(*id);
    }
    if (vocab_.empty())
      throw std::runtime_error("Smolvlm2Runner: empty tokenizer vocab");

    // Merge ranks: "a b" -> rank (lower rank merges earlier).
    if (const llvm::json::Array *merges = model->getArray("merges")) {
      for (size_t i = 0; i < merges->size(); ++i) {
        if (auto s = (*merges)[i].getAsString())
          mergeRanks_[s->str()] = static_cast<int>(i);
      }
    }

    // Special / added tokens are matched verbatim before BPE.
    if (const llvm::json::Array *added = root->getArray("added_tokens")) {
      for (const auto &entry : *added) {
        const llvm::json::Object *obj = entry.getAsObject();
        if (!obj)
          continue;
        auto id = obj->getInteger("id");
        auto content = obj->getString("content");
        if (id && content)
          specialToId_[content->str()] = static_cast<int>(*id);
      }
    }
    if (specialToId_.empty()) {
      // Fall back: treat "<...>" keys in the base vocab as special.
      for (const auto &kv : vocab_)
        if (kv.first.size() > 2 && kv.first.front() == '<' &&
            kv.first.back() == '>')
          specialToId_[kv.first] = kv.second;
    }

    auto padIt = vocab_.find("<|im_end|>");
    if (padIt == vocab_.end())
      padIt = vocab_.find("<pad>");
    padId_ = padIt != vocab_.end() ? padIt->second : 2;
  }

  // Encode with special-token matching plus byte-level BPE.
  void encodeImpl(const std::string &text, std::vector<int64_t> &ids) const {
    static const ByteEncoder encoder;
    size_t i = 0;
    std::string regular;
    auto flushRegular = [&]() {
      if (!regular.empty()) {
        encodeWords(splitWords(regular), ids);
        regular.clear();
      }
    };

    // Greedy longest special-token match at each position.
    while (i < text.size()) {
      bool matched = false;
      size_t bestLen = 0;
      for (const auto &kv : specialToId_) {
        const std::string &tok = kv.first;
        if (tok.size() > bestLen && text.compare(i, tok.size(), tok) == 0) {
          bestLen = tok.size();
          matched = true;
        }
      }
      if (matched) {
        flushRegular();
        ids.push_back(specialToId_.at(text.substr(i, bestLen)));
        i += bestLen;
      } else {
        regular += text[i];
        ++i;
      }
    }
    flushRegular();
  }

  // GPT2-like word splitting (approximation of the ByteLevel + Digits
  // pretokenizers).  Whitespace runs are attached to the following word so
  // merges see the "Ġ..." prefix; digit runs split into individual digits.
  std::vector<std::string> splitWords(const std::string &text) const {
    std::vector<std::string> words;
    const size_t n = text.size();
    auto isSpace = [](char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
             c == '\v';
    };
    auto isDigit = [](char c) { return c >= '0' && c <= '9'; };
    auto isWordChar = [&](char c) {
      if (isSpace(c))
        return false;
      // Approximate \p{L}\p{N}: ASCII alnum/underscore plus any non-ASCII
      // (multi-byte UTF-8) byte, which groups CJK/emoji into words.
      if (isDigit(c))
        return true;
      if (c == '_')
        return true;
      if (static_cast<unsigned char>(c) >= 0x80)
        return true;
      return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };

    std::vector<std::string> groups;
    size_t i = 0;
    while (i < n) {
      char c = text[i];
      size_t j = i;
      if (isSpace(c)) {
        while (j < n && isSpace(text[j]))
          ++j;
      } else if (isWordChar(c)) {
        while (j < n && isWordChar(text[j]))
          ++j;
      } else {
        while (j < n && !isSpace(text[j]) && !isWordChar(text[j]))
          ++j;
      }
      groups.push_back(text.substr(i, j - i));
      i = j;
    }

    for (size_t g = 0; g < groups.size(); ++g) {
      const std::string &grp = groups[g];
      // Attach a whitespace run to the following group (the " ?" prefix).
      if (!grp.empty() && isSpace(grp[0]) && g + 1 < groups.size()) {
        groups[g + 1] = grp + groups[g + 1];
        continue;
      }
      if (!grp.empty() && isSpace(grp[0]))
        continue; // trailing whitespace: dropped
      // Split digit runs into individual digits within a word.
      std::string cur;
      for (size_t k = 0; k < grp.size(); ++k) {
        if (isDigit(grp[k])) {
          if (!cur.empty()) {
            words.push_back(cur);
            cur.clear();
          }
          words.push_back(std::string(1, grp[k]));
        } else {
          cur += grp[k];
        }
      }
      if (!cur.empty())
        words.push_back(cur);
    }
    return words;
  }

  // Byte-encode + merge each word and append ids.
  void encodeWords(const std::vector<std::string> &words,
                   std::vector<int64_t> &ids) const {
    static const ByteEncoder encoder;
    for (const std::string &word : words) {
      // Byte-encode: one unicode char per input byte.
      std::vector<std::string> symbols;
      symbols.reserve(word.size());
      for (unsigned char b : word)
        symbols.push_back(encoder.byteToChar[b]);

      // Apply the merge table until no pair can merge.
      bool changed = true;
      while (changed && symbols.size() > 1) {
        changed = false;
        int bestRank = std::numeric_limits<int>::max();
        size_t bestIdx = 0;
        for (size_t k = 0; k + 1 < symbols.size(); ++k) {
          auto it = mergeRanks_.find(symbols[k] + " " + symbols[k + 1]);
          if (it != mergeRanks_.end() && it->second < bestRank) {
            bestRank = it->second;
            bestIdx = k;
            changed = true;
          }
        }
        if (!changed)
          break;
        std::string merged = symbols[bestIdx] + symbols[bestIdx + 1];
        symbols[bestIdx] = std::move(merged);
        symbols.erase(symbols.begin() + static_cast<long>(bestIdx + 1));
      }

      for (const std::string &sym : symbols) {
        auto it = vocab_.find(sym);
        ids.push_back(it != vocab_.end() ? it->second : 0);
      }
    }
  }

  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> mergeRanks_;
  std::unordered_map<std::string, int> specialToId_;
  int padId_ = 2;
};

} // namespace

void Smolvlm2Runner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "Smolvlm2Runner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "Smolvlm2Runner: only single-prompt inference is implemented");

  std::string soPath;
  std::string weightsPath;
  std::string tokenizerPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t paramsSize = kDefaultParamsSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("Smolvlm2Runner: manifest has no weight file");
    tokenizerPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    if (auto it = manifest.moduleAttrs.find("params_size");
        it != manifest.moduleAttrs.end() && !it->second.empty())
      paramsSize = static_cast<size_t>(std::stoull(it->second));
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("Smolvlm2Runner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
  }

  if (tokenizerPath.empty())
    throw std::runtime_error("Smolvlm2Runner: tokenizer path is empty");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Tokenizer : " + tokenizerPath, suppress);

  ByteLevelBPETokenizer tokenizer(tokenizerPath);
  const int64_t padId = tokenizer.padId();

  // Minimal chat scaffold (instruct model): user prompt, assistant turn.
  std::string scaffold = "<|im_start|>user\n" + prompt +
                         "<|im_end|>\n<|im_start|>assistant\n";
  std::vector<int64_t> tokenIds = tokenizer.encode(scaffold, maxSeqLen);
  printLog("Tokenization complete: " + std::to_string(tokenIds.size()) +
               " tokens (max " + std::to_string(maxSeqLen) + ")",
           suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("Smolvlm2Runner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward = reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "Smolvlm2Runner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("Smolvlm2Runner: weight file is not f32-aligned");
  }
  const size_t actualElems = weightBytes / sizeof(float);
  if (actualElems != paramsSize)
    printLog("Note: weight file has " + std::to_string(actualElems) +
                 " f32, spec params_size = " + std::to_string(paramsSize),
             suppress);
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({actualElems});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  // Fixed-shape inputs: 1 x max_seq_len i64.
  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  int64_t *idData = inputIds.getData();
  int64_t *maskData = attentionMask.getData();
  for (size_t i = 0; i < maxSeqLen; ++i) {
    const bool real = i < tokenIds.size();
    idData[i] = real ? tokenIds[i] : padId;
    maskData[i] = real ? 1 : 0;
  }

  MemRef<float, 3> lastHiddenState({1, maxSeqLen, hiddenSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(&lastHiddenState, &paramsContainer, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mSmolVLM2 Last-Hidden-State\033[0m\n";
    std::cerr << "  seq_len  : " << maxSeqLen << "\n";
    std::cerr << "  hidden   : " << hiddenSize << "\n";
    std::cerr << "  time     : " << seconds << "s\n";
  }

  // Emit the final real token's 960-dim hidden-state vector plus summary
  // statistics (deterministic, mirrors validate_accuracy.py's last-position
  // reference).
  const float *data = lastHiddenState.getData();
  const size_t lastReal =
      tokenIds.empty() ? 0 : std::min(tokenIds.size(), maxSeqLen) - 1;
  const float *vec = data + lastReal * hiddenSize;

  double sum = 0.0, sumsq = 0.0;
  size_t argmax = 0;
  for (size_t d = 0; d < hiddenSize; ++d) {
    sum += vec[d];
    sumsq += static_cast<double>(vec[d]) * vec[d];
    if (vec[d] > vec[argmax])
      argmax = d;
  }
  const double mean = sum / static_cast<double>(hiddenSize);
  const double l2 = std::sqrt(sumsq);

  std::cout << "{\n";
  std::cout << "  \"token_index\": " << lastReal << ",\n";
  std::cout << "  \"shape\": [1, " << maxSeqLen << ", " << hiddenSize << "],\n";
  std::cout << "  \"sum\": " << sum << ",\n";
  std::cout << "  \"mean\": " << mean << ",\n";
  std::cout << "  \"l2_norm\": " << l2 << ",\n";
  std::cout << "  \"argmax_dim\": " << argmax << ",\n";
  std::cout << "  \"vector\": [";
  for (size_t d = 0; d < hiddenSize; ++d) {
    if (d)
      std::cout << ", ";
    std::cout << vec[d];
  }
  std::cout << "]\n";
  std::cout << "}\n";

  free(lastHiddenState.release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
