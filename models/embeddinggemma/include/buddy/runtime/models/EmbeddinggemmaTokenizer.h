//===- EmbeddinggemmaTokenizer.h - Gemma BPE tokenizer, pure C++ ---------===//
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
// google/embeddinggemma-300m ships a Gemma3TextModel whose tokenizer is the
// Gemma byte-level BPE exported by HuggingFace `tokenizers` as tokenizer.json:
//
//   model:        {type: "BPE", vocab: {piece: id}, merges: [[a, b], ...],
//                  unk_token: "<unk>", byte_fallback: true, fuse_unk: true}
//   normalizer:   Replace(" " -> "▁" U+2581)
//   pre_tokenizer:Split(pattern=" ", behavior=MergedWithPrevious)  -- after
//                  the normalizer no literal space remains, so the whole
//                  input becomes a single pre-token (as with the real model)
//   post_processor: <bos> A <eos>
//
// This is a dependency-free reimplementation of that BPE pipeline:
//   1. extract added tokens, then replace ASCII spaces with "▁" in other text,
//   2. split the normalized UTF-8 string into codepoint symbols,
//      using byte fallback for symbols outside the vocabulary,
//   3. greedily apply the ranked merges table (lowest rank wins) until no
//      adjacent pair is mergeable,
//   4. map each resulting symbol to its vocab id (fusing consecutive
//      unknown tokens into a single <unk>, mirroring fuse_unk),
//   5. wrap as [<bos>, content..., <eos>] and right-pad with <pad>.
//
// It reproduces `AutoTokenizer(text, padding="max_length", truncation=True,
// max_length=N)` without a Python/`transformers` runtime dependency. The
// algorithm was validated token-for-token against GemmaTokenizerFast.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMATOKENIZER_H
#define BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMATOKENIZER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {

class EmbeddinggemmaTokenizer {
public:
  static EmbeddinggemmaTokenizer
  loadFromFile(const std::string &tokenizerJsonPath) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(tokenizerJsonPath);
    if (!bufOrErr)
      throw std::runtime_error("EmbeddinggemmaTokenizer: cannot open " +
                               tokenizerJsonPath + ": " +
                               bufOrErr.getError().message());

    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*bufOrErr)->getBuffer());
    if (!parsed)
      throw std::runtime_error("EmbeddinggemmaTokenizer: failed to parse " +
                               tokenizerJsonPath + ": " +
                               llvm::toString(parsed.takeError()));

    const llvm::json::Object *root = parsed->getAsObject();
    if (!root)
      throw std::runtime_error("EmbeddinggemmaTokenizer: " + tokenizerJsonPath +
                               " root is not a JSON object");

    EmbeddinggemmaTokenizer tok;
    tok.loadModel(*root);
    tok.loadSpecialTokens(*root);
    return tok;
  }

  // Reproduces AutoTokenizer(text, padding="max_length", truncation=True,
  // max_length=maxSeqLen): [<bos>, content..., <eos>], content truncated so
  // the wrapped sequence fits maxSeqLen, right-padded with padId_ / mask 0.
  void encode(const std::string &text, size_t maxSeqLen,
              std::vector<int64_t> &inputIds,
              std::vector<int64_t> &attentionMask) const {
    if (maxSeqLen < 2)
      throw std::runtime_error(
          "EmbeddinggemmaTokenizer: maxSeqLen must be >= 2");

    std::vector<int64_t> content = tokenize(text);
    size_t maxContent = maxSeqLen - 2;
    if (content.size() > maxContent)
      content.resize(maxContent);

    inputIds.clear();
    attentionMask.clear();
    inputIds.reserve(maxSeqLen);
    attentionMask.reserve(maxSeqLen);

    inputIds.push_back(bosId_);
    inputIds.insert(inputIds.end(), content.begin(), content.end());
    inputIds.push_back(eosId_);
    attentionMask.assign(inputIds.size(), 1);

    while (inputIds.size() < maxSeqLen) {
      inputIds.push_back(padId_);
      attentionMask.push_back(0);
    }
  }

private:
  using SymbolId = uint32_t;
  static constexpr SymbolId kUnknownSymbol = 0;

  struct Merge {
    SymbolId result;
    size_t rank;
  };

  // Internal IDs identify strings independently of their output token IDs.
  // Merge operands and results can exist even without a vocabulary entry.
  SymbolId internSymbol(llvm::StringRef symbol) {
    if (tokenIds_.size() > std::numeric_limits<SymbolId>::max())
      throw std::runtime_error("EmbeddinggemmaTokenizer: too many BPE symbols");
    auto [it, inserted] =
        symbols_.try_emplace(symbol, static_cast<SymbolId>(tokenIds_.size()));
    if (inserted)
      tokenIds_.emplace_back();
    return it->getValue();
  }

  SymbolId findSymbol(llvm::StringRef symbol) const {
    auto it = symbols_.find(symbol);
    return it == symbols_.end() ? kUnknownSymbol : it->getValue();
  }

  // ── Loading ───────────────────────────────────────────────────────────

  void loadModel(const llvm::json::Object &root) {
    const llvm::json::Object *model = root.getObject("model");
    if (!model)
      throw std::runtime_error("EmbeddinggemmaTokenizer: missing \"model\"");

    auto type = model->getString("type");
    if (!type || *type != "BPE")
      throw std::runtime_error(
          "EmbeddinggemmaTokenizer: unsupported tokenizer model type "
          "(expected BPE)");

    const llvm::json::Object *vocabObj = model->getObject("vocab");
    if (!vocabObj)
      throw std::runtime_error("EmbeddinggemmaTokenizer: missing "
                               "\"model.vocab\"");
    symbols_ = llvm::StringMap<SymbolId>(vocabObj->size());
    tokenIds_.reserve(vocabObj->size() + 1);
    for (const auto &kv : *vocabObj) {
      auto id = kv.second.getAsInteger();
      if (!id)
        throw std::runtime_error("EmbeddinggemmaTokenizer: vocab id is not an "
                                 "integer");
      tokenIds_[internSymbol(kv.first)] = *id;
    }

    const llvm::json::Array *mergesArr = model->getArray("merges");
    if (!mergesArr)
      throw std::runtime_error("EmbeddinggemmaTokenizer: missing "
                               "\"model.merges\"");
    merges_.reserve(mergesArr->size());
    for (size_t i = 0; i < mergesArr->size(); ++i) {
      const llvm::json::Value &entry = (*mergesArr)[i];
      const llvm::json::Array *pair = entry.getAsArray();
      if (!pair || pair->size() != 2)
        throw std::runtime_error("EmbeddinggemmaTokenizer: malformed merge");
      auto a = (*pair)[0].getAsString();
      auto b = (*pair)[1].getAsString();
      if (!a || !b)
        throw std::runtime_error("EmbeddinggemmaTokenizer: malformed merge");
      const SymbolId left = internSymbol(*a);
      const SymbolId right = internSymbol(*b);
      std::string merged = a->str();
      merged.append(b->data(), b->size());
      merges_.try_emplace({left, right}, Merge{internSymbol(merged), i});
    }

    if (auto unk = model->getString("unk_token"))
      unkId_ = tokenIds_[findSymbol(*unk)].value_or(unkId_);
    byteFallback_ = model->getBoolean("byte_fallback").value_or(false);
  }

  void loadSpecialTokens(const llvm::json::Object &root) {
    bosId_ = findTokenId(root, "<bos>", 2);
    eosId_ = findTokenId(root, "<eos>", 1);
    padId_ = findTokenId(root, "<pad>", 0);
    if (const auto *tokens = root.getArray("added_tokens")) {
      for (const auto &value : *tokens) {
        const auto *token = value.getAsObject();
        if (!token || !token->getString("content") || !token->getInteger("id"))
          throw std::runtime_error(
              "EmbeddinggemmaTokenizer: malformed added token");
        // EmbeddingGemma's added tokens use literal, unnormalized matching.
        if (token->getBoolean("single_word").value_or(false) ||
            token->getBoolean("lstrip").value_or(false) ||
            token->getBoolean("rstrip").value_or(false) ||
            token->getBoolean("normalized").value_or(false))
          throw std::runtime_error(
              "EmbeddinggemmaTokenizer: unsupported added token options");
        std::string content = token->getString("content")->str();
        if (!content.empty())
          addedTokens_[content.front()].emplace_back(content,
                                                     *token->getInteger("id"));
      }
    }
  }

  // Reads a special token's id via post_processor.special_tokens (authoritative
  // for bos/eos), falling back to `fallback`.
  static int64_t findTokenId(const llvm::json::Object &root,
                             llvm::StringRef content, int64_t fallback) {
    if (const llvm::json::Object *pp = root.getObject("post_processor")) {
      if (const llvm::json::Object *special = pp->getObject("special_tokens")) {
        if (const llvm::json::Object *entry = special->getObject(content)) {
          if (const llvm::json::Array *ids = entry->getArray("ids")) {
            if (!ids->empty())
              if (auto id = (*ids)[0].getAsInteger())
                return *id;
          }
        }
      }
    }
    return fallback;
  }

  // ── Tokenization ──────────────────────────────────────────────────────

  std::vector<int64_t> tokenize(const std::string &text) const {
    std::vector<int64_t> ids;
    size_t start = 0;
    auto appendText = [&](size_t end) {
      auto pieces = tokenizeText(text.substr(start, end - start));
      ids.insert(ids.end(), pieces.begin(), pieces.end());
    };
    for (size_t pos = 0; pos < text.size();) {
      const std::pair<std::string, int64_t> *match = nullptr;
      auto candidates = addedTokens_.find(text[pos]);
      if (candidates != addedTokens_.end()) {
        for (const auto &token : candidates->second) {
          if ((!match || token.first.size() > match->first.size()) &&
              text.compare(pos, token.first.size(), token.first) == 0)
            match = &token;
        }
      }
      if (!match) {
        ++pos;
        continue;
      }
      appendText(pos);
      ids.push_back(match->second);
      pos += match->first.size();
      start = pos;
    }
    appendText(text.size());
    return ids;
  }

  std::vector<int64_t> tokenizeText(const std::string &text) const {
    // Normalizer: " " -> "▁" (U+2581, UTF-8 E2 96 81).
    std::string norm;
    norm.reserve(text.size() + 4);
    for (char c : text) {
      if (c == ' ')
        norm += kMetaspace;
      else
        norm.push_back(c);
    }

    // The pre-tokenizer is a no-op after normalization (no literal space
    // remains), so the whole string is one pre-token. Run BPE on it.
    auto codepoints = splitCodepoints(norm);
    std::vector<SymbolId> word;
    word.reserve(codepoints.size());
    for (const auto &sym : codepoints) {
      const SymbolId symbol = findSymbol(sym);
      std::vector<SymbolId> bytes;
      if (byteFallback_ && !tokenIds_[symbol]) {
        constexpr char hex[] = "0123456789ABCDEF";
        for (unsigned char byte : sym) {
          std::string piece = "<0x00>";
          piece[3] = hex[byte >> 4];
          piece[4] = hex[byte & 15];
          const SymbolId byteSymbol = findSymbol(piece);
          if (!tokenIds_[byteSymbol])
            break;
          bytes.push_back(byteSymbol);
        }
      }
      if (!bytes.empty() && bytes.size() == sym.size())
        word.insert(word.end(), bytes.begin(), bytes.end());
      else
        word.push_back(symbol);
    }
    bpeMerge(word);

    std::vector<int64_t> ids;
    ids.reserve(word.size());
    bool prevUnknown = false;
    for (SymbolId symbol : word) {
      if (auto id = tokenIds_[symbol]) {
        ids.push_back(*id);
        prevUnknown = false;
      } else {
        // fuse_unk=true: consecutive unknown symbols collapse to one <unk>.
        if (!prevUnknown)
          ids.push_back(unkId_);
        prevUnknown = true;
      }
    }
    return ids;
  }

  static std::vector<std::string> splitCodepoints(const std::string &s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
      unsigned char c = static_cast<unsigned char>(s[i]);
      size_t n;
      if ((c & 0x80) == 0x00)
        n = 1;
      else if ((c & 0xE0) == 0xC0)
        n = 2;
      else if ((c & 0xF0) == 0xE0)
        n = 3;
      else if ((c & 0xF8) == 0xF0)
        n = 4;
      else
        n = 1;
      if (i + n > s.size())
        n = s.size() - i;
      out.push_back(s.substr(i, n));
      i += n;
    }
    return out;
  }

  // Greedy byte-pair merging: repeatedly merge the adjacent pair with the
  // lowest rank present in the merges table.
  void bpeMerge(std::vector<SymbolId> &word) const {
    while (word.size() > 1) {
      size_t bestIdx = 0;
      const Merge *best = nullptr;
      for (size_t i = 0; i + 1 < word.size(); ++i) {
        auto it = merges_.find({word[i], word[i + 1]});
        if (it != merges_.end() && (!best || it->second.rank < best->rank)) {
          bestIdx = i;
          best = &it->second;
        }
      }
      if (!best)
        break;
      word[bestIdx] = best->result;
      word.erase(word.begin() + bestIdx + 1);
    }
  }

  static const char *kMetaspace;

  llvm::StringMap<SymbolId> symbols_;
  std::vector<std::optional<int64_t>> tokenIds_{std::nullopt};
  llvm::DenseMap<std::pair<SymbolId, SymbolId>, Merge> merges_;
  std::unordered_map<char, std::vector<std::pair<std::string, int64_t>>>
      addedTokens_;
  bool byteFallback_ = false;

  int64_t bosId_ = 2;
  int64_t eosId_ = 1;
  int64_t padId_ = 0;
  int64_t unkId_ = 3;
};

inline const char *EmbeddinggemmaTokenizer::kMetaspace = "\xE2\x96\x81";

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_EMBEDDINGGEMMATOKENIZER_H
