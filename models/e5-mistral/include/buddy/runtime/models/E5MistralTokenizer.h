//===- E5MistralTokenizer.h - Llama BPE tokenizer, pure C++ ---------------===//
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
// E5-Mistral-7B-Instruct ships the Llama-2 style SentencePiece BPE tokenizer
// (tokenizer_class = LlamaTokenizer, tokenizer.json "model.type" = "BPE" with
// byte_fallback). This loads it straight from the HuggingFace fast-tokenizer
// export and reproduces:
//
//   AutoTokenizer(text, padding="max_length", truncation=True, max_length=N)
//
// which, for this model, produces a LEFT-padded sequence:
//   [</s>] * pad + [<s>] + bpe(text) + [</s>]
// (pad_token == eos_token == "</s>"; e5-mistral's tokenizer_config only sets
// add_eos_token=True and leaves add_bos_token unset, so the Llama tokenizer's
// default -- prepend "<s>" -- applies and IS included in the output).
//
// Algorithm (independent reimplementation of SentencePiece BPE):
//   * Normalizer: prepend U+2581 ("▁"), then replace every ASCII space with
//     U+2581.
//   * Pre-tokenizer is null, so the whole normalized string is one segment.
//   * Initial pieces are UTF-8 code points that exist in the vocab; any other
//     code point is split into its bytes as "<0xXX>" (uppercase hex) via the
//     byte_fallback tokens.
//   * BPE merge: repeatedly merge the adjacent pair with the lowest merge rank
//     (global minimum over the current piece list) until no pair remains in the
//     merge table.
//   * added_tokens (special) substrings are matched on the RAW input before
//     normalization; each surrounding plain-text run is normalized + BPE'd
//     independently (matches HF `tokenizers`).
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_E5MISTRALTOKENIZER_H
#define BUDDY_RUNTIME_MODELS_E5MISTRALTOKENIZER_H

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {

class E5MistralTokenizer {
public:
  static E5MistralTokenizer loadFromFile(const std::string &tokenizerJsonPath) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(tokenizerJsonPath);
    if (!bufOrErr)
      throw std::runtime_error("E5MistralTokenizer: cannot open " +
                               tokenizerJsonPath + ": " +
                               bufOrErr.getError().message());

    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*bufOrErr)->getBuffer());
    if (!parsed)
      throw std::runtime_error("E5MistralTokenizer: failed to parse " +
                               tokenizerJsonPath + ": " +
                               llvm::toString(parsed.takeError()));

    const llvm::json::Object *root = parsed->getAsObject();
    if (!root)
      throw std::runtime_error("E5MistralTokenizer: " + tokenizerJsonPath +
                               " root is not a JSON object");

    E5MistralTokenizer tok;
    tok.loadVocabAndMerges(*root);
    tok.loadAddedTokens(*root);
    return tok;
  }

  // Reproduces AutoTokenizer(text, padding="max_length", truncation=True,
  // max_length=maxSeqLen): [</s>] * pad + [<s>] + [content...] + [</s>],
  // LEFT-padded to maxSeqLen with "</s>" (pad token). e5-mistral's
  // tokenizer_config only sets add_eos_token=True; add_bos_token is unset, so
  // the Llama tokenizer DEFAULT (add BOS) applies and "<s>" IS prepended --
  // verified empirically: HF "hello world" -> [1, 6312, 28709, 1526, 2].
  // attention_mask = 1 on the content positions.
  void encode(const std::string &text, size_t maxSeqLen,
              std::vector<int64_t> &inputIds,
              std::vector<int64_t> &attentionMask) const {
    if (maxSeqLen == 0)
      throw std::runtime_error("E5MistralTokenizer: maxSeqLen must be > 0");

    std::vector<int64_t> content = tokenize(text);
    size_t maxContent = maxSeqLen >= 2 ? maxSeqLen - 2 : 0;
    if (content.size() > maxContent)
      content.resize(maxContent);

    std::vector<int64_t> seq;
    seq.reserve(content.size() + 2);
    seq.push_back(bosId_);
    seq.insert(seq.end(), content.begin(), content.end());
    seq.push_back(eosId_);
    if (seq.size() > maxSeqLen)
      seq.resize(maxSeqLen);

    const size_t n = seq.size();
    const size_t pad = maxSeqLen - n;
    inputIds.assign(maxSeqLen, padId_);
    attentionMask.assign(maxSeqLen, 0);
    std::copy(seq.begin(), seq.end(), inputIds.begin() + static_cast<long>(pad));
    std::fill(attentionMask.begin() + static_cast<long>(pad),
              attentionMask.end(), 1);
  }

private:
  static constexpr const char *kMetaSpace = "\xE2\x96\x81"; // U+2581 "▁"

  // ── Loading ───────────────────────────────────────────────────────────

  void loadVocabAndMerges(const llvm::json::Object &root) {
    const llvm::json::Object *model = root.getObject("model");
    if (!model)
      throw std::runtime_error("E5MistralTokenizer: missing \"model\"");
    auto type = model->getString("type");
    if (!type || *type != "BPE")
      throw std::runtime_error(
          "E5MistralTokenizer: unsupported tokenizer model type (expected BPE)");

    const llvm::json::Object *vocabObj = model->getObject("vocab");
    if (!vocabObj)
      throw std::runtime_error("E5MistralTokenizer: missing \"model.vocab\"");
    vocab_.reserve(vocabObj->size());
    for (const auto &kv : *vocabObj) {
      auto id = kv.second.getAsInteger();
      if (!id)
        throw std::runtime_error("E5MistralTokenizer: malformed vocab entry");
      vocab_[kv.first.str()] = *id;
    }

    if (auto unkToken = model->getString("unk_token")) {
      auto it = vocab_.find(unkToken->str());
      if (it != vocab_.end())
        unkId_ = it->second;
    }

    const llvm::json::Array *mergesArr = model->getArray("merges");
    if (!mergesArr)
      throw std::runtime_error("E5MistralTokenizer: missing \"model.merges\"");
    mergeRank_.reserve(mergesArr->size());
    for (size_t i = 0; i < mergesArr->size(); ++i) {
      auto piece = (*mergesArr)[i].getAsString();
      if (!piece)
        throw std::runtime_error("E5MistralTokenizer: malformed merge entry");
      mergeRank_[piece->str()] = static_cast<int64_t>(i);
    }
  }

  void loadAddedTokens(const llvm::json::Object &root) {
    const llvm::json::Array *added = root.getArray("added_tokens");
    if (added) {
      for (const llvm::json::Value &v : *added) {
        const llvm::json::Object *o = v.getAsObject();
        if (!o)
          continue;
        auto content = o->getString("content");
        auto id = o->getInteger("id");
        auto special = o->getBoolean("special");
        if (content && id && special && *special)
          specialTokens_.emplace_back(content->str(), *id);
      }
    }
    // Longest-first so e.g. "</s>" is matched before any prefix confusion.
    std::sort(specialTokens_.begin(), specialTokens_.end(),
              [](const auto &a, const auto &b) {
                return a.first.size() > b.first.size();
              });

    bosId_ = findSpecialId("<s>", 1);
    eosId_ = findSpecialId("</s>", 2);
    padId_ = findSpecialId("</s>", 2);
  }

  int64_t findSpecialId(const std::string &content, int64_t fallback) const {
    for (const auto &sp : specialTokens_)
      if (sp.first == content)
        return sp.second;
    return fallback;
  }

  // ── UTF-8 helpers ─────────────────────────────────────────────────────

  static size_t utf8Len(unsigned char c) {
    if ((c & 0x80) == 0x00)
      return 1;
    if ((c & 0xE0) == 0xC0)
      return 2;
    if ((c & 0xF0) == 0xE0)
      return 3;
    if ((c & 0xF8) == 0xF0)
      return 4;
    return 1; // invalid leading byte
  }

  static bool tryUtf8Len(const std::string &input, size_t offset, size_t &len) {
    if (offset >= input.size())
      return false;
    size_t n = utf8Len(static_cast<unsigned char>(input[offset]));
    if (offset + n > input.size())
      return false;
    for (size_t i = 1; i < n; ++i)
      if ((static_cast<unsigned char>(input[offset + i]) & 0xC0) != 0x80)
        return false;
    len = n;
    return true;
  }

  // ── Normalization ─────────────────────────────────────────────────────

  std::string normalize(const std::string &text) const {
    // Prepend U+2581, then replace every ASCII space with U+2581.
    std::string out;
    out.reserve(text.size() + 4);
    out += kMetaSpace;
    for (char c : text) {
      if (c == ' ')
        out += kMetaSpace;
      else
        out.push_back(c);
    }
    return out;
  }

  // ── Piece splitting + BPE merge ───────────────────────────────────────

  std::vector<std::string> initialPieces(const std::string &text) const {
    std::vector<std::string> pieces;
    size_t offset = 0;
    while (offset < text.size()) {
      size_t len;
      if (!tryUtf8Len(text, offset, len))
        len = 1;
      std::string cp = text.substr(offset, len);
      if (vocab_.count(cp)) {
        pieces.push_back(std::move(cp));
      } else {
        // byte_fallback: split the code point into "<0xXX>" byte tokens.
        char buf[8];
        for (size_t i = 0; i < len; ++i) {
          unsigned char b = static_cast<unsigned char>(text[offset + i]);
          std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<int>(b));
          pieces.emplace_back(buf);
        }
      }
      offset += len;
    }
    return pieces;
  }

  std::vector<std::string> bpeMerge(std::vector<std::string> pieces) const {
    while (pieces.size() >= 2) {
      int64_t bestRank = -1;
      size_t bestI = 0;
      for (size_t i = 0; i + 1 < pieces.size(); ++i) {
        std::string key = pieces[i];
        key += ' ';
        key += pieces[i + 1];
        auto it = mergeRank_.find(key);
        if (it == mergeRank_.end())
          continue;
        if (bestRank < 0 || it->second < bestRank) {
          bestRank = it->second;
          bestI = i;
        }
      }
      if (bestRank < 0)
        break;
      pieces[bestI] += pieces[bestI + 1];
      pieces.erase(pieces.begin() + static_cast<long>(bestI) + 1);
    }
    return pieces;
  }

  std::vector<int64_t> toIds(const std::vector<std::string> &pieces) const {
    std::vector<int64_t> ids;
    ids.reserve(pieces.size());
    for (const auto &p : pieces) {
      auto it = vocab_.find(p);
      ids.push_back(it != vocab_.end() ? it->second : unkId_);
    }
    return ids;
  }

  // Normalize + BPE one plain-text run (contains no special tokens).
  std::vector<int64_t> tokenizeSegment(const std::string &text) const {
    return toIds(bpeMerge(initialPieces(normalize(text))));
  }

  bool isSpecialAt(const std::string &text, size_t pos) const {
    for (const auto &sp : specialTokens_)
      if (pos + sp.first.size() <= text.size() &&
          text.compare(pos, sp.first.size(), sp.first) == 0)
        return true;
    return false;
  }

  // Split raw text on special-token substrings; each plain run is processed
  // independently (matches HF `tokenizers`).
  std::vector<int64_t> tokenize(const std::string &text) const {
    std::vector<int64_t> ids;
    size_t i = 0;
    while (i < text.size()) {
      int64_t specialId = -1;
      size_t specialLen = 0;
      for (const auto &sp : specialTokens_) {
        if (i + sp.first.size() <= text.size() &&
            text.compare(i, sp.first.size(), sp.first) == 0) {
          specialId = sp.second;
          specialLen = sp.first.size();
          break;
        }
      }
      if (specialId >= 0) {
        ids.push_back(specialId);
        i += specialLen;
        continue;
      }
      size_t j = i;
      while (j < text.size() && !isSpecialAt(text, j))
        ++j;
      if (j > i) {
        std::vector<int64_t> seg = tokenizeSegment(text.substr(i, j - i));
        ids.insert(ids.end(), seg.begin(), seg.end());
      }
      i = j;
    }
    return ids;
  }

  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<std::string, int64_t> mergeRank_;
  std::vector<std::pair<std::string, int64_t>> specialTokens_;
  int64_t bosId_ = 1;
  int64_t eosId_ = 2;
  int64_t padId_ = 2;
  int64_t unkId_ = 0;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_E5MISTRALTOKENIZER_H
