//===- BgeM3Tokenizer.h - SentencePiece Unigram tokenizer, pure C++ ------===//
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
// BGE-M3's tokenizer is XLM-RoBERTa's: a SentencePiece Unigram model with a
// Metaspace pre-tokenizer. This loads it straight from the HuggingFace
// `tokenizers` fast-tokenizer export (tokenizer.json) and reproduces
// `AutoTokenizer(text, padding="max_length", truncation=True,
// max_length=N)` without a Python/`transformers` runtime dependency.
//
// The normalizer (XCDA-trie prefix replacement, "nmt_nfkc") and the Viterbi
// unigram segmentation are SentencePiece's own algorithms; this is an
// independent reimplementation following the same approach used by
// llama.cpp's `llm_tokenizer_ugm` for SentencePiece-Unigram vocabularies,
// adapted to read tokenizer.json directly (so no protobuf dependency is
// needed to parse the raw .model file either). See:
//   https://github.com/google/sentencepiece
//   third_party llama.cpp src/llama-vocab.cpp (llm_tokenizer_ugm)
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_BGEM3TOKENIZER_H
#define BUDDY_RUNTIME_MODELS_BGEM3TOKENIZER_H

#include <jsoncons/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace buddy {
namespace runtime {

// Minimal standard base64 decoder; avoids depending on llvm::decodeBase64.
inline void decodeBase64(std::string_view input, std::vector<char> &output) {
  auto valueOf = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    return -1;
  };

  output.clear();
  uint32_t buffer = 0;
  int bits = 0;
  for (char ch : input) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t')
      continue;
    int value = valueOf(c);
    if (value < 0)
      throw std::runtime_error("BgeM3Tokenizer: invalid base64 input");
    buffer = (buffer << 6) | static_cast<uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      output.push_back(static_cast<char>((buffer >> bits) & 0xFF));
    }
  }
}

class BgeM3Tokenizer {
public:
  static BgeM3Tokenizer loadFromFile(const std::string &tokenizerJsonPath) {
    std::ifstream in(tokenizerJsonPath, std::ios::binary);
    if (!in)
      throw std::runtime_error("BgeM3Tokenizer: cannot open " +
                               tokenizerJsonPath);
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());

    jsoncons::json root;
    try {
      root = jsoncons::json::parse(text);
    } catch (const std::exception &e) {
      throw std::runtime_error("BgeM3Tokenizer: failed to parse " +
                               tokenizerJsonPath + ": " + e.what());
    }
    if (!root.is_object())
      throw std::runtime_error("BgeM3Tokenizer: " + tokenizerJsonPath +
                               " root is not a JSON object");

    BgeM3Tokenizer tok;
    tok.loadVocab(root);
    tok.loadCharsmap(root);
    tok.loadSpecialTokens(root);
    tok.buildMatchers();
    return tok;
  }

  // Reproduces AutoTokenizer(text, padding="max_length", truncation=True,
  // max_length=maxSeqLen): [bos, content..., eos], content truncated so the
  // wrapped sequence fits maxSeqLen, right-padded with padId_ / mask 0.
  size_t tokenCount(const std::string &text) const {
    return tokenize(text).size();
  }

  void encode(const std::string &text, size_t maxSeqLen,
              std::vector<int64_t> &inputIds,
              std::vector<int64_t> &attentionMask) const {
    if (maxSeqLen < 2)
      throw std::runtime_error("BgeM3Tokenizer: maxSeqLen must be >= 2");

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
  // Minimal byte trie: piece text (UTF-8 bytes) -> vocab id.
  struct TrieNode {
    std::map<char, TrieNode> children;
    bool hasValue = false;
    int64_t value = 0;

    void insert(const std::string &key, int64_t val) {
      TrieNode *node = this;
      for (char c : key)
        node = &node->children[c];
      node->hasValue = true;
      node->value = val;
    }
    const TrieNode *traverse(char c) const {
      auto it = children.find(c);
      return it == children.end() ? nullptr : &it->second;
    }
    // Longest prefix of key[offset..offset+len) with a stored value; 0 if
    // none. Mirrors SentencePiece's "user defined token" longest match.
    size_t longestPrefix(const std::string &key, size_t offset) const {
      const TrieNode *node = this;
      size_t best = 0;
      for (size_t i = offset; i < key.size(); ++i) {
        auto it = node->children.find(key[i]);
        if (it == node->children.end())
          break;
        node = &it->second;
        if (node->hasValue)
          best = i + 1 - offset;
      }
      return best;
    }
  };

  struct NormalizedPrefix {
    const char *data;
    size_t len;      // bytes of replacement/pass-through text
    size_t consumed; // bytes consumed from the input
  };

  // ── Loading ───────────────────────────────────────────────────────────

  void loadVocab(const jsoncons::json &root) {
    if (!root.contains("model") || !root["model"].is_object())
      throw std::runtime_error("BgeM3Tokenizer: missing \"model\"");
    const jsoncons::json &model = root["model"];
    if (!model.contains("type") || !model["type"].is_string() ||
        model["type"].as<std::string>() != "Unigram")
      throw std::runtime_error(
          "BgeM3Tokenizer: unsupported tokenizer model type (expected "
          "Unigram)");

    if (model.contains("unk_id") && model["unk_id"].is_number())
      unkId_ = model["unk_id"].as<int64_t>();

    if (!model.contains("vocab") || !model["vocab"].is_array())
      throw std::runtime_error("BgeM3Tokenizer: missing \"model.vocab\"");
    const jsoncons::json &vocabArr = model["vocab"];
    vocab_.reserve(vocabArr.size());
    for (const jsoncons::json &entry : vocabArr.array_range()) {
      if (!entry.is_array() || entry.size() != 2)
        throw std::runtime_error("BgeM3Tokenizer: malformed vocab entry");
      const jsoncons::json &piece = entry[0];
      const jsoncons::json &score = entry[1];
      if (!piece.is_string() || !score.is_number())
        throw std::runtime_error("BgeM3Tokenizer: malformed vocab entry");
      vocab_.emplace_back(piece.as<std::string>(), score.as<double>());
    }
  }

  void loadCharsmap(const jsoncons::json &root) {
    if (!root.contains("normalizer") || !root["normalizer"].is_object())
      return;
    const jsoncons::json &normalizer = root["normalizer"];
    std::vector<const jsoncons::json *> flat;
    if (normalizer.contains("normalizers") &&
        normalizer["normalizers"].is_array()) {
      for (const jsoncons::json &s : normalizer["normalizers"].array_range())
        if (s.is_object())
          flat.push_back(&s);
    } else {
      flat.push_back(&normalizer);
    }

    for (const jsoncons::json *step : flat) {
      if (!step->contains("type") || !(*step)["type"].is_string() ||
          (*step)["type"].as<std::string>() != "Precompiled")
        continue;
      if (!step->contains("precompiled_charsmap") ||
          !(*step)["precompiled_charsmap"].is_string())
        continue;
      try {
        decodeBase64((*step)["precompiled_charsmap"].as<std::string>(),
                     charsmap_);
      } catch (const std::exception &e) {
        throw std::runtime_error(
            "BgeM3Tokenizer: failed to decode precompiled_charsmap: " +
            std::string(e.what()));
      }
      break;
    }

    if (charsmap_.empty())
      return;
    if (charsmap_.size() < sizeof(uint32_t))
      throw std::runtime_error("BgeM3Tokenizer: precompiled_charsmap too "
                               "small");

    uint32_t xcdaBlobSize;
    std::memcpy(&xcdaBlobSize, charsmap_.data(), sizeof(uint32_t));
    size_t offset = sizeof(uint32_t);
    if (static_cast<size_t>(xcdaBlobSize) + offset > charsmap_.size())
      throw std::runtime_error(
          "BgeM3Tokenizer: precompiled_charsmap XCDA blob out of bounds");

    xcdaArrayOffset_ = offset;
    xcdaArraySize_ = xcdaBlobSize / sizeof(uint32_t);
    offset += xcdaBlobSize;
    prefixReplacementsOffset_ = offset;
    prefixReplacementsSize_ = charsmap_.size() - offset;
  }

  void loadSpecialTokens(const jsoncons::json &root) {
    bosId_ = findTokenId(root, "<s>", 0);
    eosId_ = findTokenId(root, "</s>", 2);
    padId_ = findTokenId(root, "<pad>", 1);

    if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
      for (const jsoncons::json &o : root["added_tokens"].array_range()) {
        if (!o.is_object())
          continue;
        if (o.contains("content") && o.contains("id") &&
            o.contains("special") && o["special"].is_bool() &&
            o["special"].as<bool>())
          specialIds_.push_back(o["id"].as<int64_t>());
      }
    }
  }

  // Looks up a special token's id via post_processor.special_tokens first
  // (authoritative for bos/eos), falling back to `fallback`.
  static int64_t findTokenId(const jsoncons::json &root,
                             const std::string &content, int64_t fallback) {
    if (root.contains("post_processor") && root["post_processor"].is_object()) {
      const jsoncons::json &pp = root["post_processor"];
      if (pp.contains("special_tokens") && pp["special_tokens"].is_object()) {
        const jsoncons::json &special = pp["special_tokens"];
        if (special.contains(content) && special[content].is_object()) {
          const jsoncons::json &entry = special[content];
          if (entry.contains("ids") && entry["ids"].is_array() &&
              !entry["ids"].empty() && entry["ids"][0].is_number())
            return entry["ids"][0].as<int64_t>();
        }
      }
    }
    return fallback;
  }

  void buildMatchers() {
    // Special tokens never participate in Viterbi segmentation: tokenize()
    // splits them out via specialMatcher_ before normalize()/tokenizeSegment
    // ever see them (see tokenize()'s doc comment), so they're excluded here
    // both from tokenMatcher_ and from the min/max score computation.
    for (size_t id = 0; id < vocab_.size(); ++id) {
      if (isSpecialId(static_cast<int64_t>(id)))
        continue;
      tokenMatcher_.insert(vocab_[id].first, static_cast<int64_t>(id));
      minScore_ = std::min(minScore_, vocab_[id].second);
    }
    unknownTokenScore_ = minScore_ - 10.0;

    for (int64_t id : specialIds_)
      if (static_cast<size_t>(id) < vocab_.size())
        specialMatcher_.insert(vocab_[id].first, id);
  }

  bool isSpecialId(int64_t id) const {
    return std::find(specialIds_.begin(), specialIds_.end(), id) !=
           specialIds_.end();
  }

  // ── XCDA (XOR-compressed compact double array) trie accessors ──────────
  // Bit layout matches SentencePiece's own precompiled_charsmap format: each
  // 32-bit entry packs BASE (bits 10-30), LCHECK (bits 0-7 + sign bit 31),
  // and a LEAF flag (bit 8).

  uint32_t xcdaNode(size_t index) const {
    if (index >= xcdaArraySize_)
      throw std::runtime_error("BgeM3Tokenizer: XCDA index out of bounds");
    uint32_t value;
    std::memcpy(&value,
                charsmap_.data() + xcdaArrayOffset_ + index * sizeof(uint32_t),
                sizeof(uint32_t));
    return value;
  }
  uint32_t xcdaBase(size_t index) const {
    uint32_t n = xcdaNode(index);
    return (n >> 10) << ((n & (1U << 9)) >> 6);
  }
  uint32_t xcdaLcheck(size_t index) const {
    return xcdaNode(index) & ((1U << 31) | 0xffU);
  }
  bool xcdaLeaf(size_t index) const { return (xcdaNode(index) >> 8) & 1U; }
  uint32_t xcdaValue(size_t index) const {
    return xcdaNode(index) & ((1U << 31) - 1);
  }

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

  // True if input[offset..offset+len) is a well-formed UTF-8 sequence.
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

  // ── Normalization: SentencePiece "nmt_nfkc" + dummy prefix + escaped
  // whitespace + collapse of extra whitespace, all folded into one pass
  // (matches sentencepiece_model.proto NormalizerSpec semantics for this
  // model: add_dummy_prefix=true, remove_extra_whitespaces=true,
  // escape_whitespaces=true). ─────────────────────────────────────────────

  NormalizedPrefix normalizePrefix(const std::string &input,
                                   size_t offset) const {
    if (offset == input.size())
      return {&input[offset], 0, 0};

    size_t longestLen = 0;
    size_t longestReplOffset = 0;
    if (xcdaArraySize_ > 0) {
      uint32_t nodeIndex = xcdaBase(0);
      for (size_t p = offset; p < input.size(); ++p) {
        unsigned char c = static_cast<unsigned char>(input[p]);
        if (c == 0)
          break;
        nodeIndex ^= c;
        if (xcdaLcheck(nodeIndex) != c)
          break;
        bool leaf = xcdaLeaf(nodeIndex);
        nodeIndex ^= xcdaBase(nodeIndex);
        if (leaf) {
          longestLen = p - offset + 1;
          longestReplOffset = xcdaValue(nodeIndex);
        }
      }
    }

    if (longestLen > 0) {
      if (longestReplOffset >= prefixReplacementsSize_)
        throw std::runtime_error(
            "BgeM3Tokenizer: charsmap replacement offset out of bounds");
      const char *repl =
          charsmap_.data() + prefixReplacementsOffset_ + longestReplOffset;
      return {repl, std::strlen(repl), longestLen};
    }

    size_t len;
    if (tryUtf8Len(input, offset, len))
      return {&input[offset], len, len};
    static const char kReplacementChar[] = "\xEF\xBF\xBD"; // U+FFFD
    return {kReplacementChar, 3, 1};
  }

  std::string normalize(const std::string &input) const {
    std::string out;
    out.reserve(input.size() * 3);
    static const std::string kEscapedSpace = "\xE2\x96\x81"; // U+2581 "▁"

    bool processingNonWs = false;

    for (size_t offset = 0; offset < input.size();) {
      NormalizedPrefix res = normalizePrefix(input, offset);
      for (size_t i = 0; i < res.len; ++i) {
        char c = res.data[i];
        if (c != ' ') {
          if (!processingNonWs) {
            processingNonWs = true;
            // add_dummy_prefix and remove_extra_whitespaces are both true
            // for this model, so a "▁" is unconditionally (re)inserted at
            // every word start, not just the first (SentencePiece folds the
            // leading dummy prefix and inter-word separators into the same
            // rule once remove_extra_whitespaces collapses runs of spaces).
            out += kEscapedSpace;
          }
          out.push_back(c);
        } else {
          processingNonWs = false;
          // remove_extra_whitespaces=true: literal spaces are dropped here;
          // the next word boundary re-inserts a single "▁" above.
        }
      }
      offset += res.consumed;
    }
    return out;
  }

  // ── Viterbi unigram segmentation (SentencePiece's optimized DP): walk the
  // normalized string one UTF-8 codepoint at a time, at each position try
  // every vocab piece starting there via the byte trie, and keep the
  // highest-scoring segmentation ending at each offset. Operates on one
  // plain-text segment at a time (see tokenize() below for why). ─────────

  std::vector<int64_t> tokenizeSegment(const std::string &text) const {
    std::string normalized = normalize(text);
    size_t n = normalized.size();
    if (n == 0)
      return {};

    struct Best {
      int64_t tokenId;
      size_t inputOffset;
      double scoreSum;
    };
    std::vector<Best> best(n + 1,
                           {unkId_, 0, -std::numeric_limits<double>::max()});
    best[0] = {unkId_, 0, 0.0};

    for (size_t inputOffset = 0; inputOffset < n;) {
      size_t prefixOffset = inputOffset;
      size_t codepointLen =
          std::min(utf8Len(static_cast<unsigned char>(normalized[inputOffset])),
                   n - inputOffset);

      bool singleCodepointFound = false;
      const Best &curBest = best[inputOffset];
      const TrieNode *node = tokenMatcher_.traverse(normalized[prefixOffset++]);

      while (prefixOffset <= n && node != nullptr) {
        if (node->hasValue) {
          if (prefixOffset - inputOffset == codepointLen)
            singleCodepointFound = true;
          int64_t tokenId = node->value;
          double challengerScore = curBest.scoreSum + vocab_[tokenId].second;
          Best &champ = best[prefixOffset];
          if (challengerScore > champ.scoreSum)
            champ = {tokenId, inputOffset, challengerScore};
        }
        node = node->traverse(normalized[prefixOffset++]);
      }

      if (!singleCodepointFound) {
        size_t unkOffset = inputOffset + codepointLen;
        double challengerScore = curBest.scoreSum + unknownTokenScore_;
        Best &champ = best[unkOffset];
        if (challengerScore > champ.scoreSum)
          champ = {unkId_, inputOffset, challengerScore};
      }

      inputOffset += codepointLen;
    }

    std::vector<int64_t> output;
    bool prevUnknown = false;
    size_t pos = n;
    while (true) {
      const Best &b = best[pos];
      bool isUnknown = (b.tokenId == unkId_);
      if (!(prevUnknown && isUnknown))
        output.push_back(b.tokenId);
      if (b.inputOffset == 0)
        break;
      prevUnknown = isUnknown;
      pos = b.inputOffset;
    }
    std::reverse(output.begin(), output.end());
    return output;
  }

  // HF's `tokenizers` library splits the raw string on added-token (special
  // token) substrings *before* normalization, and normalizes each resulting
  // plain-text run independently — so e.g. "before <mask> after" becomes
  // segments ["before ", "<mask>", " after"], each of "before "/" after"
  // getting its own fresh dummy-prefix, and "<mask>" resolving straight to
  // its id without ever going through the SentencePiece model. This must be
  // done as a separate pass (not inline within normalize()/normalizePrefix)
  // because folding a special token into one continuous normalization pass
  // would incorrectly attach a "▁" word-start marker to it, e.g. producing
  // "▁<mask>" -> ["▁", "<mask>"] instead of the correct ["<mask>"].
  std::vector<int64_t> tokenize(const std::string &text) const {
    std::vector<int64_t> output;
    size_t segStart = 0;
    for (size_t offset = 0; offset < text.size();) {
      size_t specialLen = specialMatcher_.longestPrefix(text, offset);
      if (specialLen == 0) {
        ++offset;
        continue;
      }
      if (offset > segStart) {
        std::vector<int64_t> seg =
            tokenizeSegment(text.substr(segStart, offset - segStart));
        output.insert(output.end(), seg.begin(), seg.end());
      }
      const TrieNode *node = specialMatcher_.traverse(text[offset]);
      for (size_t i = 1; i < specialLen; ++i)
        node = node->traverse(text[offset + i]);
      output.push_back(node->value);
      offset += specialLen;
      segStart = offset;
    }
    if (segStart < text.size()) {
      std::vector<int64_t> seg = tokenizeSegment(text.substr(segStart));
      output.insert(output.end(), seg.begin(), seg.end());
    }
    return output;
  }

  std::vector<std::pair<std::string, double>> vocab_;
  std::vector<int64_t> specialIds_;
  TrieNode tokenMatcher_;
  TrieNode specialMatcher_;

  // XCDA array / prefix replacement blob are stored as byte offsets into
  // charsmap_ (not raw pointers) so this object stays safe to copy/move: a
  // copied charsmap_ gets its own buffer at a new address, and offsets are
  // still valid into it, whereas pointers computed from the original buffer
  // would silently dangle.
  std::vector<char> charsmap_;
  size_t xcdaArrayOffset_ = 0;
  size_t xcdaArraySize_ = 0;
  size_t prefixReplacementsOffset_ = 0;
  size_t prefixReplacementsSize_ = 0;

  int64_t bosId_ = 0;
  int64_t eosId_ = 2;
  int64_t padId_ = 1;
  int64_t unkId_ = 3;
  double minScore_ = std::numeric_limits<double>::max();
  double unknownTokenScore_ = 0.0;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_BGEM3TOKENIZER_H
