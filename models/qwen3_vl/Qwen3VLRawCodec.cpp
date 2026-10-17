//===- Qwen3VLRawCodec.cpp - Raw Qwen3-VL vocabulary codec ---------------===//
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

#include "buddy/runtime/models/Qwen3VLRawCodec.h"

#include "buddy/runtime/llm/TextGeneration.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace buddy {
namespace runtime {
namespace {

bool isSpecial(const std::string &piece) {
  // These are the special=true entries in Qwen3-VL's tokenizer profile.
  // Tool/thinking/FIM tokens are added tokens, but are ordinary candidates.
  static constexpr std::string_view spellings[] = {
      "<|endoftext|>",        "<|im_start|>",       "<|im_end|>",
      "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
      "<|box_end|>",          "<|quad_start|>",     "<|quad_end|>",
      "<|vision_start|>",     "<|vision_end|>",     "<|vision_pad|>",
      "<|image_pad|>",        "<|video_pad|>"};
  return std::find(std::begin(spellings), std::end(spellings), piece) !=
         std::end(spellings);
}

// Invert the GPT-2/Qwen byte-to-Unicode alphabet after validating each UTF-8
// spelling. Raw Unicode pieces beyond that alphabet are retained as bytes.
std::string decodePiece(const std::string &piece) {
  static const auto extraBytes = [] {
    std::array<unsigned char, 68> result{};
    std::size_t next = 0;
    for (unsigned int byte = 0; byte < 256; ++byte)
      if (!((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
            (byte >= 174 && byte <= 255)))
        result[next++] = static_cast<unsigned char>(byte);
    return result;
  }();

  std::string result;
  for (std::size_t i = 0; i < piece.size();) {
    const auto first = static_cast<unsigned char>(piece[i]);
    unsigned int code = 0;
    std::size_t width = 0;
    if (first < 0x80) {
      code = first;
      width = 1;
    } else if (first >= 0xc2 && first <= 0xdf) {
      code = first & 0x1f;
      width = 2;
    } else if (first >= 0xe0 && first <= 0xef) {
      code = first & 0x0f;
      width = 3;
    } else if (first >= 0xf0 && first <= 0xf4) {
      code = first & 0x07;
      width = 4;
    } else {
      throw std::invalid_argument("qwen3_vl: invalid UTF-8 in vocabulary");
    }
    if (i + width > piece.size())
      throw std::invalid_argument("qwen3_vl: incomplete vocabulary piece");
    for (std::size_t j = 1; j < width; ++j) {
      const auto byte = static_cast<unsigned char>(piece[i + j]);
      if ((byte & 0xc0) != 0x80)
        throw std::invalid_argument("qwen3_vl: invalid vocabulary piece");
      code = (code << 6) | (byte & 0x3f);
    }
    if ((width == 3 && code < 0x800) || (width == 4 && code < 0x10000) ||
        code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
      throw std::invalid_argument("qwen3_vl: invalid vocabulary code point");
    if (code >= 256 && code <= 323)
      result.push_back(static_cast<char>(extraBytes[code - 256]));
    else if (code <= 255)
      result.push_back(static_cast<char>(code));
    else
      result.append(piece, i, width);
    i += width;
  }
  return result;
}

std::string repairUtf8(const std::string &bytes) {
  Utf8StreamDecoder decoder;
  std::string result = decoder.append(bytes);
  result += decoder.finish();
  return result;
}

} // namespace

class Qwen3VLRawCodec::Impl {
public:
  explicit Impl(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("qwen3_vl: cannot open vocabulary: " + path);
    std::string piece;
    while (std::getline(input, piece)) {
      if (piece.empty())
        throw std::invalid_argument("qwen3_vl: empty vocabulary piece");
      if (pieces.size() >=
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("qwen3_vl: vocabulary is too large");
      const int id = static_cast<int>(pieces.size());
      if (!ids.emplace(piece, id).second)
        throw std::invalid_argument("qwen3_vl: duplicate vocabulary piece");
      pieces.push_back(decodePiece(piece));
      if (isSpecial(piece)) {
        specialIds.push_back(id);
      } else {
        ordinary.emplace(pieces.back(), id);
        maxPieceLength = std::max(maxPieceLength, pieces.back().size());
      }
    }
    if (input.bad() || pieces.empty())
      throw std::invalid_argument(
          "qwen3_vl: vocabulary is empty or unreadable");
    std::sort(specialIds.begin(), specialIds.end(), [&](int lhs, int rhs) {
      if (pieces[lhs].size() != pieces[rhs].size())
        return pieces[lhs].size() > pieces[rhs].size();
      return lhs < rhs;
    });
  }

  void encodeOrdinary(const std::string &content, std::size_t begin,
                      std::size_t end, std::vector<int> &output) const {
    const std::size_t length = end - begin;
    std::vector<long long> scores(length + 1, -1);
    std::vector<int> previousToken(length + 1, -1);
    std::vector<std::size_t> previousPosition(length + 1, 0);
    scores[0] = 0;
    for (std::size_t i = 0; i < length; ++i) {
      if (scores[i] < 0)
        continue;
      for (std::size_t width = 1; width <= std::min(maxPieceLength, length - i);
           ++width) {
        const auto found = ordinary.find(content.substr(begin + i, width));
        if (found == ordinary.end())
          continue;
        const long long score =
            scores[i] + static_cast<long long>(width * width);
        if (score > scores[i + width] ||
            (score == scores[i + width] &&
             found->second < previousToken[i + width])) {
          scores[i + width] = score;
          previousToken[i + width] = found->second;
          previousPosition[i + width] = i;
        }
      }
    }
    if (scores[length] < 0)
      throw std::invalid_argument(
          "content: text cannot be represented by the loaded vocabulary");
    std::vector<int> reversed;
    for (std::size_t position = length; position > 0;
         position = previousPosition[position])
      reversed.push_back(previousToken[position]);
    output.insert(output.end(), reversed.rbegin(), reversed.rend());
  }

  std::vector<int> encode(const std::string &content, bool parseSpecial) const {
    std::vector<int> output;
    std::size_t begin = 0;
    std::size_t position = 0;
    while (parseSpecial && position < content.size()) {
      int found = -1;
      for (int id : specialIds)
        if (content.compare(position, pieces[id].size(), pieces[id]) == 0) {
          found = id;
          break;
        }
      if (found < 0) {
        ++position;
        continue;
      }
      encodeOrdinary(content, begin, position, output);
      output.push_back(found);
      position += pieces[found].size();
      begin = position;
    }
    encodeOrdinary(content, begin, content.size(), output);
    return output;
  }

  std::vector<std::string> pieces;
  std::unordered_map<std::string, int> ids;
  std::unordered_map<std::string, int> ordinary;
  std::vector<int> specialIds;
  std::size_t maxPieceLength = 0;
};

Qwen3VLRawCodec::Qwen3VLRawCodec(const std::string &vocabPath)
    : impl(std::make_unique<Impl>(vocabPath)) {}
Qwen3VLRawCodec::~Qwen3VLRawCodec() = default;

std::vector<int> Qwen3VLRawCodec::encode(const std::string &content,
                                         bool addSpecial,
                                         bool parseSpecial) const {
  // The Qwen tokenizer profile has add_bos_token=false and bos_token=null.
  // Chat/image wrappers belong to the caller, independent of this option.
  (void)addSpecial;
  return impl->encode(content, parseSpecial);
}

const std::string &Qwen3VLRawCodec::tokenBytes(int id) const {
  if (id < 0 || id >= vocabSize())
    throw std::invalid_argument("tokens: token ID is outside the vocabulary");
  return impl->pieces[id];
}

int Qwen3VLRawCodec::vocabSize() const {
  return static_cast<int>(impl->pieces.size());
}

std::optional<int> Qwen3VLRawCodec::tokenId(const std::string &spelling) const {
  const auto found = impl->ids.find(spelling);
  return found == impl->ids.end() ? std::nullopt
                                  : std::optional<int>(found->second);
}

TokenizeResult Qwen3VLRawCodec::tokenize(const TokenizeRequest &request) const {
  TokenizeResult result;
  result.tokens =
      encode(request.content, request.addSpecial, request.parseSpecial);
  result.count = result.tokens.size();
  if (request.countOnly) {
    result.tokens.clear();
  } else if (request.withPieces) {
    for (int id : result.tokens)
      result.pieces.push_back(repairUtf8(tokenBytes(id)));
  }
  return result;
}

DetokenizeResult
Qwen3VLRawCodec::detokenize(const DetokenizeRequest &request) const {
  for (int id : request.tokens)
    (void)tokenBytes(id);
  std::string bytes;
  for (int id : request.tokens)
    bytes += tokenBytes(id);
  return {repairUtf8(bytes)};
}

} // namespace runtime
} // namespace buddy
