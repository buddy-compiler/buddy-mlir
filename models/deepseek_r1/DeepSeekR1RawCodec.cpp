//===- DeepSeekR1RawCodec.cpp - Raw DeepSeek vocabulary codec ------------===//
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

#include "buddy/runtime/models/DeepSeekR1RawCodec.h"

#include "buddy/runtime/llm/TextGeneration.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace buddy {
namespace runtime {
namespace {

bool isSpecial(const std::string &piece) {
  return (piece.size() >= 4 && piece.compare(0, 2, "<|") == 0 &&
          piece.compare(piece.size() - 2, 2, "|>") == 0) ||
         piece == "<think>" || piece == "</think>" || piece == "<tool_call>" ||
         piece == "</tool_call>";
}

// Buddy's vocabulary contains both byte-BPE symbols and raw UTF-8 CJK pieces.
// Use the same byte mapping as the legacy decoder, without its leading-space
// removal or EOS truncation. Validate the vocabulary before indexing UTF-8.
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
      throw std::invalid_argument("deepseek_r1: invalid UTF-8 in vocabulary");
    }
    if (i + width > piece.size())
      throw std::invalid_argument("deepseek_r1: incomplete vocabulary piece");
    for (std::size_t j = 1; j < width; ++j) {
      const auto byte = static_cast<unsigned char>(piece[i + j]);
      if ((byte & 0xc0) != 0x80)
        throw std::invalid_argument("deepseek_r1: invalid vocabulary piece");
      code = (code << 6) | (byte & 0x3f);
    }
    if ((width == 3 && code < 0x800) || (width == 4 && code < 0x10000) ||
        code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
      throw std::invalid_argument("deepseek_r1: invalid vocabulary code point");
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

} // namespace

class DeepSeekR1RawCodec::Impl {
public:
  explicit Impl(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
      throw std::runtime_error("deepseek_r1: cannot open vocabulary: " + path);
    std::string piece;
    while (std::getline(input, piece)) {
      if (pieces.size() >=
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("deepseek_r1: vocabulary is too large");
      const int id = static_cast<int>(pieces.size());
      ids.emplace(piece, id);
      const bool special = isSpecial(piece);
      pieces.push_back(special ? piece : decodePiece(piece));
      if (special) {
        specialIds.push_back(id);
      } else if (!pieces.back().empty()) {
        ordinary.emplace(pieces.back(), id);
        maxPieceLength = std::max(maxPieceLength, pieces.back().size());
      }
    }
    if (input.bad() || pieces.empty())
      throw std::invalid_argument(
          "deepseek_r1: vocabulary is empty or unreadable");
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

  std::vector<int> encode(const std::string &content, bool addSpecial,
                          bool parseSpecial) const {
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
    if (addSpecial) {
      const auto bos = ids.find("<|begin▁of▁sentence|>");
      if (bos == ids.end())
        throw std::invalid_argument(
            "add_special: loaded vocabulary has no BOS token");
      if (output.empty() || output.front() != bos->second)
        output.insert(output.begin(), bos->second);
    }
    return output;
  }

  std::vector<std::string> pieces;
  std::unordered_map<std::string, int> ids;
  std::unordered_map<std::string, int> ordinary;
  std::vector<int> specialIds;
  std::size_t maxPieceLength = 0;
};

DeepSeekR1RawCodec::DeepSeekR1RawCodec(const std::string &vocabPath)
    : impl(std::make_unique<Impl>(vocabPath)) {}
DeepSeekR1RawCodec::~DeepSeekR1RawCodec() = default;

std::vector<int> DeepSeekR1RawCodec::encode(const std::string &content,
                                            bool addSpecial,
                                            bool parseSpecial) const {
  return impl->encode(content, addSpecial, parseSpecial);
}

std::string DeepSeekR1RawCodec::tokenBytes(int id) const {
  if (id < 0 || id >= vocabSize())
    throw std::invalid_argument("tokens: token ID is outside the vocabulary");
  return impl->pieces[id];
}

int DeepSeekR1RawCodec::tokenId(const std::string &piece) const {
  const auto found = impl->ids.find(piece);
  return found == impl->ids.end() ? -1 : found->second;
}

int DeepSeekR1RawCodec::vocabSize() const {
  return static_cast<int>(impl->pieces.size());
}

TokenizeResult
DeepSeekR1RawCodec::tokenize(const TokenizeRequest &request) const {
  if (request.withPieces)
    throw std::invalid_argument("with_pieces: mode is not supported");
  TokenizeResult result;
  result.tokens =
      encode(request.content, request.addSpecial, request.parseSpecial);
  result.count = result.tokens.size();
  if (request.countOnly)
    result.tokens.clear();
  return result;
}

DetokenizeResult
DeepSeekR1RawCodec::detokenize(const DetokenizeRequest &request) const {
  // Validate every ID before decoding any piece; EOS does not terminate this
  // path. Decode UTF-8 only after all token byte fragments have been joined.
  for (int id : request.tokens)
    if (id < 0 || id >= vocabSize())
      throw std::invalid_argument("tokens: token ID is outside the vocabulary");
  std::string bytes;
  for (int id : request.tokens)
    bytes += impl->pieces[id];
  Utf8StreamDecoder decoder;
  std::string content = decoder.append(bytes);
  content += decoder.finish();
  return {std::move(content)};
}

} // namespace runtime
} // namespace buddy
