//===- Qwen3VLRawCodecTest.cpp - Qwen3-VL raw codec contract checks -------===//
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

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace buddy::runtime;
namespace fs = std::filesystem;

namespace {

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

template <typename Function>
void requireFailure(Function &&function, const std::string &part) {
  try {
    function();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(part) != std::string::npos,
            "unexpected failure: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected failure containing " + part);
}

class Fixtures {
public:
  explicit Fixtures(const fs::path &parent)
      : root(parent / ("qwen3-vl-raw-codec-" + std::to_string(getpid()))) {
    fs::create_directories(root);
  }
  ~Fixtures() {
    std::error_code error;
    fs::remove_all(root, error);
  }

  std::string write(const std::string &name,
                    const std::string &contents) const {
    const fs::path path = root / name;
    std::ofstream file(path, std::ios::binary);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    require(static_cast<bool>(file), "cannot write vocabulary fixture");
    return path.string();
  }

private:
  fs::path root;
};

void checkProductionVocabulary(const Qwen3VLRawCodec &codec) {
  require(codec.vocabSize() == 151669, "unexpected production Qwen vocabulary");
  require(codec.tokenId("Ġ") == 220 && codec.tokenBytes(220) == " " &&
              codec.tokenId("Ċ") == 198 && codec.tokenBytes(198) == "\n" &&
              codec.tokenId("ĉ") == 197 && codec.tokenBytes(197) == "\t" &&
              codec.tokenId("Ā") == 188 &&
              codec.tokenBytes(188) == std::string(1, '\0'),
          "Qwen byte alphabet did not preserve whitespace or NUL");
  require(!codec.tokenId("missing token spelling"),
          "unknown spelling unexpectedly had a token ID");
  require(codec.encode("").empty() && codec.encode("", true).empty(),
          "raw empty text added BOS/chat/image tokens");

  struct Example {
    std::string text;
    std::vector<int> tokens;
  };
  const std::vector<Example> examples = {
      {"Hello, world!", {9707, 11, 1879, 0}},
      {"  Hello\n\tworld", {220, 21927, 198, 76508}},
      {"你好，中\n", {108386, 3837, 15946, 198}},
      {"antidisestablishmentarianism",
       {15359, 4243, 33400, 76, 79680, 72, 67813}},
      {"e\xcc\x81", {68, 53839}},
      {"é", {963}}};
  for (const auto &example : examples) {
    require(codec.encode(example.text) == example.tokens,
            "unexpected raw segmentation for " + example.text);
    require(codec.encode(example.text, true) == example.tokens,
            "add_special changed Qwen's no-BOS raw encoding");
    require(codec.detokenize({example.tokens}).content == example.text,
            "raw text did not round trip");
  }

  // Official tokenizer.json at Qwen/Qwen3-VL-2B-Instruct revision
  // 89644892e4d85e24eaac8bacfd4f463576704203 has BPE merges, regex splitting,
  // and NFC normalization absent from vocab.txt. These known differences are
  // intentional raw-byte/segmentation boundaries, not equivalence claims.
  const std::vector<int> canonicalWord = {517, 84242, 33400, 478, 8821, 2142};
  require(codec.encode("antidisestablishmentarianism") != canonicalWord,
          "known canonical merge difference unexpectedly disappeared");
  require(codec.detokenize({canonicalWord}).content ==
              "antidisestablishmentarianism",
          "canonical comparison IDs did not describe the same word");
  require(codec.encode("e\xcc\x81") != std::vector<int>{963},
          "raw encoding unexpectedly normalized decomposed Unicode");

  const std::vector<std::string> specials = {
      "<|endoftext|>",        "<|im_start|>",       "<|im_end|>",
      "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
      "<|box_end|>",          "<|quad_start|>",     "<|quad_end|>",
      "<|vision_start|>",     "<|vision_end|>",     "<|vision_pad|>",
      "<|image_pad|>",        "<|video_pad|>"};
  for (std::size_t i = 0; i < specials.size(); ++i) {
    const int id = 151643 + static_cast<int>(i);
    require(codec.tokenId(specials[i]) == id &&
                codec.encode(specials[i]) == std::vector<int>{id},
            "special-token parsing lost its ID/spelling");
    const auto ordinary = codec.encode(specials[i], false, false);
    require(std::find(ordinary.begin(), ordinary.end(), id) == ordinary.end() &&
                codec.detokenize({ordinary}).content == specials[i],
            "parse_special=false did not preserve special spelling as text");
  }
  for (const auto &entry : {std::make_pair("<tool_call>", 151657),
                            std::make_pair("<|fim_prefix|>", 151659),
                            std::make_pair("<tool_response>", 151665),
                            std::make_pair("</tool_response>", 151666),
                            std::make_pair("<think>", 151667)})
    require(codec.encode(entry.first, false, false) ==
                    std::vector<int>{entry.second} &&
                codec.encode(entry.first) == std::vector<int>{entry.second},
            "ordinary added token was incorrectly gated by parse_special");

  TokenizeRequest request;
  request.content = "Hello, world!";
  request.addSpecial = false; // HTTP defaults independently select this value.
  auto result = codec.tokenize(request);
  require(result.tokens == examples.front().tokens && result.count == 4 &&
              result.pieces.empty(),
          "raw HTTP tokenizer defaults changed");
  request.addSpecial = true;
  request.withPieces = true;
  result = codec.tokenize(request);
  require(result.tokens == examples.front().tokens &&
              result.pieces ==
                  std::vector<std::string>{"Hello", ",", " world", "!"},
          "requested token pieces or add_special no-op changed");
  request.countOnly = true;
  result = codec.tokenize(request);
  require(result.count == 4 && result.tokens.empty() && result.pieces.empty(),
          "count_only did not suppress token IDs and pieces");
  request = {};
  result = codec.tokenize(request);
  require(result.count == 0 && result.tokens.empty() && result.pieces.empty(),
          "empty tokenization added a scaffold");

  std::string allBytes;
  for (int byte = 0; byte < 256; ++byte)
    allBytes.push_back(static_cast<char>(byte));
  std::string joined;
  for (int id : codec.encode(allBytes))
    joined += codec.tokenBytes(id);
  require(joined == allBytes, "raw encoding lost an input byte");
  const std::string longText(1024, '\0');
  const auto longTokens = codec.encode(longText);
  require(longTokens.size() == longText.size() &&
              codec.detokenize({longTokens}).content == longText,
          "raw encoding applied a model context limit or truncation");

  std::array<int, 256> byteIds;
  byteIds.fill(-1);
  for (int id = 0; id < codec.vocabSize(); ++id) {
    const auto &bytes = codec.tokenBytes(id);
    if (bytes.size() == 1)
      byteIds[static_cast<unsigned char>(bytes.front())] = id;
  }
  require(std::none_of(byteIds.begin(), byteIds.end(),
                       [](int id) { return id < 0; }),
          "production vocabulary does not cover every byte");
  const std::vector<int> chinese = {byteIds[0xe4], byteIds[0xb8],
                                    byteIds[0xad]};
  const std::string replacement = "\xef\xbf\xbd";
  require(codec.detokenize({chinese}).content == "中",
          "UTF-8 fragments were repaired before joining");
  require(codec.detokenize({{byteIds[0xe4], byteIds[0xb8]}}).content ==
              replacement,
          "incomplete UTF-8 tail did not produce one replacement");
  require(codec.detokenize({{byteIds[0xff]}}).content == replacement,
          "invalid UTF-8 byte did not produce a replacement");
  const std::string afterEos = std::string("  <|endoftext|>A") + '\0';
  require(codec.detokenize({{220, 220, 151643, byteIds['A'], 188}}).content ==
              afterEos,
          "raw decoding trimmed spaces, stopped at EOS, or lost NUL");
  require(codec.detokenize({{}}).content.empty(),
          "empty detokenize did not produce empty text");
  for (int id : {-1, codec.vocabSize(), std::numeric_limits<int>::max()}) {
    requireFailure([&] { (void)codec.tokenBytes(id); }, "tokens:");
    requireFailure([&] { (void)codec.detokenize({{151643, id}}); }, "tokens:");
  }
}

void checkSyntheticVocabulary(const Fixtures &fixtures) {
  // Arbitrary fixture IDs demonstrate that parsing uses spelling, not the
  // production 151643..151656 numbering. These byte spellings mean E4 B8 AD.
  Qwen3VLRawCodec codec(
      fixtures.write("small.txt", "Ġ\nA\n<|im_start|>\nä\n¸\nŃ\n"));
  require(codec.vocabSize() == 6 && codec.tokenId("<|im_start|>") == 2 &&
              codec.encode("<|im_start|>") == std::vector<int>{2},
          "synthetic special-token IDs were not recognized by spelling");
  require(codec.encode("中") == std::vector<int>({3, 4, 5}) &&
              codec.detokenize({{3, 4, 5}}).content == "中",
          "synthetic byte fragments did not jointly decode");
  requireFailure([&] { (void)codec.encode("B"); }, "content:");
  TokenizeRequest request;
  request.content = "中";
  request.withPieces = true;
  const auto result = codec.tokenize(request);
  require(result.tokens == std::vector<int>({3, 4, 5}) &&
              result.pieces == std::vector<std::string>(3, "\xef\xbf\xbd"),
          "pieces were not independently repaired for display");
  requireFailure(
      [&] { Qwen3VLRawCodec empty(fixtures.write("empty.txt", "")); },
      "vocabulary");
  requireFailure(
      [&] { Qwen3VLRawCodec blank(fixtures.write("blank.txt", "A\n\n")); },
      "empty vocabulary piece");
  requireFailure(
      [&] {
        Qwen3VLRawCodec repeated(fixtures.write("duplicate.txt", "A\nA\n"));
      },
      "duplicate vocabulary piece");
  const std::vector<std::string> invalid = {
      std::string("\xc0\x80", 2), std::string("\xed\xa0\x80", 3),
      std::string("\xf4\x90\x80\x80", 4), std::string("\xe4\xb8", 2),
      std::string("\xe4", 1) + "A"};
  for (std::size_t i = 0; i < invalid.size(); ++i)
    requireFailure(
        [&] {
          Qwen3VLRawCodec bad(fixtures.write("invalid-" + std::to_string(i),
                                             invalid[i] + "\n"));
        },
        "vocabulary");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2 || argc == 3,
            "usage: Qwen3VLRawCodecTest <vocab.txt> [fixture-parent]");
    Qwen3VLRawCodec codec(argv[1]);
    checkProductionVocabulary(codec);
    const Fixtures fixtures(argc == 3 ? fs::path(argv[2]) : fs::current_path());
    checkSyntheticVocabulary(fixtures);
    std::puts("Qwen3-VL raw codec contract checks passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Qwen3-VL raw codec checks failed: %s\n",
                 error.what());
    return 1;
  }
}
