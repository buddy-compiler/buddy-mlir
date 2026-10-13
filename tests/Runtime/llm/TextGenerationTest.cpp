//===- TextGenerationTest.cpp --------------------------------------------===//
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

// RUN: buddy-text-generation-test 2>&1 | FileCheck %s

#include <buddy/runtime/llm/TextGeneration.h>

#include <cstdio>
#include <string>

using buddy::runtime::Utf8StreamDecoder;

int main() {
  Utf8StreamDecoder chineseDecoder;
  std::string chinese = chineseDecoder.append("prefix ");
  const std::string chineseByte1 =
      chineseDecoder.append(std::string("\xE4", 1));
  const std::string chineseByte2 =
      chineseDecoder.append(std::string("\xB8", 1));
  const std::string chineseByte3 =
      chineseDecoder.append(std::string("\xAD", 1));
  // CHECK: Chinese chunks: 0 0 3
  std::fprintf(stderr, "Chinese chunks: %zu %zu %zu\n", chineseByte1.size(),
               chineseByte2.size(), chineseByte3.size());
  chinese += chineseByte1;
  chinese += chineseByte2;
  chinese += chineseByte3;
  chinese += chineseDecoder.append(" suffix");
  chinese += chineseDecoder.finish();
  // CHECK: split Chinese: prefix 中 suffix
  std::fprintf(stderr, "split Chinese: %s\n", chinese.c_str());

  Utf8StreamDecoder emojiDecoder;
  const std::string emojiBytes12 =
      emojiDecoder.append(std::string("\xF0\x9F", 2));
  const std::string emojiByte3 = emojiDecoder.append(std::string("\x98", 1));
  const std::string emojiByte4 = emojiDecoder.append(std::string("\x80", 1));
  // CHECK: emoji chunks: 0 0 4
  std::fprintf(stderr, "emoji chunks: %zu %zu %zu\n", emojiBytes12.size(),
               emojiByte3.size(), emojiByte4.size());
  std::string emoji = emojiBytes12 + emojiByte3 + emojiByte4;
  emoji += emojiDecoder.finish();
  // CHECK: split emoji: 😀
  std::fprintf(stderr, "split emoji: %s\n", emoji.c_str());

  Utf8StreamDecoder malformedDecoder;
  std::string malformed = malformedDecoder.append(std::string("\x80", 1));
  malformed += malformedDecoder.append("valid");
  malformed += malformedDecoder.finish();
  // CHECK: malformed: �valid
  std::fprintf(stderr, "malformed: %s\n", malformed.c_str());

  Utf8StreamDecoder truncatedDecoder;
  std::string truncated =
      truncatedDecoder.append(std::string("tail \xE4\xB8", 7));
  truncated += truncatedDecoder.finish();
  // CHECK: truncated: tail �
  std::fprintf(stderr, "truncated: %s\n", truncated.c_str());

  return 0;
}
