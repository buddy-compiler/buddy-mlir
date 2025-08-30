//===- TextContainerTest.cpp ----------------------------------------------===//
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
// This is the text container test file.
//
//===----------------------------------------------------------------------===//

// RUN: buddy-text-container-test 2>&1 | FileCheck %s

#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>

using namespace buddy;

int main() {
  // The map of string-to-id used in the test cases:
  // buddy: 8937, compiler: 21624, is: 2003, a: 1037, domain: 5884
  // specific: 3563, ":": 1024, "!": 999 ,",":1989
  // 我:1855, 中:1746, 国:1799, 北:1781, 京:1755, 人:1756
  // it:2009, colour:6120, ##less:3238
  //
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // vocabulary directory uses the following relative path.
  std::string vocabDir = "../../../../tests/Interface/core/vocab_bert.txt";
  //===--------------------------------------------------------------------===//
  // Test text constructor for pure string.
  //===--------------------------------------------------------------------===//
  std::string pureStr = "buddy compiler is a domain specific compiler";
  Text<size_t, 2> pureStrBertContainer(pureStr);
  pureStrBertContainer.tokenizeBert(vocabDir, 12);
  // CHECK: 101
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[0]);
  // CHECK: 8937
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[1]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[2]);
  // CHECK: 2003
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[3]);
  // CHECK: 1037
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[4]);
  // CHECK: 5884
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[5]);
  // CHECK: 3563
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[6]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[7]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[8]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", pureStrBertContainer.getData()[11]);

  //===--------------------------------------------------------------------===//
  // Test text constructor for punctuation.
  //===--------------------------------------------------------------------===//
  std::string puncStr = "buddy compiler: a domain specific compiler!";
  Text<size_t, 2> puncStrBertContainer(puncStr);
  puncStrBertContainer.tokenizeBert(vocabDir, 12);
  // CHECK: 101
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[0]);
  // CHECK: 8937
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[1]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[2]);
  // CHECK: 1024
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[3]);
  // CHECK: 1037
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[4]);
  // CHECK: 5884
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[5]);
  // CHECK: 3563
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[6]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[7]);
  // CHECK: 999
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[8]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", puncStrBertContainer.getData()[11]);

  //===--------------------------------------------------------------------===//
  // Test text constructor for corner cases.
  //===--------------------------------------------------------------------===//
  std::string cornerStr = "  buddy compiler : a domain specific compiler!!  ";
  Text<size_t, 2> cornerStrContainer(cornerStr);
  cornerStrContainer.tokenizeBert(vocabDir, 12);
  // CHECK: 101
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[0]);
  // CHECK: 8937
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[1]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[2]);
  // CHECK: 1024
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[3]);
  // CHECK: 1037
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[4]);
  // CHECK: 5884
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[5]);
  // CHECK: 3563
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[6]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[7]);
  // CHECK: 999
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[8]);
  // CHECK: 999
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", cornerStrContainer.getData()[11]);

  //===--------------------------------------------------------------------===//
  // Test text constructor for mutibyteschar cases.
  // Specially, the Chinese characters are included.
  // Select Chinese characters for testing.
  //===--------------------------------------------------------------------===//
  std::string chineseStr = "我，中国北京人！";
  Text<size_t, 2> chineseStrBertContainer(chineseStr);
  chineseStrBertContainer.tokenizeBert(vocabDir, 12);
  // CHECK: 101
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[0]);
  // CHECK: 1855
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[1]);
  // CHECK: 1989
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[2]);
  // CHECK: 1746
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[3]);
  // CHECK: 1799
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[4]);
  // CHECK: 1781
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[5]);
  // CHECK: 1755
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[6]);
  // CHECK: 1756
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[7]);
  // CHECK: 1986
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[8]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", chineseStrBertContainer.getData()[11]);
  //===--------------------------------------------------------------------===//
  // Test text constructor for lower cases.
  //===--------------------------------------------------------------------===//
  std::string toLowerStr = "BUDDY COMPILER IS A DOMAIN SPECIFIC COMPILER";
  Text<size_t, 2> toLowerStrBertContainer(toLowerStr);
  toLowerStrBertContainer.tokenizeBert(vocabDir, 12, true);
  // CHECK: 101
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[0]);
  // CHECK: 8937
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[1]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[2]);
  // CHECK: 2003
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[3]);
  // CHECK: 1037
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[4]);
  // CHECK: 5884
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[5]);
  // CHECK: 3563
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[6]);
  // CHECK: 21624
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[7]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[8]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", toLowerStrBertContainer.getData()[11]);
  //===--------------------------------------------------------------------===//
  // Test text constructor for root and affix process.
  //===--------------------------------------------------------------------===//
  std::string affixStr = "it is colourless";
  Text<size_t, 2> affixStrBertContainer(affixStr);
  affixStrBertContainer.tokenizeBert(vocabDir, 12, false, true);
  // CHECK: 101
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[0]);
  // CHECK: 2009
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[1]);
  // CHECK: 2003
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[2]);
  // CHECK: 6120
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[3]);
  // CHECK: 3238
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[4]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[5]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[6]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[7]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[8]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[9]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[10]);
  // CHECK: 102
  fprintf(stderr, "%ld\n", affixStrBertContainer.getData()[11]);
  // The map of string-to-id used in the test cases:
  // bud: 8619, dy:4518, compiler: 6516, is: 338, a: 263, domain: 5354
  // specific: 2702, ":": 29901, "!": 29991
  //
  // The test running directory is in <build dir>/tests/Interface/core, so the
  // vocabulary directory uses the following relative path.
  vocabDir = "../../../../tests/Interface/core/vocab_llama.txt";
  //===--------------------------------------------------------------------===//
  // Test text constructor for pure string using Llama tokenizer.
  //===--------------------------------------------------------------------===//
  std::string pureStrLlama = "buddy compiler is a domain specific compiler";
  Text<size_t, 2> pureStrLlamaContainer(pureStrLlama);
  pureStrLlamaContainer.tokenizeLlama(vocabDir, 12);
  std::string pureStrLlamaResult =
      pureStrLlamaContainer.revertLlama();
  // CHECK: 1
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[0]);
  // CHECK: 8619
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[1]);
  // CHECK: 4518
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[2]);
  // CHECK: 6516
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[3]);
  // CHECK: 338
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[4]);
  // CHECK: 263
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[5]);
  // CHECK: 5354
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[6]);
  // CHECK: 2702
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[7]);
  // CHECK: 6516
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[8]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[9]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[10]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", pureStrLlamaContainer.getData()[11]);
  // CHECK: buddy compiler is a domain specific compiler
  fprintf(stderr, "%s\n", pureStrLlamaResult.c_str());
  //===--------------------------------------------------------------------===//
  // Test text constructor for punctuation.
  //===--------------------------------------------------------------------===//
  std::string puncStrLlama = "buddy compiler: a domain specific compiler!";
  Text<size_t, 2> puncStrLlamaContainer(puncStrLlama);
  puncStrLlamaContainer.tokenizeLlama(vocabDir, 12);
  std::string puncStrLlamaResult =
      puncStrLlamaContainer.revertLlama();
  // CHECK: 1
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[0]);
  // CHECK: 8619
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[1]);
  // CHECK: 4518
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[2]);
  // CHECK: 6516
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[3]);
  // CHECK: 29901
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[4]);
  // CHECK: 263
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[5]);
  // CHECK: 5354
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[6]);
  // CHECK: 2702
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[7]);
  // CHECK: 6516
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[8]);
  // CHECK: 29991
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[9]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[10]);
  // CHECK: 2
  fprintf(stderr, "%ld\n", puncStrLlamaContainer.getData()[11]);
  // CHECK: buddy compiler: a domain specific compiler!
  fprintf(stderr, "%s\n", puncStrLlamaResult.c_str());
}
