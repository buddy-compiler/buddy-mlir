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
    std::string vocabDir = "../../../../tests/Interface/core/vocab.txt";
    //===--------------------------------------------------------------------===//
    // Test text constructor for pure string.
    //===--------------------------------------------------------------------===//
    std::string pureStr = "buddy compiler is a domain specific compiler";
    Text<long long, 2> pureStrContainer(pureStr);
    pureStrContainer.tokenize(vocabDir, 12);
    pureStrContainer.revert();
    // CHECK: 101
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[0]);
    // CHECK: 8937
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[1]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[2]);
    // CHECK: 2003
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[3]);
    // CHECK: 1037
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[4]);
    // CHECK: 5884
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[5]);
    // CHECK: 3563
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[6]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[7]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[8]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[9]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", pureStrContainer.getData()[11]);

    //===--------------------------------------------------------------------===//
    // Test text constructor for punctuation.
    //===--------------------------------------------------------------------===//
    std::string puncStr = "buddy compiler: a domain specific compiler!";
    Text<long long, 2> puncStrContainer(puncStr);
    puncStrContainer.tokenize(vocabDir, 12);
    // CHECK: 101
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[0]);
    // CHECK: 8937
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[1]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[2]);
    // CHECK: 1024
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[3]);
    // CHECK: 1037
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[4]);
    // CHECK: 5884
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[5]);
    // CHECK: 3563
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[6]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[7]);
    // CHECK: 999
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[8]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[9]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", puncStrContainer.getData()[11]);

    //===--------------------------------------------------------------------===//
    // Test text constructor for corner cases.
    //===--------------------------------------------------------------------===//
    std::string cornerStr = "  buddy compiler : a domain specific compiler!!  ";
    Text<long long, 2> cornerStrContainer(cornerStr);
    cornerStrContainer.tokenize(vocabDir, 12);
    // CHECK: 101
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[0]);
    // CHECK: 8937
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[1]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[2]);
    // CHECK: 1024
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[3]);
    // CHECK: 1037
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[4]);
    // CHECK: 5884
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[5]);
    // CHECK: 3563
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[6]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[7]);
    // CHECK: 999
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[8]);
    // CHECK: 999
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[9]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", cornerStrContainer.getData()[11]);

    //===--------------------------------------------------------------------===//
    // Test text constructor for chinese cases.
    //===--------------------------------------------------------------------===//
    std::string chineseStr = "我，中国北京人！";
    Text<long long, 2> chineseStrContainer(chineseStr);
    chineseStrContainer.tokenize(vocabDir, 12);
    // CHECK: 101
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[0]);
    // CHECK: 1855
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[1]);
    // CHECK: 1989
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[2]);
    // CHECK: 1746
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[3]);
    // CHECK: 1799
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[4]);
    // CHECK: 1781
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[5]);
    // CHECK: 1755
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[6]);
    // CHECK: 1756
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[7]);
    // CHECK: 1986
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[8]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[9]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", chineseStrContainer.getData()[11]);
    //===--------------------------------------------------------------------===//
    // Test text constructor for lower cases.
    //===--------------------------------------------------------------------===//
    std::string toLowerStr = "BUDDY COMPILER IS A DOMAIN SPECIFIC COMPILER";
    Text<long long, 2> toLowerStrContainer(toLowerStr);
    toLowerStrContainer.tokenize(vocabDir, 12, true);
    // CHECK: 101
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[0]);
    // CHECK: 8937
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[1]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[2]);
    // CHECK: 2003
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[3]);
    // CHECK: 1037
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[4]);
    // CHECK: 5884
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[5]);
    // CHECK: 3563
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[6]);
    // CHECK: 21624
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[7]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[8]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[9]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", toLowerStrContainer.getData()[11]);
    //===--------------------------------------------------------------------===//
    // Test text constructor for root and affix process.
    //===--------------------------------------------------------------------===//
    std::string affixStr = "it is colourless";
    Text<long long, 2> affixStrContainer(affixStr);
    affixStrContainer.tokenize(vocabDir, 12,false,true);
    // CHECK: 101
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[0]);
    // CHECK: 2009
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[1]);
    // CHECK: 2003
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[2]);
    // CHECK: 6120
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[3]);
    // CHECK: 3238
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[4]);
    // CHECK: 102
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[5]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[6]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[7]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[8]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[9]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[10]);
    // CHECK: 0
    fprintf(stderr, "%lld\n", affixStrContainer.getData()[11]);
}
