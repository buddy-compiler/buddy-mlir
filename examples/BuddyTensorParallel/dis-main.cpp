//===- dis-main.cpp -----------------------------------------------------===//
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

#include <algorithm>
#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

using namespace buddy;
double total_time = 0;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;
constexpr size_t SubMaxTokenLength = 512;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HiddenSize0 = 1536;
constexpr size_t HeadNum = 2;

struct MemRefContainerPrefill0 {
  MemRef<float, 3> data;
  MemRef<int8_t, 2> mask;
  MemRef<float, 3> cos;
  MemRef<float, 3> sin;

  MemRefContainerPrefill0(MemRef<float, 3> m1, MemRef<int8_t, 2> m2,
                          MemRef<float, 3> m3, MemRef<float, 3> m4)
      : data(m1), mask(m2), cos(m3), sin(m4) {}
};

struct MemRefContainerPrefill2 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 2> data;

  MemRefContainerPrefill2(MemRef<float, 4> m1, MemRef<float, 4> m2,
                          MemRef<float, 2> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};

struct MemRefContainerDecode1 {
  MemRef<float, 4> kcache;
  MemRef<float, 4> vcache;
  MemRef<float, 3> data;

  MemRefContainerDecode1(MemRef<float, 4> m1, MemRef<float, 4> m2,
                         MemRef<float, 3> m3)
      : kcache(m1), vcache(m2), data(m3) {}
};

/// Declare DeepSeekR1 forward function.
extern "C" {
void _mlir_ciface_forward_prefill0(MemRefContainerPrefill0 *,
                                   MemRef<float, 1> *, Text<size_t, 2> *);
void _mlir_ciface_forward_prefill1(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill2(MemRefContainerPrefill2 *,
                                   MemRef<float, 1> *, MemRef<int8_t, 4> *,
                                   MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill3(MemRef<float, 3> *, MemRef<float, 3> *,
                                   MemRef<float, 2> *);
void _mlir_ciface_forward_prefill5(MemRef<float, 2> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
void _mlir_ciface_forward_prefill169(MemRef<float, 3> *, MemRef<float, 1> *,
                                     MemRef<float, 3> *);

void _mlir_ciface_forward_decode0(MemRefContainerPrefill0 *, MemRef<float, 1> *,
                                  MemRef<long long, 2> *,
                                  MemRef<long long, 1> *);
void _mlir_ciface_forward_decode1(MemRefContainerDecode1 *, MemRef<float, 1> *,
                                  MemRef<long long, 1> *, MemRef<float, 4> *,
                                  MemRef<float, 4> *, MemRef<float, 3> *,
                                  MemRef<int8_t, 4> *, MemRef<float, 3> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode2(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<float, 3> *);
void _mlir_ciface_forward_decode57(MemRef<float, 3> *, MemRef<float, 1> *,
                                   MemRef<float, 3> *);
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Capture input message.
void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

/// Print [Log] label in bold blue format.
void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

/// Print information for each iteration.
void printIterInfo(size_t iterIdx, std::string str, double time) {
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

/// Tokenize input data in the container.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    std::cout << paramFilePath << std::endl;
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  printLogLabel();
  std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
            << "s\n"
            << std::endl;
}

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

/// Expand 2D attention mask to 4D attention mask.
void expandMask2DTo4D(MemRef<int8_t, 2> &src, MemRef<int8_t, 4> &dst,
                      int maxTokenLength) {
  auto srcStrides = src.getStrides();
  auto dstStrides = dst.getStrides();

  for (int i = 0; i < maxTokenLength; ++i) {
    for (int j = 0; j < maxTokenLength; ++j) {

      size_t srcIdx = i * srcStrides[0] + j * srcStrides[1];

      size_t dstIdx = 0 * dstStrides[0] + 0 * dstStrides[1] +
                      i * dstStrides[2] + j * dstStrides[3];

      dst[dstIdx] = src[srcIdx];
    }
  }
}

/// Expand 2D attention mask to 4D attention mask for 1 length input.
void expandMask2D_1L_to_4D(MemRef<int8_t, 2> &src, MemRef<int8_t, 4> &dst,
                           int maxTokenLength) {

  auto srcStrides = src.getStrides();
  auto dstStrides = dst.getStrides();

  for (int j = 0; j < maxTokenLength; ++j) {
    size_t srcIdx = 0 * srcStrides[0] + j * srcStrides[1];

    size_t dstIdx = 0 * dstStrides[0] + 0 * dstStrides[1] + 0 * dstStrides[2] +
                    j * dstStrides[3];

    dst[dstIdx] = src[srcIdx];
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "DeepSeekR1  Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DSR1TP_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DSR1TP_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "/vocab.txt";

  int splitGroupPrefill[] = {
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
      1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,
      2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1,
      1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1,
      2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2,
      1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1};
  constexpr size_t paramSizeGroupPrefill[] = {
      233373760, 1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 1536,     2753536,
      0,         1536, 20643840, 0, 1536, 2753536,  0, 1536,     20643840,
      0,         1536, 2753536,  0, 1536, 20643840, 0, 233375232};
  std::vector<std::string> paramsDirsPrefill;
  for (int i = 0; i <= 169; i++) {
    for (int j = 0; j < splitGroupPrefill[i]; j++) {
      paramsDirsPrefill.emplace_back(deepSeekR1BuildDir + "/subgraph0_prefill" +
                                     std::to_string(i) + "_arg" +
                                     std::to_string(j) + ".data");
    }
  }
  int splitGroupDecode[] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1};
  constexpr size_t paramSizeGroupDecode[] = {
      233373760, 2755072,  20645376, 2755072,  20645376, 2755072,  20645376,
      2755072,   20645376, 2755072,  20645376, 2755072,  20645376, 2755072,
      20645376,  2755072,  20645376, 2755072,  20645376, 2755072,  20645376,
      2755072,   20645376, 2755072,  20645376, 2755072,  20645376, 2755072,
      20645376,  2755072,  20645376, 2755072,  20645376, 2755072,  20645376,
      2755072,   20645376, 2755072,  20645376, 2755072,  20645376, 2755072,
      20645376,  2755072,  20645376, 2755072,  20645376, 2755072,  20645376,
      2755072,   20645376, 2755072,  20645376, 2755072,  20645376, 2755072,
      20645376,  233375232};
  std::vector<std::string> paramsDirsDecode;
  for (int i = 0; i <= 57; i++) {
    for (int j = 0; j < splitGroupDecode[i]; j++) {
      paramsDirsDecode.emplace_back(deepSeekR1BuildDir + "/subgraph0_decode" +
                                    std::to_string(i) + "_arg" +
                                    std::to_string(j) + ".data");
    }
  }

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Output container.
  //  - Parameters container.
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  std::vector<MemRef<float, 1>> paramsContainersPrefill;
  std::vector<MemRef<float, 1>> paramsContainersDecode;
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize0});
  MemRef<int8_t, 2> myMemRef2({MaxTokenLength, MaxTokenLength});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize});
  MemRefContainerPrefill0 resultContainer(myMemRef1, myMemRef2, myMemRef3,
                                          myMemRef4);
  MemRefContainerPrefill0 *resultContainerPtr = &resultContainer;
  MemRef<float, 3> resultContainer0({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> resultPrefill({1, MaxTokenLength, MaxVocabSize});
  MemRef<float, 3> resultDecode({1, 1, MaxVocabSize});
  MemRef<float, 3> subResultContainer0({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> subResultContainer1({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> sub3DContainer0({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> sub3DContainer1({1, SubMaxTokenLength, HiddenSize0});
  MemRef<float, 3> tmp3DContainer({1, MaxTokenLength, HiddenSize0});

  std::vector<MemRef<float, 4>> kv0;
  kv0.reserve(56);
  for (int i = 0; i < 56; ++i) {
    kv0.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
  }

  std::vector<MemRef<float, 4>> kv1;
  kv1.reserve(56);
  for (int i = 0; i < 56; ++i) {
    kv1.emplace_back(std::vector<size_t>{1, 1, MaxTokenLength, HiddenSize});
  }

  MemRef<int8_t, 4> mask4DContainer({1, 1, MaxTokenLength, MaxTokenLength});
  MemRef<float, 2> tmp2DContainer0({MaxTokenLength, HiddenSize0});
  MemRef<float, 2> tmp2DContainer1({MaxTokenLength, HiddenSize0});
  MemRef<float, 2> sub2DContainer0({SubMaxTokenLength, HiddenSize0});
  MemRef<float, 2> sub2DContainer1({SubMaxTokenLength, HiddenSize0});

  // MemRef<float, 3> myMemRef_decode1({1, 1, HiddenSize0});
  // MemRef<int8_t, 2> myMemRef_decode2({1, MaxTokenLength});
  // MemRef<float, 3> myMemRef_decode3({1, 1, HiddenSize});
  // MemRef<float, 3> myMemRef_decode4({1, 1, HiddenSize});
  // MemRefContainerPrefill0 resultContainer_decode(
  // myMemRef_decode1, myMemRef_decode2, myMemRef_decode3, myMemRef_decode4);
  
  // MemRefContainerPrefill0 *resultContainerPtr_decode = &resultContainer_decode;
  MemRef<float, 3> decodeData({1, 1, HiddenSize0});
  MemRef<float, 3> mhaDecodeData0({1, 1, HiddenSize0});
  MemRef<float, 3> mhaDecodeData1({1, 1, HiddenSize0});
  MemRef<float, 3> mlpData1({1, 1, HiddenSize0});
  MemRef<int8_t, 4> mask4DContainerDecode({1, 1, 1, MaxTokenLength});

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);

  MemRef<float, 1> paramsContainer0({paramSizeGroupPrefill[0]});
  loadParameters(paramsDirsPrefill[0], paramsContainer0);

  int params_count = 1;
  for (int i = 1; i < 169; i++) {
    for (int j = 0; j < splitGroupPrefill[i]; j++) {
      if (paramSizeGroupPrefill[i] > 0) {
        MemRef<float, 1> paramsContainer1({paramSizeGroupPrefill[i]});
        loadParameters(paramsDirsPrefill[params_count], paramsContainer1);
        paramsContainersPrefill.push_back(paramsContainer1);
      }
      params_count++;
    }
  }
  MemRef<float, 1> paramsContainer2({paramSizeGroupPrefill[169]});
  loadParameters(paramsDirsPrefill[params_count], paramsContainer2);

  /// Run DeepSeekR1 Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();

  _mlir_ciface_forward_prefill0(resultContainerPtr, &paramsContainer0,
                                &inputContainerPrefill);
  resultContainer0 = resultContainerPtr->data;
  auto resultContainer1 = resultContainerPtr->mask;
  auto resultContainer2 = resultContainerPtr->cos;
  auto resultContainer3 = resultContainerPtr->sin;
  expandMask2DTo4D(resultContainer1, mask4DContainer, MaxTokenLength);
  resultContainer0.splitMemRef(std::move(resultContainer0), subResultContainer0,
                               subResultContainer1, 1, 512);
  for (int m = 0; m < 28; m++) {
    _mlir_ciface_forward_prefill1(&sub3DContainer0,
                                  &paramsContainersPrefill[m * 6],
                                  &subResultContainer0);
    _mlir_ciface_forward_prefill1(&sub3DContainer1,
                                  &paramsContainersPrefill[m * 6],
                                  &subResultContainer1);
    tmp3DContainer.concatenateMemRefs(sub3DContainer0, sub3DContainer1,
                                      tmp3DContainer, 1);
    MemRef<float, 4> &k0 = kv0[2 * m];
    MemRef<float, 4> &v0 = kv0[2 * m + 1];
    MemRef<float, 2> mhaData0({MaxTokenLength, HiddenSize0});
    MemRefContainerPrefill2 kvContainer0(k0, v0, mhaData0);
    MemRefContainerPrefill2 *kvContainerPtr0 = &kvContainer0;
    _mlir_ciface_forward_prefill2(
        kvContainerPtr0, &paramsContainersPrefill[m * 6 + 1], &mask4DContainer,
        &resultContainer2, &resultContainer3, &tmp3DContainer);
    MemRef<float, 4> &k1 = kv1[2 * m];
    MemRef<float, 4> &v1 = kv1[2 * m + 1];
    MemRef<float, 2> mhaData1({MaxTokenLength, HiddenSize0});
    MemRefContainerPrefill2 kvContainer1(k1, v1, mhaData1);
    MemRefContainerPrefill2 *kvContainerPtr1 = &kvContainer1;
    _mlir_ciface_forward_prefill2(
        kvContainerPtr1, &paramsContainersPrefill[m * 6 + 2], &mask4DContainer,
        &resultContainer2, &resultContainer3, &tmp3DContainer);
    tmp2DContainer0 = kvContainerPtr0->data;
    tmp2DContainer1 = kvContainerPtr1->data;
    tmp2DContainer0.addMemRef(tmp2DContainer0, tmp2DContainer1);
    tmp2DContainer0.splitMemRef(std::move(tmp2DContainer0), sub2DContainer0,
                                sub2DContainer1, 0, 512);
    _mlir_ciface_forward_prefill3(&subResultContainer0, &subResultContainer0,
                                  &sub2DContainer0);
    _mlir_ciface_forward_prefill3(&subResultContainer1, &subResultContainer1,
                                  &sub2DContainer1);
    _mlir_ciface_forward_prefill1(&sub3DContainer0,
                                  &paramsContainersPrefill[m * 6 + 3],
                                  &subResultContainer0);
    _mlir_ciface_forward_prefill1(&sub3DContainer1,
                                  &paramsContainersPrefill[m * 6 + 3],
                                  &subResultContainer1);
    tmp3DContainer.concatenateMemRefs(sub3DContainer0, sub3DContainer1,
                                      tmp3DContainer, 1);
    _mlir_ciface_forward_prefill5(
        &tmp2DContainer0, &paramsContainersPrefill[m * 6 + 4], &tmp3DContainer);
    _mlir_ciface_forward_prefill5(
        &tmp2DContainer1, &paramsContainersPrefill[m * 6 + 5], &tmp3DContainer);
    tmp2DContainer0.addMemRef(tmp2DContainer0, tmp2DContainer1);
    tmp2DContainer0.splitMemRef(std::move(tmp2DContainer0), sub2DContainer0,
                                sub2DContainer1, 0, 512);
    _mlir_ciface_forward_prefill3(&subResultContainer0, &subResultContainer0,
                                  &sub2DContainer0);
    _mlir_ciface_forward_prefill3(&subResultContainer1, &subResultContainer1,
                                  &sub2DContainer1);
  }
  tmp3DContainer.concatenateMemRefs(subResultContainer0, subResultContainer1,
                                    tmp3DContainer, 1);
  _mlir_ciface_forward_prefill169(&resultPrefill, &paramsContainer2,
                                  &tmp3DContainer);

  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  // Determine the generated token.
  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const float *startPtr = resultPrefill.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  // Print the generated token and inference time.
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }

  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);
  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  MemRef<float, 1> paramsContainerDecode0({paramSizeGroupDecode[0]});
  loadParameters(paramsDirsDecode[0], paramsContainerDecode0);

  int current_file_index = 1;
  for (int i = 1; i < 57; i++) {
    for (int j = 0; j < splitGroupDecode[i]; j++) {
      if (paramSizeGroupDecode[i] > 0) {
        MemRef<float, 1> paramsContainer_decode1({paramSizeGroupDecode[i]});
        loadParameters(paramsDirsDecode[current_file_index],
                       paramsContainer_decode1);
        paramsContainersDecode.push_back(paramsContainer_decode1);
        current_file_index++;
      }
    }
  }
  MemRef<float, 1> paramsContainerDecode57({paramSizeGroupDecode[57]});
  loadParameters(paramsDirsDecode[current_file_index], paramsContainerDecode57);


  // Decode phase
  for (int i = 1; i <= generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode0(resultContainerPtr,
                                 &paramsContainerDecode0,
                                 &inputContainerDecode, &cachePosition);
    decodeData = resultContainerPtr->data;
    auto resultContainer1 = resultContainerPtr->mask;
    auto resultContainer2 = resultContainerPtr->cos;
    auto resultContainer3 = resultContainerPtr->sin;
    expandMask2D_1L_to_4D(resultContainer1, mask4DContainerDecode,
                          MaxTokenLength);
    for (int m = 0; m < 28; m++) {
      MemRef<float, 4> &k0 = kv0[2 * m];
      MemRef<float, 4> &v0 = kv0[2 * m + 1];
      MemRef<float, 3> mhaData0({1, 1, HiddenSize0});
      MemRefContainerDecode1 kvContainer0(k0, v0, mhaData0);
      MemRefContainerDecode1 *kvContainerPtr0 = &kvContainer0;
      _mlir_ciface_forward_decode1(
          kvContainerPtr0, &paramsContainersDecode[m * 4], &cachePosition, 
          &k0, &v0, &decodeData, &mask4DContainerDecode, &resultContainer2,
          &resultContainer3);
      MemRef<float, 4> &k1 = kv1[2 * m];
      MemRef<float, 4> &v1 = kv1[2 * m + 1];
      MemRef<float, 3> mhaData1({1, 1, HiddenSize0});
      MemRefContainerDecode1 kvContainer1(k1, v1, mhaData1);
      MemRefContainerDecode1 *kvContainerPtr1 = &kvContainer1;
      _mlir_ciface_forward_decode1(
          kvContainerPtr1, &paramsContainersDecode[m * 4 + 1], &cachePosition,
          &k1, &v1, &decodeData, &mask4DContainerDecode, &resultContainer2,
          &resultContainer3);
      mhaDecodeData0 = kvContainerPtr0->data;
      mhaDecodeData1 = kvContainerPtr1->data;
      mhaDecodeData0.addMemRef(mhaDecodeData0, mhaDecodeData1);
      _mlir_ciface_forward_decode2(
          &decodeData, &paramsContainersDecode[m * 4 + 2], &mhaDecodeData0);
      _mlir_ciface_forward_decode2(
          &mlpData1, &paramsContainersDecode[m * 4 + 3], &mhaDecodeData0);
      decodeData.addMemRef(decodeData, mlpData1);
    }

    _mlir_ciface_forward_decode57(&resultDecode, &paramsContainerDecode57,
                                  &decodeData);
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;
    // Determine the generated token.
    const float *startPtr = resultDecode.getData();
    const float *endPtr = startPtr + MaxVocabSize;
    int maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);
    if (maxIndex == 151643) {
      break;
    }

    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;
  }

  double decodeSeconds = decodeTimeAccumMs / 1000.0;
  const double decodeTokensPerSec =
      decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                          : 0.0;

  //

  /// Print the final result
  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  return 0;
}
