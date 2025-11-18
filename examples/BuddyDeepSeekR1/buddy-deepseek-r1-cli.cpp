//===- buddy-deepseek-r1-cli.cpp ----------------------------------------===//
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
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace buddy;
namespace fs = std::filesystem;

namespace {

constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;
constexpr long long DefaultEosToken = 151643;

struct MemRefContainer {
  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<float, 4> kv2;
  MemRef<float, 4> kv3;
  MemRef<float, 4> kv4;
  MemRef<float, 4> kv5;
  MemRef<float, 4> kv6;
  MemRef<float, 4> kv7;
  MemRef<float, 4> kv8;
  MemRef<float, 4> kv9;
  MemRef<float, 4> kv10;
  MemRef<float, 4> kv11;
  MemRef<float, 4> kv12;
  MemRef<float, 4> kv13;
  MemRef<float, 4> kv14;
  MemRef<float, 4> kv15;
  MemRef<float, 4> kv16;
  MemRef<float, 4> kv17;
  MemRef<float, 4> kv18;
  MemRef<float, 4> kv19;
  MemRef<float, 4> kv20;
  MemRef<float, 4> kv21;
  MemRef<float, 4> kv22;
  MemRef<float, 4> kv23;
  MemRef<float, 4> kv24;
  MemRef<float, 4> kv25;
  MemRef<float, 4> kv26;
  MemRef<float, 4> kv27;
  MemRef<float, 4> kv28;
  MemRef<float, 4> kv29;
  MemRef<float, 4> kv30;
  MemRef<float, 4> kv31;
  MemRef<float, 4> kv32;
  MemRef<float, 4> kv33;
  MemRef<float, 4> kv34;
  MemRef<float, 4> kv35;
  MemRef<float, 4> kv36;
  MemRef<float, 4> kv37;
  MemRef<float, 4> kv38;
  MemRef<float, 4> kv39;
  MemRef<float, 4> kv40;
  MemRef<float, 4> kv41;
  MemRef<float, 4> kv42;
  MemRef<float, 4> kv43;
  MemRef<float, 4> kv44;
  MemRef<float, 4> kv45;
  MemRef<float, 4> kv46;
  MemRef<float, 4> kv47;
  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;

  MemRef<float, 3> logits;

  std::array<MemRef<float, 4> *, 56> kvPtrs;

  MemRefContainer(
      MemRef<float, 4> k0, MemRef<float, 4> k1, MemRef<float, 4> k2,
      MemRef<float, 4> k3, MemRef<float, 4> k4, MemRef<float, 4> k5,
      MemRef<float, 4> k6, MemRef<float, 4> k7, MemRef<float, 4> k8,
      MemRef<float, 4> k9, MemRef<float, 4> k10, MemRef<float, 4> k11,
      MemRef<float, 4> k12, MemRef<float, 4> k13, MemRef<float, 4> k14,
      MemRef<float, 4> k15, MemRef<float, 4> k16, MemRef<float, 4> k17,
      MemRef<float, 4> k18, MemRef<float, 4> k19, MemRef<float, 4> k20,
      MemRef<float, 4> k21, MemRef<float, 4> k22, MemRef<float, 4> k23,
      MemRef<float, 4> k24, MemRef<float, 4> k25, MemRef<float, 4> k26,
      MemRef<float, 4> k27, MemRef<float, 4> k28, MemRef<float, 4> k29,
      MemRef<float, 4> k30, MemRef<float, 4> k31, MemRef<float, 4> k32,
      MemRef<float, 4> k33, MemRef<float, 4> k34, MemRef<float, 4> k35,
      MemRef<float, 4> k36, MemRef<float, 4> k37, MemRef<float, 4> k38,
      MemRef<float, 4> k39, MemRef<float, 4> k40, MemRef<float, 4> k41,
      MemRef<float, 4> k42, MemRef<float, 4> k43, MemRef<float, 4> k44,
      MemRef<float, 4> k45, MemRef<float, 4> k46, MemRef<float, 4> k47,
      MemRef<float, 4> k48, MemRef<float, 4> k49, MemRef<float, 4> k50,
      MemRef<float, 4> k51, MemRef<float, 4> k52, MemRef<float, 4> k53,
      MemRef<float, 4> k54, MemRef<float, 4> k55, MemRef<float, 3> l)
      : kv0(k0), kv1(k1), kv2(k2), kv3(k3), kv4(k4), kv5(k5), kv6(k6), kv7(k7),
        kv8(k8), kv9(k9), kv10(k10), kv11(k11), kv12(k12), kv13(k13), kv14(k14),
        kv15(k15), kv16(k16), kv17(k17), kv18(k18), kv19(k19), kv20(k20),
        kv21(k21), kv22(k22), kv23(k23), kv24(k24), kv25(k25), kv26(k26),
        kv27(k27), kv28(k28), kv29(k29), kv30(k30), kv31(k31), kv32(k32),
        kv33(k33), kv34(k34), kv35(k35), kv36(k36), kv37(k37), kv38(k38),
        kv39(k39), kv40(k40), kv41(k41), kv42(k42), kv43(k43), kv44(k44),
        kv45(k45), kv46(k46), kv47(k47), kv48(k48), kv49(k49), kv50(k50),
        kv51(k51), kv52(k52), kv53(k53), kv54(k54), kv55(k55), logits(l),
        kvPtrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,
               &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,
               &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,
               &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,
               &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,
               &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,
               &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
                                             MemRef<float, 1> *arg0,
                                             Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<float, 1> *arg0, MemRef<long long, 2> *arg1,
    MemRef<long long, 1> *arg2, MemRef<float, 4> *kv0, MemRef<float, 4> *kv1,
    MemRef<float, 4> *kv2, MemRef<float, 4> *kv3, MemRef<float, 4> *kv4,
    MemRef<float, 4> *kv5, MemRef<float, 4> *kv6, MemRef<float, 4> *kv7,
    MemRef<float, 4> *kv8, MemRef<float, 4> *kv9, MemRef<float, 4> *kv10,
    MemRef<float, 4> *kv11, MemRef<float, 4> *kv12, MemRef<float, 4> *kv13,
    MemRef<float, 4> *kv14, MemRef<float, 4> *kv15, MemRef<float, 4> *kv16,
    MemRef<float, 4> *kv17, MemRef<float, 4> *kv18, MemRef<float, 4> *kv19,
    MemRef<float, 4> *kv20, MemRef<float, 4> *kv21, MemRef<float, 4> *kv22,
    MemRef<float, 4> *kv23, MemRef<float, 4> *kv24, MemRef<float, 4> *kv25,
    MemRef<float, 4> *kv26, MemRef<float, 4> *kv27, MemRef<float, 4> *kv28,
    MemRef<float, 4> *kv29, MemRef<float, 4> *kv30, MemRef<float, 4> *kv31,
    MemRef<float, 4> *kv32, MemRef<float, 4> *kv33, MemRef<float, 4> *kv34,
    MemRef<float, 4> *kv35, MemRef<float, 4> *kv36, MemRef<float, 4> *kv37,
    MemRef<float, 4> *kv38, MemRef<float, 4> *kv39, MemRef<float, 4> *kv40,
    MemRef<float, 4> *kv41, MemRef<float, 4> *kv42, MemRef<float, 4> *kv43,
    MemRef<float, 4> *kv44, MemRef<float, 4> *kv45, MemRef<float, 4> *kv46,
    MemRef<float, 4> *kv47, MemRef<float, 4> *kv48, MemRef<float, 4> *kv49,
    MemRef<float, 4> *kv50, MemRef<float, 4> *kv51, MemRef<float, 4> *kv52,
    MemRef<float, 4> *kv53, MemRef<float, 4> *kv54, MemRef<float, 4> *kv55);

int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

void copyKVByCachePositionBlock(const MemRefContainer &prefill,
                                MemRefContainer &decode, int cachePosition) {
  constexpr int numKV = 56;
  const int copyLen = std::min(cachePosition, static_cast<int>(MaxTokenLength));

  for (int k = 0; k < numKV; ++k) {
    auto &src = *prefill.kvPtrs[k];
    auto &dst = *decode.kvPtrs[k];

    for (int h = 0; h < static_cast<int>(HeadNum); ++h) {
      const size_t bytesToCopy =
          static_cast<size_t>(copyLen) * HiddenSize * sizeof(float);
      float *srcPtr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dstPtr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dstPtr, srcPtr, bytesToCopy);
    }
  }
}

std::string readPromptFromFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open prompt file: " + filePath);
  }
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

std::string readPromptFromStdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("Failed to open model parameter file: " +
                             paramFilePath);
  }
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (paramFile.fail()) {
    throw std::runtime_error("Failed to read model parameters: " +
                             paramFilePath);
  }
}

struct GenerationResult {
  size_t promptTokens = 0;
  size_t generatedTokens = 0;
  double prefillTokensPerSec = 0.0;
  double decodeTokensPerSec = 0.0;
  double totalSeconds = 0.0;
  std::string finalText;
};

std::string getDefaultModelPath() {
#ifdef DEEPSEEKR1_EXAMPLE_BUILD_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_BUILD_PATH) + "arg0.data";
#else
  return "arg0.data";
#endif
}

std::string getDefaultVocabPath() {
#ifdef DEEPSEEKR1_EXAMPLE_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_PATH) + "vocab.txt";
#else
  return "vocab.txt";
#endif
}

void streamNewText(Text<size_t, 2> &outputContainer, std::string &lastPrinted,
                   std::ostream &tokenStream) {
  std::string current = outputContainer.revertDeepSeekR1();
  if (current.size() > lastPrinted.size()) {
    tokenStream.write(current.data() + lastPrinted.size(),
                      current.size() - lastPrinted.size());
    tokenStream.flush();
  }
  lastPrinted = std::move(current);
}

GenerationResult runGeneration(const std::string &prompt,
                               MemRef<float, 1> &paramsContainer,
                               const std::string &vocabPath, int maxNewTokens,
                               long long eosTokenId,
                               std::ostream &tokenStream) {
  GenerationResult stats;

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(prompt);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logitsPrefill({1, MaxTokenLength, MaxVocabSize});

  auto makeKV = []() {
    return MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  };

  MemRef<float, 4> kv0 = makeKV();
  MemRef<float, 4> kv1 = makeKV();
  MemRef<float, 4> kv2 = makeKV();
  MemRef<float, 4> kv3 = makeKV();
  MemRef<float, 4> kv4 = makeKV();
  MemRef<float, 4> kv5 = makeKV();
  MemRef<float, 4> kv6 = makeKV();
  MemRef<float, 4> kv7 = makeKV();
  MemRef<float, 4> kv8 = makeKV();
  MemRef<float, 4> kv9 = makeKV();
  MemRef<float, 4> kv10 = makeKV();
  MemRef<float, 4> kv11 = makeKV();
  MemRef<float, 4> kv12 = makeKV();
  MemRef<float, 4> kv13 = makeKV();
  MemRef<float, 4> kv14 = makeKV();
  MemRef<float, 4> kv15 = makeKV();
  MemRef<float, 4> kv16 = makeKV();
  MemRef<float, 4> kv17 = makeKV();
  MemRef<float, 4> kv18 = makeKV();
  MemRef<float, 4> kv19 = makeKV();
  MemRef<float, 4> kv20 = makeKV();
  MemRef<float, 4> kv21 = makeKV();
  MemRef<float, 4> kv22 = makeKV();
  MemRef<float, 4> kv23 = makeKV();
  MemRef<float, 4> kv24 = makeKV();
  MemRef<float, 4> kv25 = makeKV();
  MemRef<float, 4> kv26 = makeKV();
  MemRef<float, 4> kv27 = makeKV();
  MemRef<float, 4> kv28 = makeKV();
  MemRef<float, 4> kv29 = makeKV();
  MemRef<float, 4> kv30 = makeKV();
  MemRef<float, 4> kv31 = makeKV();
  MemRef<float, 4> kv32 = makeKV();
  MemRef<float, 4> kv33 = makeKV();
  MemRef<float, 4> kv34 = makeKV();
  MemRef<float, 4> kv35 = makeKV();
  MemRef<float, 4> kv36 = makeKV();
  MemRef<float, 4> kv37 = makeKV();
  MemRef<float, 4> kv38 = makeKV();
  MemRef<float, 4> kv39 = makeKV();
  MemRef<float, 4> kv40 = makeKV();
  MemRef<float, 4> kv41 = makeKV();
  MemRef<float, 4> kv42 = makeKV();
  MemRef<float, 4> kv43 = makeKV();
  MemRef<float, 4> kv44 = makeKV();
  MemRef<float, 4> kv45 = makeKV();
  MemRef<float, 4> kv46 = makeKV();
  MemRef<float, 4> kv47 = makeKV();
  MemRef<float, 4> kv48 = makeKV();
  MemRef<float, 4> kv49 = makeKV();
  MemRef<float, 4> kv50 = makeKV();
  MemRef<float, 4> kv51 = makeKV();
  MemRef<float, 4> kv52 = makeKV();
  MemRef<float, 4> kv53 = makeKV();
  MemRef<float, 4> kv54 = makeKV();
  MemRef<float, 4> kv55 = makeKV();

  MemRefContainer prefillResult(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logitsPrefill);
  MemRefContainer *prefillPtr = &prefillResult;

  outputContainer.loadVocab(vocabPath);

  inputContainerPrefill.tokenizeDeepSeekR1(vocabPath, MaxTokenLength);
  if (inputContainerPrefill.getTokenCnt() == 0) {
    tokenStream << std::endl;
    stats.finalText.clear();
    return stats;
  }
  // Prefill graph always runs with a fixed sequence length, so report that.
  stats.promptTokens = MaxTokenLength;

  const auto prefillStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(prefillPtr, &paramsContainer,
                               &inputContainerPrefill);
  const auto prefillEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> prefillMs =
      prefillEnd - prefillStart;
  const double prefillSeconds = prefillMs.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    stats.prefillTokensPerSec =
        static_cast<double>(MaxTokenLength) / prefillSeconds;
  }

  std::string streamed;
  int availableByContext =
      std::max(0, static_cast<int>(MaxTokenLength) -
                      static_cast<int>(inputContainerPrefill.getTokenCnt()));
  if (availableByContext == 0 || maxNewTokens == 0) {
    tokenStream << std::endl;
    stats.totalSeconds = prefillSeconds;
    stats.finalText = streamed;
    return stats;
  }

  int remainingBudget = std::min(maxNewTokens, availableByContext);

  MemRef<float, 3> logitsDecode({1, 1, MaxVocabSize});
  MemRefContainer decodeResult(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logitsDecode);
  MemRefContainer *decodePtr = &decodeResult;

  const int tokenIndex =
      static_cast<int>(inputContainerPrefill.getTokenCnt()) - 1;
  const float *startPtr =
      prefillPtr->logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);

  copyKVByCachePositionBlock(prefillResult, decodeResult,
                             inputContainerPrefill.getTokenCnt());

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  inputContainerDecode.getData()[0] = static_cast<long long>(maxIndex);
  if (maxIndex == eosTokenId) {
    tokenStream << std::endl;
    stats.totalSeconds = prefillSeconds;
    stats.finalText = streamed;
    return stats;
  }

  if (remainingBudget > 0) {
    outputContainer.appendTokenIdx(maxIndex);
    streamNewText(outputContainer, streamed, tokenStream);
    --remainingBudget;
  }

  if (remainingBudget == 0) {
    tokenStream << std::endl;
    stats.generatedTokens = outputContainer.getTokenCnt();
    stats.finalText = streamed;
    stats.totalSeconds = prefillSeconds;
    return stats;
  }

  const int decodeBudget = remainingBudget;
  const auto maxDecodeSteps = std::min(
      decodeBudget,
      std::max(0, static_cast<int>(MaxTokenLength) -
                      static_cast<int>(inputContainerPrefill.getTokenCnt())));

  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  for (int i = 0; i < maxDecodeSteps; ++i) {
    const auto decodeStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        decodePtr, &paramsContainer, &inputContainerDecode, &cachePosition,
        &decodePtr->kv0, &decodePtr->kv1, &decodePtr->kv2, &decodePtr->kv3,
        &decodePtr->kv4, &decodePtr->kv5, &decodePtr->kv6, &decodePtr->kv7,
        &decodePtr->kv8, &decodePtr->kv9, &decodePtr->kv10, &decodePtr->kv11,
        &decodePtr->kv12, &decodePtr->kv13, &decodePtr->kv14, &decodePtr->kv15,
        &decodePtr->kv16, &decodePtr->kv17, &decodePtr->kv18, &decodePtr->kv19,
        &decodePtr->kv20, &decodePtr->kv21, &decodePtr->kv22, &decodePtr->kv23,
        &decodePtr->kv24, &decodePtr->kv25, &decodePtr->kv26, &decodePtr->kv27,
        &decodePtr->kv28, &decodePtr->kv29, &decodePtr->kv30, &decodePtr->kv31,
        &decodePtr->kv32, &decodePtr->kv33, &decodePtr->kv34, &decodePtr->kv35,
        &decodePtr->kv36, &decodePtr->kv37, &decodePtr->kv38, &decodePtr->kv39,
        &decodePtr->kv40, &decodePtr->kv41, &decodePtr->kv42, &decodePtr->kv43,
        &decodePtr->kv44, &decodePtr->kv45, &decodePtr->kv46, &decodePtr->kv47,
        &decodePtr->kv48, &decodePtr->kv49, &decodePtr->kv50, &decodePtr->kv51,
        &decodePtr->kv52, &decodePtr->kv53, &decodePtr->kv54, &decodePtr->kv55);
    const auto decodeEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> decodeTime =
        decodeEnd - decodeStart;
    decodeTimeAccumMs += decodeTime.count();
    ++decodeTokens;

    const float *decodeStartPtr = decodePtr->logits.getData();
    const float *decodeEndPtr = decodeStartPtr + MaxVocabSize;
    maxIndex = findMaxIndex(decodeStartPtr, decodeEndPtr);

    if (maxIndex == eosTokenId) {
      break;
    }

    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    streamNewText(outputContainer, streamed, tokenStream);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  if (decodeSeconds > 0.0) {
    stats.decodeTokensPerSec =
        decodeTokens > 0 ? static_cast<double>(decodeTokens) / decodeSeconds
                         : 0.0;
  }

  tokenStream << std::endl;
  stats.generatedTokens = outputContainer.getTokenCnt();
  stats.finalText = streamed;
  stats.totalSeconds = prefillSeconds + decodeSeconds;
  return stats;
}

void printStats(const GenerationResult &result) {
  llvm::errs() << "Prompt tokens: " << result.promptTokens << "\n";
  llvm::errs() << "Generated tokens: " << result.generatedTokens << "\n";
  llvm::errs() << "Prefill throughput: "
               << llvm::formatv("{0:F3}", result.prefillTokensPerSec)
               << " tokens/s\n";
  llvm::errs() << "Decode throughput: "
               << llvm::formatv("{0:F3}", result.decodeTokensPerSec)
               << " tokens/s\n";
  llvm::errs() << "Total time: " << llvm::formatv("{0:F2}", result.totalSeconds)
               << " s\n";
}

void runInteractiveSession(const std::string &systemPrompt,
                           MemRef<float, 1> &paramsContainer,
                           const std::string &vocabPath, int maxNewTokens,
                           long long eosTokenId, bool suppressStats) {
  llvm::errs()
      << "Entering interactive mode. Type :exit or :quit to end the session\n";
  std::string userInput;
  while (true) {
    std::cout << ">>> " << std::flush;
    if (!std::getline(std::cin, userInput)) {
      llvm::errs() << "Input stream ended. Leaving interactive mode\n";
      break;
    }
    if (userInput == ":exit" || userInput == ":quit") {
      llvm::errs() << "Leaving interactive mode\n";
      break;
    }
    if (userInput.empty()) {
      continue;
    }
    std::string finalPrompt = userInput;
    if (!systemPrompt.empty()) {
      finalPrompt = systemPrompt + "\n\n" + userInput;
    }
    GenerationResult result =
        runGeneration(finalPrompt, paramsContainer, vocabPath, maxNewTokens,
                      eosTokenId, std::cout);
    if (!suppressStats) {
      printStats(result);
    }
  }
}

} // namespace

static llvm::cl::opt<std::string>
    ModelPathOpt("model",
                 llvm::cl::desc("Path to the model parameter file (arg0.data)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    VocabPathOpt("vocab",
                 llvm::cl::desc("Path to the vocabulary file (vocab.txt)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptOpt("prompt", llvm::cl::desc("Prompt text passed directly"),
              llvm::cl::value_desc("text"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptFileOpt("prompt-file", llvm::cl::desc("File containing prompt text"),
                  llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    InteractiveOpt("interactive",
                   llvm::cl::desc("Start REPL-style interactive mode (combine "
                                  "with --prompt for a system prompt)"),
                   llvm::cl::init(false));

static llvm::cl::opt<unsigned>
    MaxTokensOpt("max-tokens",
                 llvm::cl::desc("Maximum number of tokens to generate "
                                "(including the first decoded token)"),
                 llvm::cl::init(256));

static llvm::cl::opt<double> TemperatureOpt(
    "temperature",
    llvm::cl::desc(
        "Sampling temperature (currently only greedy decoding is supported)"),
    llvm::cl::init(0.0));

static llvm::cl::opt<int>
    TopKOpt("top-k",
            llvm::cl::desc(
                "Top-k sampling (currently only greedy decoding is supported)"),
            llvm::cl::init(1));

static llvm::cl::opt<double>
    TopPOpt("top-p",
            llvm::cl::desc(
                "Top-p sampling (currently only greedy decoding is supported)"),
            llvm::cl::init(1.0));

static llvm::cl::opt<long long>
    EosIdOpt("eos-id", llvm::cl::desc("ID of the end-of-sequence token"),
             llvm::cl::init(DefaultEosToken));

static llvm::cl::opt<bool> SuppressStatsOpt(
    "no-stats",
    llvm::cl::desc("Output text only and hide performance statistics"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "buddy DeepSeek R1 CLI\n");

  if (!PromptOpt.empty() && !PromptFileOpt.empty()) {
    llvm::errs() << "Cannot use --prompt and --prompt-file at the same time\n";
    return 1;
  }

  std::string prompt;
  try {
    if (!PromptOpt.empty()) {
      prompt = PromptOpt;
    } else if (!PromptFileOpt.empty()) {
      prompt = readPromptFromFile(PromptFileOpt);
    } else if (!InteractiveOpt) {
      prompt = readPromptFromStdin();
    }
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return 1;
  }

  if (!InteractiveOpt && prompt.empty()) {
    llvm::errs() << "Prompt cannot be empty\n";
    return 1;
  }

  std::string modelPath =
      ModelPathOpt.empty() ? getDefaultModelPath() : ModelPathOpt;
  std::string vocabPath =
      VocabPathOpt.empty() ? getDefaultVocabPath() : VocabPathOpt;

  if (!fs::exists(modelPath)) {
    llvm::errs() << "Model parameter file not found: " << modelPath << "\n";
    return 1;
  }
  if (!fs::exists(vocabPath)) {
    llvm::errs() << "Vocabulary file not found: " << vocabPath << "\n";
    return 1;
  }

  if (TemperatureOpt != 0.0 || TopKOpt != 1 || TopPOpt != 1.0) {
    llvm::errs() << "Only greedy decoding is implemented; "
                    "temperature/top-k/top-p arguments are ignored for now\n";
  }

  const unsigned maxNewTokens =
      std::min(MaxTokensOpt.getValue(), static_cast<unsigned>(MaxTokenLength));

  MemRef<float, 1> paramsContainer({ParamsSize});
  try {
    loadParameters(modelPath, paramsContainer);
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return 1;
  }

  try {
    if (InteractiveOpt) {
      runInteractiveSession(prompt, paramsContainer, vocabPath,
                            static_cast<int>(maxNewTokens), EosIdOpt,
                            SuppressStatsOpt);
    } else {
      GenerationResult result =
          runGeneration(prompt, paramsContainer, vocabPath,
                        static_cast<int>(maxNewTokens), EosIdOpt, std::cout);
      if (!SuppressStatsOpt) {
        printStats(result);
      }
    }
  } catch (const std::exception &ex) {
    llvm::errs() << "Inference failed: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
