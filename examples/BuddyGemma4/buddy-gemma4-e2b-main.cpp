//===- buddy-gemma4-e2b-main.cpp ------------------------------------------===//
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

#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace buddy;

double total_time = 0;
constexpr size_t ParamsSize = 4628569765;
constexpr size_t MaxVocabSize = 262144;
constexpr size_t MaxTokenLength = 512;
constexpr size_t SlidingCacheLen = 512;
constexpr size_t SlidingHeadDim = 256;
constexpr size_t FullCacheLen = 512;
constexpr size_t FullHeadDim = 512;
constexpr size_t KVHeadNum = 1;

extern "C" double _mlir_ciface_rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif
}

// Gemma4 uses hybrid attention (sliding window + full) with 3 groups of 5
// layers. Each group: 4 sliding layers (K,V,cumlen each) + 1 full layer
// (cumlen,K,V). The result container layout must match the MLIR function return
// order.
struct MemRefContainer {
  // Group 0 (sliding layers 0-3 + full layer 0)
  MemRef<float, 4> sk0k, sk0v;
  MemRef<long long, 1> cl0;
  MemRef<float, 4> sk1k, sk1v;
  MemRef<long long, 1> cl1;
  MemRef<float, 4> sk2k, sk2v;
  MemRef<long long, 1> cl2;
  MemRef<float, 4> sk3k, sk3v;
  MemRef<long long, 1> cl3;
  MemRef<long long, 1> fcl0;
  MemRef<float, 4> fk0k, fk0v;
  // Group 1 (sliding layers 4-7 + full layer 1)
  MemRef<float, 4> sk4k, sk4v;
  MemRef<long long, 1> cl4;
  MemRef<float, 4> sk5k, sk5v;
  MemRef<long long, 1> cl5;
  MemRef<float, 4> sk6k, sk6v;
  MemRef<long long, 1> cl6;
  MemRef<float, 4> sk7k, sk7v;
  MemRef<long long, 1> cl7;
  MemRef<long long, 1> fcl1;
  MemRef<float, 4> fk1k, fk1v;
  // Group 2 (sliding layers 8-11 + full layer 2)
  MemRef<float, 4> sk8k, sk8v;
  MemRef<long long, 1> cl8;
  MemRef<float, 4> sk9k, sk9v;
  MemRef<long long, 1> cl9;
  MemRef<float, 4> sk10k, sk10v;
  MemRef<long long, 1> cl10;
  MemRef<float, 4> sk11k, sk11v;
  MemRef<long long, 1> cl11;
  MemRef<long long, 1> fcl2;
  MemRef<float, 4> fk2k, fk2v;
  // Logits
  MemRef<float, 3> logits;
};

extern "C" void _mlir_ciface_forward_prefill(
    MemRefContainer *result, MemRef<float, 1> *params,
    Text<size_t, 2> *input_ids, MemRef<long long, 1> *cl0,
    MemRef<float, 4> *sk0k, MemRef<float, 4> *sk0v, MemRef<long long, 1> *cl1,
    MemRef<float, 4> *sk1k, MemRef<float, 4> *sk1v, MemRef<long long, 1> *cl2,
    MemRef<float, 4> *sk2k, MemRef<float, 4> *sk2v, MemRef<long long, 1> *cl3,
    MemRef<float, 4> *sk3k, MemRef<float, 4> *sk3v, MemRef<long long, 1> *fcl0,
    MemRef<float, 4> *fk0k, MemRef<float, 4> *fk0v, MemRef<long long, 1> *cl4,
    MemRef<float, 4> *sk4k, MemRef<float, 4> *sk4v, MemRef<long long, 1> *cl5,
    MemRef<float, 4> *sk5k, MemRef<float, 4> *sk5v, MemRef<long long, 1> *cl6,
    MemRef<float, 4> *sk6k, MemRef<float, 4> *sk6v, MemRef<long long, 1> *cl7,
    MemRef<float, 4> *sk7k, MemRef<float, 4> *sk7v, MemRef<long long, 1> *fcl1,
    MemRef<float, 4> *fk1k, MemRef<float, 4> *fk1v, MemRef<long long, 1> *cl8,
    MemRef<float, 4> *sk8k, MemRef<float, 4> *sk8v, MemRef<long long, 1> *cl9,
    MemRef<float, 4> *sk9k, MemRef<float, 4> *sk9v, MemRef<long long, 1> *cl10,
    MemRef<float, 4> *sk10k, MemRef<float, 4> *sk10v,
    MemRef<long long, 1> *cl11, MemRef<float, 4> *sk11k,
    MemRef<float, 4> *sk11v, MemRef<long long, 1> *fcl2, MemRef<float, 4> *fk2k,
    MemRef<float, 4> *fk2v);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<float, 1> *params,
    MemRef<long long, 2> *input_ids, MemRef<long long, 1> *cache_position,
    MemRef<float, 4> *sk0k, MemRef<float, 4> *sk0v, MemRef<long long, 1> *cl0,
    MemRef<float, 4> *sk1k, MemRef<float, 4> *sk1v, MemRef<long long, 1> *cl1,
    MemRef<float, 4> *sk2k, MemRef<float, 4> *sk2v, MemRef<long long, 1> *cl2,
    MemRef<float, 4> *sk3k, MemRef<float, 4> *sk3v, MemRef<long long, 1> *cl3,
    MemRef<float, 4> *fk0k, MemRef<float, 4> *fk0v, MemRef<long long, 1> *fcl0,
    MemRef<float, 4> *sk4k, MemRef<float, 4> *sk4v, MemRef<long long, 1> *cl4,
    MemRef<float, 4> *sk5k, MemRef<float, 4> *sk5v, MemRef<long long, 1> *cl5,
    MemRef<float, 4> *sk6k, MemRef<float, 4> *sk6v, MemRef<long long, 1> *cl6,
    MemRef<float, 4> *sk7k, MemRef<float, 4> *sk7v, MemRef<long long, 1> *cl7,
    MemRef<float, 4> *fk1k, MemRef<float, 4> *fk1v, MemRef<long long, 1> *fcl1,
    MemRef<float, 4> *sk8k, MemRef<float, 4> *sk8v, MemRef<long long, 1> *cl8,
    MemRef<float, 4> *sk9k, MemRef<float, 4> *sk9v, MemRef<long long, 1> *cl9,
    MemRef<float, 4> *sk10k, MemRef<float, 4> *sk10v,
    MemRef<long long, 1> *cl10, MemRef<float, 4> *sk11k,
    MemRef<float, 4> *sk11v, MemRef<long long, 1> *cl11, MemRef<float, 4> *fk2k,
    MemRef<float, 4> *fk2v);

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

void printIterInfo(size_t iterIdx, std::string str, double time) {
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | "
            << "Time: " << time << "s" << std::endl;
}

void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer) {
  printLogLabel();
  std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
            << std::endl;
  const auto start = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeGemma4(vocabFile, MaxTokenLength);
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = end - start;
  printLogLabel();
  std::cout << "Tokenize time: " << elapsed.count() << "ms" << std::endl;
}

void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  printLogLabel();
  std::cout << "Loading params..." << std::endl;
  printLogLabel();
  std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
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

int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

void copyKV4(MemRef<float, 4> &dst, MemRef<float, 4> &src) {
  std::memcpy(dst.getData(), src.getData(), sizeof(float) * dst.getSize());
}

// Copy KV caches from prefill result to decode container, truncated to
// actual token count (like BuddyQwen3's copy_kv_by_cache_position_block).
void copyKVFromPrefill(MemRefContainer &prefill, MemRefContainer &decode,
                       int cachePos) {
  auto copyOne = [&](MemRef<float, 4> &dst, MemRef<float, 4> &src) {
    int headDim = dst.getSizes()[3];
    int copyLen = std::min(cachePos, (int)dst.getSizes()[2]);
    for (int h = 0; h < (int)KVHeadNum; h++) {
      std::memcpy(dst.getData() + h * dst.getSizes()[2] * headDim,
                  src.getData() + h * src.getSizes()[2] * headDim,
                  sizeof(float) * copyLen * headDim);
    }
  };
  // Sliding KV
  copyOne(decode.sk0k, prefill.sk0k);
  copyOne(decode.sk0v, prefill.sk0v);
  copyOne(decode.sk1k, prefill.sk1k);
  copyOne(decode.sk1v, prefill.sk1v);
  copyOne(decode.sk2k, prefill.sk2k);
  copyOne(decode.sk2v, prefill.sk2v);
  copyOne(decode.sk3k, prefill.sk3k);
  copyOne(decode.sk3v, prefill.sk3v);
  copyOne(decode.sk4k, prefill.sk4k);
  copyOne(decode.sk4v, prefill.sk4v);
  copyOne(decode.sk5k, prefill.sk5k);
  copyOne(decode.sk5v, prefill.sk5v);
  copyOne(decode.sk6k, prefill.sk6k);
  copyOne(decode.sk6v, prefill.sk6v);
  copyOne(decode.sk7k, prefill.sk7k);
  copyOne(decode.sk7v, prefill.sk7v);
  copyOne(decode.sk8k, prefill.sk8k);
  copyOne(decode.sk8v, prefill.sk8v);
  copyOne(decode.sk9k, prefill.sk9k);
  copyOne(decode.sk9v, prefill.sk9v);
  copyOne(decode.sk10k, prefill.sk10k);
  copyOne(decode.sk10v, prefill.sk10v);
  copyOne(decode.sk11k, prefill.sk11k);
  copyOne(decode.sk11v, prefill.sk11v);
  // Full KV
  copyOne(decode.fk0k, prefill.fk0k);
  copyOne(decode.fk0v, prefill.fk0v);
  copyOne(decode.fk1k, prefill.fk1k);
  copyOne(decode.fk1v, prefill.fk1v);
  copyOne(decode.fk2k, prefill.fk2k);
  copyOne(decode.fk2v, prefill.fk2v);
  // Cumlens
  decode.cl0.getData()[0] = cachePos;
  decode.cl1.getData()[0] = cachePos;
  decode.cl2.getData()[0] = cachePos;
  decode.cl3.getData()[0] = cachePos;
  decode.cl4.getData()[0] = cachePos;
  decode.cl5.getData()[0] = cachePos;
  decode.cl6.getData()[0] = cachePos;
  decode.cl7.getData()[0] = cachePos;
  decode.cl8.getData()[0] = cachePos;
  decode.cl9.getData()[0] = cachePos;
  decode.cl10.getData()[0] = cachePos;
  decode.cl11.getData()[0] = cachePos;
  decode.fcl0.getData()[0] = cachePos;
  decode.fcl1.getData()[0] = cachePos;
  decode.fcl2.getData()[0] = cachePos;
}

// Helper to create a sliding KV MemRef
#define SLIDING_KV {1, KVHeadNum, SlidingCacheLen, SlidingHeadDim}
#define FULL_KV {1, KVHeadNum, FullCacheLen, FullHeadDim}

// -----------------------------------------------------------------------------
// Gemma4-E2B Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title = "Gemma4-E2B Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string exampleDir = GEMMA4_E2B_EXAMPLE_PATH;
  std::string buildDir = GEMMA4_E2B_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = exampleDir + "vocab.txt";
  const std::string paramsDir = buildDir + "arg0_e2b.data";

  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logitsPrefill({1, MaxTokenLength, MaxVocabSize});

  // KV cache buffers for prefill input (zero-initialized)
  MemRef<float, 4> sk0k(SLIDING_KV, 0.f), sk0v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk1k(SLIDING_KV, 0.f), sk1v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk2k(SLIDING_KV, 0.f), sk2v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk3k(SLIDING_KV, 0.f), sk3v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk4k(SLIDING_KV, 0.f), sk4v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk5k(SLIDING_KV, 0.f), sk5v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk6k(SLIDING_KV, 0.f), sk6v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk7k(SLIDING_KV, 0.f), sk7v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk8k(SLIDING_KV, 0.f), sk8v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk9k(SLIDING_KV, 0.f), sk9v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk10k(SLIDING_KV, 0.f), sk10v(SLIDING_KV, 0.f);
  MemRef<float, 4> sk11k(SLIDING_KV, 0.f), sk11v(SLIDING_KV, 0.f);
  MemRef<float, 4> fk0k(FULL_KV, 0.f), fk0v(FULL_KV, 0.f);
  MemRef<float, 4> fk1k(FULL_KV, 0.f), fk1v(FULL_KV, 0.f);
  MemRef<float, 4> fk2k(FULL_KV, 0.f), fk2v(FULL_KV, 0.f);
  MemRef<long long, 1> cl0({1}, 0LL), cl1({1}, 0LL), cl2({1}, 0LL),
      cl3({1}, 0LL), cl4({1}, 0LL), cl5({1}, 0LL), cl6({1}, 0LL), cl7({1}, 0LL),
      cl8({1}, 0LL), cl9({1}, 0LL), cl10({1}, 0LL), cl11({1}, 0LL),
      fcl0({1}, 0LL), fcl1({1}, 0LL), fcl2({1}, 0LL);

  // Prefill result container (MLIR writes return values here via sret)
  MemRefContainer prefillResultContainer = {
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      logitsPrefill};

  /// Fill data into containers
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  // ---- Prefill ----
  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(
      &prefillResultContainer, &ParamsContainer, &inputContainerPrefill, &cl0,
      &sk0k, &sk0v, &cl1, &sk1k, &sk1v, &cl2, &sk2k, &sk2v, &cl3, &sk3k, &sk3v,
      &fcl0, &fk0k, &fk0v, &cl4, &sk4k, &sk4v, &cl5, &sk5k, &sk5v, &cl6, &sk6k,
      &sk6v, &cl7, &sk7k, &sk7v, &fcl1, &fk1k, &fk1v, &cl8, &sk8k, &sk8v, &cl9,
      &sk9k, &sk9v, &cl10, &sk10k, &sk10v, &cl11, &sk11k, &sk11v, &fcl2, &fk2k,
      &fk2v);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const float *startPtr =
      prefillResultContainer.logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);

  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  // ---- Copy KV from prefill to decode ----
  MemRef<float, 3> logitsDecode({1, 1, MaxVocabSize});
  MemRefContainer decodeResultContainer = {
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {SLIDING_KV, 0.f}, {SLIDING_KV, 0.f}, {{1}, 0LL},
      {{1}, 0LL},        {FULL_KV, 0.f},    {FULL_KV, 0.f},
      logitsDecode};
  MemRefContainer *ptrDecodeResultContainer = &decodeResultContainer;

  copyKVFromPrefill(prefillResultContainer, decodeResultContainer,
                    inputContainerPrefill.getTokenCnt() + 1);

  // ---- Decode loop ----
  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt() + 1;
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  for (int i = 1; i <= generateLen; i++) {
    const auto decStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        ptrDecodeResultContainer, &ParamsContainer, &inputContainerDecode,
        &cachePosition, &ptrDecodeResultContainer->sk0k,
        &ptrDecodeResultContainer->sk0v, &ptrDecodeResultContainer->cl0,
        &ptrDecodeResultContainer->sk1k, &ptrDecodeResultContainer->sk1v,
        &ptrDecodeResultContainer->cl1, &ptrDecodeResultContainer->sk2k,
        &ptrDecodeResultContainer->sk2v, &ptrDecodeResultContainer->cl2,
        &ptrDecodeResultContainer->sk3k, &ptrDecodeResultContainer->sk3v,
        &ptrDecodeResultContainer->cl3, &ptrDecodeResultContainer->fk0k,
        &ptrDecodeResultContainer->fk0v, &ptrDecodeResultContainer->fcl0,
        &ptrDecodeResultContainer->sk4k, &ptrDecodeResultContainer->sk4v,
        &ptrDecodeResultContainer->cl4, &ptrDecodeResultContainer->sk5k,
        &ptrDecodeResultContainer->sk5v, &ptrDecodeResultContainer->cl5,
        &ptrDecodeResultContainer->sk6k, &ptrDecodeResultContainer->sk6v,
        &ptrDecodeResultContainer->cl6, &ptrDecodeResultContainer->sk7k,
        &ptrDecodeResultContainer->sk7v, &ptrDecodeResultContainer->cl7,
        &ptrDecodeResultContainer->fk1k, &ptrDecodeResultContainer->fk1v,
        &ptrDecodeResultContainer->fcl1, &ptrDecodeResultContainer->sk8k,
        &ptrDecodeResultContainer->sk8v, &ptrDecodeResultContainer->cl8,
        &ptrDecodeResultContainer->sk9k, &ptrDecodeResultContainer->sk9v,
        &ptrDecodeResultContainer->cl9, &ptrDecodeResultContainer->sk10k,
        &ptrDecodeResultContainer->sk10v, &ptrDecodeResultContainer->cl10,
        &ptrDecodeResultContainer->sk11k, &ptrDecodeResultContainer->sk11v,
        &ptrDecodeResultContainer->cl11, &ptrDecodeResultContainer->fk2k,
        &ptrDecodeResultContainer->fk2v);
    const auto decEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> decTime = decEnd - decStart;
    decodeTimeAccumMs += decTime.count();
    decodeTokens += 1;

    const float *logitStart = ptrDecodeResultContainer->logits.getData();
    const float *logitEnd = logitStart + MaxVocabSize;
    maxIndex = findMaxIndex(logitStart, logitEnd);
    tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(i, tok, decTime.count() / 1000);

    if (maxIndex == 1) {
      break;
    }
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  const double decodeTokensPerSec =
      decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                          : 0.0;

  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertGemma4()
            << std::endl;

  return 0;
}
