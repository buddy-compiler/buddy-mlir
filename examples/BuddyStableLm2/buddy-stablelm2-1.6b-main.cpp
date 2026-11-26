//===- buddy-stablelm2-1.6b-main.cpp
//-----------------------------------------===//
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

#include <array>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace buddy;
double total_time = 0;
constexpr size_t ParamsSize = 1644515336;
constexpr size_t MaxVocabSize = 100352;
// Embedding rows 100289..100351 are padding; clamp argmax to real vocab
constexpr size_t RealVocabSize = 100289;

constexpr size_t MaxTokenLength = 1024;

constexpr size_t HiddenSize = 64;
constexpr size_t HeadNum = 32;

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
#endif // _WIN32
}

struct PrefillReturns {
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
  MemRef<float, 3> logits;
};

struct DecodeReturns {
  MemRef<long long, 1> cache_position_out;
  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<long long, 1> ret_dummy0;
  MemRef<float, 4> kv2;
  MemRef<float, 4> kv3;
  MemRef<long long, 1> ret_dummy1;
  MemRef<float, 4> kv4;
  MemRef<float, 4> kv5;
  MemRef<long long, 1> ret_dummy2;
  MemRef<float, 4> kv6;
  MemRef<float, 4> kv7;
  MemRef<long long, 1> ret_dummy3;
  MemRef<float, 4> kv8;
  MemRef<float, 4> kv9;
  MemRef<long long, 1> ret_dummy4;
  MemRef<float, 4> kv10;
  MemRef<float, 4> kv11;
  MemRef<long long, 1> ret_dummy5;
  MemRef<float, 4> kv12;
  MemRef<float, 4> kv13;
  MemRef<long long, 1> ret_dummy6;
  MemRef<float, 4> kv14;
  MemRef<float, 4> kv15;
  MemRef<long long, 1> ret_dummy7;
  MemRef<float, 4> kv16;
  MemRef<float, 4> kv17;
  MemRef<long long, 1> ret_dummy8;
  MemRef<float, 4> kv18;
  MemRef<float, 4> kv19;
  MemRef<long long, 1> ret_dummy9;
  MemRef<float, 4> kv20;
  MemRef<float, 4> kv21;
  MemRef<long long, 1> ret_dummy10;
  MemRef<float, 4> kv22;
  MemRef<float, 4> kv23;
  MemRef<long long, 1> ret_dummy11;
  MemRef<float, 4> kv24;
  MemRef<float, 4> kv25;
  MemRef<long long, 1> ret_dummy12;
  MemRef<float, 4> kv26;
  MemRef<float, 4> kv27;
  MemRef<long long, 1> ret_dummy13;
  MemRef<float, 4> kv28;
  MemRef<float, 4> kv29;
  MemRef<long long, 1> ret_dummy14;
  MemRef<float, 4> kv30;
  MemRef<float, 4> kv31;
  MemRef<long long, 1> ret_dummy15;
  MemRef<float, 4> kv32;
  MemRef<float, 4> kv33;
  MemRef<long long, 1> ret_dummy16;
  MemRef<float, 4> kv34;
  MemRef<float, 4> kv35;
  MemRef<long long, 1> ret_dummy17;
  MemRef<float, 4> kv36;
  MemRef<float, 4> kv37;
  MemRef<long long, 1> ret_dummy18;
  MemRef<float, 4> kv38;
  MemRef<float, 4> kv39;
  MemRef<long long, 1> ret_dummy19;
  MemRef<float, 4> kv40;
  MemRef<float, 4> kv41;
  MemRef<long long, 1> ret_dummy20;
  MemRef<float, 4> kv42;
  MemRef<float, 4> kv43;
  MemRef<long long, 1> ret_dummy21;
  MemRef<float, 4> kv44;
  MemRef<float, 4> kv45;
  MemRef<long long, 1> ret_dummy22;
  MemRef<float, 4> kv46;
  MemRef<float, 4> kv47;
  MemRef<float, 3> logits;
};

/// Declare StableLM2 forward functions.
extern "C" void _mlir_ciface_forward_prefill(PrefillReturns *result,
                                             MemRef<float, 1> *arg0,
                                             Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    DecodeReturns *result, MemRef<float, 1> *arg0, MemRef<long long, 2> *arg1,
    MemRef<long long, 1> *arg2,
    // Group 1
    MemRef<float, 4> *kv0, MemRef<float, 4> *kv1, MemRef<long long, 1> *dummy0,
    // Group 2
    MemRef<float, 4> *kv2, MemRef<float, 4> *kv3, MemRef<long long, 1> *dummy1,
    // Group 3
    MemRef<float, 4> *kv4, MemRef<float, 4> *kv5, MemRef<long long, 1> *dummy2,
    // Group 4
    MemRef<float, 4> *kv6, MemRef<float, 4> *kv7, MemRef<long long, 1> *dummy3,
    // Group 5
    MemRef<float, 4> *kv8, MemRef<float, 4> *kv9, MemRef<long long, 1> *dummy4,
    // Group 6
    MemRef<float, 4> *kv10, MemRef<float, 4> *kv11,
    MemRef<long long, 1> *dummy5,
    // Group 7
    MemRef<float, 4> *kv12, MemRef<float, 4> *kv13,
    MemRef<long long, 1> *dummy6,
    // Group 8
    MemRef<float, 4> *kv14, MemRef<float, 4> *kv15,
    MemRef<long long, 1> *dummy7,
    // Group 9
    MemRef<float, 4> *kv16, MemRef<float, 4> *kv17,
    MemRef<long long, 1> *dummy8,
    // Group 10
    MemRef<float, 4> *kv18, MemRef<float, 4> *kv19,
    MemRef<long long, 1> *dummy9,
    // Group 11
    MemRef<float, 4> *kv20, MemRef<float, 4> *kv21,
    MemRef<long long, 1> *dummy10,
    // Group 12
    MemRef<float, 4> *kv22, MemRef<float, 4> *kv23,
    MemRef<long long, 1> *dummy11,
    // Group 13
    MemRef<float, 4> *kv24, MemRef<float, 4> *kv25,
    MemRef<long long, 1> *dummy12,
    // Group 14
    MemRef<float, 4> *kv26, MemRef<float, 4> *kv27,
    MemRef<long long, 1> *dummy13,
    // Group 15
    MemRef<float, 4> *kv28, MemRef<float, 4> *kv29,
    MemRef<long long, 1> *dummy14,
    // Group 16
    MemRef<float, 4> *kv30, MemRef<float, 4> *kv31,
    MemRef<long long, 1> *dummy15,
    // Group 17
    MemRef<float, 4> *kv32, MemRef<float, 4> *kv33,
    MemRef<long long, 1> *dummy16,
    // Group 18
    MemRef<float, 4> *kv34, MemRef<float, 4> *kv35,
    MemRef<long long, 1> *dummy17,
    // Group 19
    MemRef<float, 4> *kv36, MemRef<float, 4> *kv37,
    MemRef<long long, 1> *dummy18,
    // Group 20
    MemRef<float, 4> *kv38, MemRef<float, 4> *kv39,
    MemRef<long long, 1> *dummy19,
    // Group 21
    MemRef<float, 4> *kv40, MemRef<float, 4> *kv41,
    MemRef<long long, 1> *dummy20,
    // Group 22
    MemRef<float, 4> *kv42, MemRef<float, 4> *kv43,
    MemRef<long long, 1> *dummy21,
    // Group 23
    MemRef<float, 4> *kv44, MemRef<float, 4> *kv45,
    MemRef<long long, 1> *dummy22,
    // Group 24 (no dummy)
    MemRef<float, 4> *kv46, MemRef<float, 4> *kv47);
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
  inputContainer.tokenizeStableLm2(vocabFile, MaxTokenLength);
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

int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

// -----------------------------------------------------------------------------
// StableLM2-1.6B Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title =
      "StableLM2-1.6B Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string stablelm2_Dir = STABLELM2_1_6B_EXAMPLE_PATH;
  std::string stablelm2_BuildDir = STABLELM2_1_6B_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = stablelm2_Dir + "vocab.txt";
  const std::string paramsDir = stablelm2_BuildDir + "arg0_stablelm2_1_6b.data";

  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize containers
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  // Define logits memory
  MemRef<float, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});
  MemRef<float, 3> logits_decode({1, 1, MaxVocabSize});

  // ========== Initialize Prefill Returns Container ==========
  // All KV fields are constructed directly with aggregate initialization,
  // values set to 0
  PrefillReturns prefillRet = {
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      logits_prefill};

  // ========== Initialize Decode Returns Container ==========
  DecodeReturns decodeRet = {
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      logits_decode};

  /// Load vocab and parameters
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  /// Execute Prefill
  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(&prefillRet, &ParamsContainer,
                               &inputContainerPrefill);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const float *startPtr =
      prefillRet.logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + RealVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  // Copy KV cache
  size_t copy_len =
      std::min(static_cast<size_t>(inputContainerPrefill.getTokenCnt()),
               static_cast<size_t>(MaxTokenLength));
  auto copyOne = [&](MemRef<float, 4> &dst, MemRef<float, 4> &src) {
    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes = static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);
      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dst_ptr, src_ptr, bytes);
    }
  };

  // Copy 48 KV pairs one by one
  copyOne(decodeRet.kv0, prefillRet.kv0);
  copyOne(decodeRet.kv1, prefillRet.kv1);
  copyOne(decodeRet.kv2, prefillRet.kv2);
  copyOne(decodeRet.kv3, prefillRet.kv3);
  copyOne(decodeRet.kv4, prefillRet.kv4);
  copyOne(decodeRet.kv5, prefillRet.kv5);
  copyOne(decodeRet.kv6, prefillRet.kv6);
  copyOne(decodeRet.kv7, prefillRet.kv7);
  copyOne(decodeRet.kv8, prefillRet.kv8);
  copyOne(decodeRet.kv9, prefillRet.kv9);
  copyOne(decodeRet.kv10, prefillRet.kv10);
  copyOne(decodeRet.kv11, prefillRet.kv11);
  copyOne(decodeRet.kv12, prefillRet.kv12);
  copyOne(decodeRet.kv13, prefillRet.kv13);
  copyOne(decodeRet.kv14, prefillRet.kv14);
  copyOne(decodeRet.kv15, prefillRet.kv15);
  copyOne(decodeRet.kv16, prefillRet.kv16);
  copyOne(decodeRet.kv17, prefillRet.kv17);
  copyOne(decodeRet.kv18, prefillRet.kv18);
  copyOne(decodeRet.kv19, prefillRet.kv19);
  copyOne(decodeRet.kv20, prefillRet.kv20);
  copyOne(decodeRet.kv21, prefillRet.kv21);
  copyOne(decodeRet.kv22, prefillRet.kv22);
  copyOne(decodeRet.kv23, prefillRet.kv23);
  copyOne(decodeRet.kv24, prefillRet.kv24);
  copyOne(decodeRet.kv25, prefillRet.kv25);
  copyOne(decodeRet.kv26, prefillRet.kv26);
  copyOne(decodeRet.kv27, prefillRet.kv27);
  copyOne(decodeRet.kv28, prefillRet.kv28);
  copyOne(decodeRet.kv29, prefillRet.kv29);
  copyOne(decodeRet.kv30, prefillRet.kv30);
  copyOne(decodeRet.kv31, prefillRet.kv31);
  copyOne(decodeRet.kv32, prefillRet.kv32);
  copyOne(decodeRet.kv33, prefillRet.kv33);
  copyOne(decodeRet.kv34, prefillRet.kv34);
  copyOne(decodeRet.kv35, prefillRet.kv35);
  copyOne(decodeRet.kv36, prefillRet.kv36);
  copyOne(decodeRet.kv37, prefillRet.kv37);
  copyOne(decodeRet.kv38, prefillRet.kv38);
  copyOne(decodeRet.kv39, prefillRet.kv39);
  copyOne(decodeRet.kv40, prefillRet.kv40);
  copyOne(decodeRet.kv41, prefillRet.kv41);
  copyOne(decodeRet.kv42, prefillRet.kv42);
  copyOne(decodeRet.kv43, prefillRet.kv43);
  copyOne(decodeRet.kv44, prefillRet.kv44);
  copyOne(decodeRet.kv45, prefillRet.kv45);
  copyOne(decodeRet.kv46, prefillRet.kv46);
  copyOne(decodeRet.kv47, prefillRet.kv47);

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  int generateLen = 29; // 30 tokens total incl. prefill; re-prefill mode grows
                        // ~0.8 GiB/step, fits 31 GiB RAM
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  // NOTE: the single-token decode graph mis-masks pad positions at runtime
  // for this model family (verified by poisoning: output changes when pad KV
  // changes, layer 0 K/V itself is exact). Until that compiler issue is
  // fixed, each step re-runs the (proven-correct) prefill graph over the
  // extended sequence. Cost: O(step) full forward per token.
  for (int i = 1; i <= generateLen; i++) {
    const auto loopStart = std::chrono::high_resolution_clock::now();

    inputContainerPrefill.appendTokenIdx(static_cast<size_t>(maxIndex));
    _mlir_ciface_forward_prefill(&prefillRet, &ParamsContainer,
                                 &inputContainerPrefill);

    const auto loopEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> loopTime =
        loopEnd - loopStart;
    decodeTimeAccumMs += loopTime.count();
    decodeTokens += 1;

    int lastIdx = static_cast<int>(inputContainerPrefill.getTokenCnt()) - 1;
    const float *stepStart = prefillRet.logits.getData() +
                             static_cast<size_t>(lastIdx) * MaxVocabSize;
    const float *stepEnd = stepStart + RealVocabSize;

    maxIndex = findMaxIndex(stepStart, stepEnd);
    tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(i, tok, loopTime.count() / 1000);

    outputContainer.appendTokenIdx(maxIndex);
    if (maxIndex == 100257) {
      break; // EOS (StableLM2: eos == pad == 100257)
    }
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
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertStableLm2()
            << std::endl;

  return 0;
}
