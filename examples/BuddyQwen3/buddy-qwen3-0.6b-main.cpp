//===- buddy-qwen3-0.6b-main.cpp -----------------------------------------===//
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
constexpr size_t ParamsSize = 596049984;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 8;

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
  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;
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
  MemRef<long long, 1> ret_dummy23;

  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<long long, 1> ret_dummy24;

  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<long long, 1> ret_dummy25;

  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<long long, 1> ret_dummy26;

  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;

  MemRef<float, 3> logits;
};

// Pointer array for traversing KV fields (externally defined, does not intrude
// into struct)
using KVPtrArray = std::array<MemRef<float, 4> *, 56>;

KVPtrArray buildPrefillKVPtrs(PrefillReturns &ret) {
  return {&ret.kv0,  &ret.kv1,  &ret.kv2,  &ret.kv3,  &ret.kv4,  &ret.kv5,
          &ret.kv6,  &ret.kv7,  &ret.kv8,  &ret.kv9,  &ret.kv10, &ret.kv11,
          &ret.kv12, &ret.kv13, &ret.kv14, &ret.kv15, &ret.kv16, &ret.kv17,
          &ret.kv18, &ret.kv19, &ret.kv20, &ret.kv21, &ret.kv22, &ret.kv23,
          &ret.kv24, &ret.kv25, &ret.kv26, &ret.kv27, &ret.kv28, &ret.kv29,
          &ret.kv30, &ret.kv31, &ret.kv32, &ret.kv33, &ret.kv34, &ret.kv35,
          &ret.kv36, &ret.kv37, &ret.kv38, &ret.kv39, &ret.kv40, &ret.kv41,
          &ret.kv42, &ret.kv43, &ret.kv44, &ret.kv45, &ret.kv46, &ret.kv47,
          &ret.kv48, &ret.kv49, &ret.kv50, &ret.kv51, &ret.kv52, &ret.kv53,
          &ret.kv54, &ret.kv55};
}

KVPtrArray buildDecodeKVPtrs(DecodeReturns &ret) {
  return {&ret.kv0,  &ret.kv1,  &ret.kv2,  &ret.kv3,  &ret.kv4,  &ret.kv5,
          &ret.kv6,  &ret.kv7,  &ret.kv8,  &ret.kv9,  &ret.kv10, &ret.kv11,
          &ret.kv12, &ret.kv13, &ret.kv14, &ret.kv15, &ret.kv16, &ret.kv17,
          &ret.kv18, &ret.kv19, &ret.kv20, &ret.kv21, &ret.kv22, &ret.kv23,
          &ret.kv24, &ret.kv25, &ret.kv26, &ret.kv27, &ret.kv28, &ret.kv29,
          &ret.kv30, &ret.kv31, &ret.kv32, &ret.kv33, &ret.kv34, &ret.kv35,
          &ret.kv36, &ret.kv37, &ret.kv38, &ret.kv39, &ret.kv40, &ret.kv41,
          &ret.kv42, &ret.kv43, &ret.kv44, &ret.kv45, &ret.kv46, &ret.kv47,
          &ret.kv48, &ret.kv49, &ret.kv50, &ret.kv51, &ret.kv52, &ret.kv53,
          &ret.kv54, &ret.kv55};
}

/// Declare Qwen3 forward functions.
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
    // Group 24
    MemRef<float, 4> *kv46, MemRef<float, 4> *kv47,
    MemRef<long long, 1> *dummy23,
    // Group 25
    MemRef<float, 4> *kv48, MemRef<float, 4> *kv49,
    MemRef<long long, 1> *dummy24,
    // Group 26
    MemRef<float, 4> *kv50, MemRef<float, 4> *kv51,
    MemRef<long long, 1> *dummy25,
    // Group 27
    MemRef<float, 4> *kv52, MemRef<float, 4> *kv53,
    MemRef<long long, 1> *dummy26,
    // Group 28 (no dummy)
    MemRef<float, 4> *kv54, MemRef<float, 4> *kv55);

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
  inputContainer.tokenizeQwen3(vocabFile, MaxTokenLength);
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
// Qwen3-0.6B Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title = "Qwen3-0.6B Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string qwen3_0_6b_Dir = QWEN3_0_6B_EXAMPLE_PATH;
  std::string qwen3_0_6b_BuildDir = QWEN3_0_6B_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = qwen3_0_6b_Dir + "vocab.txt";
  const std::string paramsDir = qwen3_0_6b_BuildDir + "arg0_0_6b.data";

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
      // kv0 ~ kv55
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
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f),
      // logits
      logits_prefill};

  // ========== Initialize Decode Returns Container ==========
  DecodeReturns decodeRet = {

      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv0
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv1
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv2
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv3
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv4
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv5
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv6
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv7
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv8
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv9
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv10
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv11
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv12
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv13
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv14
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv15
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv16
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv17
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv18
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv19
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv20
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv21
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv22
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv23
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv24
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv25
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv26
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv27
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv28
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv29
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv30
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv31
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv32
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv33
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv34
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv35
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv36
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv37
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv38
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv39
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv40
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv41
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv42
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv43
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv44
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv45
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv46
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv47
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv48
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv49
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv50
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv51
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv52
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv53
      MemRef<long long, 1>({1}, 0LL),

      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv54
      MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0.f), // kv55

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

  // Copy KV cache
  size_t copy_len =
      std::min(static_cast<size_t>(inputContainerPrefill.getTokenCnt() + 1),
               static_cast<size_t>(MaxTokenLength));
  auto copyOne = [&](MemRef<float, 4> &dst, MemRef<float, 4> &src) {
    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes = static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);
      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dst_ptr, src_ptr, bytes);
    }
  };

  // Copy 56 KV pairs one by one
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
  copyOne(decodeRet.kv48, prefillRet.kv48);
  copyOne(decodeRet.kv49, prefillRet.kv49);
  copyOne(decodeRet.kv50, prefillRet.kv50);
  copyOne(decodeRet.kv51, prefillRet.kv51);
  copyOne(decodeRet.kv52, prefillRet.kv52);
  copyOne(decodeRet.kv53, prefillRet.kv53);
  copyOne(decodeRet.kv54, prefillRet.kv54);
  copyOne(decodeRet.kv55, prefillRet.kv55);

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt() + 1;
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  for (int i = 1; i <= generateLen; i++) {
    const auto loopStart = std::chrono::high_resolution_clock::now();

    decodeRet.ret_dummy0.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy1.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy2.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy3.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy4.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy5.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy6.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy7.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy8.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy9.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy10.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy11.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy12.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy13.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy14.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy15.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy16.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy17.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy18.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy19.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy20.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy21.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy22.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy23.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy24.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy25.getData()[0] = cachePosition.getData()[0];
    decodeRet.ret_dummy26.getData()[0] = cachePosition.getData()[0];

    _mlir_ciface_forward_decode(
        &decodeRet, &ParamsContainer, &inputContainerDecode, &cachePosition,
        // Group 1
        &decodeRet.kv0, &decodeRet.kv1, &decodeRet.ret_dummy0,
        // Group 2
        &decodeRet.kv2, &decodeRet.kv3, &decodeRet.ret_dummy1,
        // Group 3
        &decodeRet.kv4, &decodeRet.kv5, &decodeRet.ret_dummy2,
        // Group 4
        &decodeRet.kv6, &decodeRet.kv7, &decodeRet.ret_dummy3,
        // Group 5
        &decodeRet.kv8, &decodeRet.kv9, &decodeRet.ret_dummy4,
        // Group 6
        &decodeRet.kv10, &decodeRet.kv11, &decodeRet.ret_dummy5,
        // Group 7
        &decodeRet.kv12, &decodeRet.kv13, &decodeRet.ret_dummy6,
        // Group 8
        &decodeRet.kv14, &decodeRet.kv15, &decodeRet.ret_dummy7,
        // Group 9
        &decodeRet.kv16, &decodeRet.kv17, &decodeRet.ret_dummy8,
        // Group 10
        &decodeRet.kv18, &decodeRet.kv19, &decodeRet.ret_dummy9,
        // Group 11
        &decodeRet.kv20, &decodeRet.kv21, &decodeRet.ret_dummy10,
        // Group 12
        &decodeRet.kv22, &decodeRet.kv23, &decodeRet.ret_dummy11,
        // Group 13
        &decodeRet.kv24, &decodeRet.kv25, &decodeRet.ret_dummy12,
        // Group 14
        &decodeRet.kv26, &decodeRet.kv27, &decodeRet.ret_dummy13,
        // Group 15
        &decodeRet.kv28, &decodeRet.kv29, &decodeRet.ret_dummy14,
        // Group 16
        &decodeRet.kv30, &decodeRet.kv31, &decodeRet.ret_dummy15,
        // Group 17
        &decodeRet.kv32, &decodeRet.kv33, &decodeRet.ret_dummy16,
        // Group 18
        &decodeRet.kv34, &decodeRet.kv35, &decodeRet.ret_dummy17,
        // Group 19
        &decodeRet.kv36, &decodeRet.kv37, &decodeRet.ret_dummy18,
        // Group 20
        &decodeRet.kv38, &decodeRet.kv39, &decodeRet.ret_dummy19,
        // Group 21
        &decodeRet.kv40, &decodeRet.kv41, &decodeRet.ret_dummy20,
        // Group 22
        &decodeRet.kv42, &decodeRet.kv43, &decodeRet.ret_dummy21,
        // Group 23
        &decodeRet.kv44, &decodeRet.kv45, &decodeRet.ret_dummy22,
        // Group 24
        &decodeRet.kv46, &decodeRet.kv47, &decodeRet.ret_dummy23,
        // Group 25
        &decodeRet.kv48, &decodeRet.kv49, &decodeRet.ret_dummy24,
        // Group 26
        &decodeRet.kv50, &decodeRet.kv51, &decodeRet.ret_dummy25,
        // Group 27
        &decodeRet.kv52, &decodeRet.kv53, &decodeRet.ret_dummy26,
        // Group 28 (no dummy)
        &decodeRet.kv54, &decodeRet.kv55);

    const auto loopEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> loopTime =
        loopEnd - loopStart;
    decodeTimeAccumMs += loopTime.count();
    decodeTokens += 1;

    const float *logitsStart = decodeRet.logits.getData();
    const float *logitsEnd = logitsStart + MaxVocabSize;

    maxIndex = findMaxIndex(logitsStart, logitsEnd);
    tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(i, tok, loopTime.count() / 1000);

    if (maxIndex == 151643) {
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
  std::cout << "\033[33;1m[Output]\033[0m " << outputContainer.revertQwen3()
            << std::endl;

  return 0;
}
