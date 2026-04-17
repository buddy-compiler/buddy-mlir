//===- buddy-deepseek-r1-w8a32-main.cpp -----------------------------------===//
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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace buddy;
double total_time = 0;
constexpr size_t F32ParamsSize = 322338752;
constexpr size_t I8ParamsSize = 1455489024;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

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

/// Prefill returns: 56 KV caches followed by logits.
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

/// Decode returns: updated cache_position, then 27 groups of (kv, kv, dummy),
/// followed by the final two kvs and logits. Total 85 fields.
struct DecodeReturns {
  // First return value: updated cache_position (memref<1xi64>)
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

// Type alias for a pointer array to all 56 KV fields.
using KVPtrArray = std::array<MemRef<float, 4> *, 56>;

// Build KV pointer array for PrefillReturns.
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

// Build KV pointer array for DecodeReturns.
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

// ============================================================================
// MLIR function declarations.
// ============================================================================

extern "C" void _mlir_ciface_forward_prefill(PrefillReturns *result,
                                             MemRef<float, 1> *f32_params,
                                             MemRef<int8_t, 1> *i8_params,
                                             Text<size_t, 2> *input);

extern "C" void _mlir_ciface_forward_decode(
    DecodeReturns *result, MemRef<float, 1> *f32_params,
    MemRef<int8_t, 1> *i8_params, MemRef<long long, 2> *input,
    MemRef<long long, 1> *cache_pos, MemRef<float, 4> *kv0,
    MemRef<float, 4> *kv1, MemRef<long long, 1> *dummy0, MemRef<float, 4> *kv2,
    MemRef<float, 4> *kv3, MemRef<long long, 1> *dummy1, MemRef<float, 4> *kv4,
    MemRef<float, 4> *kv5, MemRef<long long, 1> *dummy2, MemRef<float, 4> *kv6,
    MemRef<float, 4> *kv7, MemRef<long long, 1> *dummy3, MemRef<float, 4> *kv8,
    MemRef<float, 4> *kv9, MemRef<long long, 1> *dummy4, MemRef<float, 4> *kv10,
    MemRef<float, 4> *kv11, MemRef<long long, 1> *dummy5,
    MemRef<float, 4> *kv12, MemRef<float, 4> *kv13,
    MemRef<long long, 1> *dummy6, MemRef<float, 4> *kv14,
    MemRef<float, 4> *kv15, MemRef<long long, 1> *dummy7,
    MemRef<float, 4> *kv16, MemRef<float, 4> *kv17,
    MemRef<long long, 1> *dummy8, MemRef<float, 4> *kv18,
    MemRef<float, 4> *kv19, MemRef<long long, 1> *dummy9,
    MemRef<float, 4> *kv20, MemRef<float, 4> *kv21,
    MemRef<long long, 1> *dummy10, MemRef<float, 4> *kv22,
    MemRef<float, 4> *kv23, MemRef<long long, 1> *dummy11,
    MemRef<float, 4> *kv24, MemRef<float, 4> *kv25,
    MemRef<long long, 1> *dummy12, MemRef<float, 4> *kv26,
    MemRef<float, 4> *kv27, MemRef<long long, 1> *dummy13,
    MemRef<float, 4> *kv28, MemRef<float, 4> *kv29,
    MemRef<long long, 1> *dummy14, MemRef<float, 4> *kv30,
    MemRef<float, 4> *kv31, MemRef<long long, 1> *dummy15,
    MemRef<float, 4> *kv32, MemRef<float, 4> *kv33,
    MemRef<long long, 1> *dummy16, MemRef<float, 4> *kv34,
    MemRef<float, 4> *kv35, MemRef<long long, 1> *dummy17,
    MemRef<float, 4> *kv36, MemRef<float, 4> *kv37,
    MemRef<long long, 1> *dummy18, MemRef<float, 4> *kv38,
    MemRef<float, 4> *kv39, MemRef<long long, 1> *dummy19,
    MemRef<float, 4> *kv40, MemRef<float, 4> *kv41,
    MemRef<long long, 1> *dummy20, MemRef<float, 4> *kv42,
    MemRef<float, 4> *kv43, MemRef<long long, 1> *dummy21,
    MemRef<float, 4> *kv44, MemRef<float, 4> *kv45,
    MemRef<long long, 1> *dummy22, MemRef<float, 4> *kv46,
    MemRef<float, 4> *kv47, MemRef<long long, 1> *dummy23,
    MemRef<float, 4> *kv48, MemRef<float, 4> *kv49,
    MemRef<long long, 1> *dummy24, MemRef<float, 4> *kv50,
    MemRef<float, 4> *kv51, MemRef<long long, 1> *dummy25,
    MemRef<float, 4> *kv52, MemRef<float, 4> *kv53,
    MemRef<long long, 1> *dummy26, MemRef<float, 4> *kv54,
    MemRef<float, 4> *kv55);

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
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

template <typename T>
void loadParameters(const std::string &paramFilePath, MemRef<T, 1> &params) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file: " +
                             paramFilePath);
  }
  printLogLabel();
  std::cout << "Loading params: " << std::filesystem::canonical(paramFilePath)
            << std::endl;
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(T) * (params.getSize()));
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

void copy_kv_by_cache_position_block(const KVPtrArray &prefillPtrs,
                                     const KVPtrArray &decodePtrs,
                                     int cache_position) {
  constexpr int num_kv = 56;
  const size_t copy_len = std::min<size_t>(static_cast<size_t>(cache_position),
                                           static_cast<size_t>(MaxTokenLength));

  for (int k = 0; k < num_kv; ++k) {
    auto &src = *prefillPtrs[k];
    auto &dst = *decodePtrs[k];

    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes_to_copy = copy_len * HiddenSize * sizeof(float);

      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 W8A32 Quantized Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title =
      "DeepSeekR1 W8A32 Quantized Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string f32ParamsDir = deepSeekR1BuildDir + "arg0-w8a32-f32.data";
  const std::string i8ParamsDir = deepSeekR1BuildDir + "arg0-w8a32-i8.data";

  std::string inputStr;
  getUserInput(inputStr);

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> F32ParamsContainer({F32ParamsSize});
  MemRef<int8_t, 1> I8ParamsContainer({I8ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  // Helper lambda to create a zero-initialized KV MemRef.
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

  // Initialize Prefill returns (aggregate initialization).
  PrefillReturns prefillRet = {
      kv0,  kv1,  kv2,  kv3,  kv4,  kv5,  kv6,           kv7,  kv8,  kv9,
      kv10, kv11, kv12, kv13, kv14, kv15, kv16,          kv17, kv18, kv19,
      kv20, kv21, kv22, kv23, kv24, kv25, kv26,          kv27, kv28, kv29,
      kv30, kv31, kv32, kv33, kv34, kv35, kv36,          kv37, kv38, kv39,
      kv40, kv41, kv42, kv43, kv44, kv45, kv46,          kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill};

  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(f32ParamsDir, F32ParamsContainer);
  loadParameters(i8ParamsDir, I8ParamsContainer);

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(&prefillRet, &F32ParamsContainer,
                               &I8ParamsContainer, &inputContainerPrefill);
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

  // Build Prefill KV pointer array for copying.
  KVPtrArray prefillPtrs = buildPrefillKVPtrs(prefillRet);

  // Initialize Decode returns.
  MemRef<float, 3> logits_decode({1, 1, MaxVocabSize});
  DecodeReturns decodeRet = {
      MemRef<long long, 1>({1}, 0LL), // cache_position_out
      kv0,
      kv1,
      MemRef<long long, 1>({1}, 0LL),
      kv2,
      kv3,
      MemRef<long long, 1>({1}, 0LL),
      kv4,
      kv5,
      MemRef<long long, 1>({1}, 0LL),
      kv6,
      kv7,
      MemRef<long long, 1>({1}, 0LL),
      kv8,
      kv9,
      MemRef<long long, 1>({1}, 0LL),
      kv10,
      kv11,
      MemRef<long long, 1>({1}, 0LL),
      kv12,
      kv13,
      MemRef<long long, 1>({1}, 0LL),
      kv14,
      kv15,
      MemRef<long long, 1>({1}, 0LL),
      kv16,
      kv17,
      MemRef<long long, 1>({1}, 0LL),
      kv18,
      kv19,
      MemRef<long long, 1>({1}, 0LL),
      kv20,
      kv21,
      MemRef<long long, 1>({1}, 0LL),
      kv22,
      kv23,
      MemRef<long long, 1>({1}, 0LL),
      kv24,
      kv25,
      MemRef<long long, 1>({1}, 0LL),
      kv26,
      kv27,
      MemRef<long long, 1>({1}, 0LL),
      kv28,
      kv29,
      MemRef<long long, 1>({1}, 0LL),
      kv30,
      kv31,
      MemRef<long long, 1>({1}, 0LL),
      kv32,
      kv33,
      MemRef<long long, 1>({1}, 0LL),
      kv34,
      kv35,
      MemRef<long long, 1>({1}, 0LL),
      kv36,
      kv37,
      MemRef<long long, 1>({1}, 0LL),
      kv38,
      kv39,
      MemRef<long long, 1>({1}, 0LL),
      kv40,
      kv41,
      MemRef<long long, 1>({1}, 0LL),
      kv42,
      kv43,
      MemRef<long long, 1>({1}, 0LL),
      kv44,
      kv45,
      MemRef<long long, 1>({1}, 0LL),
      kv46,
      kv47,
      MemRef<long long, 1>({1}, 0LL),
      kv48,
      kv49,
      MemRef<long long, 1>({1}, 0LL),
      kv50,
      kv51,
      MemRef<long long, 1>({1}, 0LL),
      kv52,
      kv53,
      MemRef<long long, 1>({1}, 0LL),
      kv54,
      kv55,
      logits_decode};

  KVPtrArray decodePtrs = buildDecodeKVPtrs(decodeRet);

  // Copy KV cache from prefill to decode.
  copy_kv_by_cache_position_block(
      prefillPtrs, decodePtrs,
      static_cast<int>(inputContainerPrefill.getTokenCnt()));

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  for (int i = 1; i <= generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    // Update dummy fields with current cache position.
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
        &decodeRet, &F32ParamsContainer, &I8ParamsContainer,
        &inputContainerDecode, &cachePosition, &decodeRet.kv0, &decodeRet.kv1,
        &decodeRet.ret_dummy0, &decodeRet.kv2, &decodeRet.kv3,
        &decodeRet.ret_dummy1, &decodeRet.kv4, &decodeRet.kv5,
        &decodeRet.ret_dummy2, &decodeRet.kv6, &decodeRet.kv7,
        &decodeRet.ret_dummy3, &decodeRet.kv8, &decodeRet.kv9,
        &decodeRet.ret_dummy4, &decodeRet.kv10, &decodeRet.kv11,
        &decodeRet.ret_dummy5, &decodeRet.kv12, &decodeRet.kv13,
        &decodeRet.ret_dummy6, &decodeRet.kv14, &decodeRet.kv15,
        &decodeRet.ret_dummy7, &decodeRet.kv16, &decodeRet.kv17,
        &decodeRet.ret_dummy8, &decodeRet.kv18, &decodeRet.kv19,
        &decodeRet.ret_dummy9, &decodeRet.kv20, &decodeRet.kv21,
        &decodeRet.ret_dummy10, &decodeRet.kv22, &decodeRet.kv23,
        &decodeRet.ret_dummy11, &decodeRet.kv24, &decodeRet.kv25,
        &decodeRet.ret_dummy12, &decodeRet.kv26, &decodeRet.kv27,
        &decodeRet.ret_dummy13, &decodeRet.kv28, &decodeRet.kv29,
        &decodeRet.ret_dummy14, &decodeRet.kv30, &decodeRet.kv31,
        &decodeRet.ret_dummy15, &decodeRet.kv32, &decodeRet.kv33,
        &decodeRet.ret_dummy16, &decodeRet.kv34, &decodeRet.kv35,
        &decodeRet.ret_dummy17, &decodeRet.kv36, &decodeRet.kv37,
        &decodeRet.ret_dummy18, &decodeRet.kv38, &decodeRet.kv39,
        &decodeRet.ret_dummy19, &decodeRet.kv40, &decodeRet.kv41,
        &decodeRet.ret_dummy20, &decodeRet.kv42, &decodeRet.kv43,
        &decodeRet.ret_dummy21, &decodeRet.kv44, &decodeRet.kv45,
        &decodeRet.ret_dummy22, &decodeRet.kv46, &decodeRet.kv47,
        &decodeRet.ret_dummy23, &decodeRet.kv48, &decodeRet.kv49,
        &decodeRet.ret_dummy24, &decodeRet.kv50, &decodeRet.kv51,
        &decodeRet.ret_dummy25, &decodeRet.kv52, &decodeRet.kv53,
        &decodeRet.ret_dummy26, &decodeRet.kv54, &decodeRet.kv55);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    const float *startPtr = decodeRet.logits.getData();
    const float *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(i, tok, inferenceTime.count() / 1000);

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
  std::cout << "\033[33;1m[Output]\033[0m "
            << outputContainer.revertDeepSeekR1() << std::endl;

  return 0;
}
