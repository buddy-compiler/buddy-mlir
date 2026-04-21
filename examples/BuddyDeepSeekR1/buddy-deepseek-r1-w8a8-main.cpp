//===- buddy-deepseek-r1-w8a8-main.cpp ------------------------------------===//
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

  std::array<MemRef<float, 4> *, 56> kv_ptrs;

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
        kv_ptrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,

                &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,

                &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,

                &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,

                &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,

                &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,

                &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

/// Decode returns layout for quantized variants:
/// cache_position_out, kv0, kv1, dummy0..dummy26, kv2..kv55, logits.
struct DecodeReturns {
  MemRef<long long, 1> cache_position_out;
  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<long long, 1> ret_dummy0;
  MemRef<long long, 1> ret_dummy1;
  MemRef<long long, 1> ret_dummy2;
  MemRef<long long, 1> ret_dummy3;
  MemRef<long long, 1> ret_dummy4;
  MemRef<long long, 1> ret_dummy5;
  MemRef<long long, 1> ret_dummy6;
  MemRef<long long, 1> ret_dummy7;
  MemRef<long long, 1> ret_dummy8;
  MemRef<long long, 1> ret_dummy9;
  MemRef<long long, 1> ret_dummy10;
  MemRef<long long, 1> ret_dummy11;
  MemRef<long long, 1> ret_dummy12;
  MemRef<long long, 1> ret_dummy13;
  MemRef<long long, 1> ret_dummy14;
  MemRef<long long, 1> ret_dummy15;
  MemRef<long long, 1> ret_dummy16;
  MemRef<long long, 1> ret_dummy17;
  MemRef<long long, 1> ret_dummy18;
  MemRef<long long, 1> ret_dummy19;
  MemRef<long long, 1> ret_dummy20;
  MemRef<long long, 1> ret_dummy21;
  MemRef<long long, 1> ret_dummy22;
  MemRef<long long, 1> ret_dummy23;
  MemRef<long long, 1> ret_dummy24;
  MemRef<long long, 1> ret_dummy25;
  MemRef<long long, 1> ret_dummy26;
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

using KVPtrArray = std::array<MemRef<float, 4> *, 56>;

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

/// W8A8 quantization (int8 weights + int8 activation path) with f32 host-side
/// buffers. Forward functions take two param memrefs: f32 and i8.
extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
                                             MemRef<float, 1> *f32_params,
                                             MemRef<int8_t, 1> *i8_params,
                                             MemRef<long long, 2> *input);

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

void copy_kv_by_cache_position_block(const MemRefContainer &prefill,
                                     const KVPtrArray &decodePtrs,
                                     int cache_position) {
  constexpr int num_kv = 56;
  int copy_len = std::min(cache_position, (int)MaxTokenLength);

  for (int k = 0; k < num_kv; ++k) {
    auto &src = *prefill.kv_ptrs[k];
    auto &dst = *decodePtrs[k];

    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes_to_copy =
          static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);

      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 W8A8 Quantized Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title =
      "DeepSeekR1 W8A8 Quantized Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string f32ParamsDir = deepSeekR1BuildDir + "arg0-w8a8-f32.data";
  const std::string i8ParamsDir = deepSeekR1BuildDir + "arg0-w8a8-i8.data";

  std::string inputStr;
  getUserInput(inputStr);

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> tokenizedText(inputStr);
  MemRef<long long, 2> inputContainerPrefill({1, MaxTokenLength}, 0LL);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> F32ParamsContainer({F32ParamsSize});
  MemRef<int8_t, 1> I8ParamsContainer({I8ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  MemRef<float, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<float, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRefContainer prefillResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
  MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

  tokenizeInput(vocabDir, tokenizedText);
  {
    size_t *src = tokenizedText.getData();
    long long *dst = inputContainerPrefill.getData();
    for (size_t i = 0; i < MaxTokenLength; ++i)
      dst[i] = static_cast<long long>(src[i]);
  }
  outputContainer.loadVocab(vocabDir);
  loadParameters(f32ParamsDir, F32ParamsContainer);
  loadParameters(i8ParamsDir, I8ParamsContainer);

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(ptrPrefillResultContainer, &F32ParamsContainer,
                               &I8ParamsContainer, &inputContainerPrefill);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = tokenizedText.getTokenCnt() - 1;
  const float *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = tokenizedText.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  MemRef<float, 3> logits_decode({1, 1, MaxVocabSize});
  DecodeReturns decodeResultContainer = {
      MemRef<long long, 1>({1}, 0LL), // cache_position_out
      kv0,
      kv1,
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      MemRef<long long, 1>({1}, 0LL),
      kv2,
      kv3,
      kv4,
      kv5,
      kv6,
      kv7,
      kv8,
      kv9,
      kv10,
      kv11,
      kv12,
      kv13,
      kv14,
      kv15,
      kv16,
      kv17,
      kv18,
      kv19,
      kv20,
      kv21,
      kv22,
      kv23,
      kv24,
      kv25,
      kv26,
      kv27,
      kv28,
      kv29,
      kv30,
      kv31,
      kv32,
      kv33,
      kv34,
      kv35,
      kv36,
      kv37,
      kv38,
      kv39,
      kv40,
      kv41,
      kv42,
      kv43,
      kv44,
      kv45,
      kv46,
      kv47,
      kv48,
      kv49,
      kv50,
      kv51,
      kv52,
      kv53,
      kv54,
      kv55,
      logits_decode};

  KVPtrArray decodePtrs = buildDecodeKVPtrs(decodeResultContainer);

  copy_kv_by_cache_position_block(
      prefillResultContainer, decodePtrs,
      static_cast<int>(tokenizedText.getTokenCnt()));

  cachePosition.getData()[0] = tokenizedText.getTokenCnt();
  int generateLen = MaxTokenLength - tokenizedText.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  for (int i = 1; i <= generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    long long curPos = cachePosition.getData()[0];
    decodeResultContainer.ret_dummy0.getData()[0] = curPos;
    decodeResultContainer.ret_dummy1.getData()[0] = curPos;
    decodeResultContainer.ret_dummy2.getData()[0] = curPos;
    decodeResultContainer.ret_dummy3.getData()[0] = curPos;
    decodeResultContainer.ret_dummy4.getData()[0] = curPos;
    decodeResultContainer.ret_dummy5.getData()[0] = curPos;
    decodeResultContainer.ret_dummy6.getData()[0] = curPos;
    decodeResultContainer.ret_dummy7.getData()[0] = curPos;
    decodeResultContainer.ret_dummy8.getData()[0] = curPos;
    decodeResultContainer.ret_dummy9.getData()[0] = curPos;
    decodeResultContainer.ret_dummy10.getData()[0] = curPos;
    decodeResultContainer.ret_dummy11.getData()[0] = curPos;
    decodeResultContainer.ret_dummy12.getData()[0] = curPos;
    decodeResultContainer.ret_dummy13.getData()[0] = curPos;
    decodeResultContainer.ret_dummy14.getData()[0] = curPos;
    decodeResultContainer.ret_dummy15.getData()[0] = curPos;
    decodeResultContainer.ret_dummy16.getData()[0] = curPos;
    decodeResultContainer.ret_dummy17.getData()[0] = curPos;
    decodeResultContainer.ret_dummy18.getData()[0] = curPos;
    decodeResultContainer.ret_dummy19.getData()[0] = curPos;
    decodeResultContainer.ret_dummy20.getData()[0] = curPos;
    decodeResultContainer.ret_dummy21.getData()[0] = curPos;
    decodeResultContainer.ret_dummy22.getData()[0] = curPos;
    decodeResultContainer.ret_dummy23.getData()[0] = curPos;
    decodeResultContainer.ret_dummy24.getData()[0] = curPos;
    decodeResultContainer.ret_dummy25.getData()[0] = curPos;
    decodeResultContainer.ret_dummy26.getData()[0] = curPos;

    _mlir_ciface_forward_decode(
        &decodeResultContainer, &F32ParamsContainer, &I8ParamsContainer,
        &inputContainerDecode, &cachePosition, &decodeResultContainer.kv0,
        &decodeResultContainer.kv1, &decodeResultContainer.ret_dummy0,
        &decodeResultContainer.kv2, &decodeResultContainer.kv3,
        &decodeResultContainer.ret_dummy1, &decodeResultContainer.kv4,
        &decodeResultContainer.kv5, &decodeResultContainer.ret_dummy2,
        &decodeResultContainer.kv6, &decodeResultContainer.kv7,
        &decodeResultContainer.ret_dummy3, &decodeResultContainer.kv8,
        &decodeResultContainer.kv9, &decodeResultContainer.ret_dummy4,
        &decodeResultContainer.kv10, &decodeResultContainer.kv11,
        &decodeResultContainer.ret_dummy5, &decodeResultContainer.kv12,
        &decodeResultContainer.kv13, &decodeResultContainer.ret_dummy6,
        &decodeResultContainer.kv14, &decodeResultContainer.kv15,
        &decodeResultContainer.ret_dummy7, &decodeResultContainer.kv16,
        &decodeResultContainer.kv17, &decodeResultContainer.ret_dummy8,
        &decodeResultContainer.kv18, &decodeResultContainer.kv19,
        &decodeResultContainer.ret_dummy9, &decodeResultContainer.kv20,
        &decodeResultContainer.kv21, &decodeResultContainer.ret_dummy10,
        &decodeResultContainer.kv22, &decodeResultContainer.kv23,
        &decodeResultContainer.ret_dummy11, &decodeResultContainer.kv24,
        &decodeResultContainer.kv25, &decodeResultContainer.ret_dummy12,
        &decodeResultContainer.kv26, &decodeResultContainer.kv27,
        &decodeResultContainer.ret_dummy13, &decodeResultContainer.kv28,
        &decodeResultContainer.kv29, &decodeResultContainer.ret_dummy14,
        &decodeResultContainer.kv30, &decodeResultContainer.kv31,
        &decodeResultContainer.ret_dummy15, &decodeResultContainer.kv32,
        &decodeResultContainer.kv33, &decodeResultContainer.ret_dummy16,
        &decodeResultContainer.kv34, &decodeResultContainer.kv35,
        &decodeResultContainer.ret_dummy17, &decodeResultContainer.kv36,
        &decodeResultContainer.kv37, &decodeResultContainer.ret_dummy18,
        &decodeResultContainer.kv38, &decodeResultContainer.kv39,
        &decodeResultContainer.ret_dummy19, &decodeResultContainer.kv40,
        &decodeResultContainer.kv41, &decodeResultContainer.ret_dummy20,
        &decodeResultContainer.kv42, &decodeResultContainer.kv43,
        &decodeResultContainer.ret_dummy21, &decodeResultContainer.kv44,
        &decodeResultContainer.kv45, &decodeResultContainer.ret_dummy22,
        &decodeResultContainer.kv46, &decodeResultContainer.kv47,
        &decodeResultContainer.ret_dummy23, &decodeResultContainer.kv48,
        &decodeResultContainer.kv49, &decodeResultContainer.ret_dummy24,
        &decodeResultContainer.kv50, &decodeResultContainer.kv51,
        &decodeResultContainer.ret_dummy25, &decodeResultContainer.kv52,
        &decodeResultContainer.kv53, &decodeResultContainer.ret_dummy26,
        &decodeResultContainer.kv54, &decodeResultContainer.kv55);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    const float *startPtr = decodeResultContainer.logits.getData();
    const float *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = tokenizedText.getStr(maxIndex);
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
