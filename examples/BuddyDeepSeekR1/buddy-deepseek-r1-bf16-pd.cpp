//===- buddy-deepseek-r1-bf16-pd.cpp --------------------------------------===//
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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace buddy;
double total_time = 0;
constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 20;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

struct MemRefContainer {

  MemRef<uint16_t, 4> kv0;
  MemRef<uint16_t, 4> kv1;
  MemRef<uint16_t, 4> kv2;
  MemRef<uint16_t, 4> kv3;
  MemRef<uint16_t, 4> kv4;
  MemRef<uint16_t, 4> kv5;
  MemRef<uint16_t, 4> kv6;
  MemRef<uint16_t, 4> kv7;
  MemRef<uint16_t, 4> kv8;
  MemRef<uint16_t, 4> kv9;
  MemRef<uint16_t, 4> kv10;
  MemRef<uint16_t, 4> kv11;
  MemRef<uint16_t, 4> kv12;
  MemRef<uint16_t, 4> kv13;
  MemRef<uint16_t, 4> kv14;
  MemRef<uint16_t, 4> kv15;
  MemRef<uint16_t, 4> kv16;
  MemRef<uint16_t, 4> kv17;
  MemRef<uint16_t, 4> kv18;
  MemRef<uint16_t, 4> kv19;
  MemRef<uint16_t, 4> kv20;
  MemRef<uint16_t, 4> kv21;
  MemRef<uint16_t, 4> kv22;
  MemRef<uint16_t, 4> kv23;
  MemRef<uint16_t, 4> kv24;
  MemRef<uint16_t, 4> kv25;
  MemRef<uint16_t, 4> kv26;
  MemRef<uint16_t, 4> kv27;
  MemRef<uint16_t, 4> kv28;
  MemRef<uint16_t, 4> kv29;
  MemRef<uint16_t, 4> kv30;
  MemRef<uint16_t, 4> kv31;
  MemRef<uint16_t, 4> kv32;
  MemRef<uint16_t, 4> kv33;
  MemRef<uint16_t, 4> kv34;
  MemRef<uint16_t, 4> kv35;
  MemRef<uint16_t, 4> kv36;
  MemRef<uint16_t, 4> kv37;
  MemRef<uint16_t, 4> kv38;
  MemRef<uint16_t, 4> kv39;
  MemRef<uint16_t, 4> kv40;
  MemRef<uint16_t, 4> kv41;
  MemRef<uint16_t, 4> kv42;
  MemRef<uint16_t, 4> kv43;
  MemRef<uint16_t, 4> kv44;
  MemRef<uint16_t, 4> kv45;
  MemRef<uint16_t, 4> kv46;
  MemRef<uint16_t, 4> kv47;
  MemRef<uint16_t, 4> kv48;
  MemRef<uint16_t, 4> kv49;
  MemRef<uint16_t, 4> kv50;
  MemRef<uint16_t, 4> kv51;
  MemRef<uint16_t, 4> kv52;
  MemRef<uint16_t, 4> kv53;
  MemRef<uint16_t, 4> kv54;
  MemRef<uint16_t, 4> kv55;

  MemRef<uint16_t, 3> logits;

  std::array<MemRef<uint16_t, 4> *, 56> kv_ptrs;

  MemRefContainer(MemRef<uint16_t, 4> &kv0_, MemRef<uint16_t, 4> &kv1_,
                  MemRef<uint16_t, 4> &kv2_, MemRef<uint16_t, 4> &kv3_,
                  MemRef<uint16_t, 4> &kv4_, MemRef<uint16_t, 4> &kv5_,
                  MemRef<uint16_t, 4> &kv6_, MemRef<uint16_t, 4> &kv7_,
                  MemRef<uint16_t, 4> &kv8_, MemRef<uint16_t, 4> &kv9_,
                  MemRef<uint16_t, 4> &kv10_, MemRef<uint16_t, 4> &kv11_,
                  MemRef<uint16_t, 4> &kv12_, MemRef<uint16_t, 4> &kv13_,
                  MemRef<uint16_t, 4> &kv14_, MemRef<uint16_t, 4> &kv15_,
                  MemRef<uint16_t, 4> &kv16_, MemRef<uint16_t, 4> &kv17_,
                  MemRef<uint16_t, 4> &kv18_, MemRef<uint16_t, 4> &kv19_,
                  MemRef<uint16_t, 4> &kv20_, MemRef<uint16_t, 4> &kv21_,
                  MemRef<uint16_t, 4> &kv22_, MemRef<uint16_t, 4> &kv23_,
                  MemRef<uint16_t, 4> &kv24_, MemRef<uint16_t, 4> &kv25_,
                  MemRef<uint16_t, 4> &kv26_, MemRef<uint16_t, 4> &kv27_,
                  MemRef<uint16_t, 4> &kv28_, MemRef<uint16_t, 4> &kv29_,
                  MemRef<uint16_t, 4> &kv30_, MemRef<uint16_t, 4> &kv31_,
                  MemRef<uint16_t, 4> &kv32_, MemRef<uint16_t, 4> &kv33_,
                  MemRef<uint16_t, 4> &kv34_, MemRef<uint16_t, 4> &kv35_,
                  MemRef<uint16_t, 4> &kv36_, MemRef<uint16_t, 4> &kv37_,
                  MemRef<uint16_t, 4> &kv38_, MemRef<uint16_t, 4> &kv39_,
                  MemRef<uint16_t, 4> &kv40_, MemRef<uint16_t, 4> &kv41_,
                  MemRef<uint16_t, 4> &kv42_, MemRef<uint16_t, 4> &kv43_,
                  MemRef<uint16_t, 4> &kv44_, MemRef<uint16_t, 4> &kv45_,
                  MemRef<uint16_t, 4> &kv46_, MemRef<uint16_t, 4> &kv47_,
                  MemRef<uint16_t, 4> &kv48_, MemRef<uint16_t, 4> &kv49_,
                  MemRef<uint16_t, 4> &kv50_, MemRef<uint16_t, 4> &kv51_,
                  MemRef<uint16_t, 4> &kv52_, MemRef<uint16_t, 4> &kv53_,
                  MemRef<uint16_t, 4> &kv54_, MemRef<uint16_t, 4> &kv55_,
                  MemRef<uint16_t, 3> &logits_)
      : kv0(kv0_), kv1(kv1_), kv2(kv2_), kv3(kv3_), kv4(kv4_), kv5(kv5_),
        kv6(kv6_), kv7(kv7_), kv8(kv8_), kv9(kv9_), kv10(kv10_), kv11(kv11_),
        kv12(kv12_), kv13(kv13_), kv14(kv14_), kv15(kv15_), kv16(kv16_),
        kv17(kv17_), kv18(kv18_), kv19(kv19_), kv20(kv20_), kv21(kv21_),
        kv22(kv22_), kv23(kv23_), kv24(kv24_), kv25(kv25_), kv26(kv26_),
        kv27(kv27_), kv28(kv28_), kv29(kv29_), kv30(kv30_), kv31(kv31_),
        kv32(kv32_), kv33(kv33_), kv34(kv34_), kv35(kv35_), kv36(kv36_),
        kv37(kv37_), kv38(kv38_), kv39(kv39_), kv40(kv40_), kv41(kv41_),
        kv42(kv42_), kv43(kv43_), kv44(kv44_), kv45(kv45_), kv46(kv46_),
        kv47(kv47_), kv48(kv48_), kv49(kv49_), kv50(kv50_), kv51(kv51_),
        kv52(kv52_), kv53(kv53_), kv54(kv54_), kv55(kv55_),
        logits(logits_), kv_ptrs{
                             &kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,
                             &kv7,  &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13,
                             &kv14, &kv15, &kv16, &kv17, &kv18, &kv19, &kv20,
                             &kv21, &kv22, &kv23, &kv24, &kv25, &kv26, &kv27,
                             &kv28, &kv29, &kv30, &kv31, &kv32, &kv33, &kv34,
                             &kv35, &kv36, &kv37, &kv38, &kv39, &kv40, &kv41,
                             &kv42, &kv43, &kv44, &kv45, &kv46, &kv47, &kv48,
                             &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

/// Declare DeepSeekR1 forward function.
extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
                                             MemRef<uint16_t, 1> *arg0,
                                             Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<uint16_t, 1> *arg0,
    MemRef<long long, 2> *arg1, MemRef<long long, 1> *arg2,
    MemRef<uint16_t, 4> *kv0, MemRef<uint16_t, 4> *kv1,
    MemRef<uint16_t, 4> *kv2, MemRef<uint16_t, 4> *kv3,
    MemRef<uint16_t, 4> *kv4, MemRef<uint16_t, 4> *kv5,
    MemRef<uint16_t, 4> *kv6, MemRef<uint16_t, 4> *kv7,
    MemRef<uint16_t, 4> *kv8, MemRef<uint16_t, 4> *kv9,
    MemRef<uint16_t, 4> *kv10, MemRef<uint16_t, 4> *kv11,
    MemRef<uint16_t, 4> *kv12, MemRef<uint16_t, 4> *kv13,
    MemRef<uint16_t, 4> *kv14, MemRef<uint16_t, 4> *kv15,
    MemRef<uint16_t, 4> *kv16, MemRef<uint16_t, 4> *kv17,
    MemRef<uint16_t, 4> *kv18, MemRef<uint16_t, 4> *kv19,
    MemRef<uint16_t, 4> *kv20, MemRef<uint16_t, 4> *kv21,
    MemRef<uint16_t, 4> *kv22, MemRef<uint16_t, 4> *kv23,
    MemRef<uint16_t, 4> *kv24, MemRef<uint16_t, 4> *kv25,
    MemRef<uint16_t, 4> *kv26, MemRef<uint16_t, 4> *kv27,
    MemRef<uint16_t, 4> *kv28, MemRef<uint16_t, 4> *kv29,
    MemRef<uint16_t, 4> *kv30, MemRef<uint16_t, 4> *kv31,
    MemRef<uint16_t, 4> *kv32, MemRef<uint16_t, 4> *kv33,
    MemRef<uint16_t, 4> *kv34, MemRef<uint16_t, 4> *kv35,
    MemRef<uint16_t, 4> *kv36, MemRef<uint16_t, 4> *kv37,
    MemRef<uint16_t, 4> *kv38, MemRef<uint16_t, 4> *kv39,
    MemRef<uint16_t, 4> *kv40, MemRef<uint16_t, 4> *kv41,
    MemRef<uint16_t, 4> *kv42, MemRef<uint16_t, 4> *kv43,
    MemRef<uint16_t, 4> *kv44, MemRef<uint16_t, 4> *kv45,
    MemRef<uint16_t, 4> *kv46, MemRef<uint16_t, 4> *kv47,
    MemRef<uint16_t, 4> *kv48, MemRef<uint16_t, 4> *kv49,
    MemRef<uint16_t, 4> *kv50, MemRef<uint16_t, 4> *kv51,
    MemRef<uint16_t, 4> *kv52, MemRef<uint16_t, 4> *kv53,
    MemRef<uint16_t, 4> *kv54, MemRef<uint16_t, 4> *kv55);

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
  total_time += time;
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
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  printLogLabel();
  std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
            << std::endl;
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath,
                    MemRef<uint16_t, 1> &params) {
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
                 sizeof(uint16_t) * (params.getSize()));
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

/// Convert BF16 (uint16_t) to float for finding max index
float bf16_to_float(uint16_t bf16_val) {
  uint32_t float_bits = static_cast<uint32_t>(bf16_val) << 16;
  float result;
  std::memcpy(&result, &float_bits, sizeof(float));
  return result;
}

/// Find the index of the max value in BF16 array
int findMaxIndex(const uint16_t *start, const uint16_t *end) {
  size_t size = end - start;
  float max_val = bf16_to_float(start[0]);
  int max_idx = 0;

  for (size_t i = 1; i < size; ++i) {
    float val = bf16_to_float(start[i]);
    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }
  return max_idx;
}

void copy_kv_by_cache_position_block(const MemRefContainer &prefill,
                                     MemRefContainer &decode,
                                     int cache_position) {
  constexpr int num_kv = 56;
  int copy_len = std::min(cache_position, (int)MaxTokenLength);

  for (int k = 0; k < num_kv; ++k) {
    auto &src = *prefill.kv_ptrs[k];
    auto &dst = *decode.kv_ptrs[k];

    for (int h = 0; h < (int)HeadNum; ++h) {
      size_t bytes_to_copy =
          static_cast<size_t>(copy_len) * HiddenSize * sizeof(uint16_t);

      uint16_t *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      uint16_t *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title =
      "DeepSeekR1 BF16 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0-bf16.data";

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
  MemRef<uint16_t, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<uint16_t, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  MemRef<uint16_t, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  MemRef<uint16_t, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

  MemRefContainer prefillResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
  MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  /// Run DeepSeekR1 Inference
  //  - Perform the forward function.
  //  - Find and append the generated token.
  //  - Continue iterating until the terminal condition is met.

  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(ptrPrefillResultContainer, &ParamsContainer,
                               &inputContainerPrefill);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const uint16_t *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
  const uint16_t *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);
  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  MemRef<uint16_t, 3> logits_decode({1, 1, MaxVocabSize});

  MemRefContainer decodeResultContainer(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_decode);

  MemRefContainer *ptrDecodeResultContainer = &decodeResultContainer;

  copy_kv_by_cache_position_block(prefillResultContainer, decodeResultContainer,
                                  inputContainerPrefill.getTokenCnt());

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  for (int i = 1; i <= generateLen; i++) {
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        ptrDecodeResultContainer, &ParamsContainer, &inputContainerDecode,
        &cachePosition, &ptrDecodeResultContainer->kv0,
        &ptrDecodeResultContainer->kv1, &ptrDecodeResultContainer->kv2,
        &ptrDecodeResultContainer->kv3, &ptrDecodeResultContainer->kv4,
        &ptrDecodeResultContainer->kv5, &ptrDecodeResultContainer->kv6,
        &ptrDecodeResultContainer->kv7, &ptrDecodeResultContainer->kv8,
        &ptrDecodeResultContainer->kv9, &ptrDecodeResultContainer->kv10,
        &ptrDecodeResultContainer->kv11, &ptrDecodeResultContainer->kv12,
        &ptrDecodeResultContainer->kv13, &ptrDecodeResultContainer->kv14,
        &ptrDecodeResultContainer->kv15, &ptrDecodeResultContainer->kv16,
        &ptrDecodeResultContainer->kv17, &ptrDecodeResultContainer->kv18,
        &ptrDecodeResultContainer->kv19, &ptrDecodeResultContainer->kv20,
        &ptrDecodeResultContainer->kv21, &ptrDecodeResultContainer->kv22,
        &ptrDecodeResultContainer->kv23, &ptrDecodeResultContainer->kv24,
        &ptrDecodeResultContainer->kv25, &ptrDecodeResultContainer->kv26,
        &ptrDecodeResultContainer->kv27, &ptrDecodeResultContainer->kv28,
        &ptrDecodeResultContainer->kv29, &ptrDecodeResultContainer->kv30,
        &ptrDecodeResultContainer->kv31, &ptrDecodeResultContainer->kv32,
        &ptrDecodeResultContainer->kv33, &ptrDecodeResultContainer->kv34,
        &ptrDecodeResultContainer->kv35, &ptrDecodeResultContainer->kv36,
        &ptrDecodeResultContainer->kv37, &ptrDecodeResultContainer->kv38,
        &ptrDecodeResultContainer->kv39, &ptrDecodeResultContainer->kv40,
        &ptrDecodeResultContainer->kv41, &ptrDecodeResultContainer->kv42,
        &ptrDecodeResultContainer->kv43, &ptrDecodeResultContainer->kv44,
        &ptrDecodeResultContainer->kv45, &ptrDecodeResultContainer->kv46,
        &ptrDecodeResultContainer->kv47, &ptrDecodeResultContainer->kv48,
        &ptrDecodeResultContainer->kv49, &ptrDecodeResultContainer->kv50,
        &ptrDecodeResultContainer->kv51, &ptrDecodeResultContainer->kv52,
        &ptrDecodeResultContainer->kv53, &ptrDecodeResultContainer->kv54,
        &ptrDecodeResultContainer->kv55);

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    // Determine the generated token.
    const uint16_t *startPtr = ptrDecodeResultContainer->logits.getData();
    const uint16_t *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    // Stop if a <|end▁of▁sentence|> token is generated.
    if (maxIndex == 151643) {
      break;
    }
    // Append the generated token into the input and output container.
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  const double decodeTokensPerSec =
      decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
                          : 0.0;

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
