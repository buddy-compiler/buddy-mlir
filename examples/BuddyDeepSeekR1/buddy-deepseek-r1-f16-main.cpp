// //===- buddy-deepseek-r1-f16-main.cpp
// -------------------------------------===//
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// //===----------------------------------------------------------------------===//

// #include <array>
// #include <buddy/Core/Container.h>
// #include <buddy/LLM/TextContainer.h>
// #include <chrono>
// #include <cmath>
// #include <cstddef>
// #include <cstdint>
// #include <cstring>
// #include <filesystem>
// #include <fstream>
// #include <iostream>
// #include <limits>

// using namespace buddy;
// double total_time = 0;
// constexpr size_t ParamsSize = 1777088064;
// constexpr size_t MaxVocabSize = 151936;
// constexpr size_t MaxTokenLength = 1024;

// constexpr size_t NUM_LAYERS = 56;
// constexpr size_t HiddenSize = 128;
// constexpr size_t HeadNum = 2;

// struct MemRefContainer {

//   MemRef<uint16_t, 4> kv0;
//   MemRef<uint16_t, 4> kv1;
//   MemRef<uint16_t, 4> kv2;
//   MemRef<uint16_t, 4> kv3;
//   MemRef<uint16_t, 4> kv4;
//   MemRef<uint16_t, 4> kv5;
//   MemRef<uint16_t, 4> kv6;
//   MemRef<uint16_t, 4> kv7;
//   MemRef<uint16_t, 4> kv8;
//   MemRef<uint16_t, 4> kv9;
//   MemRef<uint16_t, 4> kv10;
//   MemRef<uint16_t, 4> kv11;
//   MemRef<uint16_t, 4> kv12;
//   MemRef<uint16_t, 4> kv13;
//   MemRef<uint16_t, 4> kv14;
//   MemRef<uint16_t, 4> kv15;
//   MemRef<uint16_t, 4> kv16;
//   MemRef<uint16_t, 4> kv17;
//   MemRef<uint16_t, 4> kv18;
//   MemRef<uint16_t, 4> kv19;
//   MemRef<uint16_t, 4> kv20;
//   MemRef<uint16_t, 4> kv21;
//   MemRef<uint16_t, 4> kv22;
//   MemRef<uint16_t, 4> kv23;
//   MemRef<uint16_t, 4> kv24;
//   MemRef<uint16_t, 4> kv25;
//   MemRef<uint16_t, 4> kv26;
//   MemRef<uint16_t, 4> kv27;
//   MemRef<uint16_t, 4> kv28;
//   MemRef<uint16_t, 4> kv29;
//   MemRef<uint16_t, 4> kv30;
//   MemRef<uint16_t, 4> kv31;
//   MemRef<uint16_t, 4> kv32;
//   MemRef<uint16_t, 4> kv33;
//   MemRef<uint16_t, 4> kv34;
//   MemRef<uint16_t, 4> kv35;
//   MemRef<uint16_t, 4> kv36;
//   MemRef<uint16_t, 4> kv37;
//   MemRef<uint16_t, 4> kv38;
//   MemRef<uint16_t, 4> kv39;
//   MemRef<uint16_t, 4> kv40;
//   MemRef<uint16_t, 4> kv41;
//   MemRef<uint16_t, 4> kv42;
//   MemRef<uint16_t, 4> kv43;
//   MemRef<uint16_t, 4> kv44;
//   MemRef<uint16_t, 4> kv45;
//   MemRef<uint16_t, 4> kv46;
//   MemRef<uint16_t, 4> kv47;
//   MemRef<uint16_t, 4> kv48;
//   MemRef<uint16_t, 4> kv49;
//   MemRef<uint16_t, 4> kv50;
//   MemRef<uint16_t, 4> kv51;
//   MemRef<uint16_t, 4> kv52;
//   MemRef<uint16_t, 4> kv53;
//   MemRef<uint16_t, 4> kv54;
//   MemRef<uint16_t, 4> kv55;

//   MemRef<uint16_t, 3> logits;

//   std::array<MemRef<uint16_t, 4> *, 56> kv_ptrs;

//   MemRefContainer(
//       MemRef<uint16_t, 4> k0, MemRef<uint16_t, 4> k1, MemRef<uint16_t, 4> k2,
//       MemRef<uint16_t, 4> k3, MemRef<uint16_t, 4> k4, MemRef<uint16_t, 4> k5,
//       MemRef<uint16_t, 4> k6, MemRef<uint16_t, 4> k7, MemRef<uint16_t, 4> k8,
//       MemRef<uint16_t, 4> k9, MemRef<uint16_t, 4> k10, MemRef<uint16_t, 4>
//       k11, MemRef<uint16_t, 4> k12, MemRef<uint16_t, 4> k13, MemRef<uint16_t,
//       4> k14, MemRef<uint16_t, 4> k15, MemRef<uint16_t, 4> k16,
//       MemRef<uint16_t, 4> k17, MemRef<uint16_t, 4> k18, MemRef<uint16_t, 4>
//       k19, MemRef<uint16_t, 4> k20, MemRef<uint16_t, 4> k21, MemRef<uint16_t,
//       4> k22, MemRef<uint16_t, 4> k23, MemRef<uint16_t, 4> k24,
//       MemRef<uint16_t, 4> k25, MemRef<uint16_t, 4> k26, MemRef<uint16_t, 4>
//       k27, MemRef<uint16_t, 4> k28, MemRef<uint16_t, 4> k29, MemRef<uint16_t,
//       4> k30, MemRef<uint16_t, 4> k31, MemRef<uint16_t, 4> k32,
//       MemRef<uint16_t, 4> k33, MemRef<uint16_t, 4> k34, MemRef<uint16_t, 4>
//       k35, MemRef<uint16_t, 4> k36, MemRef<uint16_t, 4> k37, MemRef<uint16_t,
//       4> k38, MemRef<uint16_t, 4> k39, MemRef<uint16_t, 4> k40,
//       MemRef<uint16_t, 4> k41, MemRef<uint16_t, 4> k42, MemRef<uint16_t, 4>
//       k43, MemRef<uint16_t, 4> k44, MemRef<uint16_t, 4> k45, MemRef<uint16_t,
//       4> k46, MemRef<uint16_t, 4> k47, MemRef<uint16_t, 4> k48,
//       MemRef<uint16_t, 4> k49, MemRef<uint16_t, 4> k50, MemRef<uint16_t, 4>
//       k51, MemRef<uint16_t, 4> k52, MemRef<uint16_t, 4> k53, MemRef<uint16_t,
//       4> k54, MemRef<uint16_t, 4> k55, MemRef<uint16_t, 3> l) : kv0(k0),
//       kv1(k1), kv2(k2), kv3(k3), kv4(k4), kv5(k5), kv6(k6), kv7(k7),
//         kv8(k8), kv9(k9), kv10(k10), kv11(k11), kv12(k12), kv13(k13),
//         kv14(k14), kv15(k15), kv16(k16), kv17(k17), kv18(k18), kv19(k19),
//         kv20(k20), kv21(k21), kv22(k22), kv23(k23), kv24(k24), kv25(k25),
//         kv26(k26), kv27(k27), kv28(k28), kv29(k29), kv30(k30), kv31(k31),
//         kv32(k32), kv33(k33), kv34(k34), kv35(k35), kv36(k36), kv37(k37),
//         kv38(k38), kv39(k39), kv40(k40), kv41(k41), kv42(k42), kv43(k43),
//         kv44(k44), kv45(k45), kv46(k46), kv47(k47), kv48(k48), kv49(k49),
//         kv50(k50), kv51(k51), kv52(k52), kv53(k53), kv54(k54), kv55(k55),
//         logits(l), kv_ptrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,
//         &kv7,

//                 &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,

//                 &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,

//                 &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,

//                 &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,

//                 &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,

//                 &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
// };

// /// Declare DeepSeekR1 forward function.
// extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
//                                              MemRef<uint16_t, 1> *arg0,
//                                              Text<size_t, 2> *arg1);

// extern "C" void _mlir_ciface_forward_decode(
//     MemRefContainer *result, MemRef<uint16_t, 1> *arg0,
//     MemRef<long long, 2> *arg1, MemRef<long long, 1> *arg2,
//     MemRef<uint16_t, 4> *kv0, MemRef<uint16_t, 4> *kv1,
//     MemRef<uint16_t, 4> *kv2, MemRef<uint16_t, 4> *kv3,
//     MemRef<uint16_t, 4> *kv4, MemRef<uint16_t, 4> *kv5,
//     MemRef<uint16_t, 4> *kv6, MemRef<uint16_t, 4> *kv7,
//     MemRef<uint16_t, 4> *kv8, MemRef<uint16_t, 4> *kv9,
//     MemRef<uint16_t, 4> *kv10, MemRef<uint16_t, 4> *kv11,
//     MemRef<uint16_t, 4> *kv12, MemRef<uint16_t, 4> *kv13,
//     MemRef<uint16_t, 4> *kv14, MemRef<uint16_t, 4> *kv15,
//     MemRef<uint16_t, 4> *kv16, MemRef<uint16_t, 4> *kv17,
//     MemRef<uint16_t, 4> *kv18, MemRef<uint16_t, 4> *kv19,
//     MemRef<uint16_t, 4> *kv20, MemRef<uint16_t, 4> *kv21,
//     MemRef<uint16_t, 4> *kv22, MemRef<uint16_t, 4> *kv23,
//     MemRef<uint16_t, 4> *kv24, MemRef<uint16_t, 4> *kv25,
//     MemRef<uint16_t, 4> *kv26, MemRef<uint16_t, 4> *kv27,
//     MemRef<uint16_t, 4> *kv28, MemRef<uint16_t, 4> *kv29,
//     MemRef<uint16_t, 4> *kv30, MemRef<uint16_t, 4> *kv31,
//     MemRef<uint16_t, 4> *kv32, MemRef<uint16_t, 4> *kv33,
//     MemRef<uint16_t, 4> *kv34, MemRef<uint16_t, 4> *kv35,
//     MemRef<uint16_t, 4> *kv36, MemRef<uint16_t, 4> *kv37,
//     MemRef<uint16_t, 4> *kv38, MemRef<uint16_t, 4> *kv39,
//     MemRef<uint16_t, 4> *kv40, MemRef<uint16_t, 4> *kv41,
//     MemRef<uint16_t, 4> *kv42, MemRef<uint16_t, 4> *kv43,
//     MemRef<uint16_t, 4> *kv44, MemRef<uint16_t, 4> *kv45,
//     MemRef<uint16_t, 4> *kv46, MemRef<uint16_t, 4> *kv47,
//     MemRef<uint16_t, 4> *kv48, MemRef<uint16_t, 4> *kv49,
//     MemRef<uint16_t, 4> *kv50, MemRef<uint16_t, 4> *kv51,
//     MemRef<uint16_t, 4> *kv52, MemRef<uint16_t, 4> *kv53,
//     MemRef<uint16_t, 4> *kv54, MemRef<uint16_t, 4> *kv55);

// //
// -----------------------------------------------------------------------------
// // Helper Functions
// //
// -----------------------------------------------------------------------------

// /// Capture input message.
// void getUserInput(std::string &inputStr) {
//   std::cout << "\nPlease send a message:" << std::endl;
//   std::cout << ">>> ";
//   getline(std::cin, inputStr);
//   std::cout << std::endl;
// }

// /// Print [Log] label in bold blue format.
// void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

// /// Print information for each iteration.
// void printIterInfo(size_t iterIdx, std::string str, double time) {
//   total_time += time;
//   std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
//   std::cout << "Token: " << str << " | "
//             << "Time: " << time << "s" << std::endl;
// }

// /// Tokenize input data in the container.
// void tokenizeInput(const std::string &vocabFile,
//                    Text<size_t, 2> &inputContainer) {
//   printLogLabel();
//   std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
//             << std::endl;
//   const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
//   inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
//   const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
//   const std::chrono::duration<double, std::milli> buddyTokenizeTime =
//       buddyTokenizeEnd - buddyTokenizeStart;
//   printLogLabel();
//   std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
//             << std::endl;
// }

// /// Load parameters into data container.
// void loadParameters(const std::string &paramFilePath,
//                     MemRef<uint16_t, 1> &params) {
//   const auto loadStart = std::chrono::high_resolution_clock::now();
//   std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
//   if (!paramFile.is_open()) {
//     throw std::runtime_error("[Error] Failed to open params file!");
//   }
//   printLogLabel();
//   std::cout << "Loading params..." << std::endl;
//   printLogLabel();
//   std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
//             << std::endl;
//   paramFile.read(reinterpret_cast<char *>(params.getData()),
//                  sizeof(uint16_t) * (params.getSize()));
//   if (paramFile.fail()) {
//     throw std::runtime_error("Error occurred while reading params file!");
//   }
//   paramFile.close();
//   const auto loadEnd = std::chrono::high_resolution_clock::now();
//   const std::chrono::duration<double, std::milli> loadTime =
//       loadEnd - loadStart;
//   printLogLabel();
//   std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
//             << "s\n"
//             << std::endl;
// }

// // f16 to f32 conversion function (IEEE 754 half precision -> single
// precision) float decode_f16(uint16_t h) {
//   uint16_t h_exp = (h & 0x7C00) >> 10;
//   uint16_t h_sig = h & 0x03FF;
//   uint16_t h_sign = h >> 15;

//   if (h_exp == 0) {
//     // subnormal
//     float f = std::ldexp((float)h_sig, -24);
//     return h_sign ? -f : f;
//   } else if (h_exp == 0x1F) {
//     // Inf/NaN
//     return h_sig ? std::numeric_limits<float>::quiet_NaN()
//                  : (h_sign ? -INFINITY : INFINITY);
//   } else {
//     float f = std::ldexp((float)(h_sig | 0x0400), h_exp - 25);
//     return h_sign ? -f : f;
//   }
// }

// int findMaxIndex(const uint16_t *start, size_t length) {
//   int maxIdx = 0;
//   float maxVal = decode_f16(start[0]);
//   for (int i = 1; i < (int)length; ++i) {
//     float val = decode_f16(start[i]);
//     if (val > maxVal) {
//       maxVal = val;
//       maxIdx = i;
//     }
//   }
//   return maxIdx;
// }

// void copy_kv_by_cache_position_block(const MemRefContainer &prefill,
//                                      MemRefContainer &decode,
//                                      int cache_position) {
//   constexpr int num_kv = 56;
//   int copy_len = std::min(cache_position, (int)MaxTokenLength);

//   for (int k = 0; k < num_kv; ++k) {
//     auto &src = *prefill.kv_ptrs[k];
//     auto &dst = *decode.kv_ptrs[k];

//     for (int h = 0; h < (int)HeadNum; ++h) {
//       size_t bytes_to_copy =
//           static_cast<size_t>(copy_len) * HiddenSize * sizeof(uint16_t);

//       uint16_t *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
//       uint16_t *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

//       std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
//     }
//   }
// }

// //
// -----------------------------------------------------------------------------
// // DeepSeekR1 Inference Main Entry
// //
// -----------------------------------------------------------------------------

// int main() {
//   /// Print the title of this example.
//   const std::string title = "DeepSeekR1 Inference Powered by Buddy Compiler";
//   std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

//   /// Define directories of vacabulary and parameter file.
//   std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
//   std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
//   const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
//   const std::string paramsDir = deepSeekR1BuildDir + "arg0-f16.data";

//   /// Get user message.
//   std::string inputStr;
//   getUserInput(inputStr);

//   /// Initialize data containers
//   //  - Input container.
//   //  - Result container
//   //  - Output container.
//   //  - Parameters container.
//   Text<size_t, 2> outputContainer;
//   Text<size_t, 2> inputContainerPrefill(inputStr);
//   MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
//   MemRef<uint16_t, 1> ParamsContainer({ParamsSize});
//   MemRef<long long, 1> cachePosition({1}, 0LL);

//   MemRef<uint16_t, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

//   MemRef<uint16_t, 4> kv0({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv1({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv2({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv3({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv4({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv5({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv6({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv7({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv8({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv9({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv10({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv11({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv12({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv13({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv14({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv15({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv16({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv17({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv18({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv19({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv20({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv21({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv22({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv23({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv24({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv25({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv26({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv27({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv28({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv29({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv30({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv31({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv32({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv33({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv34({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv35({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv36({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv37({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv38({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv39({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv40({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv41({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv42({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv43({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv44({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv45({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv46({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv47({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRef<uint16_t, 4> kv48({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv49({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv50({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv51({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv52({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv53({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv54({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
//   MemRef<uint16_t, 4> kv55({1, HeadNum, MaxTokenLength, HiddenSize}, 0);

//   MemRefContainer prefillResultContainer(
//       kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12,
//       kv13, kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24,
//       kv25, kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36,
//       kv37, kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48,
//       kv49, kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill);
//   MemRefContainer *ptrPrefillResultContainer = &prefillResultContainer;

//   /// Fill data into containers
//   //  - Input: register vocabulary and tokenize the input string.
//   //  - Output: register vocabulary.
//   //  - Parameters: load parameters from the `arg0` file into the container.
//   tokenizeInput(vocabDir, inputContainerPrefill);
//   outputContainer.loadVocab(vocabDir);
//   loadParameters(paramsDir, ParamsContainer);

//   /// Run DeepSeekR1 Inference
//   //  - Perform the forward function.
//   //  - Find and append the generated token.
//   //  - Continue iterating until the terminal condition is met.

//   double prefillTokensPerSec = 0.0;
//   const auto inferenceStart = std::chrono::high_resolution_clock::now();
//   _mlir_ciface_forward_prefill(ptrPrefillResultContainer, &ParamsContainer,
//                                &inputContainerPrefill);
//   const auto inferenceEnd = std::chrono::high_resolution_clock::now();
//   const std::chrono::duration<double, std::milli> inferenceTime =
//       inferenceEnd - inferenceStart;

//   int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
//   const uint16_t *startPtr =
//       ptrPrefillResultContainer->logits.getData() + tokenIndex *
//       MaxVocabSize;
//   int maxIndex = findMaxIndex(startPtr, MaxVocabSize);
//   std::string tok = inputContainerPrefill.getStr(maxIndex);
//   printIterInfo(0, tok, inferenceTime.count() / 1000);
//   const double prefillSeconds = inferenceTime.count() / 1000.0;
//   if (prefillSeconds > 0.0) {
//     prefillTokensPerSec = static_cast<double>(MaxTokenLength) /
//     prefillSeconds;
//   }
//   inputContainerDecode.getData()[0] = (long long)maxIndex;
//   outputContainer.appendTokenIdx(maxIndex);

//   MemRef<uint16_t, 3> logits_decode({1, 1, MaxVocabSize});

//   MemRefContainer decodeResultContainer(
//       kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12,
//       kv13, kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24,
//       kv25, kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36,
//       kv37, kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48,
//       kv49, kv50, kv51, kv52, kv53, kv54, kv55, logits_decode);

//   MemRefContainer *ptrDecodeResultContainer = &decodeResultContainer;

//   copy_kv_by_cache_position_block(prefillResultContainer,
//   decodeResultContainer,
//                                   inputContainerPrefill.getTokenCnt());

//   cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
//   int generateLen = MaxTokenLength - inputContainerPrefill.getTokenCnt();
//   double decodeTimeAccumMs = 0.0;
//   size_t decodeTokens = 0;
//   for (int i = 1; i <= generateLen; i++) {
//     const auto inferenceStart = std::chrono::high_resolution_clock::now();
//     _mlir_ciface_forward_decode(
//         ptrDecodeResultContainer, &ParamsContainer, &inputContainerDecode,
//         &cachePosition, &ptrDecodeResultContainer->kv0,
//         &ptrDecodeResultContainer->kv1, &ptrDecodeResultContainer->kv2,
//         &ptrDecodeResultContainer->kv3, &ptrDecodeResultContainer->kv4,
//         &ptrDecodeResultContainer->kv5, &ptrDecodeResultContainer->kv6,
//         &ptrDecodeResultContainer->kv7, &ptrDecodeResultContainer->kv8,
//         &ptrDecodeResultContainer->kv9, &ptrDecodeResultContainer->kv10,
//         &ptrDecodeResultContainer->kv11, &ptrDecodeResultContainer->kv12,
//         &ptrDecodeResultContainer->kv13, &ptrDecodeResultContainer->kv14,
//         &ptrDecodeResultContainer->kv15, &ptrDecodeResultContainer->kv16,
//         &ptrDecodeResultContainer->kv17, &ptrDecodeResultContainer->kv18,
//         &ptrDecodeResultContainer->kv19, &ptrDecodeResultContainer->kv20,
//         &ptrDecodeResultContainer->kv21, &ptrDecodeResultContainer->kv22,
//         &ptrDecodeResultContainer->kv23, &ptrDecodeResultContainer->kv24,
//         &ptrDecodeResultContainer->kv25, &ptrDecodeResultContainer->kv26,
//         &ptrDecodeResultContainer->kv27, &ptrDecodeResultContainer->kv28,
//         &ptrDecodeResultContainer->kv29, &ptrDecodeResultContainer->kv30,
//         &ptrDecodeResultContainer->kv31, &ptrDecodeResultContainer->kv32,
//         &ptrDecodeResultContainer->kv33, &ptrDecodeResultContainer->kv34,
//         &ptrDecodeResultContainer->kv35, &ptrDecodeResultContainer->kv36,
//         &ptrDecodeResultContainer->kv37, &ptrDecodeResultContainer->kv38,
//         &ptrDecodeResultContainer->kv39, &ptrDecodeResultContainer->kv40,
//         &ptrDecodeResultContainer->kv41, &ptrDecodeResultContainer->kv42,
//         &ptrDecodeResultContainer->kv43, &ptrDecodeResultContainer->kv44,
//         &ptrDecodeResultContainer->kv45, &ptrDecodeResultContainer->kv46,
//         &ptrDecodeResultContainer->kv47, &ptrDecodeResultContainer->kv48,
//         &ptrDecodeResultContainer->kv49, &ptrDecodeResultContainer->kv50,
//         &ptrDecodeResultContainer->kv51, &ptrDecodeResultContainer->kv52,
//         &ptrDecodeResultContainer->kv53, &ptrDecodeResultContainer->kv54,
//         &ptrDecodeResultContainer->kv55);

//     const auto inferenceEnd = std::chrono::high_resolution_clock::now();
//     const std::chrono::duration<double, std::milli> inferenceTime =
//         inferenceEnd - inferenceStart;
//     decodeTimeAccumMs += inferenceTime.count();
//     decodeTokens += 1;

//     // Determine the generated token.
//     const uint16_t *startPtr = ptrDecodeResultContainer->logits.getData();
//     maxIndex = findMaxIndex(startPtr, MaxVocabSize);
//     std::string tok = inputContainerPrefill.getStr(maxIndex);
//     // Print the generated token and inference time.
//     printIterInfo(i, tok, inferenceTime.count() / 1000);

//     // Stop if a <|end▁of▁sentence|> token is generated.
//     if (maxIndex == 151643) {
//       break;
//     }
//     // Append the generated token into the input and output container.
//     inputContainerDecode.getData()[0] = maxIndex;
//     outputContainer.appendTokenIdx(maxIndex);
//     cachePosition.getData()[0] += 1;
//   }

//   const double decodeSeconds = decodeTimeAccumMs / 1000.0;
//   const double decodeTokensPerSec =
//       decodeSeconds > 0.0 ? static_cast<double>(decodeTokens) / decodeSeconds
//                           : 0.0;

//   /// Print the final result
//   std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
//   std::cout << "\033[33;1m[Prefilling]\033[0m " << prefillTokensPerSec
//             << " tokens/s" << std::endl;
//   std::cout << "\033[33;1m[Decoding]\033[0m " << decodeTokensPerSec
//             << " tokens/s" << std::endl;
//   std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
//   std::cout << "\033[33;1m[Output]\033[0m "
//             << outputContainer.revertDeepSeekR1() << std::endl;

//   return 0;
// }

//===- buddy-deepseek-r1-bf16-main.cpp ------------------------------------===//
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace buddy;
double total_time = 0;
constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

// ============================================================================
// Pure return value containers (no extra members) to match MLIR layout exactly.
// ============================================================================

/// Prefill returns: 56 KV caches followed by logits.
struct PrefillReturns {
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
};

/// Decode returns: updated cache_position, then 27 groups of (kv, kv, dummy),
/// followed by the final two kvs and logits. Total 85 fields.
struct DecodeReturns {
  // First return value: updated cache_position (memref<1xi64>)
  MemRef<long long, 1> cache_position_out;

  // Group 1
  MemRef<uint16_t, 4> kv0;
  MemRef<uint16_t, 4> kv1;
  MemRef<long long, 1> ret_dummy0;
  // Group 2
  MemRef<uint16_t, 4> kv2;
  MemRef<uint16_t, 4> kv3;
  MemRef<long long, 1> ret_dummy1;
  // Group 3
  MemRef<uint16_t, 4> kv4;
  MemRef<uint16_t, 4> kv5;
  MemRef<long long, 1> ret_dummy2;
  // Group 4
  MemRef<uint16_t, 4> kv6;
  MemRef<uint16_t, 4> kv7;
  MemRef<long long, 1> ret_dummy3;
  // Group 5
  MemRef<uint16_t, 4> kv8;
  MemRef<uint16_t, 4> kv9;
  MemRef<long long, 1> ret_dummy4;
  // Group 6
  MemRef<uint16_t, 4> kv10;
  MemRef<uint16_t, 4> kv11;
  MemRef<long long, 1> ret_dummy5;
  // Group 7
  MemRef<uint16_t, 4> kv12;
  MemRef<uint16_t, 4> kv13;
  MemRef<long long, 1> ret_dummy6;
  // Group 8
  MemRef<uint16_t, 4> kv14;
  MemRef<uint16_t, 4> kv15;
  MemRef<long long, 1> ret_dummy7;
  // Group 9
  MemRef<uint16_t, 4> kv16;
  MemRef<uint16_t, 4> kv17;
  MemRef<long long, 1> ret_dummy8;
  // Group 10
  MemRef<uint16_t, 4> kv18;
  MemRef<uint16_t, 4> kv19;
  MemRef<long long, 1> ret_dummy9;
  // Group 11
  MemRef<uint16_t, 4> kv20;
  MemRef<uint16_t, 4> kv21;
  MemRef<long long, 1> ret_dummy10;
  // Group 12
  MemRef<uint16_t, 4> kv22;
  MemRef<uint16_t, 4> kv23;
  MemRef<long long, 1> ret_dummy11;
  // Group 13
  MemRef<uint16_t, 4> kv24;
  MemRef<uint16_t, 4> kv25;
  MemRef<long long, 1> ret_dummy12;
  // Group 14
  MemRef<uint16_t, 4> kv26;
  MemRef<uint16_t, 4> kv27;
  MemRef<long long, 1> ret_dummy13;
  // Group 15
  MemRef<uint16_t, 4> kv28;
  MemRef<uint16_t, 4> kv29;
  MemRef<long long, 1> ret_dummy14;
  // Group 16
  MemRef<uint16_t, 4> kv30;
  MemRef<uint16_t, 4> kv31;
  MemRef<long long, 1> ret_dummy15;
  // Group 17
  MemRef<uint16_t, 4> kv32;
  MemRef<uint16_t, 4> kv33;
  MemRef<long long, 1> ret_dummy16;
  // Group 18
  MemRef<uint16_t, 4> kv34;
  MemRef<uint16_t, 4> kv35;
  MemRef<long long, 1> ret_dummy17;
  // Group 19
  MemRef<uint16_t, 4> kv36;
  MemRef<uint16_t, 4> kv37;
  MemRef<long long, 1> ret_dummy18;
  // Group 20
  MemRef<uint16_t, 4> kv38;
  MemRef<uint16_t, 4> kv39;
  MemRef<long long, 1> ret_dummy19;
  // Group 21
  MemRef<uint16_t, 4> kv40;
  MemRef<uint16_t, 4> kv41;
  MemRef<long long, 1> ret_dummy20;
  // Group 22
  MemRef<uint16_t, 4> kv42;
  MemRef<uint16_t, 4> kv43;
  MemRef<long long, 1> ret_dummy21;
  // Group 23
  MemRef<uint16_t, 4> kv44;
  MemRef<uint16_t, 4> kv45;
  MemRef<long long, 1> ret_dummy22;
  // Group 24
  MemRef<uint16_t, 4> kv46;
  MemRef<uint16_t, 4> kv47;
  MemRef<long long, 1> ret_dummy23;
  // Group 25
  MemRef<uint16_t, 4> kv48;
  MemRef<uint16_t, 4> kv49;
  MemRef<long long, 1> ret_dummy24;
  // Group 26
  MemRef<uint16_t, 4> kv50;
  MemRef<uint16_t, 4> kv51;
  MemRef<long long, 1> ret_dummy25;
  // Group 27
  MemRef<uint16_t, 4> kv52;
  MemRef<uint16_t, 4> kv53;
  MemRef<long long, 1> ret_dummy26;
  // Group 28 (no dummy)
  MemRef<uint16_t, 4> kv54;
  MemRef<uint16_t, 4> kv55;
  // Logits
  MemRef<uint16_t, 3> logits;
};

// Type alias for a pointer array to all 56 KV fields.
using KVPtrArray = std::array<MemRef<uint16_t, 4> *, 56>;

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
                                             MemRef<uint16_t, 1> *arg0,
                                             Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    DecodeReturns *result, MemRef<uint16_t, 1> *arg0,
    MemRef<long long, 2> *arg1, MemRef<long long, 1> *arg2,
    // Group 1
    MemRef<uint16_t, 4> *kv0, MemRef<uint16_t, 4> *kv1,
    MemRef<long long, 1> *dummy0,
    // Group 2
    MemRef<uint16_t, 4> *kv2, MemRef<uint16_t, 4> *kv3,
    MemRef<long long, 1> *dummy1,
    // Group 3
    MemRef<uint16_t, 4> *kv4, MemRef<uint16_t, 4> *kv5,
    MemRef<long long, 1> *dummy2,
    // Group 4
    MemRef<uint16_t, 4> *kv6, MemRef<uint16_t, 4> *kv7,
    MemRef<long long, 1> *dummy3,
    // Group 5
    MemRef<uint16_t, 4> *kv8, MemRef<uint16_t, 4> *kv9,
    MemRef<long long, 1> *dummy4,
    // Group 6
    MemRef<uint16_t, 4> *kv10, MemRef<uint16_t, 4> *kv11,
    MemRef<long long, 1> *dummy5,
    // Group 7
    MemRef<uint16_t, 4> *kv12, MemRef<uint16_t, 4> *kv13,
    MemRef<long long, 1> *dummy6,
    // Group 8
    MemRef<uint16_t, 4> *kv14, MemRef<uint16_t, 4> *kv15,
    MemRef<long long, 1> *dummy7,
    // Group 9
    MemRef<uint16_t, 4> *kv16, MemRef<uint16_t, 4> *kv17,
    MemRef<long long, 1> *dummy8,
    // Group 10
    MemRef<uint16_t, 4> *kv18, MemRef<uint16_t, 4> *kv19,
    MemRef<long long, 1> *dummy9,
    // Group 11
    MemRef<uint16_t, 4> *kv20, MemRef<uint16_t, 4> *kv21,
    MemRef<long long, 1> *dummy10,
    // Group 12
    MemRef<uint16_t, 4> *kv22, MemRef<uint16_t, 4> *kv23,
    MemRef<long long, 1> *dummy11,
    // Group 13
    MemRef<uint16_t, 4> *kv24, MemRef<uint16_t, 4> *kv25,
    MemRef<long long, 1> *dummy12,
    // Group 14
    MemRef<uint16_t, 4> *kv26, MemRef<uint16_t, 4> *kv27,
    MemRef<long long, 1> *dummy13,
    // Group 15
    MemRef<uint16_t, 4> *kv28, MemRef<uint16_t, 4> *kv29,
    MemRef<long long, 1> *dummy14,
    // Group 16
    MemRef<uint16_t, 4> *kv30, MemRef<uint16_t, 4> *kv31,
    MemRef<long long, 1> *dummy15,
    // Group 17
    MemRef<uint16_t, 4> *kv32, MemRef<uint16_t, 4> *kv33,
    MemRef<long long, 1> *dummy16,
    // Group 18
    MemRef<uint16_t, 4> *kv34, MemRef<uint16_t, 4> *kv35,
    MemRef<long long, 1> *dummy17,
    // Group 19
    MemRef<uint16_t, 4> *kv36, MemRef<uint16_t, 4> *kv37,
    MemRef<long long, 1> *dummy18,
    // Group 20
    MemRef<uint16_t, 4> *kv38, MemRef<uint16_t, 4> *kv39,
    MemRef<long long, 1> *dummy19,
    // Group 21
    MemRef<uint16_t, 4> *kv40, MemRef<uint16_t, 4> *kv41,
    MemRef<long long, 1> *dummy20,
    // Group 22
    MemRef<uint16_t, 4> *kv42, MemRef<uint16_t, 4> *kv43,
    MemRef<long long, 1> *dummy21,
    // Group 23
    MemRef<uint16_t, 4> *kv44, MemRef<uint16_t, 4> *kv45,
    MemRef<long long, 1> *dummy22,
    // Group 24
    MemRef<uint16_t, 4> *kv46, MemRef<uint16_t, 4> *kv47,
    MemRef<long long, 1> *dummy23,
    // Group 25
    MemRef<uint16_t, 4> *kv48, MemRef<uint16_t, 4> *kv49,
    MemRef<long long, 1> *dummy24,
    // Group 26
    MemRef<uint16_t, 4> *kv50, MemRef<uint16_t, 4> *kv51,
    MemRef<long long, 1> *dummy25,
    // Group 27
    MemRef<uint16_t, 4> *kv52, MemRef<uint16_t, 4> *kv53,
    MemRef<long long, 1> *dummy26,
    // Group 28 (no dummy)
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

// bf16 to f32 conversion function (Brain floating point -> single precision)
float decode_bf16(uint16_t h) {
  uint32_t f32_bits = static_cast<uint32_t>(h) << 16;
  float out;
  std::memcpy(&out, &f32_bits, sizeof(out));
  return out;
}

int findMaxIndex(const uint16_t *start, size_t length) {
  int maxIdx = 0;
  float maxVal = decode_bf16(start[0]);
  for (int i = 1; i < (int)length; ++i) {
    float val = decode_bf16(start[i]);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
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
      size_t bytes_to_copy = copy_len * HiddenSize * sizeof(uint16_t);

      uint16_t *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      uint16_t *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 BF16 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title =
      "DeepSeekR1 BF16 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vocabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0-bf16.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<uint16_t, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<uint16_t, 3> logits_prefill({1, MaxTokenLength, MaxVocabSize});

  // Helper lambda to create a zero-initialized KV MemRef.
  auto makeKV = []() {
    return MemRef<uint16_t, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  };

  MemRef<uint16_t, 4> kv0 = makeKV();
  MemRef<uint16_t, 4> kv1 = makeKV();
  MemRef<uint16_t, 4> kv2 = makeKV();
  MemRef<uint16_t, 4> kv3 = makeKV();
  MemRef<uint16_t, 4> kv4 = makeKV();
  MemRef<uint16_t, 4> kv5 = makeKV();
  MemRef<uint16_t, 4> kv6 = makeKV();
  MemRef<uint16_t, 4> kv7 = makeKV();
  MemRef<uint16_t, 4> kv8 = makeKV();
  MemRef<uint16_t, 4> kv9 = makeKV();
  MemRef<uint16_t, 4> kv10 = makeKV();
  MemRef<uint16_t, 4> kv11 = makeKV();
  MemRef<uint16_t, 4> kv12 = makeKV();
  MemRef<uint16_t, 4> kv13 = makeKV();
  MemRef<uint16_t, 4> kv14 = makeKV();
  MemRef<uint16_t, 4> kv15 = makeKV();
  MemRef<uint16_t, 4> kv16 = makeKV();
  MemRef<uint16_t, 4> kv17 = makeKV();
  MemRef<uint16_t, 4> kv18 = makeKV();
  MemRef<uint16_t, 4> kv19 = makeKV();
  MemRef<uint16_t, 4> kv20 = makeKV();
  MemRef<uint16_t, 4> kv21 = makeKV();
  MemRef<uint16_t, 4> kv22 = makeKV();
  MemRef<uint16_t, 4> kv23 = makeKV();
  MemRef<uint16_t, 4> kv24 = makeKV();
  MemRef<uint16_t, 4> kv25 = makeKV();
  MemRef<uint16_t, 4> kv26 = makeKV();
  MemRef<uint16_t, 4> kv27 = makeKV();
  MemRef<uint16_t, 4> kv28 = makeKV();
  MemRef<uint16_t, 4> kv29 = makeKV();
  MemRef<uint16_t, 4> kv30 = makeKV();
  MemRef<uint16_t, 4> kv31 = makeKV();
  MemRef<uint16_t, 4> kv32 = makeKV();
  MemRef<uint16_t, 4> kv33 = makeKV();
  MemRef<uint16_t, 4> kv34 = makeKV();
  MemRef<uint16_t, 4> kv35 = makeKV();
  MemRef<uint16_t, 4> kv36 = makeKV();
  MemRef<uint16_t, 4> kv37 = makeKV();
  MemRef<uint16_t, 4> kv38 = makeKV();
  MemRef<uint16_t, 4> kv39 = makeKV();
  MemRef<uint16_t, 4> kv40 = makeKV();
  MemRef<uint16_t, 4> kv41 = makeKV();
  MemRef<uint16_t, 4> kv42 = makeKV();
  MemRef<uint16_t, 4> kv43 = makeKV();
  MemRef<uint16_t, 4> kv44 = makeKV();
  MemRef<uint16_t, 4> kv45 = makeKV();
  MemRef<uint16_t, 4> kv46 = makeKV();
  MemRef<uint16_t, 4> kv47 = makeKV();
  MemRef<uint16_t, 4> kv48 = makeKV();
  MemRef<uint16_t, 4> kv49 = makeKV();
  MemRef<uint16_t, 4> kv50 = makeKV();
  MemRef<uint16_t, 4> kv51 = makeKV();
  MemRef<uint16_t, 4> kv52 = makeKV();
  MemRef<uint16_t, 4> kv53 = makeKV();
  MemRef<uint16_t, 4> kv54 = makeKV();
  MemRef<uint16_t, 4> kv55 = makeKV();

  // Initialize Prefill returns (aggregate initialization).
  PrefillReturns prefillRet = {
      kv0,  kv1,  kv2,  kv3,  kv4,  kv5,  kv6,           kv7,  kv8,  kv9,
      kv10, kv11, kv12, kv13, kv14, kv15, kv16,          kv17, kv18, kv19,
      kv20, kv21, kv22, kv23, kv24, kv25, kv26,          kv27, kv28, kv29,
      kv30, kv31, kv32, kv33, kv34, kv35, kv36,          kv37, kv38, kv39,
      kv40, kv41, kv42, kv43, kv44, kv45, kv46,          kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logits_prefill};

  /// Fill data into containers
  tokenizeInput(vocabDir, inputContainerPrefill);
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  /// Run DeepSeekR1 Inference - Prefill phase
  double prefillTokensPerSec = 0.0;
  const auto inferenceStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(&prefillRet, &ParamsContainer,
                               &inputContainerPrefill);
  const auto inferenceEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> inferenceTime =
      inferenceEnd - inferenceStart;

  int tokenIndex = inputContainerPrefill.getTokenCnt() - 1;
  const uint16_t *startPtr =
      prefillRet.logits.getData() + tokenIndex * MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, MaxVocabSize);
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
  MemRef<uint16_t, 3> logits_decode({1, 1, MaxVocabSize});
  DecodeReturns decodeRet = {
      MemRef<long long, 1>({1}, 0LL), // cache_position_out
      kv0,
      kv1,
      MemRef<long long, 1>({1}, 0LL), // Group 1
      kv2,
      kv3,
      MemRef<long long, 1>({1}, 0LL), // Group 2
      kv4,
      kv5,
      MemRef<long long, 1>({1}, 0LL), // Group 3
      kv6,
      kv7,
      MemRef<long long, 1>({1}, 0LL), // Group 4
      kv8,
      kv9,
      MemRef<long long, 1>({1}, 0LL), // Group 5
      kv10,
      kv11,
      MemRef<long long, 1>({1}, 0LL), // Group 6
      kv12,
      kv13,
      MemRef<long long, 1>({1}, 0LL), // Group 7
      kv14,
      kv15,
      MemRef<long long, 1>({1}, 0LL), // Group 8
      kv16,
      kv17,
      MemRef<long long, 1>({1}, 0LL), // Group 9
      kv18,
      kv19,
      MemRef<long long, 1>({1}, 0LL), // Group 10
      kv20,
      kv21,
      MemRef<long long, 1>({1}, 0LL), // Group 11
      kv22,
      kv23,
      MemRef<long long, 1>({1}, 0LL), // Group 12
      kv24,
      kv25,
      MemRef<long long, 1>({1}, 0LL), // Group 13
      kv26,
      kv27,
      MemRef<long long, 1>({1}, 0LL), // Group 14
      kv28,
      kv29,
      MemRef<long long, 1>({1}, 0LL), // Group 15
      kv30,
      kv31,
      MemRef<long long, 1>({1}, 0LL), // Group 16
      kv32,
      kv33,
      MemRef<long long, 1>({1}, 0LL), // Group 17
      kv34,
      kv35,
      MemRef<long long, 1>({1}, 0LL), // Group 18
      kv36,
      kv37,
      MemRef<long long, 1>({1}, 0LL), // Group 19
      kv38,
      kv39,
      MemRef<long long, 1>({1}, 0LL), // Group 20
      kv40,
      kv41,
      MemRef<long long, 1>({1}, 0LL), // Group 21
      kv42,
      kv43,
      MemRef<long long, 1>({1}, 0LL), // Group 22
      kv44,
      kv45,
      MemRef<long long, 1>({1}, 0LL), // Group 23
      kv46,
      kv47,
      MemRef<long long, 1>({1}, 0LL), // Group 24
      kv48,
      kv49,
      MemRef<long long, 1>({1}, 0LL), // Group 25
      kv50,
      kv51,
      MemRef<long long, 1>({1}, 0LL), // Group 26
      kv52,
      kv53,
      MemRef<long long, 1>({1}, 0LL), // Group 27
      kv54,
      kv55, // Group 28 (no dummy)
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

  /// Decode loop.
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

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    // Determine the generated token.
    const uint16_t *startPtr = decodeRet.logits.getData();
    maxIndex = findMaxIndex(startPtr, MaxVocabSize);
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
