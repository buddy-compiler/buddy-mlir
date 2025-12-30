//===- buddy-deepseek-r1-main.cpp -----------------------------------------===//
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
#include <buddy/Core/Container.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ===== Operator Timing Infrastructure =====

// Timing data structure
struct TimingRecord {
  std::string op_name;
  std::vector<double> times_ms;

  void add_time(double time_sec) {
    times_ms.push_back(time_sec * 1000.0); // Convert to milliseconds
  }

  double get_total() const {
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
  }
};

// Global timing data storage
static std::map<std::string, TimingRecord> g_timing_data;

// Timing functions called from MLIR
extern "C" {
// Get current time in seconds
double rtclock() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// MLIR C interface wrapper for rtclock
double _mlir_ciface_rtclock() { return rtclock(); }

// Record timing for an operator
void record_timing(const char *op_name, double duration_sec) {
  std::string name(op_name);
  g_timing_data[name].op_name = name;
  g_timing_data[name].add_time(duration_sec);
}

// MLIR C interface wrapper for record_timing
void _mlir_ciface_record_timing(void *op_name_ptr, double duration_sec) {
  const char *op_name = reinterpret_cast<const char *>(op_name_ptr);
  record_timing(op_name, duration_sec);
}
}

void print_timing_report() {
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "     Operator Timing Report\n";
  std::cout << "========================================\n";
  std::cout << std::fixed << std::setprecision(4);

  // compute total time
  double total_time = 0.0;
  for (const auto &[name, record] : g_timing_data) {
    total_time += record.get_total();
  }

  // print table header
  std::cout << std::left << std::setw(30) << "Operator" << std::right
            << std::setw(16) << "Total (ms)" << std::setw(12) << "% Total"
            << "\n";
  std::cout << "----------------------------------------"
            << "------------------------------\n";

  // print each operator time
  for (const auto &[name, record] : g_timing_data) {
    double total = record.get_total();
    double percentage = (total_time > 0) ? (total / total_time * 100.0) : 0.0;

    std::cout << std::left << std::setw(30) << name << std::right
              << std::setw(16) << total << std::setw(11) << percentage << "%\n";
  }

  // print total time
  std::cout << "----------------------------------------"
            << "------------------------------\n";
  std::cout << std::left << std::setw(30) << "TOTAL" << std::right
            << std::setw(16) << total_time << std::setw(11) << "100.0%\n";
  std::cout << "========================================\n\n";
}

// Clear timing data (for warmup)
void clear_timing_data() { g_timing_data.clear(); }

// ===== End of Timing Infrastructure =====

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
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

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

/// Declare DeepSeekR1 forward function.
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

/// Find the index of the max value.
int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
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
          static_cast<size_t>(copy_len) * HiddenSize * sizeof(float);

      float *src_ptr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dst_ptr = dst.getData() + h * MaxTokenLength * HiddenSize;

      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// -----------------------------------------------------------------------------
// DeepSeekR1 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "DeepSeekR1 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Define directories of vacabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0.data";

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
  MemRef<float, 1> ParamsContainer({ParamsSize});
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
  const float *startPtr =
      ptrPrefillResultContainer->logits.getData() + tokenIndex * MaxVocabSize;
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

  MemRef<float, 3> logits_decode({1, 1, MaxVocabSize});

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
    const float *startPtr = ptrDecodeResultContainer->logits.getData();
    const float *endPtr = startPtr + MaxVocabSize;
    maxIndex = findMaxIndex(startPtr, endPtr);
    std::string tok = inputContainerPrefill.getStr(maxIndex);
    // Print the generated token and inference time.
    printIterInfo(i, tok, inferenceTime.count() / 1000);

    print_timing_report();
    clear_timing_data();

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
