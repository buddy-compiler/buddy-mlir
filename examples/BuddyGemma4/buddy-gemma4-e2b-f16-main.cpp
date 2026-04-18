//===- buddy-gemma4-e2b-f16-main.cpp
//---------------------------------------===//
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
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ===== Operator Timing Infrastructure =====

struct TimingRecord {
  std::string op_name;
  std::vector<double> times_ms;

  void add_time(double time_sec) { times_ms.push_back(time_sec * 1000.0); }

  double get_total() const {
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
  }
};

static std::map<std::string, TimingRecord> g_timing_data;

extern "C" {
double rtclock() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void record_timing(const char *op_name, double duration_sec) {
  std::string name(op_name);
  g_timing_data[name].op_name = name;
  g_timing_data[name].add_time(duration_sec);
}

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

  double total_time = 0.0;
  for (const auto &[name, record] : g_timing_data)
    total_time += record.get_total();

  std::cout << std::left << std::setw(30) << "Operator" << std::right
            << std::setw(16) << "Total (ms)" << std::setw(12) << "% Total"
            << "\n";
  std::cout << "----------------------------------------"
            << "------------------------------\n";

  for (const auto &[name, record] : g_timing_data) {
    double total = record.get_total();
    double percentage = (total_time > 0) ? (total / total_time * 100.0) : 0.0;
    std::cout << std::left << std::setw(30) << name << std::right
              << std::setw(16) << total << std::setw(11) << percentage << "%\n";
  }

  std::cout << "----------------------------------------"
            << "------------------------------\n";
  std::cout << std::left << std::setw(30) << "TOTAL" << std::right
            << std::setw(16) << total_time << std::setw(11) << "100.0%\n";
  std::cout << "========================================\n\n";
}

void clear_timing_data() { g_timing_data.clear(); }

// ===== End of Timing Infrastructure =====

#include <buddy/LLM/TextContainer.h>
#include <cstddef>
#include <filesystem>
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

struct MemRefContainer {
  // Group 0 (sliding layers 0-3 + full layer 0)
  MemRef<uint16_t, 4> sk0k, sk0v;
  MemRef<long long, 1> cl0;
  MemRef<uint16_t, 4> sk1k, sk1v;
  MemRef<long long, 1> cl1;
  MemRef<uint16_t, 4> sk2k, sk2v;
  MemRef<long long, 1> cl2;
  MemRef<uint16_t, 4> sk3k, sk3v;
  MemRef<long long, 1> cl3;
  MemRef<long long, 1> fcl0;
  MemRef<uint16_t, 4> fk0k, fk0v;
  // Group 1 (sliding layers 4-7 + full layer 1)
  MemRef<uint16_t, 4> sk4k, sk4v;
  MemRef<long long, 1> cl4;
  MemRef<uint16_t, 4> sk5k, sk5v;
  MemRef<long long, 1> cl5;
  MemRef<uint16_t, 4> sk6k, sk6v;
  MemRef<long long, 1> cl6;
  MemRef<uint16_t, 4> sk7k, sk7v;
  MemRef<long long, 1> cl7;
  MemRef<long long, 1> fcl1;
  MemRef<uint16_t, 4> fk1k, fk1v;
  // Group 2 (sliding layers 8-11 + full layer 2)
  MemRef<uint16_t, 4> sk8k, sk8v;
  MemRef<long long, 1> cl8;
  MemRef<uint16_t, 4> sk9k, sk9v;
  MemRef<long long, 1> cl9;
  MemRef<uint16_t, 4> sk10k, sk10v;
  MemRef<long long, 1> cl10;
  MemRef<uint16_t, 4> sk11k, sk11v;
  MemRef<long long, 1> cl11;
  MemRef<long long, 1> fcl2;
  MemRef<uint16_t, 4> fk2k, fk2v;
  // Logits
  MemRef<uint16_t, 3> logits;
};

extern "C" void _mlir_ciface_forward_prefill(
    MemRefContainer *result, MemRef<uint16_t, 1> *params,
    Text<size_t, 2> *input_ids, MemRef<long long, 1> *cl0,
    MemRef<uint16_t, 4> *sk0k, MemRef<uint16_t, 4> *sk0v,
    MemRef<long long, 1> *cl1, MemRef<uint16_t, 4> *sk1k,
    MemRef<uint16_t, 4> *sk1v, MemRef<long long, 1> *cl2,
    MemRef<uint16_t, 4> *sk2k, MemRef<uint16_t, 4> *sk2v,
    MemRef<long long, 1> *cl3, MemRef<uint16_t, 4> *sk3k,
    MemRef<uint16_t, 4> *sk3v, MemRef<long long, 1> *fcl0,
    MemRef<uint16_t, 4> *fk0k, MemRef<uint16_t, 4> *fk0v,
    MemRef<long long, 1> *cl4, MemRef<uint16_t, 4> *sk4k,
    MemRef<uint16_t, 4> *sk4v, MemRef<long long, 1> *cl5,
    MemRef<uint16_t, 4> *sk5k, MemRef<uint16_t, 4> *sk5v,
    MemRef<long long, 1> *cl6, MemRef<uint16_t, 4> *sk6k,
    MemRef<uint16_t, 4> *sk6v, MemRef<long long, 1> *cl7,
    MemRef<uint16_t, 4> *sk7k, MemRef<uint16_t, 4> *sk7v,
    MemRef<long long, 1> *fcl1, MemRef<uint16_t, 4> *fk1k,
    MemRef<uint16_t, 4> *fk1v, MemRef<long long, 1> *cl8,
    MemRef<uint16_t, 4> *sk8k, MemRef<uint16_t, 4> *sk8v,
    MemRef<long long, 1> *cl9, MemRef<uint16_t, 4> *sk9k,
    MemRef<uint16_t, 4> *sk9v, MemRef<long long, 1> *cl10,
    MemRef<uint16_t, 4> *sk10k, MemRef<uint16_t, 4> *sk10v,
    MemRef<long long, 1> *cl11, MemRef<uint16_t, 4> *sk11k,
    MemRef<uint16_t, 4> *sk11v, MemRef<long long, 1> *fcl2,
    MemRef<uint16_t, 4> *fk2k, MemRef<uint16_t, 4> *fk2v);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<uint16_t, 1> *params,
    MemRef<long long, 2> *input_ids, MemRef<long long, 1> *cache_position,
    MemRef<uint16_t, 4> *sk0k, MemRef<uint16_t, 4> *sk0v,
    MemRef<long long, 1> *cl0, MemRef<uint16_t, 4> *sk1k,
    MemRef<uint16_t, 4> *sk1v, MemRef<long long, 1> *cl1,
    MemRef<uint16_t, 4> *sk2k, MemRef<uint16_t, 4> *sk2v,
    MemRef<long long, 1> *cl2, MemRef<uint16_t, 4> *sk3k,
    MemRef<uint16_t, 4> *sk3v, MemRef<long long, 1> *cl3,
    MemRef<uint16_t, 4> *fk0k, MemRef<uint16_t, 4> *fk0v,
    MemRef<long long, 1> *fcl0, MemRef<uint16_t, 4> *sk4k,
    MemRef<uint16_t, 4> *sk4v, MemRef<long long, 1> *cl4,
    MemRef<uint16_t, 4> *sk5k, MemRef<uint16_t, 4> *sk5v,
    MemRef<long long, 1> *cl5, MemRef<uint16_t, 4> *sk6k,
    MemRef<uint16_t, 4> *sk6v, MemRef<long long, 1> *cl6,
    MemRef<uint16_t, 4> *sk7k, MemRef<uint16_t, 4> *sk7v,
    MemRef<long long, 1> *cl7, MemRef<uint16_t, 4> *fk1k,
    MemRef<uint16_t, 4> *fk1v, MemRef<long long, 1> *fcl1,
    MemRef<uint16_t, 4> *sk8k, MemRef<uint16_t, 4> *sk8v,
    MemRef<long long, 1> *cl8, MemRef<uint16_t, 4> *sk9k,
    MemRef<uint16_t, 4> *sk9v, MemRef<long long, 1> *cl9,
    MemRef<uint16_t, 4> *sk10k, MemRef<uint16_t, 4> *sk10v,
    MemRef<long long, 1> *cl10, MemRef<uint16_t, 4> *sk11k,
    MemRef<uint16_t, 4> *sk11v, MemRef<long long, 1> *cl11,
    MemRef<uint16_t, 4> *fk2k, MemRef<uint16_t, 4> *fk2v);

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
                 sizeof(uint16_t) * params.getSize());
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

// IEEE 754 half precision -> single precision decode for argmax
float decode_f16(uint16_t h) {
  uint16_t h_exp = (h & 0x7C00) >> 10;
  uint16_t h_sig = h & 0x03FF;
  uint16_t h_sign = h >> 15;
  if (h_exp == 0) {
    float f = std::ldexp((float)h_sig, -24);
    return h_sign ? -f : f;
  } else if (h_exp == 0x1F) {
    return h_sig ? std::numeric_limits<float>::quiet_NaN()
                 : (h_sign ? -INFINITY : INFINITY);
  } else {
    float f = std::ldexp((float)(h_sig | 0x0400), h_exp - 25);
    return h_sign ? -f : f;
  }
}

int findMaxIndex(const uint16_t *start, size_t length) {
  int maxIdx = 0;
  float maxVal = decode_f16(start[0]);
  for (size_t i = 1; i < length; ++i) {
    float val = decode_f16(start[i]);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = (int)i;
    }
  }
  return maxIdx;
}

void copyKVFromPrefill(MemRefContainer &prefill, MemRefContainer &decode,
                       int cachePos) {
  auto copyOne = [&](MemRef<uint16_t, 4> &dst, MemRef<uint16_t, 4> &src) {
    int headDim = dst.getSizes()[3];
    int copyLen = std::min(cachePos, (int)dst.getSizes()[2]);
    for (int h = 0; h < (int)KVHeadNum; h++) {
      std::memcpy(dst.getData() + h * dst.getSizes()[2] * headDim,
                  src.getData() + h * src.getSizes()[2] * headDim,
                  sizeof(uint16_t) * copyLen * headDim);
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

#define SLIDING_KV {1, KVHeadNum, SlidingCacheLen, SlidingHeadDim}
#define FULL_KV {1, KVHeadNum, FullCacheLen, FullHeadDim}

// -----------------------------------------------------------------------------
// Gemma4-E2B F16 Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  const std::string title =
      "Gemma4-E2B F16 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  std::string exampleDir = GEMMA4_E2B_EXAMPLE_PATH;
  std::string buildDir = GEMMA4_E2B_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = buildDir + "vocab.txt";
  const std::string paramsDir = buildDir + "arg0_e2b-f16.data";

  std::string inputStr;
  getUserInput(inputStr);

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<uint16_t, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<uint16_t, 3> logitsPrefill({1, MaxTokenLength, MaxVocabSize});

  // KV cache buffers for prefill input (zero-initialized)
  MemRef<uint16_t, 4> sk0k(SLIDING_KV, (uint16_t)0),
      sk0v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk1k(SLIDING_KV, (uint16_t)0),
      sk1v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk2k(SLIDING_KV, (uint16_t)0),
      sk2v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk3k(SLIDING_KV, (uint16_t)0),
      sk3v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk4k(SLIDING_KV, (uint16_t)0),
      sk4v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk5k(SLIDING_KV, (uint16_t)0),
      sk5v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk6k(SLIDING_KV, (uint16_t)0),
      sk6v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk7k(SLIDING_KV, (uint16_t)0),
      sk7v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk8k(SLIDING_KV, (uint16_t)0),
      sk8v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk9k(SLIDING_KV, (uint16_t)0),
      sk9v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk10k(SLIDING_KV, (uint16_t)0),
      sk10v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> sk11k(SLIDING_KV, (uint16_t)0),
      sk11v(SLIDING_KV, (uint16_t)0);
  MemRef<uint16_t, 4> fk0k(FULL_KV, (uint16_t)0), fk0v(FULL_KV, (uint16_t)0);
  MemRef<uint16_t, 4> fk1k(FULL_KV, (uint16_t)0), fk1v(FULL_KV, (uint16_t)0);
  MemRef<uint16_t, 4> fk2k(FULL_KV, (uint16_t)0), fk2v(FULL_KV, (uint16_t)0);
  MemRef<long long, 1> cl0({1}, 0LL), cl1({1}, 0LL), cl2({1}, 0LL),
      cl3({1}, 0LL), cl4({1}, 0LL), cl5({1}, 0LL), cl6({1}, 0LL), cl7({1}, 0LL),
      cl8({1}, 0LL), cl9({1}, 0LL), cl10({1}, 0LL), cl11({1}, 0LL),
      fcl0({1}, 0LL), fcl1({1}, 0LL), fcl2({1}, 0LL);

  MemRefContainer prefillResultContainer = {{SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {{1}, 0LL},
                                            {FULL_KV, (uint16_t)0},
                                            {FULL_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {{1}, 0LL},
                                            {FULL_KV, (uint16_t)0},
                                            {FULL_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {SLIDING_KV, (uint16_t)0},
                                            {SLIDING_KV, (uint16_t)0},
                                            {{1}, 0LL},
                                            {{1}, 0LL},
                                            {FULL_KV, (uint16_t)0},
                                            {FULL_KV, (uint16_t)0},
                                            logitsPrefill};

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
  const uint16_t *startPtr =
      prefillResultContainer.logits.getData() + tokenIndex * MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, MaxVocabSize);

  std::string tok = inputContainerPrefill.getStr(maxIndex);
  printIterInfo(0, tok, inferenceTime.count() / 1000);

  const double prefillSeconds = inferenceTime.count() / 1000.0;
  if (prefillSeconds > 0.0)
    prefillTokensPerSec = static_cast<double>(MaxTokenLength) / prefillSeconds;
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  // ---- Copy KV from prefill to decode ----
  MemRef<uint16_t, 3> logitsDecode({1, 1, MaxVocabSize});
  MemRefContainer decodeResultContainer = {{SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {{1}, 0LL},
                                           {FULL_KV, (uint16_t)0},
                                           {FULL_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {{1}, 0LL},
                                           {FULL_KV, (uint16_t)0},
                                           {FULL_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {SLIDING_KV, (uint16_t)0},
                                           {SLIDING_KV, (uint16_t)0},
                                           {{1}, 0LL},
                                           {{1}, 0LL},
                                           {FULL_KV, (uint16_t)0},
                                           {FULL_KV, (uint16_t)0},
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

    const uint16_t *logitStart = ptrDecodeResultContainer->logits.getData();
    maxIndex = findMaxIndex(logitStart, MaxVocabSize);
    tok = inputContainerPrefill.getStr(maxIndex);
    printIterInfo(i, tok, decTime.count() / 1000);

    print_timing_report();
    clear_timing_data();

    if (maxIndex == 1 || maxIndex == 106) {
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
