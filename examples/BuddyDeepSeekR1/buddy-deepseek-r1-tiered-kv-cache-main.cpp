//===- buddy-deepseek-r1-tiered-kv-cache-main.cpp -------------------------===//
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
//
// This is the tiered KV cache version of DeepSeekR1 inference runtime.
// It dynamically selects the appropriate prefill and decode subgraph based on
// the input sequence length and current KV cache position to minimize
// computation waste.
//
// Supported cache sizes for both prefill and decode: 32, 64, 128, 256, 512,
// 1024
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
#include <functional>
#include <iostream>
#include <sys/time.h>

using namespace buddy;

double total_time = 0;
constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t NUM_LAYERS = 56;
constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;

// Supported KV cache sizes (must match generated subgraphs)
constexpr std::array<size_t, 6> KV_CACHE_SIZES = {32, 64, 128, 256, 512, 1024};

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

// Template MemRefContainer for different cache sizes and logits sequence
// lengths CacheLen: KV cache dimension (used in attention) LogitsSeqLen: output
// logits sequence length (CacheLen for prefill, 1 for decode)
template <size_t CacheLen, size_t LogitsSeqLen = 1> struct MemRefContainerT {
  MemRef<float, 4> kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7;
  MemRef<float, 4> kv8, kv9, kv10, kv11, kv12, kv13, kv14, kv15;
  MemRef<float, 4> kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23;
  MemRef<float, 4> kv24, kv25, kv26, kv27, kv28, kv29, kv30, kv31;
  MemRef<float, 4> kv32, kv33, kv34, kv35, kv36, kv37, kv38, kv39;
  MemRef<float, 4> kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47;
  MemRef<float, 4> kv48, kv49, kv50, kv51, kv52, kv53, kv54, kv55;
  MemRef<float, 3> logits;
  std::array<MemRef<float, 4> *, 56> kv_ptrs;

  MemRefContainerT()
      : kv0({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv1({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv2({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv3({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv4({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv5({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv6({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv7({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv8({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv9({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv10({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv11({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv12({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv13({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv14({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv15({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv16({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv17({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv18({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv19({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv20({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv21({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv22({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv23({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv24({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv25({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv26({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv27({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv28({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv29({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv30({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv31({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv32({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv33({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv34({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv35({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv36({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv37({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv38({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv39({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv40({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv41({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv42({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv43({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv44({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv45({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv46({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv47({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv48({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv49({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv50({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv51({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv52({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv53({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv54({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv55({1, HeadNum, CacheLen, HiddenSize}, 0),
        logits({1, LogitsSeqLen, MaxVocabSize}),
        kv_ptrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,
                &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,
                &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,
                &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,
                &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,
                &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,
                &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}

  static constexpr size_t getCacheLen() { return CacheLen; }
  static constexpr size_t getLogitsSeqLen() { return LogitsSeqLen; }
};

// Original MemRefContainer for backward compatibility (uses MaxTokenLength)
using MemRefContainer = MemRefContainerT<MaxTokenLength, MaxTokenLength>;

// Prefill containers (logits seq_len = cache_len)
using PrefillContainer32 = MemRefContainerT<32, 32>;
using PrefillContainer64 = MemRefContainerT<64, 64>;
using PrefillContainer128 = MemRefContainerT<128, 128>;
using PrefillContainer256 = MemRefContainerT<256, 256>;
using PrefillContainer512 = MemRefContainerT<512, 512>;
using PrefillContainer1024 = MemRefContainerT<1024, 1024>;

// Decode containers (logits seq_len = 1)
using DecodeContainer32 = MemRefContainerT<32, 1>;
using DecodeContainer64 = MemRefContainerT<64, 1>;
using DecodeContainer128 = MemRefContainerT<128, 1>;
using DecodeContainer256 = MemRefContainerT<256, 1>;
using DecodeContainer512 = MemRefContainerT<512, 1>;
using DecodeContainer1024 = MemRefContainerT<1024, 1>;

// Declare prefill functions for each cache size
#define DECLARE_PREFILL_FUNC(SIZE)                                             \
  extern "C" void _mlir_ciface_forward_prefill_##SIZE(                         \
      PrefillContainer##SIZE *result, MemRef<float, 1> *arg0,                  \
      Text<size_t, 2> *arg1)

DECLARE_PREFILL_FUNC(32);
DECLARE_PREFILL_FUNC(64);
DECLARE_PREFILL_FUNC(128);
DECLARE_PREFILL_FUNC(256);
DECLARE_PREFILL_FUNC(512);
DECLARE_PREFILL_FUNC(1024);

#undef DECLARE_PREFILL_FUNC

// Declare decode functions for each cache size
#define DECLARE_DECODE_FUNC(SIZE)                                              \
  extern "C" void _mlir_ciface_forward_decode_##SIZE(                          \
      DecodeContainer##SIZE *result, MemRef<float, 1> *arg0,                   \
      MemRef<long long, 2> *arg1, MemRef<long long, 1> *arg2,                  \
      MemRef<float, 4> *kv0, MemRef<float, 4> *kv1, MemRef<float, 4> *kv2,     \
      MemRef<float, 4> *kv3, MemRef<float, 4> *kv4, MemRef<float, 4> *kv5,     \
      MemRef<float, 4> *kv6, MemRef<float, 4> *kv7, MemRef<float, 4> *kv8,     \
      MemRef<float, 4> *kv9, MemRef<float, 4> *kv10, MemRef<float, 4> *kv11,   \
      MemRef<float, 4> *kv12, MemRef<float, 4> *kv13, MemRef<float, 4> *kv14,  \
      MemRef<float, 4> *kv15, MemRef<float, 4> *kv16, MemRef<float, 4> *kv17,  \
      MemRef<float, 4> *kv18, MemRef<float, 4> *kv19, MemRef<float, 4> *kv20,  \
      MemRef<float, 4> *kv21, MemRef<float, 4> *kv22, MemRef<float, 4> *kv23,  \
      MemRef<float, 4> *kv24, MemRef<float, 4> *kv25, MemRef<float, 4> *kv26,  \
      MemRef<float, 4> *kv27, MemRef<float, 4> *kv28, MemRef<float, 4> *kv29,  \
      MemRef<float, 4> *kv30, MemRef<float, 4> *kv31, MemRef<float, 4> *kv32,  \
      MemRef<float, 4> *kv33, MemRef<float, 4> *kv34, MemRef<float, 4> *kv35,  \
      MemRef<float, 4> *kv36, MemRef<float, 4> *kv37, MemRef<float, 4> *kv38,  \
      MemRef<float, 4> *kv39, MemRef<float, 4> *kv40, MemRef<float, 4> *kv41,  \
      MemRef<float, 4> *kv42, MemRef<float, 4> *kv43, MemRef<float, 4> *kv44,  \
      MemRef<float, 4> *kv45, MemRef<float, 4> *kv46, MemRef<float, 4> *kv47,  \
      MemRef<float, 4> *kv48, MemRef<float, 4> *kv49, MemRef<float, 4> *kv50,  \
      MemRef<float, 4> *kv51, MemRef<float, 4> *kv52, MemRef<float, 4> *kv53,  \
      MemRef<float, 4> *kv54, MemRef<float, 4> *kv55)

DECLARE_DECODE_FUNC(32);
DECLARE_DECODE_FUNC(64);
DECLARE_DECODE_FUNC(128);
DECLARE_DECODE_FUNC(256);
DECLARE_DECODE_FUNC(512);
DECLARE_DECODE_FUNC(1024);

#undef DECLARE_DECODE_FUNC

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
void printIterInfo(size_t iterIdx, std::string str, double time,
                   size_t cacheSize = 0) {
  total_time += time;
  std::cout << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cout << "Token: " << str << " | Time: " << time << "s";
  if (cacheSize > 0) {
    std::cout << " | Cache: " << cacheSize;
  }
  std::cout << std::endl;
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

/// Select appropriate cache size based on current position (for decode stage).
size_t selectCacheSize(size_t currentPos) {
  for (size_t size : KV_CACHE_SIZES) {
    if (currentPos < size) {
      return size;
    }
  }
  return KV_CACHE_SIZES.back();
}

/// Select appropriate prefill size based on actual token count.
size_t selectPrefillSize(size_t tokenCount) {
  for (size_t size : KV_CACHE_SIZES) {
    if (tokenCount <= size) {
      return size;
    }
  }
  return KV_CACHE_SIZES.back();
}

/// Copy KV cache from source to destination container.
/// Works across different container types (prefill/decode) with potentially
/// different cache sizes and logits sequence lengths.
template <size_t SrcLen, size_t SrcLogitsLen, size_t DstLen,
          size_t DstLogitsLen>
void copyKVCache(const MemRefContainerT<SrcLen, SrcLogitsLen> &src,
                 MemRefContainerT<DstLen, DstLogitsLen> &dst,
                 size_t validTokens) {
  constexpr int num_kv = 56;
  size_t copy_len = std::min({validTokens, SrcLen, DstLen});

  for (int k = 0; k < num_kv; ++k) {
    auto &src_kv = *src.kv_ptrs[k];
    auto &dst_kv = *dst.kv_ptrs[k];

    for (size_t h = 0; h < HeadNum; ++h) {
      size_t bytes_to_copy = copy_len * HiddenSize * sizeof(float);
      float *src_ptr = src_kv.getData() + h * SrcLen * HiddenSize;
      float *dst_ptr = dst_kv.getData() + h * DstLen * HiddenSize;
      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

// Macro to generate prefill call for specific size
#define CALL_PREFILL(SIZE, container, paramsPtr, inputPtr)                     \
  _mlir_ciface_forward_prefill_##SIZE(&container, paramsPtr, inputPtr)

// Macro to generate decode call for specific size
#define CALL_DECODE(SIZE, container, paramsPtr, inputPtr, cachePtr)            \
  _mlir_ciface_forward_decode_##SIZE(                                          \
      &container, paramsPtr, inputPtr, cachePtr, &container.kv0,               \
      &container.kv1, &container.kv2, &container.kv3, &container.kv4,          \
      &container.kv5, &container.kv6, &container.kv7, &container.kv8,          \
      &container.kv9, &container.kv10, &container.kv11, &container.kv12,       \
      &container.kv13, &container.kv14, &container.kv15, &container.kv16,      \
      &container.kv17, &container.kv18, &container.kv19, &container.kv20,      \
      &container.kv21, &container.kv22, &container.kv23, &container.kv24,      \
      &container.kv25, &container.kv26, &container.kv27, &container.kv28,      \
      &container.kv29, &container.kv30, &container.kv31, &container.kv32,      \
      &container.kv33, &container.kv34, &container.kv35, &container.kv36,      \
      &container.kv37, &container.kv38, &container.kv39, &container.kv40,      \
      &container.kv41, &container.kv42, &container.kv43, &container.kv44,      \
      &container.kv45, &container.kv46, &container.kv47, &container.kv48,      \
      &container.kv49, &container.kv50, &container.kv51, &container.kv52,      \
      &container.kv53, &container.kv54, &container.kv55)

// -----------------------------------------------------------------------------
// DeepSeekR1 Tiered KV Cache Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title =
      "DeepSeekR1 Tiered KV Cache Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;
  std::cout << "Supported cache sizes: ";
  for (size_t i = 0; i < KV_CACHE_SIZES.size(); ++i) {
    std::cout << KV_CACHE_SIZES[i];
    if (i < KV_CACHE_SIZES.size() - 1)
      std::cout << ", ";
  }
  std::cout << std::endl;

  /// Define directories of vocabulary and parameter file.
  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0_mc.data";

  /// Get user message.
  std::string inputStr;
  getUserInput(inputStr);

  /// Initialize data containers
  Text<size_t, 2> outputContainer;
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  // Create prefill containers for each cache size (logits seq_len = cache_len)
  PrefillContainer32 prefillContainer32;
  PrefillContainer64 prefillContainer64;
  PrefillContainer128 prefillContainer128;
  PrefillContainer256 prefillContainer256;
  PrefillContainer512 prefillContainer512;
  PrefillContainer1024 prefillContainer1024;

  // Create decode containers for each cache size (logits seq_len = 1)
  DecodeContainer32 decodeContainer32;
  DecodeContainer64 decodeContainer64;
  DecodeContainer128 decodeContainer128;
  DecodeContainer256 decodeContainer256;
  DecodeContainer512 decodeContainer512;
  DecodeContainer1024 decodeContainer1024;

  /// Load vocab first to count tokens
  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  // Tokenize with maximum length first to get token count
  Text<size_t, 2> inputContainerPrefill(inputStr);
  inputContainerPrefill.loadVocab(vocabDir);

  // Perform tokenization to get actual token count
  tokenizeInput(vocabDir, inputContainerPrefill);
  size_t actualTokenCount = inputContainerPrefill.getTokenCnt();

  // Select appropriate prefill size based on actual token count
  size_t selectedPrefillSize = selectPrefillSize(actualTokenCount);
  printLogLabel();
  std::cout << "Actual token count: " << actualTokenCount
            << ", selected prefill size: " << selectedPrefillSize << std::endl;

  // Re-tokenize with the selected prefill size
  Text<size_t, 2> inputContainerTiered(inputStr);
  inputContainerTiered.tokenizeDeepSeekR1(vocabDir, selectedPrefillSize);

  /// Run prefill with dynamically selected size
  double prefillTokensPerSec = 0.0;
  printLogLabel();
  std::cout << "Running prefill with size " << selectedPrefillSize << "..."
            << std::endl;

  const float *prefillLogitsPtr = nullptr;

  const auto prefillStart = std::chrono::high_resolution_clock::now();

  switch (selectedPrefillSize) {
  case 32:
    CALL_PREFILL(32, prefillContainer32, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer32.logits.getData();
    break;
  case 64:
    CALL_PREFILL(64, prefillContainer64, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer64.logits.getData();
    break;
  case 128:
    CALL_PREFILL(128, prefillContainer128, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer128.logits.getData();
    break;
  case 256:
    CALL_PREFILL(256, prefillContainer256, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer256.logits.getData();
    break;
  case 512:
    CALL_PREFILL(512, prefillContainer512, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer512.logits.getData();
    break;
  case 1024:
    CALL_PREFILL(1024, prefillContainer1024, &ParamsContainer,
                 &inputContainerTiered);
    prefillLogitsPtr = prefillContainer1024.logits.getData();
    break;
  default:
    throw std::runtime_error("Unsupported prefill size");
  }

  const auto prefillEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> prefillTime =
      prefillEnd - prefillStart;

  int tokenIndex = inputContainerTiered.getTokenCnt() - 1;
  const float *startPtr = prefillLogitsPtr + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);

  std::string tok = inputContainerTiered.getStr(maxIndex);
  printIterInfo(0, tok, prefillTime.count() / 1000, selectedPrefillSize);
  const double prefillSeconds = prefillTime.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    prefillTokensPerSec =
        static_cast<double>(selectedPrefillSize) / prefillSeconds;
  }
  inputContainerDecode.getData()[0] = (long long)maxIndex;
  outputContainer.appendTokenIdx(maxIndex);

  /// Initialize cache position
  size_t currentPos = inputContainerTiered.getTokenCnt();
  cachePosition.getData()[0] = currentPos;

  /// Select initial decode cache size and copy KV cache from prefill
  size_t currentCacheSize = selectedPrefillSize;
  printLogLabel();
  std::cout << "Initial cache position: " << currentPos
            << ", selected decode cache size: " << currentCacheSize
            << std::endl;

  // Copy prefill KV cache to the matching decode container
  switch (selectedPrefillSize) {
  case 32:
    copyKVCache(prefillContainer32, decodeContainer32, currentPos);
    break;
  case 64:
    copyKVCache(prefillContainer64, decodeContainer64, currentPos);
    break;
  case 128:
    copyKVCache(prefillContainer128, decodeContainer128, currentPos);
    break;
  case 256:
    copyKVCache(prefillContainer256, decodeContainer256, currentPos);
    break;
  case 512:
    copyKVCache(prefillContainer512, decodeContainer512, currentPos);
    break;
  case 1024:
    copyKVCache(prefillContainer1024, decodeContainer1024, currentPos);
    break;
  }

  /// Decode loop with dynamic cache size selection
  int generateLen = MaxTokenLength - inputContainerTiered.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  size_t prevCacheSize = currentCacheSize;

  for (int i = 1; i <= generateLen; i++) {
    // Check if we need to switch to a larger cache size
    size_t neededCacheSize = selectCacheSize(currentPos + 1);

    if (neededCacheSize != prevCacheSize) {
      printLogLabel();
      std::cout << "Switching cache size from " << prevCacheSize << " to "
                << neededCacheSize << " at position " << currentPos
                << std::endl;

      // Copy KV cache from the current container to the new larger container
      // The source is determined by prevCacheSize
      switch (prevCacheSize) {
      case 32:
        switch (neededCacheSize) {
        case 64:
          copyKVCache(decodeContainer32, decodeContainer64, currentPos);
          break;
        case 128:
          copyKVCache(decodeContainer32, decodeContainer128, currentPos);
          break;
        case 256:
          copyKVCache(decodeContainer32, decodeContainer256, currentPos);
          break;
        case 512:
          copyKVCache(decodeContainer32, decodeContainer512, currentPos);
          break;
        case 1024:
          copyKVCache(decodeContainer32, decodeContainer1024, currentPos);
          break;
        }
        break;
      case 64:
        switch (neededCacheSize) {
        case 128:
          copyKVCache(decodeContainer64, decodeContainer128, currentPos);
          break;
        case 256:
          copyKVCache(decodeContainer64, decodeContainer256, currentPos);
          break;
        case 512:
          copyKVCache(decodeContainer64, decodeContainer512, currentPos);
          break;
        case 1024:
          copyKVCache(decodeContainer64, decodeContainer1024, currentPos);
          break;
        }
        break;
      case 128:
        switch (neededCacheSize) {
        case 256:
          copyKVCache(decodeContainer128, decodeContainer256, currentPos);
          break;
        case 512:
          copyKVCache(decodeContainer128, decodeContainer512, currentPos);
          break;
        case 1024:
          copyKVCache(decodeContainer128, decodeContainer1024, currentPos);
          break;
        }
        break;
      case 256:
        switch (neededCacheSize) {
        case 512:
          copyKVCache(decodeContainer256, decodeContainer512, currentPos);
          break;
        case 1024:
          copyKVCache(decodeContainer256, decodeContainer1024, currentPos);
          break;
        }
        break;
      case 512:
        switch (neededCacheSize) {
        case 1024:
          copyKVCache(decodeContainer512, decodeContainer1024, currentPos);
          break;
        }
        break;
      }
      prevCacheSize = neededCacheSize;
    }

    currentCacheSize = neededCacheSize;
    const float *logitsPtr = nullptr;

    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    // Call the appropriate decode function based on current cache size
    switch (currentCacheSize) {
    case 32:
      CALL_DECODE(32, decodeContainer32, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer32.logits.getData();
      break;
    case 64:
      CALL_DECODE(64, decodeContainer64, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer64.logits.getData();
      break;
    case 128:
      CALL_DECODE(128, decodeContainer128, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer128.logits.getData();
      break;
    case 256:
      CALL_DECODE(256, decodeContainer256, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer256.logits.getData();
      break;
    case 512:
      CALL_DECODE(512, decodeContainer512, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer512.logits.getData();
      break;
    case 1024:
      CALL_DECODE(1024, decodeContainer1024, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer1024.logits.getData();
      break;
    default:
      throw std::runtime_error("Unsupported cache size");
    }

    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inferenceTime =
        inferenceEnd - inferenceStart;
    decodeTimeAccumMs += inferenceTime.count();
    decodeTokens += 1;

    // Determine the generated token
    endPtr = logitsPtr + MaxVocabSize;
    maxIndex = findMaxIndex(logitsPtr, endPtr);
    tok = inputContainerTiered.getStr(maxIndex);

    // Print the generated token with cache size info
    printIterInfo(i, tok, inferenceTime.count() / 1000, currentCacheSize);

    // Stop if a <|end▁of▁sentence|> token is generated
    if (maxIndex == 151643) {
      break;
    }

    // Update for next iteration
    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    currentPos += 1;
    cachePosition.getData()[0] = currentPos;
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
