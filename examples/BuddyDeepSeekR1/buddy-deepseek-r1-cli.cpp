//===- buddy-deepseek-r1-cli.cpp ------------------------------------------===//
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
#include <array>
#include <atomic>
#include <buddy/Core/Container.h>
#include <buddy/LLM/ChatTemplate.h>
#include <buddy/LLM/ConversationManager.h>
#include <buddy/LLM/TextContainer.h>
#include <buddy/runtime/llm/Sampler.h>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace buddy;
namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_receivedSigInt(false);

void signalHandler(int signal) {
  if (signal == SIGINT) {
    g_receivedSigInt = true;
  }
}

constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;
constexpr long long DefaultEosToken = 151643;
constexpr float RopeTheta = 10000.0f;

using RopeFreqArray = std::array<float, HiddenSize / 2>;

// ============================================================================
// Pure return value containers (no extra members) to match MLIR layout exactly.
// ============================================================================

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
  // Final two kvs (no dummy)
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;
  // Logits
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
                                             MemRef<float, 1> *arg0,
                                             Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    DecodeReturns *result, MemRef<float, 1> *arg0, MemRef<long long, 2> *arg1,
    MemRef<long long, 1> *arg2, MemRef<float, 4> *kv0, MemRef<float, 4> *kv1,
    MemRef<long long, 1> *dummy0, MemRef<float, 4> *kv2, MemRef<float, 4> *kv3,
    MemRef<long long, 1> *dummy1, MemRef<float, 4> *kv4, MemRef<float, 4> *kv5,
    MemRef<long long, 1> *dummy2, MemRef<float, 4> *kv6, MemRef<float, 4> *kv7,
    MemRef<long long, 1> *dummy3, MemRef<float, 4> *kv8, MemRef<float, 4> *kv9,
    MemRef<long long, 1> *dummy4, MemRef<float, 4> *kv10,
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

// ============================================================================
// Command line options.
// ============================================================================

static llvm::cl::opt<std::string>
    ModelPathOpt("model",
                 llvm::cl::desc("Path to the model parameter file (arg0.data)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    VocabPathOpt("vocab",
                 llvm::cl::desc("Path to the vocabulary file (vocab.txt)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptOpt("prompt", llvm::cl::desc("Prompt text passed directly"),
              llvm::cl::value_desc("text"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptFileOpt("prompt-file", llvm::cl::desc("File containing prompt text"),
                  llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    InteractiveOpt("interactive",
                   llvm::cl::desc("Start REPL-style interactive mode (combine "
                                  "with --prompt for a system prompt)"),
                   llvm::cl::init(false));

static llvm::cl::opt<unsigned>
    KeepTokenNumOpt("keep-token-num",
                    llvm::cl::desc("Number of cached tokens to keep when the "
                                   "decode context hits the maximum length"),
                    llvm::cl::init(static_cast<unsigned>(MaxTokenLength / 4)));

static llvm::cl::opt<unsigned>
    MaxTokensOpt("max-tokens",
                 llvm::cl::desc("Maximum number of tokens to generate "
                                "(including the first decoded token)"),
                 llvm::cl::init(1024));

static llvm::cl::opt<double>
    TemperatureOpt("temperature",
                   llvm::cl::desc("Sampling temperature (0.0 = greedy)"),
                   llvm::cl::init(0.0));

static llvm::cl::opt<int>
    TopKOpt("top-k", llvm::cl::desc("Top-K candidates (0 = disabled)"),
            llvm::cl::init(0));

static llvm::cl::opt<double>
    TopPOpt("top-p",
            llvm::cl::desc("Nucleus sampling threshold (1.0 = disabled)"),
            llvm::cl::init(1.0));

static llvm::cl::opt<double>
    MinPOpt("min-p", llvm::cl::desc("Min-P threshold (0.0 = disabled)"),
            llvm::cl::init(0.0));

static llvm::cl::opt<double>
    RepeatPenaltyOpt("repeat-penalty",
                     llvm::cl::desc("Repetition penalty (1.0 = disabled)"),
                     llvm::cl::init(1.0));

static llvm::cl::opt<int>
    RepeatLastNOpt("repeat-last-n",
                   llvm::cl::desc("Repeat penalty window size"),
                   llvm::cl::init(64));

static llvm::cl::opt<unsigned long long>
    SeedOpt("seed", llvm::cl::desc("Random seed for sampling (0 = random)"),
            llvm::cl::init(0));

static llvm::cl::opt<long long>
    EosIdOpt("eos-id", llvm::cl::desc("ID of the end-of-sequence token"),
             llvm::cl::init(DefaultEosToken));

static llvm::cl::opt<bool> SuppressStatsOpt(
    "no-stats",
    llvm::cl::desc("Output text only and hide performance statistics"),
    llvm::cl::init(false));

llvm::raw_ostream &getInfoStream() {
  if (SuppressStatsOpt) {
    return llvm::nulls();
  }
  return llvm::errs();
}

static llvm::cl::opt<float>
    RopeThetaOpt("rope-theta",
                 llvm::cl::desc("RoPE Theta value (default 10000.0 for "
                                "DeepSeek/Qwen, 1000000.0 for long context)"),
                 llvm::cl::init(10000.0f));

static llvm::cl::opt<std::string>
    ChatTemplateOpt("chat-template",
                    llvm::cl::desc("Path to chat template JSON config"),
                    llvm::cl::value_desc("path"), llvm::cl::init(""));

// ============================================================================
// Helper functions.
// ============================================================================

/// Copies KV cache from prefill to decode container using the provided pointer
/// arrays.
void copyKVByCachePositionBlock(const KVPtrArray &prefillPtrs,
                                const KVPtrArray &decodePtrs,
                                int cachePosition) {
  constexpr int numKV = 56;
  const size_t copyLen = std::min<size_t>(static_cast<size_t>(cachePosition),
                                          static_cast<size_t>(MaxTokenLength));

  for (int k = 0; k < numKV; ++k) {
    auto &src = *prefillPtrs[k];
    auto &dst = *decodePtrs[k];

    for (int h = 0; h < static_cast<int>(HeadNum); ++h) {
      const size_t bytesToCopy = copyLen * HiddenSize * sizeof(float);
      float *srcPtr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dstPtr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dstPtr, srcPtr, bytesToCopy);
    }
  }
}

/// Discards tokens from KV cache to maintain fixed window size.
void discardKVByCachePositionBlock(const KVPtrArray &decodePtrs,
                                   int keepTokenNum, int discardLen,
                                   int currentTokenCount) {
  constexpr int numKV = 56;
  const int maxTokens = static_cast<int>(MaxTokenLength);
  const int headCount = static_cast<int>(HeadNum);
  currentTokenCount = std::clamp(currentTokenCount, 0, maxTokens);

  if (discardLen <= 0 || keepTokenNum < 0 ||
      keepTokenNum >= currentTokenCount) {
    llvm::errs()
        << "[Error] discardKVByCachePositionBlock: invalid parameters.\n";
    return;
  }

  const int srcStartIndex = keepTokenNum + discardLen;
  if (srcStartIndex >= currentTokenCount) {
    llvm::errs() << "[Error] discardKVByCachePositionBlock: srcStartIndex ("
                 << srcStartIndex << ") >= currentTokenCount ("
                 << currentTokenCount
                 << "). This indicates an invalid discard length.\n";
    return;
  }

  const int validTailTokens = currentTokenCount - srcStartIndex;

  for (int k = 0; k < numKV; ++k) {
    auto &kv = *decodePtrs[k];
    for (int h = 0; h < headCount; ++h) {
      float *headBase =
          kv.getData() + static_cast<size_t>(h) * MaxTokenLength * HiddenSize;

      float *dstPtr = headBase + static_cast<size_t>(keepTokenNum) * HiddenSize;
      float *srcPtr =
          headBase + static_cast<size_t>(srcStartIndex) * HiddenSize;

      const size_t bytesToMove =
          static_cast<size_t>(validTailTokens) * HiddenSize * sizeof(float);

      std::memmove(dstPtr, srcPtr, bytesToMove);

      float *clearPtr =
          dstPtr + static_cast<size_t>(validTailTokens) * HiddenSize;
      const int clearTokens = maxTokens - (keepTokenNum + validTailTokens);
      if (clearTokens > 0) {
        const size_t bytesToClear =
            static_cast<size_t>(clearTokens) * HiddenSize * sizeof(float);
        std::memset(clearPtr, 0, bytesToClear);
      }
    }
  }
}

void applyRotaryDeltaToSlice(const KVPtrArray &decodePtrs, int startToken,
                             int tokenCount, const RopeFreqArray &cosValues,
                             const RopeFreqArray &sinValues) {
  if (tokenCount <= 0)
    return;
  constexpr int numKV = 56;
  const size_t headStride = MaxTokenLength * HiddenSize;
  const size_t halfSize = HiddenSize / 2;
  for (int idx = 0; idx < numKV; idx += 2) {
    for (int h = 0; h < static_cast<int>(HeadNum); ++h) {
      float *headBase =
          decodePtrs[idx]->getData() + static_cast<size_t>(h) * headStride;
      for (int t = 0; t < tokenCount; ++t) {
        float *tokenPtr =
            headBase + static_cast<size_t>(startToken + t) * HiddenSize;
        for (size_t i = 0; i < halfSize; ++i) {
          const float val1 = tokenPtr[i];
          const float val2 = tokenPtr[i + halfSize];
          tokenPtr[i] = val1 * cosValues[i] - val2 * sinValues[i];
          tokenPtr[i + halfSize] = val1 * sinValues[i] + val2 * cosValues[i];
        }
      }
    }
  }
}

/// Adjusts RoPE for cached keys after token discard.
void adjustKeyCacheRope(const KVPtrArray &decodePtrs, int keepTokenNum,
                        int discardLen, int currentTokenCount,
                        const RopeFreqArray &inverseFreqs) {
  if (discardLen <= 0)
    return;
  const int srcStartIndex = keepTokenNum + discardLen;
  if (srcStartIndex >= currentTokenCount)
    return;
  const int tokenCount = currentTokenCount - srcStartIndex;
  RopeFreqArray cosValues{};
  RopeFreqArray sinValues{};
  // Rotate backwards to align keys with new positions.
  const float delta = -static_cast<float>(discardLen);
  for (size_t i = 0; i < inverseFreqs.size(); ++i) {
    const float angle = inverseFreqs[i] * delta;
    cosValues[i] = static_cast<float>(std::cos(angle));
    sinValues[i] = static_cast<float>(std::sin(angle));
  }

  // StaticCache stores key/value pairs; even indices correspond to key caches.
  applyRotaryDeltaToSlice(decodePtrs, keepTokenNum, tokenCount, cosValues,
                          sinValues);
}

std::string readPromptFromFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open prompt file: " + filePath);
  }
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

std::string readPromptFromStdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

void loadParameters(const std::string &paramFilePath,
                    MemRef<float, 1> &params) {
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("Failed to open model parameter file: " +
                             paramFilePath);
  }
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (paramFile.fail()) {
    throw std::runtime_error("Failed to read model parameters: " +
                             paramFilePath);
  }
}

struct GenerationResult {
  size_t promptTokens = 0;
  size_t generatedTokens = 0;
  double prefillTokensPerSec = 0.0;
  double decodeTokensPerSec = 0.0;
  double totalSeconds = 0.0;
  std::string finalText;
};

std::string getDefaultModelPath() {
#ifdef DEEPSEEKR1_EXAMPLE_BUILD_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_BUILD_PATH) + "arg0.data";
#else
  return "arg0.data";
#endif
}

std::string getDefaultVocabPath() {
#ifdef DEEPSEEKR1_EXAMPLE_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_PATH) + "vocab.txt";
#else
  return "vocab.txt";
#endif
}

void streamNewText(Text<size_t, 2> &outputContainer, std::string &lastPrinted,
                   std::ostream &tokenStream) {
  std::string current = outputContainer.revertDeepSeekR1();
  if (current.size() > lastPrinted.size()) {
    tokenStream.write(current.data() + lastPrinted.size(),
                      current.size() - lastPrinted.size());
    tokenStream.flush();
  }
  lastPrinted = std::move(current);
}

RopeFreqArray buildInverseRopeFreqs(float theta) {
  RopeFreqArray inverseFreqs{};
  for (size_t i = 0; i < inverseFreqs.size(); ++i) {
    const float exponent =
        (2.0f * static_cast<float>(i)) / static_cast<float>(HiddenSize);
    inverseFreqs[i] = 1.0f / static_cast<float>(std::pow(theta, exponent));
  }
  return inverseFreqs;
}

GenerationResult runGeneration(const std::string &prompt,
                               MemRef<float, 1> &paramsContainer,
                               const std::string &vocabPath, int maxNewTokens,
                               const std::vector<long long> &stopTokenIds,
                               std::ostream &tokenStream, Sampler &sampler) {
  GenerationResult stats;
  const RopeFreqArray ropeInverseFreqs =
      buildInverseRopeFreqs(RopeThetaOpt.getValue());

  auto isStopToken = [&](int tokenId) {
    return std::find(stopTokenIds.begin(), stopTokenIds.end(), tokenId) !=
           stopTokenIds.end();
  };

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(prompt);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logitsPrefill({1, MaxTokenLength, MaxVocabSize});

  auto makeKV = []() {
    return MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  };

  // Allocate KV cache buffers.
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
      kv0,  kv1,  kv2,  kv3,  kv4,  kv5,  kv6,          kv7,  kv8,  kv9,
      kv10, kv11, kv12, kv13, kv14, kv15, kv16,         kv17, kv18, kv19,
      kv20, kv21, kv22, kv23, kv24, kv25, kv26,         kv27, kv28, kv29,
      kv30, kv31, kv32, kv33, kv34, kv35, kv36,         kv37, kv38, kv39,
      kv40, kv41, kv42, kv43, kv44, kv45, kv46,         kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logitsPrefill};

  outputContainer.loadVocab(vocabPath);
  inputContainerPrefill.tokenizeDeepSeekR1(vocabPath, MaxTokenLength);
  if (inputContainerPrefill.getTokenCnt() > MaxTokenLength) {
    llvm::errs() << "[Error] Token count "
                 << inputContainerPrefill.getTokenCnt()
                 << " exceeds MaxTokenLength " << MaxTokenLength << ".\n";
    return stats;
  }
  if (inputContainerPrefill.getTokenCnt() == 0) {
    tokenStream << std::endl;
    stats.finalText.clear();
    return stats;
  }
  stats.promptTokens = MaxTokenLength;

  const auto prefillStart = std::chrono::high_resolution_clock::now();
  _mlir_ciface_forward_prefill(&prefillRet, &paramsContainer,
                               &inputContainerPrefill);
  const auto prefillEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> prefillMs =
      prefillEnd - prefillStart;
  const double prefillSeconds = prefillMs.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    stats.prefillTokensPerSec =
        static_cast<double>(MaxTokenLength) / prefillSeconds;
  }

  KVPtrArray prefillPtrs = buildPrefillKVPtrs(prefillRet);

  std::string streamed;
  std::vector<int> recentTokens;

  // Sample first token from prefill logits.
  const int tokenIndex =
      static_cast<int>(inputContainerPrefill.getTokenCnt()) - 1;
  const float *startPtr =
      prefillRet.logits.getData() + tokenIndex * MaxVocabSize;
  int maxIndex = sampler.sample(startPtr, MaxVocabSize, recentTokens);
  recentTokens.push_back(maxIndex);

  // Initialize Decode returns.
  MemRef<float, 3> logitsDecode({1, 1, MaxVocabSize});
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
      logitsDecode};

  KVPtrArray decodePtrs = buildDecodeKVPtrs(decodeRet);

  // Copy KV cache from prefill to decode.
  copyKVByCachePositionBlock(
      prefillPtrs, decodePtrs,
      static_cast<int>(inputContainerPrefill.getTokenCnt()));

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  inputContainerDecode.getData()[0] = static_cast<long long>(maxIndex);
  if (isStopToken(maxIndex)) {
    tokenStream << std::endl;
    stats.totalSeconds = prefillSeconds;
    stats.finalText = streamed;
    return stats;
  }
  outputContainer.appendTokenIdx(maxIndex);
  streamNewText(outputContainer, streamed, tokenStream);
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  const int keepTokenNum =
      std::clamp(static_cast<int>(KeepTokenNumOpt.getValue()), 0,
                 static_cast<int>(MaxTokenLength));

  // Decode loop.
  while (InteractiveOpt || cachePosition.getData()[0] < maxNewTokens) {
    // Discard tokens if max context length reached.
    if (cachePosition.getData()[0] >= static_cast<long long>(MaxTokenLength)) {
      const int currentTokens =
          std::min(static_cast<int>(cachePosition.getData()[0]),
                   static_cast<int>(MaxTokenLength));
      int discardTokenNum = std::max(1, (currentTokens - keepTokenNum) / 2);
      discardKVByCachePositionBlock(decodePtrs, keepTokenNum, discardTokenNum,
                                    currentTokens);
      adjustKeyCacheRope(decodePtrs, keepTokenNum, discardTokenNum,
                         currentTokens, ropeInverseFreqs);
      const long long newLength = currentTokens - discardTokenNum;
      cachePosition.getData()[0] =
          std::clamp(newLength, 0LL, static_cast<long long>(MaxTokenLength));
    }

    if (g_receivedSigInt) {
      llvm::errs() << "\n[Generation interrupted by user]\n";
      g_receivedSigInt = false;
      break;
    }

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

    const auto decodeStart = std::chrono::high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        &decodeRet, &paramsContainer, &inputContainerDecode, &cachePosition,
        &decodeRet.kv0, &decodeRet.kv1, &decodeRet.ret_dummy0, &decodeRet.kv2,
        &decodeRet.kv3, &decodeRet.ret_dummy1, &decodeRet.kv4, &decodeRet.kv5,
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
    const auto decodeEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> decodeTime =
        decodeEnd - decodeStart;
    decodeTimeAccumMs += decodeTime.count();
    ++decodeTokens;

    const float *decodeStartPtr = decodeRet.logits.getData();
    maxIndex = sampler.sample(decodeStartPtr, MaxVocabSize, recentTokens);
    recentTokens.push_back(maxIndex);

    if (isStopToken(maxIndex)) {
      break;
    }

    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    streamNewText(outputContainer, streamed, tokenStream);

    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  if (decodeSeconds > 0.0) {
    stats.decodeTokensPerSec =
        decodeTokens > 0 ? static_cast<double>(decodeTokens) / decodeSeconds
                         : 0.0;
  }

  tokenStream << std::endl;
  stats.generatedTokens = outputContainer.getTokenCnt();
  stats.finalText = streamed;
  stats.totalSeconds = prefillSeconds + decodeSeconds;
  return stats;
}

void printStats(const GenerationResult &result) {
  getInfoStream() << "Prefill tokens: " << result.promptTokens << "\n";
  getInfoStream() << "Generated tokens: " << result.generatedTokens << "\n";
  getInfoStream() << "Prefill throughput: "
                  << llvm::formatv("{0:F3}", result.prefillTokensPerSec)
                  << " tokens/s\n";
  getInfoStream() << "Decode throughput: "
                  << llvm::formatv("{0:F3}", result.decodeTokensPerSec)
                  << " tokens/s\n";
  getInfoStream() << "Total time: "
                  << llvm::formatv("{0:F2}", result.totalSeconds) << " s\n";
}

void runInteractiveSession(const std::string &systemPrompt,
                           MemRef<float, 1> &paramsContainer,
                           const std::string &vocabPath, int maxNewTokens,
                           const std::vector<long long> &stopTokenIds,
                           bool suppressStats, Sampler &sampler,
                           ConversationManager *conv) {
  const bool hasConv = (conv != nullptr);
  llvm::errs() << "Entering interactive mode.\n"
               << "  - Type your prompt and press Enter to submit\n"
               << "  - End a line with '\\' to continue to the next line\n"
               << "  - Type :paste to enter paste mode (for long text)\n"
               << "  - Type :clear to discard the "
               << (hasConv ? "conversation history" : "buffer") << "\n";
  if (hasConv) {
    llvm::errs() << "  - Type /regen to regenerate the last response\n"
                 << "  - Type /history to show conversation history\n"
                 << "  - Type /system <text> to set system prompt\n"
                 << "  - Type /read <file> to insert file content\n";
  }
  llvm::errs() << "  - Type :exit or :quit to end the session\n";

  std::string userInput;
  std::string bufferedPrompt;
  bool pasteMode = false;

  auto runAndRecord = [&](const std::string &userText) {
    std::string finalPrompt;
    if (hasConv) {
      conv->addMessage("user", userText);
      finalPrompt = conv->buildPromptWithLimit(MaxTokenLength);
    } else {
      finalPrompt = userText;
      if (!systemPrompt.empty()) {
        finalPrompt = systemPrompt + "\n\n" + finalPrompt;
      }
    }
    GenerationResult result =
        runGeneration(finalPrompt, paramsContainer, vocabPath, maxNewTokens,
                      stopTokenIds, std::cout, sampler);
    if (hasConv) {
      conv->addMessage("assistant", result.finalText);
    }
    if (!suppressStats) {
      printStats(result);
    }
  };

  auto submitBufferedPrompt = [&]() {
    if (bufferedPrompt.empty()) {
      return;
    }
    runAndRecord(bufferedPrompt);
    bufferedPrompt.clear();
  };

  while (true) {
    if (!pasteMode) {
      std::cout << (bufferedPrompt.empty() ? ">>> " : "... ") << std::flush;
    }

    if (g_receivedSigInt) {
      llvm::errs() << "\nInterrupted. Exiting...\n";
      break;
    }

    if (!std::getline(std::cin, userInput)) {
      if (g_receivedSigInt) {
        llvm::errs() << "\nInterrupted. Exiting...\n";
        break;
      }
      llvm::errs() << "Input stream ended. Leaving interactive mode\n";
      break;
    }

    if (g_receivedSigInt) {
      llvm::errs() << "\nInterrupted. Exiting...\n";
      break;
    }

    // Paste Mode Logic
    if (pasteMode) {
      if (userInput == ":end") {
        pasteMode = false;
        submitBufferedPrompt();
        continue;
      }
      if (!bufferedPrompt.empty())
        bufferedPrompt.push_back('\n');
      bufferedPrompt += userInput;
      continue;
    }

    // Command handling
    if (userInput == ":exit" || userInput == ":quit") {
      llvm::errs() << "Leaving interactive mode\n";
      break;
    }
    if (userInput == ":clear") {
      bufferedPrompt.clear();
      if (hasConv) {
        conv->clearHistory();
        llvm::errs() << "Conversation history cleared.\n";
      } else {
        llvm::errs() << "Prompt buffer cleared.\n";
      }
      continue;
    }
    if (userInput == ":paste") {
      pasteMode = true;
      llvm::errs()
          << "Entering paste mode. Type ':end' on a new line to submit.\n";
      continue;
    }

    // Conversation-aware commands (only when chat template is loaded)
    if (hasConv && userInput == "/regen") {
      // Remove the last assistant response, then re-run from the last user msg.
      if (!conv->removeLastAssistantMessage()) {
        llvm::errs() << "No assistant message to regenerate.\n";
        continue;
      }
      // The last message should now be the user message that triggered it.
      if (!conv->messages().empty() && conv->messages().back().role == "user") {
        std::string lastUserMsg = conv->messages().back().content;
        conv->removeLastMessage(); // Remove user msg; runAndRecord re-adds it.
        runAndRecord(lastUserMsg);
      } else {
        llvm::errs() << "No user message to regenerate from.\n";
      }
      continue;
    }
    if (hasConv && userInput == "/history") {
      const auto &msgs = conv->messages();
      if (msgs.empty()) {
        llvm::errs() << "(empty)\n";
      }
      for (size_t i = 0; i < msgs.size(); ++i) {
        llvm::errs() << "[" << msgs[i].role << "] ";
        if (msgs[i].content.size() > 100) {
          llvm::errs() << msgs[i].content.substr(0, 100) << "...\n";
        } else {
          llvm::errs() << msgs[i].content << "\n";
        }
      }
      continue;
    }
    if (hasConv && userInput.substr(0, 8) == "/system ") {
      conv->setSystemPrompt(userInput.substr(8));
      llvm::errs() << "System prompt updated.\n";
      continue;
    }
    if (hasConv && userInput.substr(0, 6) == "/read ") {
      std::string path = userInput.substr(6);
      try {
        std::string content = readPromptFromFile(path);
        llvm::errs() << "Read " << content.size() << " bytes from " << path
                     << "\n";
        runAndRecord(content);
      } catch (const std::exception &ex) {
        llvm::errs() << ex.what() << "\n";
      }
      continue;
    }

    if (userInput.empty()) {
      if (!bufferedPrompt.empty()) {
        submitBufferedPrompt();
      }
      continue;
    }

    // Check for continuation character '\'
    if (userInput.back() == '\\') {
      userInput.pop_back();
      if (!bufferedPrompt.empty())
        bufferedPrompt.push_back('\n');
      bufferedPrompt += userInput;
      continue;
    }

    // Append and submit immediately
    if (!bufferedPrompt.empty())
      bufferedPrompt.push_back('\n');
    bufferedPrompt += userInput;
    submitBufferedPrompt();
  }
}

} // namespace

int main(int argc, char **argv) {
  struct sigaction sa;
  sa.sa_handler = signalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);

  llvm::cl::ParseCommandLineOptions(argc, argv, "buddy DeepSeek R1 CLI\n");

  if (!PromptOpt.empty() && !PromptFileOpt.empty()) {
    llvm::errs() << "Cannot use --prompt and --prompt-file at the same time\n";
    return 1;
  }

  std::string prompt;
  try {
    if (!PromptOpt.empty()) {
      prompt = PromptOpt;
    } else if (!PromptFileOpt.empty()) {
      prompt = readPromptFromFile(PromptFileOpt);
    } else if (!InteractiveOpt) {
      prompt = readPromptFromStdin();
    }
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return 1;
  }

  if (!InteractiveOpt && prompt.empty()) {
    llvm::errs() << "Prompt cannot be empty\n";
    return 1;
  }

  if (KeepTokenNumOpt > MaxTokenLength / 2) {
    llvm::errs()
        << "--keep-token-num must be smaller than half of context window ("
        << MaxTokenLength << ")\n";
    return 1;
  }

  std::string modelPath =
      ModelPathOpt.empty() ? getDefaultModelPath() : ModelPathOpt;
  std::string vocabPath =
      VocabPathOpt.empty() ? getDefaultVocabPath() : VocabPathOpt;

  if (!fs::exists(modelPath)) {
    llvm::errs() << "Model parameter file not found: " << modelPath << "\n";
    return 1;
  }
  if (!fs::exists(vocabPath)) {
    llvm::errs() << "Vocabulary file not found: " << vocabPath << "\n";
    return 1;
  }

  const unsigned maxNewTokens =
      std::min(MaxTokensOpt.getValue(), static_cast<unsigned>(MaxTokenLength));

  SamplerConfig samplerCfg;
  samplerCfg.temperature = static_cast<float>(TemperatureOpt.getValue());
  samplerCfg.topK = TopKOpt.getValue();
  samplerCfg.topP = static_cast<float>(TopPOpt.getValue());
  samplerCfg.minP = static_cast<float>(MinPOpt.getValue());
  samplerCfg.repeatPenalty = static_cast<float>(RepeatPenaltyOpt.getValue());
  samplerCfg.repeatLastN = RepeatLastNOpt.getValue();
  samplerCfg.seed = SeedOpt.getValue();
  Sampler sampler(samplerCfg);

  MemRef<float, 1> paramsContainer({ParamsSize});
  try {
    loadParameters(modelPath, paramsContainer);
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return 1;
  }

  try {
    // Build stop token list: always include --eos-id, plus template stop
    // tokens.
    std::vector<long long> stopTokenIds = {EosIdOpt.getValue()};

    if (InteractiveOpt) {
      // Load chat template if provided; enables multi-turn conversation.
      std::unique_ptr<ConversationManager> conv;
      if (!ChatTemplateOpt.empty()) {
        ChatTemplate tmpl = ChatTemplate::fromFile(ChatTemplateOpt);
        for (int id : tmpl.stopTokenIds()) {
          if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
              stopTokenIds.end()) {
            stopTokenIds.push_back(static_cast<long long>(id));
          }
        }
        conv = std::make_unique<ConversationManager>(std::move(tmpl));
        if (!prompt.empty()) {
          conv->setSystemPrompt(prompt);
        }
        llvm::errs() << "Chat template loaded. Multi-turn conversation "
                        "enabled.\n";
      }
      runInteractiveSession(prompt, paramsContainer, vocabPath,
                            static_cast<int>(maxNewTokens), stopTokenIds,
                            SuppressStatsOpt, sampler, conv.get());
    } else {
      // In single-shot mode, apply chat template if provided.
      std::string finalPrompt = prompt;
      if (!ChatTemplateOpt.empty()) {
        ChatTemplate tmpl = ChatTemplate::fromFile(ChatTemplateOpt);
        for (int id : tmpl.stopTokenIds()) {
          if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
              stopTokenIds.end()) {
            stopTokenIds.push_back(static_cast<long long>(id));
          }
        }
        std::vector<Message> msgs = {{"user", prompt}};
        finalPrompt = tmpl.apply(msgs);
      }
      GenerationResult result = runGeneration(
          finalPrompt, paramsContainer, vocabPath,
          static_cast<int>(maxNewTokens), stopTokenIds, std::cout, sampler);
      if (!SuppressStatsOpt) {
        printStats(result);
      }
    }
  } catch (const std::exception &ex) {
    llvm::errs() << "Inference failed: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
