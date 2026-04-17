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
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

/// Prefill returns: 56 KV caches followed by logits.
template <size_t CacheLen, size_t LogitsSeqLen = CacheLen>
struct PrefillReturnsT {
  MemRef<float, 4> kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7;
  MemRef<float, 4> kv8, kv9, kv10, kv11, kv12, kv13, kv14, kv15;
  MemRef<float, 4> kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23;
  MemRef<float, 4> kv24, kv25, kv26, kv27, kv28, kv29, kv30, kv31;
  MemRef<float, 4> kv32, kv33, kv34, kv35, kv36, kv37, kv38, kv39;
  MemRef<float, 4> kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47;
  MemRef<float, 4> kv48, kv49, kv50, kv51, kv52, kv53, kv54, kv55;
  MemRef<float, 3> logits;

  PrefillReturnsT()
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
        logits({1, LogitsSeqLen, MaxVocabSize}) {}

  static constexpr size_t getCacheLen() { return CacheLen; }
};

/// Decode returns: updated cache_position, then 27 groups of (kv, kv, dummy),
/// followed by the final two kvs and logits. Total 85 fields.
template <size_t CacheLen> struct DecodeReturnsT {
  // First return value: updated cache_position
  MemRef<long long, 1> cache_position_out;

  // Group 1
  MemRef<float, 4> kv0, kv1;
  MemRef<long long, 1> ret_dummy0;
  // Group 2
  MemRef<float, 4> kv2, kv3;
  MemRef<long long, 1> ret_dummy1;
  // Group 3
  MemRef<float, 4> kv4, kv5;
  MemRef<long long, 1> ret_dummy2;
  // Group 4
  MemRef<float, 4> kv6, kv7;
  MemRef<long long, 1> ret_dummy3;
  // Group 5
  MemRef<float, 4> kv8, kv9;
  MemRef<long long, 1> ret_dummy4;
  // Group 6
  MemRef<float, 4> kv10, kv11;
  MemRef<long long, 1> ret_dummy5;
  // Group 7
  MemRef<float, 4> kv12, kv13;
  MemRef<long long, 1> ret_dummy6;
  // Group 8
  MemRef<float, 4> kv14, kv15;
  MemRef<long long, 1> ret_dummy7;
  // Group 9
  MemRef<float, 4> kv16, kv17;
  MemRef<long long, 1> ret_dummy8;
  // Group 10
  MemRef<float, 4> kv18, kv19;
  MemRef<long long, 1> ret_dummy9;
  // Group 11
  MemRef<float, 4> kv20, kv21;
  MemRef<long long, 1> ret_dummy10;
  // Group 12
  MemRef<float, 4> kv22, kv23;
  MemRef<long long, 1> ret_dummy11;
  // Group 13
  MemRef<float, 4> kv24, kv25;
  MemRef<long long, 1> ret_dummy12;
  // Group 14
  MemRef<float, 4> kv26, kv27;
  MemRef<long long, 1> ret_dummy13;
  // Group 15
  MemRef<float, 4> kv28, kv29;
  MemRef<long long, 1> ret_dummy14;
  // Group 16
  MemRef<float, 4> kv30, kv31;
  MemRef<long long, 1> ret_dummy15;
  // Group 17
  MemRef<float, 4> kv32, kv33;
  MemRef<long long, 1> ret_dummy16;
  // Group 18
  MemRef<float, 4> kv34, kv35;
  MemRef<long long, 1> ret_dummy17;
  // Group 19
  MemRef<float, 4> kv36, kv37;
  MemRef<long long, 1> ret_dummy18;
  // Group 20
  MemRef<float, 4> kv38, kv39;
  MemRef<long long, 1> ret_dummy19;
  // Group 21
  MemRef<float, 4> kv40, kv41;
  MemRef<long long, 1> ret_dummy20;
  // Group 22
  MemRef<float, 4> kv42, kv43;
  MemRef<long long, 1> ret_dummy21;
  // Group 23
  MemRef<float, 4> kv44, kv45;
  MemRef<long long, 1> ret_dummy22;
  // Group 24
  MemRef<float, 4> kv46, kv47;
  MemRef<long long, 1> ret_dummy23;
  // Group 25
  MemRef<float, 4> kv48, kv49;
  MemRef<long long, 1> ret_dummy24;
  // Group 26
  MemRef<float, 4> kv50, kv51;
  MemRef<long long, 1> ret_dummy25;
  MemRef<float, 4> kv52, kv53;
  MemRef<long long, 1> ret_dummy26;
  MemRef<float, 4> kv54, kv55;
  // Logits
  MemRef<float, 3> logits;

  DecodeReturnsT()
      : cache_position_out({1}, 0LL),
        // Group 1
        kv0({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv1({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy0({1}, 0LL),
        // Group 2
        kv2({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv3({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy1({1}, 0LL),
        // Group 3
        kv4({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv5({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy2({1}, 0LL),
        // Group 4
        kv6({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv7({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy3({1}, 0LL),
        // Group 5
        kv8({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv9({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy4({1}, 0LL),
        // Group 6
        kv10({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv11({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy5({1}, 0LL),
        // Group 7
        kv12({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv13({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy6({1}, 0LL),
        // Group 8
        kv14({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv15({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy7({1}, 0LL),
        // Group 9
        kv16({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv17({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy8({1}, 0LL),
        // Group 10
        kv18({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv19({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy9({1}, 0LL),
        // Group 11
        kv20({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv21({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy10({1}, 0LL),
        // Group 12
        kv22({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv23({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy11({1}, 0LL),
        // Group 13
        kv24({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv25({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy12({1}, 0LL),
        // Group 14
        kv26({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv27({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy13({1}, 0LL),
        // Group 15
        kv28({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv29({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy14({1}, 0LL),
        // Group 16
        kv30({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv31({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy15({1}, 0LL),
        // Group 17
        kv32({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv33({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy16({1}, 0LL),
        // Group 18
        kv34({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv35({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy17({1}, 0LL),
        // Group 19
        kv36({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv37({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy18({1}, 0LL),
        // Group 20
        kv38({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv39({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy19({1}, 0LL),
        // Group 21
        kv40({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv41({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy20({1}, 0LL),
        // Group 22
        kv42({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv43({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy21({1}, 0LL),
        // Group 23
        kv44({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv45({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy22({1}, 0LL),
        // Group 24
        kv46({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv47({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy23({1}, 0LL),
        // Group 25
        kv48({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv49({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy24({1}, 0LL),
        // Group 26
        kv50({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv51({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy25({1}, 0LL),
        // Group 27
        kv52({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv53({1, HeadNum, CacheLen, HiddenSize}, 0), ret_dummy26({1}, 0LL),
        // Group 28 (no dummy)
        kv54({1, HeadNum, CacheLen, HiddenSize}, 0),
        kv55({1, HeadNum, CacheLen, HiddenSize}, 0),
        logits({1, 1, MaxVocabSize}) {}

  static constexpr size_t getCacheLen() { return CacheLen; }
};

// Type aliases for specific sizes
using PrefillContainer32 = PrefillReturnsT<32, 32>;
using PrefillContainer64 = PrefillReturnsT<64, 64>;
using PrefillContainer128 = PrefillReturnsT<128, 128>;
using PrefillContainer256 = PrefillReturnsT<256, 256>;
using PrefillContainer512 = PrefillReturnsT<512, 512>;
using PrefillContainer1024 = PrefillReturnsT<1024, 1024>;

using DecodeContainer32 = DecodeReturnsT<32>;
using DecodeContainer64 = DecodeReturnsT<64>;
using DecodeContainer128 = DecodeReturnsT<128>;
using DecodeContainer256 = DecodeReturnsT<256>;
using DecodeContainer512 = DecodeReturnsT<512>;
using DecodeContainer1024 = DecodeReturnsT<1024>;

// KV pointer array type
using KVPtrArray = std::array<MemRef<float, 4> *, 56>;

// Helper to build KV pointer array for Prefill containers
template <size_t CacheLen, size_t LogitsSeqLen>
KVPtrArray buildPrefillKVPtrs(PrefillReturnsT<CacheLen, LogitsSeqLen> &ret) {
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

// Helper to build KV pointer array for Decode containers
template <size_t CacheLen>
KVPtrArray buildDecodeKVPtrs(DecodeReturnsT<CacheLen> &ret) {
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

#define DECLARE_DECODE_FUNC(SIZE)                                              \
  extern "C" void _mlir_ciface_forward_decode_##SIZE(                          \
      DecodeContainer##SIZE *result, MemRef<float, 1> *arg0,                   \
      MemRef<long long, 2> *arg1, MemRef<long long, 1> *arg2, /* Group 1 */    \
      MemRef<float, 4> *kv0, MemRef<float, 4> *kv1,                            \
      MemRef<long long, 1> *dummy0, /* Group 2 */                              \
      MemRef<float, 4> *kv2, MemRef<float, 4> *kv3,                            \
      MemRef<long long, 1> *dummy1, /* Group 3 */                              \
      MemRef<float, 4> *kv4, MemRef<float, 4> *kv5,                            \
      MemRef<long long, 1> *dummy2, /* Group 4 */                              \
      MemRef<float, 4> *kv6, MemRef<float, 4> *kv7,                            \
      MemRef<long long, 1> *dummy3, /* Group 5 */                              \
      MemRef<float, 4> *kv8, MemRef<float, 4> *kv9,                            \
      MemRef<long long, 1> *dummy4, /* Group 6 */                              \
      MemRef<float, 4> *kv10, MemRef<float, 4> *kv11,                          \
      MemRef<long long, 1> *dummy5, /* Group 7 */                              \
      MemRef<float, 4> *kv12, MemRef<float, 4> *kv13,                          \
      MemRef<long long, 1> *dummy6, /* Group 8 */                              \
      MemRef<float, 4> *kv14, MemRef<float, 4> *kv15,                          \
      MemRef<long long, 1> *dummy7, /* Group 9 */                              \
      MemRef<float, 4> *kv16, MemRef<float, 4> *kv17,                          \
      MemRef<long long, 1> *dummy8, /* Group 10 */                             \
      MemRef<float, 4> *kv18, MemRef<float, 4> *kv19,                          \
      MemRef<long long, 1> *dummy9, /* Group 11 */                             \
      MemRef<float, 4> *kv20, MemRef<float, 4> *kv21,                          \
      MemRef<long long, 1> *dummy10, /* Group 12 */                            \
      MemRef<float, 4> *kv22, MemRef<float, 4> *kv23,                          \
      MemRef<long long, 1> *dummy11, /* Group 13 */                            \
      MemRef<float, 4> *kv24, MemRef<float, 4> *kv25,                          \
      MemRef<long long, 1> *dummy12, /* Group 14 */                            \
      MemRef<float, 4> *kv26, MemRef<float, 4> *kv27,                          \
      MemRef<long long, 1> *dummy13, /* Group 15 */                            \
      MemRef<float, 4> *kv28, MemRef<float, 4> *kv29,                          \
      MemRef<long long, 1> *dummy14, /* Group 16 */                            \
      MemRef<float, 4> *kv30, MemRef<float, 4> *kv31,                          \
      MemRef<long long, 1> *dummy15, /* Group 17 */                            \
      MemRef<float, 4> *kv32, MemRef<float, 4> *kv33,                          \
      MemRef<long long, 1> *dummy16, /* Group 18 */                            \
      MemRef<float, 4> *kv34, MemRef<float, 4> *kv35,                          \
      MemRef<long long, 1> *dummy17, /* Group 19 */                            \
      MemRef<float, 4> *kv36, MemRef<float, 4> *kv37,                          \
      MemRef<long long, 1> *dummy18, /* Group 20 */                            \
      MemRef<float, 4> *kv38, MemRef<float, 4> *kv39,                          \
      MemRef<long long, 1> *dummy19, /* Group 21 */                            \
      MemRef<float, 4> *kv40, MemRef<float, 4> *kv41,                          \
      MemRef<long long, 1> *dummy20, /* Group 22 */                            \
      MemRef<float, 4> *kv42, MemRef<float, 4> *kv43,                          \
      MemRef<long long, 1> *dummy21, /* Group 23 */                            \
      MemRef<float, 4> *kv44, MemRef<float, 4> *kv45,                          \
      MemRef<long long, 1> *dummy22, /* Group 24 */                            \
      MemRef<float, 4> *kv46, MemRef<float, 4> *kv47,                          \
      MemRef<long long, 1> *dummy23, /* Group 25 */                            \
      MemRef<float, 4> *kv48, MemRef<float, 4> *kv49,                          \
      MemRef<long long, 1> *dummy24, /* Group 26 */                            \
      MemRef<float, 4> *kv50, MemRef<float, 4> *kv51,                          \
      MemRef<long long, 1> *dummy25, /* Group 27 */                            \
      MemRef<float, 4> *kv52, MemRef<float, 4> *kv53,                          \
      MemRef<long long, 1> *dummy26, /* Group 28 (no dummy) */                 \
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

void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  getline(std::cin, inputStr);
  std::cout << std::endl;
}

void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

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

size_t selectCacheSize(size_t currentPos) {
  for (size_t size : KV_CACHE_SIZES) {
    if (currentPos < size) {
      return size;
    }
  }
  return KV_CACHE_SIZES.back();
}

size_t selectPrefillSize(size_t tokenCount) {
  for (size_t size : KV_CACHE_SIZES) {
    if (tokenCount <= size) {
      return size;
    }
  }
  return KV_CACHE_SIZES.back();
}

// Copy KV cache using pre-built pointer arrays
void copyKVCache(const KVPtrArray &srcPtrs, const KVPtrArray &dstPtrs,
                 size_t srcCacheLen, size_t dstCacheLen, size_t validTokens) {
  size_t copy_len = std::min({validTokens, srcCacheLen, dstCacheLen});

  for (int k = 0; k < 56; ++k) {
    auto &src_kv = *srcPtrs[k];
    auto &dst_kv = *dstPtrs[k];

    for (size_t h = 0; h < HeadNum; ++h) {
      size_t bytes_to_copy = copy_len * HiddenSize * sizeof(float);
      float *src_ptr = src_kv.getData() + h * srcCacheLen * HiddenSize;
      float *dst_ptr = dst_kv.getData() + h * dstCacheLen * HiddenSize;
      std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
    }
  }
}

#define CALL_PREFILL(SIZE, container, paramsPtr, inputPtr)                     \
  _mlir_ciface_forward_prefill_##SIZE(&container, paramsPtr, inputPtr)

#define CALL_DECODE(SIZE, container, paramsPtr, inputPtr, cachePtr)            \
  _mlir_ciface_forward_decode_##SIZE(                                          \
      &container, paramsPtr, inputPtr, cachePtr,                /* Group 1 */  \
      &container.kv0, &container.kv1, &container.ret_dummy0,    /* Group 2 */  \
      &container.kv2, &container.kv3, &container.ret_dummy1,    /* Group 3 */  \
      &container.kv4, &container.kv5, &container.ret_dummy2,    /* Group 4 */  \
      &container.kv6, &container.kv7, &container.ret_dummy3,    /* Group 5 */  \
      &container.kv8, &container.kv9, &container.ret_dummy4,    /* Group 6 */  \
      &container.kv10, &container.kv11, &container.ret_dummy5,  /* Group 7 */  \
      &container.kv12, &container.kv13, &container.ret_dummy6,  /* Group 8 */  \
      &container.kv14, &container.kv15, &container.ret_dummy7,  /* Group 9 */  \
      &container.kv16, &container.kv17, &container.ret_dummy8,  /* Group 10 */ \
      &container.kv18, &container.kv19, &container.ret_dummy9,  /* Group 11 */ \
      &container.kv20, &container.kv21, &container.ret_dummy10, /* Group 12 */ \
      &container.kv22, &container.kv23, &container.ret_dummy11, /* Group 13 */ \
      &container.kv24, &container.kv25, &container.ret_dummy12, /* Group 14 */ \
      &container.kv26, &container.kv27, &container.ret_dummy13, /* Group 15 */ \
      &container.kv28, &container.kv29, &container.ret_dummy14, /* Group 16 */ \
      &container.kv30, &container.kv31, &container.ret_dummy15, /* Group 17 */ \
      &container.kv32, &container.kv33, &container.ret_dummy16, /* Group 18 */ \
      &container.kv34, &container.kv35, &container.ret_dummy17, /* Group 19 */ \
      &container.kv36, &container.kv37, &container.ret_dummy18, /* Group 20 */ \
      &container.kv38, &container.kv39, &container.ret_dummy19, /* Group 21 */ \
      &container.kv40, &container.kv41, &container.ret_dummy20, /* Group 22 */ \
      &container.kv42, &container.kv43, &container.ret_dummy21, /* Group 23 */ \
      &container.kv44, &container.kv45, &container.ret_dummy22, /* Group 24 */ \
      &container.kv46, &container.kv47, &container.ret_dummy23, /* Group 25 */ \
      &container.kv48, &container.kv49, &container.ret_dummy24, /* Group 26 */ \
      &container.kv50, &container.kv51, &container.ret_dummy25, /* Group 27 */ \
      &container.kv52, &container.kv53,                                        \
      &container.ret_dummy26, /* Group 28 (no dummy) */                        \
      &container.kv54, &container.kv55)

// -----------------------------------------------------------------------------
// DeepSeekR1 Tiered KV Cache Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
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

  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0_mc.data";

  std::string inputStr;
  getUserInput(inputStr);

  Text<size_t, 2> outputContainer;
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> ParamsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  PrefillContainer32 prefillContainer32;
  PrefillContainer64 prefillContainer64;
  PrefillContainer128 prefillContainer128;
  PrefillContainer256 prefillContainer256;
  PrefillContainer512 prefillContainer512;
  PrefillContainer1024 prefillContainer1024;

  DecodeContainer32 decodeContainer32;
  DecodeContainer64 decodeContainer64;
  DecodeContainer128 decodeContainer128;
  DecodeContainer256 decodeContainer256;
  DecodeContainer512 decodeContainer512;
  DecodeContainer1024 decodeContainer1024;

  outputContainer.loadVocab(vocabDir);
  loadParameters(paramsDir, ParamsContainer);

  Text<size_t, 2> inputContainerPrefill(inputStr);
  inputContainerPrefill.loadVocab(vocabDir);
  tokenizeInput(vocabDir, inputContainerPrefill);
  size_t actualTokenCount = inputContainerPrefill.getTokenCnt();

  size_t selectedPrefillSize = selectPrefillSize(actualTokenCount);
  printLogLabel();
  std::cout << "Actual token count: " << actualTokenCount
            << ", selected prefill size: " << selectedPrefillSize << std::endl;

  Text<size_t, 2> inputContainerTiered(inputStr);
  inputContainerTiered.tokenizeDeepSeekR1(vocabDir, selectedPrefillSize);

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

  size_t currentPos = inputContainerTiered.getTokenCnt();
  cachePosition.getData()[0] = currentPos;

  size_t currentCacheSize = selectedPrefillSize;
  printLogLabel();
  std::cout << "Initial cache position: " << currentPos
            << ", selected decode cache size: " << currentCacheSize
            << std::endl;

  // Copy prefill KV to the matching decode container
  KVPtrArray prefillPtrs;
  KVPtrArray decodePtrs;
  size_t srcCacheLen = selectedPrefillSize;

  switch (selectedPrefillSize) {
  case 32:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer32);
    decodePtrs = buildDecodeKVPtrs(decodeContainer32);
    break;
  case 64:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer64);
    decodePtrs = buildDecodeKVPtrs(decodeContainer64);
    break;
  case 128:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer128);
    decodePtrs = buildDecodeKVPtrs(decodeContainer128);
    break;
  case 256:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer256);
    decodePtrs = buildDecodeKVPtrs(decodeContainer256);
    break;
  case 512:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer512);
    decodePtrs = buildDecodeKVPtrs(decodeContainer512);
    break;
  case 1024:
    prefillPtrs = buildPrefillKVPtrs(prefillContainer1024);
    decodePtrs = buildDecodeKVPtrs(decodeContainer1024);
    break;
  }
  copyKVCache(prefillPtrs, decodePtrs, srcCacheLen, currentCacheSize,
              currentPos);

  int generateLen = MaxTokenLength - inputContainerTiered.getTokenCnt();
  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;
  size_t prevCacheSize = currentCacheSize;

  for (int i = 1; i <= generateLen; i++) {
    size_t neededCacheSize = selectCacheSize(currentPos + 1);

    if (neededCacheSize != prevCacheSize) {
      printLogLabel();
      std::cout << "Switching cache size from " << prevCacheSize << " to "
                << neededCacheSize << " at position " << currentPos
                << std::endl;

      // Copy KV cache to larger container
      KVPtrArray srcPtrs, dstPtrs;
      size_t dstCacheLen = neededCacheSize;
      size_t srcCacheLen = prevCacheSize;

      switch (prevCacheSize) {
      case 32:
        srcPtrs = buildDecodeKVPtrs(decodeContainer32);
        break;
      case 64:
        srcPtrs = buildDecodeKVPtrs(decodeContainer64);
        break;
      case 128:
        srcPtrs = buildDecodeKVPtrs(decodeContainer128);
        break;
      case 256:
        srcPtrs = buildDecodeKVPtrs(decodeContainer256);
        break;
      case 512:
        srcPtrs = buildDecodeKVPtrs(decodeContainer512);
        break;
      case 1024:
        srcPtrs = buildDecodeKVPtrs(decodeContainer1024);
        break;
      }

      switch (neededCacheSize) {
      case 64:
        dstPtrs = buildDecodeKVPtrs(decodeContainer64);
        break;
      case 128:
        dstPtrs = buildDecodeKVPtrs(decodeContainer128);
        break;
      case 256:
        dstPtrs = buildDecodeKVPtrs(decodeContainer256);
        break;
      case 512:
        dstPtrs = buildDecodeKVPtrs(decodeContainer512);
        break;
      case 1024:
        dstPtrs = buildDecodeKVPtrs(decodeContainer1024);
        break;
      default:
        break;
      }

      copyKVCache(srcPtrs, dstPtrs, srcCacheLen, dstCacheLen, currentPos);
      prevCacheSize = neededCacheSize;
    }

    currentCacheSize = neededCacheSize;
    const float *logitsPtr = nullptr;

    // Update dummy fields for current decode container
    auto updateDummy = [&](auto &container) {
      container.ret_dummy0.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy1.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy2.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy3.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy4.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy5.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy6.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy7.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy8.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy9.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy10.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy11.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy12.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy13.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy14.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy15.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy16.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy17.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy18.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy19.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy20.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy21.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy22.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy23.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy24.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy25.getData()[0] = cachePosition.getData()[0];
      container.ret_dummy26.getData()[0] = cachePosition.getData()[0];
    };

    const auto inferenceStart = std::chrono::high_resolution_clock::now();

    switch (currentCacheSize) {
    case 32:
      updateDummy(decodeContainer32);
      CALL_DECODE(32, decodeContainer32, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer32.logits.getData();
      break;
    case 64:
      updateDummy(decodeContainer64);
      CALL_DECODE(64, decodeContainer64, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer64.logits.getData();
      break;
    case 128:
      updateDummy(decodeContainer128);
      CALL_DECODE(128, decodeContainer128, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer128.logits.getData();
      break;
    case 256:
      updateDummy(decodeContainer256);
      CALL_DECODE(256, decodeContainer256, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer256.logits.getData();
      break;
    case 512:
      updateDummy(decodeContainer512);
      CALL_DECODE(512, decodeContainer512, &ParamsContainer,
                  &inputContainerDecode, &cachePosition);
      logitsPtr = decodeContainer512.logits.getData();
      break;
    case 1024:
      updateDummy(decodeContainer1024);
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

    endPtr = logitsPtr + MaxVocabSize;
    maxIndex = findMaxIndex(logitsPtr, endPtr);
    tok = inputContainerTiered.getStr(maxIndex);

    printIterInfo(i, tok, inferenceTime.count() / 1000, currentCacheSize);

    if (maxIndex == 151643) {
      break;
    }

    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    currentPos += 1;
    cachePosition.getData()[0] = currentPos;
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
