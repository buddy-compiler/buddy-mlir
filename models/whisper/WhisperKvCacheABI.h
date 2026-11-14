//===- WhisperKvCacheABI.h - Whisper Base static cache ABI
//-----------------===//
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

#ifndef BUDDY_WHISPER_KV_CACHE_ABI_H
#define BUDDY_WHISPER_KV_CACHE_ABI_H

#include "buddy/Core/Container.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

namespace buddy::runtime::detail {

// These layouts mirror the MLIR C wrapper's packed return descriptors.
// The first-token graph returns both self/cross cache states; subsequent
// calls return only updated self-attention states and one logits row.
constexpr std::size_t kWhisperDecoderLayers = 6;
using LengthRef = MemRef<std::size_t, 1>;
using CacheRef = MemRef<_Float16, 4>;
using LogitsRef = MemRef<float, 3>;
using WeightRef = MemRef<_Float16, 1>;
using HiddenRef = MemRef<float, 3>;
using TokenRef = MemRef<std::size_t, 2>;

template <typename T, std::size_t N>
MemRef<T, N> makeCacheRef(std::vector<std::size_t> shape, bool own) {
  if (own)
    return MemRef<T, N>(shape, T(0));
  return MemRef<T, N>(shape, false, 0);
}

struct InitLayer {
  LengthRef selfLength;
  CacheRef selfKey;
  CacheRef selfValue;
  LengthRef crossLength;
  CacheRef crossKey;
  CacheRef crossValue;

  InitLayer(bool own, std::size_t capacity)
      : selfLength(makeCacheRef<std::size_t, 1>({1}, own)),
        selfKey(makeCacheRef<_Float16, 4>({1, 8, capacity, 64}, own)),
        selfValue(makeCacheRef<_Float16, 4>({1, 8, capacity, 64}, own)),
        crossLength(makeCacheRef<std::size_t, 1>({1}, own)),
        crossKey(makeCacheRef<_Float16, 4>({1, 8, 1500, 64}, own)),
        crossValue(makeCacheRef<_Float16, 4>({1, 8, 1500, 64}, own)) {}
};

struct StepLayer {
  LengthRef selfLength;
  CacheRef selfKey;
  CacheRef selfValue;

  explicit StepLayer(std::size_t capacity)
      : selfLength(makeCacheRef<std::size_t, 1>({1}, false)),
        selfKey(makeCacheRef<_Float16, 4>({1, 8, capacity, 64}, false)),
        selfValue(makeCacheRef<_Float16, 4>({1, 8, capacity, 64}, false)) {}
};

#define BUDDY_WHISPER_SIX_INIT(own, capacity)                                  \
  InitLayer(own, capacity), InitLayer(own, capacity),                          \
      InitLayer(own, capacity), InitLayer(own, capacity),                      \
      InitLayer(own, capacity), InitLayer(own, capacity)
#define BUDDY_WHISPER_SIX_STEP(capacity)                                       \
  StepLayer(capacity), StepLayer(capacity), StepLayer(capacity),               \
      StepLayer(capacity), StepLayer(capacity), StepLayer(capacity)

struct CacheState {
  std::array<InitLayer, kWhisperDecoderLayers> layers;
  explicit CacheState(std::size_t capacity)
      : layers{BUDDY_WHISPER_SIX_INIT(true, capacity)} {}
};

struct InitResult {
  std::array<InitLayer, kWhisperDecoderLayers> layers;
  LogitsRef logits;
  InitResult(std::size_t capacity, std::size_t vocab)
      : layers{BUDDY_WHISPER_SIX_INIT(false, capacity)},
        logits(std::vector<std::size_t>{1, 1, vocab}, false, 0) {}
};

struct StepResult {
  std::array<StepLayer, kWhisperDecoderLayers> layers;
  LogitsRef logits;
  StepResult(std::size_t capacity, std::size_t vocab)
      : layers{BUDDY_WHISPER_SIX_STEP(capacity)},
        logits(std::vector<std::size_t>{1, 1, vocab}, false, 0) {}
};

static_assert(sizeof(InitResult) ==
              kWhisperDecoderLayers * sizeof(InitLayer) + sizeof(LogitsRef));
static_assert(sizeof(StepResult) ==
              kWhisperDecoderLayers * sizeof(StepLayer) + sizeof(LogitsRef));

#undef BUDDY_WHISPER_SIX_INIT
#undef BUDDY_WHISPER_SIX_STEP

#define BUDDY_WHISPER_INIT_TYPES                                               \
  LengthRef *, CacheRef *, CacheRef *, LengthRef *, CacheRef *, CacheRef *
#define BUDDY_WHISPER_STEP_TYPES                                               \
  LengthRef *, CacheRef *, CacheRef *, CacheRef *, CacheRef *

using InitFn = void (*)(InitResult *, WeightRef *, HiddenRef *, TokenRef *,
                        BUDDY_WHISPER_INIT_TYPES, BUDDY_WHISPER_INIT_TYPES,
                        BUDDY_WHISPER_INIT_TYPES, BUDDY_WHISPER_INIT_TYPES,
                        BUDDY_WHISPER_INIT_TYPES, BUDDY_WHISPER_INIT_TYPES);
using StepFn = void (*)(StepResult *, WeightRef *, HiddenRef *, TokenRef *,
                        BUDDY_WHISPER_STEP_TYPES, BUDDY_WHISPER_STEP_TYPES,
                        BUDDY_WHISPER_STEP_TYPES, BUDDY_WHISPER_STEP_TYPES,
                        BUDDY_WHISPER_STEP_TYPES, BUDDY_WHISPER_STEP_TYPES);

#undef BUDDY_WHISPER_INIT_TYPES
#undef BUDDY_WHISPER_STEP_TYPES

#define BUDDY_WHISPER_INIT_ARGS(state, i)                                      \
  &(state).layers[i].selfLength, &(state).layers[i].selfKey,                   \
      &(state).layers[i].selfValue, &(state).layers[i].crossLength,            \
      &(state).layers[i].crossKey, &(state).layers[i].crossValue
#define BUDDY_WHISPER_STEP_ARGS(state, i)                                      \
  &(state).layers[i].selfLength, &(state).layers[i].selfKey,                   \
      &(state).layers[i].selfValue, &(state).layers[i].crossKey,               \
      &(state).layers[i].crossValue

inline void callInit(InitFn fn, InitResult &result, WeightRef &weights,
                     HiddenRef &hidden, TokenRef &token, CacheState &state) {
  fn(&result, &weights, &hidden, &token, BUDDY_WHISPER_INIT_ARGS(state, 0),
     BUDDY_WHISPER_INIT_ARGS(state, 1), BUDDY_WHISPER_INIT_ARGS(state, 2),
     BUDDY_WHISPER_INIT_ARGS(state, 3), BUDDY_WHISPER_INIT_ARGS(state, 4),
     BUDDY_WHISPER_INIT_ARGS(state, 5));
}

inline void callStep(StepFn fn, StepResult &result, WeightRef &weights,
                     HiddenRef &hidden, TokenRef &token, CacheState &state) {
  fn(&result, &weights, &hidden, &token, BUDDY_WHISPER_STEP_ARGS(state, 0),
     BUDDY_WHISPER_STEP_ARGS(state, 1), BUDDY_WHISPER_STEP_ARGS(state, 2),
     BUDDY_WHISPER_STEP_ARGS(state, 3), BUDDY_WHISPER_STEP_ARGS(state, 4),
     BUDDY_WHISPER_STEP_ARGS(state, 5));
}

#undef BUDDY_WHISPER_INIT_ARGS
#undef BUDDY_WHISPER_STEP_ARGS

template <typename T, std::size_t N>
void copyCacheResult(MemRef<T, N> &state, MemRef<T, N> &result) {
  if (result.getData() == state.getData()) {
    (void)result.release();
  } else {
    std::memcpy(state.getData(), result.getData(), state.getSize() * sizeof(T));
  }
}

inline void updateState(CacheState &state, InitResult &result) {
  for (std::size_t i = 0; i < kWhisperDecoderLayers; ++i) {
    auto &s = state.layers[i];
    auto &r = result.layers[i];
    copyCacheResult(s.selfLength, r.selfLength);
    copyCacheResult(s.selfKey, r.selfKey);
    copyCacheResult(s.selfValue, r.selfValue);
    copyCacheResult(s.crossLength, r.crossLength);
    copyCacheResult(s.crossKey, r.crossKey);
    copyCacheResult(s.crossValue, r.crossValue);
  }
}

inline void updateState(CacheState &state, StepResult &result) {
  for (std::size_t i = 0; i < kWhisperDecoderLayers; ++i) {
    auto &s = state.layers[i];
    auto &r = result.layers[i];
    copyCacheResult(s.selfLength, r.selfLength);
    copyCacheResult(s.selfKey, r.selfKey);
    copyCacheResult(s.selfValue, r.selfValue);
  }
}

} // namespace buddy::runtime::detail

#endif // BUDDY_WHISPER_KV_CACHE_ABI_H
