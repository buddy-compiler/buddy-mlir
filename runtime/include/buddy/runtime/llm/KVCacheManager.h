//===- KVCacheManager.h - KV cache overflow and RoPE adjustment -----------===//
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
// Utilities for managing KV cache overflow in transformer inference.
//
// When the decode position reaches the maximum token length, these functions
// discard a portion of cached tokens and re-adjust RoPE (Rotary Position
// Embedding) on the surviving key cache entries so that position encodings
// remain consistent.
//
// This is model-agnostic: any RoPE-based transformer with a static KV cache
// can use these utilities.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_KVCACHEMANAGER_H
#define BUDDY_RUNTIME_LLM_KVCACHEMANAGER_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace buddy {
namespace kvcache {

/// Precompute inverse RoPE frequencies.
///
/// inverseFreqs[i] = 1 / (theta ^ (2i / hiddenSize))
/// for i in [0, hiddenSize/2).
inline std::vector<float> buildInverseRopeFreqs(float theta, int hiddenSize) {
  const int half = hiddenSize / 2;
  std::vector<float> inverseFreqs(half);
  for (int i = 0; i < half; ++i) {
    const float exponent =
        (2.0f * static_cast<float>(i)) / static_cast<float>(hiddenSize);
    inverseFreqs[i] = 1.0f / static_cast<float>(std::pow(theta, exponent));
  }
  return inverseFreqs;
}

/// Discard tokens from KV cache to maintain fixed window size.
///
/// Keeps the first `keepTokenNum` tokens intact, discards `discardLen` tokens
/// after them, and moves the remaining tail tokens forward. Stale tail regions
/// are zeroed to prevent garbage reads.
///
/// @param kvBuffers       Array of raw float pointers for each KV layer.
/// @param numLayers       Number of KV layers (e.g. 56).
/// @param headNum         Number of attention heads.
/// @param maxTokenLen     Maximum token length (cache dimension).
/// @param hiddenSize      Hidden dimension per head.
/// @param keepTokenNum    Number of initial tokens to preserve.
/// @param discardLen      Number of tokens to discard.
/// @param currentTokenCount Current number of valid tokens in cache.
inline void discardKVCache(float *kvBuffers[], int numLayers, int headNum,
                           int maxTokenLen, int hiddenSize, int keepTokenNum,
                           int discardLen, int currentTokenCount) {
  currentTokenCount = std::clamp(currentTokenCount, 0, maxTokenLen);

  if (discardLen <= 0 || keepTokenNum < 0 ||
      keepTokenNum >= currentTokenCount) {
    std::cerr << "[Error] discardKVCache: invalid parameters.\n";
    return;
  }

  const int srcStartIndex = keepTokenNum + discardLen;
  if (srcStartIndex >= currentTokenCount) {
    std::cerr << "[Error] discardKVCache: srcStartIndex (" << srcStartIndex
              << ") >= currentTokenCount (" << currentTokenCount
              << "). Invalid discard length.\n";
    return;
  }

  const int validTailTokens = currentTokenCount - srcStartIndex;
  const size_t stride = static_cast<size_t>(maxTokenLen) * hiddenSize;

  for (int k = 0; k < numLayers; ++k) {
    float *base = kvBuffers[k];
    for (int h = 0; h < headNum; ++h) {
      float *headBase = base + static_cast<size_t>(h) * stride;

      float *dstPtr = headBase + static_cast<size_t>(keepTokenNum) * hiddenSize;
      float *srcPtr =
          headBase + static_cast<size_t>(srcStartIndex) * hiddenSize;

      const size_t bytesToMove =
          static_cast<size_t>(validTailTokens) * hiddenSize * sizeof(float);
      std::memmove(dstPtr, srcPtr, bytesToMove);

      // Clear stale tail to prevent garbage reads.
      float *clearPtr =
          dstPtr + static_cast<size_t>(validTailTokens) * hiddenSize;
      const int clearTokens = maxTokenLen - (keepTokenNum + validTailTokens);
      if (clearTokens > 0) {
        const size_t bytesToClear =
            static_cast<size_t>(clearTokens) * hiddenSize * sizeof(float);
        std::memset(clearPtr, 0, bytesToClear);
      }
    }
  }
}

namespace detail {

/// Apply a rotary delta to a slice of key cache tokens.
///
/// Only operates on even-indexed layers (key caches in key/value pairs).
inline void applyRotaryDeltaToSlice(float *kvBuffers[], int numLayers,
                                    int headNum, int maxTokenLen,
                                    int hiddenSize, int startToken,
                                    int tokenCount,
                                    const std::vector<float> &cosValues,
                                    const std::vector<float> &sinValues) {
  if (tokenCount <= 0)
    return;

  const size_t headStride = static_cast<size_t>(maxTokenLen) * hiddenSize;
  const int halfSize = hiddenSize / 2;

  // Even indices are key caches; odd indices are value caches.
  for (int idx = 0; idx < numLayers; idx += 2) {
    float *base = kvBuffers[idx];
    for (int h = 0; h < headNum; ++h) {
      float *headBase = base + static_cast<size_t>(h) * headStride;
      for (int t = 0; t < tokenCount; ++t) {
        float *tokenPtr =
            headBase + static_cast<size_t>(startToken + t) * hiddenSize;
        for (int i = 0; i < halfSize; ++i) {
          const float val1 = tokenPtr[i];
          const float val2 = tokenPtr[i + halfSize];
          tokenPtr[i] = val1 * cosValues[i] - val2 * sinValues[i];
          tokenPtr[i + halfSize] = val1 * sinValues[i] + val2 * cosValues[i];
        }
      }
    }
  }
}

} // namespace detail

/// Adjust RoPE for cached keys after token discard.
///
/// After discardKVCache() has moved tail tokens forward by `discardLen`
/// positions, their RoPE embeddings are stale. This function rotates the
/// key embeddings backward by `discardLen` to realign with new positions.
///
/// IMPORTANT: Must be called AFTER discardKVCache(), not before.
/// The function uses pre-discard `currentTokenCount` to compute how many
/// tail tokens to adjust, but operates on the already-relocated data.
///
/// @param inverseFreqs  Precomputed inverse RoPE frequencies.
inline void adjustKeyCacheRope(float *kvBuffers[], int numLayers, int headNum,
                               int maxTokenLen, int hiddenSize,
                               int keepTokenNum, int discardLen,
                               int currentTokenCount,
                               const std::vector<float> &inverseFreqs) {
  if (discardLen <= 0)
    return;
  const int srcStartIndex = keepTokenNum + discardLen;
  if (srcStartIndex >= currentTokenCount)
    return;

  const int tokenCount = currentTokenCount - srcStartIndex;
  const int halfSize = hiddenSize / 2;

  // Rotate backward to align keys with new positions.
  std::vector<float> cosValues(halfSize);
  std::vector<float> sinValues(halfSize);
  const float delta = -static_cast<float>(discardLen);
  for (int i = 0; i < halfSize; ++i) {
    const float angle = inverseFreqs[i] * delta;
    cosValues[i] = std::cos(angle);
    sinValues[i] = std::sin(angle);
  }

  detail::applyRotaryDeltaToSlice(kvBuffers, numLayers, headNum, maxTokenLen,
                                  hiddenSize, keepTokenNum, tokenCount,
                                  cosValues, sinValues);
}

} // namespace kvcache
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_KVCACHEMANAGER_H
