//===- F16Bits.h - IEEE binary16 helpers for the Qwen3-VL ABI ------------===//
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
// Storage type for the compiled vision/decoder shims. Weights, activations,
// and positional tables are raw IEEE fp16 bits (uint16_t), matching
// codegen/vision_shim.cpp and codegen/decoder_shim.cpp. Host arithmetic that
// still needs a numeric compare (argmax) converts one value at a time.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_MODELS_QWEN3_VL_F16BITS_H
#define BUDDY_MODELS_QWEN3_VL_F16BITS_H

#include <cstdint>
#include <cstring>
#include <vector>

inline uint16_t f32ToF16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t expField = (bits >> 23) & 0xffu;
  const uint32_t mant = bits & 0x7fffffu;

  if (expField == 0xffu) {
    if (mant == 0)
      return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | 0x7e00u | (mant >> 13));
  }

  const int32_t exp = static_cast<int32_t>(expField) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    const uint32_t full = mant | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(1 - exp + 13);
    uint32_t half = full >> shift;
    const uint32_t remMask = (1u << shift) - 1u;
    const uint32_t rem = full & remMask;
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u)))
      ++half;
    return static_cast<uint16_t>(sign | half);
  }
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7c00u);

  uint16_t half = static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
    ++half;
  return half;
}

inline float f16BitsToF32(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
  uint32_t exp = (half >> 10) & 0x1fu;
  uint32_t mant = half & 0x3ffu;
  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 127 - 14;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      out = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float value = 0.f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

inline std::vector<uint16_t> f32ToF16Bits(const std::vector<float> &values) {
  std::vector<uint16_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    out[i] = f32ToF16Bits(values[i]);
  return out;
}

inline int argmaxF16(const uint16_t *row, size_t count) {
  int best = 0;
  float bestValue = f16BitsToF32(row[0]);
  for (size_t i = 1; i < count; ++i) {
    const float value = f16BitsToF32(row[i]);
    if (value > bestValue) {
      bestValue = value;
      best = static_cast<int>(i);
    }
  }
  return best;
}

#endif // BUDDY_MODELS_QWEN3_VL_F16BITS_H
