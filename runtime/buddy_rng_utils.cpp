//===- buddy_rng_utils.cpp ------------------------------------------------===//
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
// Runtime RNG helpers used by Buddy's external-call based lowerings.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

namespace {

bool envTruthy(const char *key) {
  const char *val = std::getenv(key);
  if (!val)
    return false;
  return val[0] == '1' || val[0] == 't' || val[0] == 'T' || val[0] == 'y' ||
         val[0] == 'Y' || val[0] == 'o' || val[0] == 'O';
}

uint32_t readSeed() {
  const char *val = std::getenv("BUDDY_RNG_SEED");
  if (!val || !val[0])
    return 0u;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(val, &end, 10);
  if (end == val)
    return 0u;
  return static_cast<uint32_t>(parsed);
}

std::mt19937 &getGenerator() {
  static thread_local std::mt19937 gen(0u);
  static thread_local bool initialized = false;
  if (!initialized) {
    gen.seed(readSeed());
    initialized = true;
  }
  return gen;
}

float nextUniformFloat(std::mt19937 &gen) {
  // Match PyTorch's float32 RNG path: take the low 24 bits of a 32-bit MT draw.
  // This aligns with `torch.rand(dtype=torch.float32)` under
  // `torch.manual_seed`.
  const uint32_t r = gen();
  const uint32_t bits = r & 0x00FFFFFFu;
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

double nextUniformDouble(std::mt19937 &gen) {
  // Match PyTorch's float64 RNG path:
  // - Build a 64-bit sample from two 32-bit draws (hi, lo).
  // - Use the low 53 bits and scale by 2^-53.
  const uint64_t hi = static_cast<uint64_t>(gen());
  const uint64_t lo = static_cast<uint64_t>(gen());
  const uint64_t x = (hi << 32) | lo;
  const uint64_t bits = x & ((1ULL << 53) - 1);
  return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
}

template <int Rank>
void initOutLike(StridedMemRefType<float, Rank> *out,
                 const StridedMemRefType<float, Rank> *like) {
  out->offset = 0;
  int64_t numel = 1;
  if constexpr (Rank > 0) {
    for (int i = 0; i < Rank; ++i) {
      out->sizes[i] = like->sizes[i];
      numel *= out->sizes[i];
    }
    for (int i = 0; i < Rank; ++i) {
      int64_t stride = 1;
      for (int j = i + 1; j < Rank; ++j)
        stride *= out->sizes[j];
      out->strides[i] = stride;
    }
  }

  // The caller provides an uninitialized result memref descriptor via the
  // C-interface wrapper. Allocate storage here and fill the descriptor.
  void *buf = nullptr;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
  if (bytes == 0) {
    out->basePtr = nullptr;
    out->data = nullptr;
    return;
  }
  if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
    buf = std::malloc(bytes);
  out->basePtr = static_cast<float *>(buf);
  out->data = static_cast<float *>(buf);
}

template <int Rank>
void bernoulliF32(StridedMemRefType<float, Rank> *out,
                  StridedMemRefType<float, Rank> *prob) {
  initOutLike<Rank>(out, prob);

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = prob->begin(); it != prob->end(); ++it) {
    const float p = *it;
    const float u = nextUniformFloat(gen);
    (*out)[it.getIndices()] = (u < p) ? 1.0f : 0.0f;
  }
}

template <int Rank>
void bernoulliF32F64Rng(StridedMemRefType<float, Rank> *out,
                        StridedMemRefType<float, Rank> *prob) {
  initOutLike<Rank>(out, prob);

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = prob->begin(); it != prob->end(); ++it) {
    const double p = static_cast<double>(*it);
    const double u = nextUniformDouble(gen);
    (*out)[it.getIndices()] = (u < p) ? 1.0f : 0.0f;
  }
}

template <int Rank>
void initOutFromDescriptor(StridedMemRefType<float, Rank> *out) {
  out->offset = 0;
  int64_t numel = 1;
  if constexpr (Rank > 0) {
    for (int i = 0; i < Rank; ++i) {
      const int64_t sz = out->sizes[i];
      if (sz <= 0) {
        numel = 0;
        break;
      }
      numel *= sz;
    }

    // Some C-interface wrappers may not provide strides; rebuild a contiguous
    // row-major layout if needed.
    bool needStrides = false;
    for (int i = 0; i < Rank; ++i)
      needStrides |= (out->strides[i] == 0);
    if (needStrides) {
      for (int i = 0; i < Rank; ++i) {
        int64_t stride = 1;
        for (int j = i + 1; j < Rank; ++j)
          stride *= out->sizes[j];
        out->strides[i] = stride;
      }
    }
  }

  if (numel == 0) {
    out->basePtr = nullptr;
    out->data = nullptr;
    return;
  }

  // If the wrapper already provided storage, reuse it.
  if (out->data != nullptr)
    return;

  void *buf = nullptr;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
  if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
    buf = std::malloc(bytes);
  out->basePtr = static_cast<float *>(buf);
  out->data = static_cast<float *>(buf);
}

template <int Rank> void randF32(StridedMemRefType<float, Rank> *out) {
  initOutFromDescriptor<Rank>(out);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = out->begin(); it != out->end(); ++it)
    (*out)[it.getIndices()] = nextUniformFloat(gen);
}

template <int Rank>
void randLikeF32(StridedMemRefType<float, Rank> *out,
                 StridedMemRefType<float, Rank> *like) {
  initOutLike<Rank>(out, like);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = out->begin(); it != out->end(); ++it)
    (*out)[it.getIndices()] = nextUniformFloat(gen);
}

template <int Rank>
void geometricF32(StridedMemRefType<float, Rank> *out,
                  StridedMemRefType<float, Rank> *prob) {
  initOutLike<Rank>(out, prob);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = prob->begin(); it != prob->end(); ++it) {
    const double p = static_cast<double>(*it);
    if (!(p > 0.0) || p >= 1.0) {
      (*out)[it.getIndices()] = 1.0f;
      continue;
    }
    const double denom = std::log1p(-p);
    double u = nextUniformDouble(gen);
    if (u <= 0.0)
      u = std::numeric_limits<double>::min();
    const double v = std::floor(std::log(u) / denom) + 1.0;
    (*out)[it.getIndices()] = static_cast<float>(v);
  }
}

template <int Rank>
void exponentialF32(StridedMemRefType<float, Rank> *out,
                    StridedMemRefType<float, Rank> *lambdTensor) {
  initOutLike<Rank>(out, lambdTensor);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = out->begin(); it != out->end(); ++it) {
    const double lambd = static_cast<double>((*lambdTensor)[it.getIndices()]);
    const double u = nextUniformDouble(gen);
    const double sample = -std::log1p(-u) / lambd;
    (*out)[it.getIndices()] = static_cast<float>(sample);
  }
}

} // namespace

extern "C" {

void _mlir_ciface_buddy_bernoulli_f32_r0_1(StridedMemRefType<float, 0> *out,
                                           StridedMemRefType<float, 0> *p) {
  bernoulliF32<0>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r0_2(
    StridedMemRefType<float, 0> *out, StridedMemRefType<float, 0> * /*self*/,
    StridedMemRefType<float, 0> *p) {
  bernoulliF32<0>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r1_1(StridedMemRefType<float, 1> *out,
                                           StridedMemRefType<float, 1> *p) {
  bernoulliF32<1>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r1_2(
    StridedMemRefType<float, 1> *out, StridedMemRefType<float, 1> * /*self*/,
    StridedMemRefType<float, 1> *p) {
  bernoulliF32<1>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r2_1(StridedMemRefType<float, 2> *out,
                                           StridedMemRefType<float, 2> *p) {
  bernoulliF32<2>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r2_2(
    StridedMemRefType<float, 2> *out, StridedMemRefType<float, 2> * /*self*/,
    StridedMemRefType<float, 2> *p) {
  bernoulliF32<2>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r3_1(StridedMemRefType<float, 3> *out,
                                           StridedMemRefType<float, 3> *p) {
  bernoulliF32<3>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r3_2(
    StridedMemRefType<float, 3> *out, StridedMemRefType<float, 3> * /*self*/,
    StridedMemRefType<float, 3> *p) {
  bernoulliF32<3>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r4_1(StridedMemRefType<float, 4> *out,
                                           StridedMemRefType<float, 4> *p) {
  bernoulliF32<4>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_r4_2(
    StridedMemRefType<float, 4> *out, StridedMemRefType<float, 4> * /*self*/,
    StridedMemRefType<float, 4> *p) {
  bernoulliF32<4>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_f64rng_r0_1(
    StridedMemRefType<float, 0> *out, StridedMemRefType<float, 0> *p) {
  bernoulliF32F64Rng<0>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_f64rng_r1_1(
    StridedMemRefType<float, 1> *out, StridedMemRefType<float, 1> *p) {
  bernoulliF32F64Rng<1>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_f64rng_r2_1(
    StridedMemRefType<float, 2> *out, StridedMemRefType<float, 2> *p) {
  bernoulliF32F64Rng<2>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_f64rng_r3_1(
    StridedMemRefType<float, 3> *out, StridedMemRefType<float, 3> *p) {
  bernoulliF32F64Rng<3>(out, p);
}

void _mlir_ciface_buddy_bernoulli_f32_f64rng_r4_1(
    StridedMemRefType<float, 4> *out, StridedMemRefType<float, 4> *p) {
  bernoulliF32F64Rng<4>(out, p);
}

void _mlir_ciface_buddy_rand_f32_r0(StridedMemRefType<float, 0> *out) {
  randF32<0>(out);
}

void _mlir_ciface_buddy_rand_f32_r1(StridedMemRefType<float, 1> *out) {
  randF32<1>(out);
}

void _mlir_ciface_buddy_rand_f32_r2(StridedMemRefType<float, 2> *out) {
  randF32<2>(out);
}

void _mlir_ciface_buddy_rand_f32_r3(StridedMemRefType<float, 3> *out) {
  randF32<3>(out);
}

void _mlir_ciface_buddy_rand_f32_r4(StridedMemRefType<float, 4> *out) {
  randF32<4>(out);
}

void _mlir_ciface_buddy_rand_f32_like_r0_1(StridedMemRefType<float, 0> *out,
                                           StridedMemRefType<float, 0> *like) {
  randLikeF32<0>(out, like);
}

void _mlir_ciface_buddy_rand_f32_like_r1_1(StridedMemRefType<float, 1> *out,
                                           StridedMemRefType<float, 1> *like) {
  randLikeF32<1>(out, like);
}

void _mlir_ciface_buddy_rand_f32_like_r2_1(StridedMemRefType<float, 2> *out,
                                           StridedMemRefType<float, 2> *like) {
  randLikeF32<2>(out, like);
}

void _mlir_ciface_buddy_rand_f32_like_r3_1(StridedMemRefType<float, 3> *out,
                                           StridedMemRefType<float, 3> *like) {
  randLikeF32<3>(out, like);
}

void _mlir_ciface_buddy_rand_f32_like_r4_1(StridedMemRefType<float, 4> *out,
                                           StridedMemRefType<float, 4> *like) {
  randLikeF32<4>(out, like);
}

void _mlir_ciface_buddy_geometric_f32_r0_1(StridedMemRefType<float, 0> *out,
                                           StridedMemRefType<float, 0> *prob) {
  geometricF32<0>(out, prob);
}

void _mlir_ciface_buddy_geometric_f32_r1_1(StridedMemRefType<float, 1> *out,
                                           StridedMemRefType<float, 1> *prob) {
  geometricF32<1>(out, prob);
}

void _mlir_ciface_buddy_geometric_f32_r2_1(StridedMemRefType<float, 2> *out,
                                           StridedMemRefType<float, 2> *prob) {
  geometricF32<2>(out, prob);
}

void _mlir_ciface_buddy_geometric_f32_r3_1(StridedMemRefType<float, 3> *out,
                                           StridedMemRefType<float, 3> *prob) {
  geometricF32<3>(out, prob);
}

void _mlir_ciface_buddy_geometric_f32_r4_1(StridedMemRefType<float, 4> *out,
                                           StridedMemRefType<float, 4> *prob) {
  geometricF32<4>(out, prob);
}

void _mlir_ciface_buddy_exponential_f32_r0_1(
    StridedMemRefType<float, 0> *out, StridedMemRefType<float, 0> *lambd) {
  exponentialF32<0>(out, lambd);
}

void _mlir_ciface_buddy_exponential_f32_r1_1(
    StridedMemRefType<float, 1> *out, StridedMemRefType<float, 1> *lambd) {
  exponentialF32<1>(out, lambd);
}

void _mlir_ciface_buddy_exponential_f32_r2_1(
    StridedMemRefType<float, 2> *out, StridedMemRefType<float, 2> *lambd) {
  exponentialF32<2>(out, lambd);
}

void _mlir_ciface_buddy_exponential_f32_r3_1(
    StridedMemRefType<float, 3> *out, StridedMemRefType<float, 3> *lambd) {
  exponentialF32<3>(out, lambd);
}

void _mlir_ciface_buddy_exponential_f32_r4_1(
    StridedMemRefType<float, 4> *out, StridedMemRefType<float, 4> *lambd) {
  exponentialF32<4>(out, lambd);
}

} // extern "C"
