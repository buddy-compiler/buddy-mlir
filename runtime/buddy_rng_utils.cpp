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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
void randnLikeF32(StridedMemRefType<float, Rank> *out,
                  StridedMemRefType<float, Rank> *like) {
  initOutLike<Rank>(out, like);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  bool hasSpare = false;
  float spare = 0.0f;
  for (auto it = out->begin(); it != out->end(); ++it) {
    if (hasSpare) {
      (*out)[it.getIndices()] = spare;
      hasSpare = false;
      continue;
    }

    double u1 = nextUniformDouble(gen);
    const double u2 = nextUniformDouble(gen);
    if (u1 <= 0.0)
      u1 = std::numeric_limits<double>::min();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * M_PI * u2;
    const float z0 = static_cast<float>(r * std::cos(theta));
    const float z1 = static_cast<float>(r * std::sin(theta));
    (*out)[it.getIndices()] = z0;
    spare = z1;
    hasSpare = true;
  }
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

template <int Rank>
void uniformF32(StridedMemRefType<float, Rank> *out,
                StridedMemRefType<float, Rank> *like,
                StridedMemRefType<float, 0> *fromTensor,
                StridedMemRefType<float, 0> *toTensor) {
  initOutLike<Rank>(out, like);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  const float fromV = fromTensor->data[fromTensor->offset];
  const float toV = toTensor->data[toTensor->offset];
  std::uniform_real_distribution<float> dist(fromV, toV);
  for (auto it = out->begin(); it != out->end(); ++it)
    (*out)[it.getIndices()] = dist(gen);
}

template <int Rank>
void cauchyF32(StridedMemRefType<float, Rank> *out,
               StridedMemRefType<float, Rank> *like,
               StridedMemRefType<float, 0> *medianTensor,
               StridedMemRefType<float, 0> *sigmaTensor) {
  initOutLike<Rank>(out, like);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  const float medianV = medianTensor->data[medianTensor->offset];
  const float sigmaV = sigmaTensor->data[sigmaTensor->offset];
  std::cauchy_distribution<double> dist(static_cast<double>(medianV),
                                        static_cast<double>(sigmaV));
  for (auto it = out->begin(); it != out->end(); ++it)
    (*out)[it.getIndices()] = static_cast<float>(dist(gen));
}

template <typename T, int Rank>
void initOutLikeTyped(StridedMemRefType<T, Rank> *out,
                      const StridedMemRefType<T, Rank> *like) {
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

  void *buf = nullptr;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(T);
  if (bytes == 0) {
    out->basePtr = nullptr;
    out->data = nullptr;
    return;
  }
  if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
    buf = std::malloc(bytes);
  out->basePtr = static_cast<T *>(buf);
  out->data = static_cast<T *>(buf);
}

template <typename T, int Rank>
void initOutFromDescriptorTyped(StridedMemRefType<T, Rank> *out) {
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

  if (out->data != nullptr)
    return;

  void *buf = nullptr;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(T);
  if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
    buf = std::malloc(bytes);
  out->basePtr = static_cast<T *>(buf);
  out->data = static_cast<T *>(buf);
}

float nextStandardNormalF32(std::mt19937 &gen, bool &hasSpare, float &spare) {
  if (hasSpare) {
    hasSpare = false;
    return spare;
  }
  double u1 = nextUniformDouble(gen);
  const double u2 = nextUniformDouble(gen);
  if (u1 <= 0.0)
    u1 = std::numeric_limits<double>::min();
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double theta = 2.0 * M_PI * u2;
  const float z0 = static_cast<float>(r * std::cos(theta));
  const float z1 = static_cast<float>(r * std::sin(theta));
  spare = z1;
  hasSpare = true;
  return z0;
}

template <typename T> T readScalar0(const StridedMemRefType<T, 0> *s) {
  return s->data[s->offset];
}

template <int Rank>
void normalF32(StridedMemRefType<float, Rank> *out,
               StridedMemRefType<float, Rank> *mean,
               StridedMemRefType<float, Rank> *stddev) {
  initOutLikeTyped<float, Rank>(out, mean);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  bool hasSpare = false;
  float spare = 0.0f;
  for (auto it = out->begin(); it != out->end(); ++it) {
    const float mu = (*mean)[it.getIndices()];
    const float sigma = (*stddev)[it.getIndices()];
    const float z = nextStandardNormalF32(gen, hasSpare, spare);
    (*out)[it.getIndices()] = mu + sigma * z;
  }
}

template <int Rank>
void logNormalF32(StridedMemRefType<float, Rank> *out,
                  StridedMemRefType<float, Rank> *mean,
                  StridedMemRefType<float, Rank> *stddev) {
  initOutLikeTyped<float, Rank>(out, mean);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  bool hasSpare = false;
  float spare = 0.0f;
  for (auto it = out->begin(); it != out->end(); ++it) {
    const float mu = (*mean)[it.getIndices()];
    const float sigma = (*stddev)[it.getIndices()];
    const float z = nextStandardNormalF32(gen, hasSpare, spare);
    (*out)[it.getIndices()] = std::exp(mu + sigma * z);
  }
}

template <int Rank>
void poissonF32(StridedMemRefType<float, Rank> *out,
                StridedMemRefType<float, Rank> *rate) {
  initOutLikeTyped<float, Rank>(out, rate);
  if (out->data == nullptr)
    return;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  for (auto it = out->begin(); it != out->end(); ++it) {
    double lambda = static_cast<double>((*rate)[it.getIndices()]);
    if (lambda < 0.0)
      lambda = 0.0;
    std::poisson_distribution<int64_t> dist(lambda);
    (*out)[it.getIndices()] = static_cast<float>(dist(gen));
  }
}

template <int Rank>
void multinomialF32I64(StridedMemRefType<int64_t, Rank> *out,
                       StridedMemRefType<float, Rank> *probs,
                       StridedMemRefType<int64_t, 0> *numSamples) {
  static_assert(Rank >= 1, "multinomial requires rank >= 1");

  const int64_t ncat = probs->sizes[Rank - 1];
  const int64_t nsamples = readScalar0<int64_t>(numSamples);
  if (ncat <= 0 || nsamples <= 0) {
    out->basePtr = nullptr;
    out->data = nullptr;
    return;
  }

  out->offset = 0;
  int64_t numel = 1;
  for (int i = 0; i < Rank - 1; ++i) {
    out->sizes[i] = probs->sizes[i];
    numel *= out->sizes[i];
  }
  out->sizes[Rank - 1] = nsamples;
  numel *= nsamples;

  for (int i = 0; i < Rank; ++i) {
    int64_t stride = 1;
    for (int j = i + 1; j < Rank; ++j)
      stride *= out->sizes[j];
    out->strides[i] = stride;
  }

  void *buf = nullptr;
  const size_t bytes = static_cast<size_t>(numel) * sizeof(int64_t);
  if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
    buf = std::malloc(bytes);
  out->basePtr = static_cast<int64_t *>(buf);
  out->data = static_cast<int64_t *>(buf);

  int64_t batch = 1;
  for (int i = 0; i < Rank - 1; ++i)
    batch *= probs->sizes[i];

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  std::array<int64_t, Rank> inIdx{};
  std::array<int64_t, Rank> outIdx{};

  for (int64_t b = 0; b < batch; ++b) {
    int64_t tmp = b;
    for (int d = Rank - 2; d >= 0; --d) {
      const int64_t sz = probs->sizes[d];
      inIdx[d] = tmp % sz;
      outIdx[d] = inIdx[d];
      tmp /= sz;
    }

    std::vector<double> weights(static_cast<size_t>(ncat), 0.0);
    for (int64_t c = 0; c < ncat; ++c) {
      inIdx[Rank - 1] = c;
      const double w = static_cast<double>((*probs)[inIdx]);
      weights[static_cast<size_t>(c)] = std::max(0.0, w);
    }

    double total = 0.0;
    for (double w : weights)
      total += w;
    if (!(total > 0.0))
      std::fill(weights.begin(), weights.end(), 1.0);

    std::vector<double> pool = weights;
    for (int64_t sample = 0; sample < nsamples; ++sample) {
      double poolTotal = 0.0;
      for (double w : pool)
        poolTotal += w;

      int64_t chosen = 0;
      if (poolTotal > 0.0) {
        const double u = nextUniformDouble(gen) * poolTotal;
        double csum = 0.0;
        for (int64_t c = 0; c < ncat; ++c) {
          csum += pool[static_cast<size_t>(c)];
          if (u <= csum) {
            chosen = c;
            break;
          }
        }
      }

      outIdx[Rank - 1] = sample;
      (*out)[outIdx] = chosen;
      pool[static_cast<size_t>(chosen)] = 0.0;
    }
  }
}

template <int Rank>
void randintLikeI64(StridedMemRefType<int64_t, Rank> *out,
                    StridedMemRefType<int64_t, Rank> *like,
                    StridedMemRefType<int64_t, 0> *low,
                    StridedMemRefType<int64_t, 0> *high) {
  initOutLikeTyped<int64_t, Rank>(out, like);
  if (out->data == nullptr)
    return;

  const int64_t lowV = readScalar0<int64_t>(low);
  const int64_t highV = readScalar0<int64_t>(high);

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  if (highV <= lowV) {
    for (auto it = out->begin(); it != out->end(); ++it)
      (*out)[it.getIndices()] = lowV;
    return;
  }

  std::uniform_int_distribution<int64_t> dist(lowV, highV - 1);
  for (auto it = out->begin(); it != out->end(); ++it)
    (*out)[it.getIndices()] = dist(gen);
}

template <int Rank>
void rreluWithNoiseF32(StridedMemRefType<float, Rank> *out,
                       StridedMemRefType<float, Rank> *self,
                       StridedMemRefType<float, Rank> *noise,
                       StridedMemRefType<float, 0> *lower,
                       StridedMemRefType<float, 0> *upper,
                       StridedMemRefType<int64_t, 0> *training) {
  initOutLikeTyped<float, Rank>(out, self);
  if (out->data == nullptr)
    return;

  const float lowerV = readScalar0<float>(lower);
  const float upperV = readScalar0<float>(upper);
  const bool isTraining = readScalar0<int64_t>(training) != 0;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  const float avgSlope = (lowerV + upperV) * 0.5f;
  const float span = upperV - lowerV;

  for (auto it = out->begin(); it != out->end(); ++it) {
    const auto idx = it.getIndices();
    const float x = (*self)[idx];

    if (x <= 0.0f) {
      const float slope =
          isTraining ? (lowerV + span * nextUniformFloat(gen)) : avgSlope;
      (*out)[idx] = x * slope;
      (*noise)[idx] = slope;
    } else {
      (*out)[idx] = x;
      (*noise)[idx] = 1.0f;
    }
  }
}

template <int Rank> struct PairMemRefF32 {
  StridedMemRefType<float, Rank> first;
  StridedMemRefType<float, Rank> second;
};

template <int Rank>
void rreluWithNoiseFunctionalF32(StridedMemRefType<float, Rank> *out,
                                 StridedMemRefType<float, Rank> *noiseOut,
                                 StridedMemRefType<float, Rank> *self,
                                 StridedMemRefType<float, Rank> * /*noise*/,
                                 StridedMemRefType<float, 0> *lower,
                                 StridedMemRefType<float, 0> *upper,
                                 StridedMemRefType<int64_t, 0> *training) {
  initOutLikeTyped<float, Rank>(out, self);
  initOutLikeTyped<float, Rank>(noiseOut, self);
  if (out->data == nullptr || noiseOut->data == nullptr)
    return;

  const float lowerV = readScalar0<float>(lower);
  const float upperV = readScalar0<float>(upper);
  const bool isTraining = readScalar0<int64_t>(training) != 0;

  auto &gen = getGenerator();
  if (envTruthy("BUDDY_OC_VALIDATE_NUMERIC"))
    gen.seed(readSeed());

  const float avgSlope = (lowerV + upperV) * 0.5f;
  const float span = upperV - lowerV;

  for (auto it = out->begin(); it != out->end(); ++it) {
    const auto idx = it.getIndices();
    const float x = (*self)[idx];

    if (x <= 0.0f) {
      const float slope =
          isTraining ? (lowerV + span * nextUniformFloat(gen)) : avgSlope;
      (*out)[idx] = x * slope;
      (*noiseOut)[idx] = slope;
    } else {
      (*out)[idx] = x;
      (*noiseOut)[idx] = 1.0f;
    }
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

void _mlir_ciface_buddy_randn_f32_like_r0_1(StridedMemRefType<float, 0> *out,
                                            StridedMemRefType<float, 0> *like) {
  randnLikeF32<0>(out, like);
}

void _mlir_ciface_buddy_randn_f32_like_r1_1(StridedMemRefType<float, 1> *out,
                                            StridedMemRefType<float, 1> *like) {
  randnLikeF32<1>(out, like);
}

void _mlir_ciface_buddy_randn_f32_like_r2_1(StridedMemRefType<float, 2> *out,
                                            StridedMemRefType<float, 2> *like) {
  randnLikeF32<2>(out, like);
}

void _mlir_ciface_buddy_randn_f32_like_r3_1(StridedMemRefType<float, 3> *out,
                                            StridedMemRefType<float, 3> *like) {
  randnLikeF32<3>(out, like);
}

void _mlir_ciface_buddy_randn_f32_like_r4_1(StridedMemRefType<float, 4> *out,
                                            StridedMemRefType<float, 4> *like) {
  randnLikeF32<4>(out, like);
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

void _mlir_ciface_buddy_uniform_f32_r0_3(StridedMemRefType<float, 0> *out,
                                         StridedMemRefType<float, 0> *like,
                                         StridedMemRefType<float, 0> *from,
                                         StridedMemRefType<float, 0> *to) {
  uniformF32<0>(out, like, from, to);
}

void _mlir_ciface_buddy_uniform_f32_r1_3(StridedMemRefType<float, 1> *out,
                                         StridedMemRefType<float, 1> *like,
                                         StridedMemRefType<float, 0> *from,
                                         StridedMemRefType<float, 0> *to) {
  uniformF32<1>(out, like, from, to);
}

void _mlir_ciface_buddy_uniform_f32_r2_3(StridedMemRefType<float, 2> *out,
                                         StridedMemRefType<float, 2> *like,
                                         StridedMemRefType<float, 0> *from,
                                         StridedMemRefType<float, 0> *to) {
  uniformF32<2>(out, like, from, to);
}

void _mlir_ciface_buddy_uniform_f32_r3_3(StridedMemRefType<float, 3> *out,
                                         StridedMemRefType<float, 3> *like,
                                         StridedMemRefType<float, 0> *from,
                                         StridedMemRefType<float, 0> *to) {
  uniformF32<3>(out, like, from, to);
}

void _mlir_ciface_buddy_uniform_f32_r4_3(StridedMemRefType<float, 4> *out,
                                         StridedMemRefType<float, 4> *like,
                                         StridedMemRefType<float, 0> *from,
                                         StridedMemRefType<float, 0> *to) {
  uniformF32<4>(out, like, from, to);
}

void _mlir_ciface_buddy_cauchy_f32_r0_3(StridedMemRefType<float, 0> *out,
                                        StridedMemRefType<float, 0> *like,
                                        StridedMemRefType<float, 0> *median,
                                        StridedMemRefType<float, 0> *sigma) {
  cauchyF32<0>(out, like, median, sigma);
}

void _mlir_ciface_buddy_cauchy_f32_r1_3(StridedMemRefType<float, 1> *out,
                                        StridedMemRefType<float, 1> *like,
                                        StridedMemRefType<float, 0> *median,
                                        StridedMemRefType<float, 0> *sigma) {
  cauchyF32<1>(out, like, median, sigma);
}

void _mlir_ciface_buddy_cauchy_f32_r2_3(StridedMemRefType<float, 2> *out,
                                        StridedMemRefType<float, 2> *like,
                                        StridedMemRefType<float, 0> *median,
                                        StridedMemRefType<float, 0> *sigma) {
  cauchyF32<2>(out, like, median, sigma);
}

void _mlir_ciface_buddy_cauchy_f32_r3_3(StridedMemRefType<float, 3> *out,
                                        StridedMemRefType<float, 3> *like,
                                        StridedMemRefType<float, 0> *median,
                                        StridedMemRefType<float, 0> *sigma) {
  cauchyF32<3>(out, like, median, sigma);
}

void _mlir_ciface_buddy_cauchy_f32_r4_3(StridedMemRefType<float, 4> *out,
                                        StridedMemRefType<float, 4> *like,
                                        StridedMemRefType<float, 0> *median,
                                        StridedMemRefType<float, 0> *sigma) {
  cauchyF32<4>(out, like, median, sigma);
}

void _mlir_ciface_buddy_normal_f32_r0_2(StridedMemRefType<float, 0> *out,
                                        StridedMemRefType<float, 0> *mean,
                                        StridedMemRefType<float, 0> *stddev) {
  normalF32<0>(out, mean, stddev);
}

void _mlir_ciface_buddy_normal_f32_r1_2(StridedMemRefType<float, 1> *out,
                                        StridedMemRefType<float, 1> *mean,
                                        StridedMemRefType<float, 1> *stddev) {
  normalF32<1>(out, mean, stddev);
}

void _mlir_ciface_buddy_normal_f32_r2_2(StridedMemRefType<float, 2> *out,
                                        StridedMemRefType<float, 2> *mean,
                                        StridedMemRefType<float, 2> *stddev) {
  normalF32<2>(out, mean, stddev);
}

void _mlir_ciface_buddy_normal_f32_r3_2(StridedMemRefType<float, 3> *out,
                                        StridedMemRefType<float, 3> *mean,
                                        StridedMemRefType<float, 3> *stddev) {
  normalF32<3>(out, mean, stddev);
}

void _mlir_ciface_buddy_normal_f32_r4_2(StridedMemRefType<float, 4> *out,
                                        StridedMemRefType<float, 4> *mean,
                                        StridedMemRefType<float, 4> *stddev) {
  normalF32<4>(out, mean, stddev);
}

void _mlir_ciface_buddy_log_normal_f32_r0_2(
    StridedMemRefType<float, 0> *out, StridedMemRefType<float, 0> *mean,
    StridedMemRefType<float, 0> *stddev) {
  logNormalF32<0>(out, mean, stddev);
}

void _mlir_ciface_buddy_log_normal_f32_r1_2(
    StridedMemRefType<float, 1> *out, StridedMemRefType<float, 1> *mean,
    StridedMemRefType<float, 1> *stddev) {
  logNormalF32<1>(out, mean, stddev);
}

void _mlir_ciface_buddy_log_normal_f32_r2_2(
    StridedMemRefType<float, 2> *out, StridedMemRefType<float, 2> *mean,
    StridedMemRefType<float, 2> *stddev) {
  logNormalF32<2>(out, mean, stddev);
}

void _mlir_ciface_buddy_log_normal_f32_r3_2(
    StridedMemRefType<float, 3> *out, StridedMemRefType<float, 3> *mean,
    StridedMemRefType<float, 3> *stddev) {
  logNormalF32<3>(out, mean, stddev);
}

void _mlir_ciface_buddy_log_normal_f32_r4_2(
    StridedMemRefType<float, 4> *out, StridedMemRefType<float, 4> *mean,
    StridedMemRefType<float, 4> *stddev) {
  logNormalF32<4>(out, mean, stddev);
}

void _mlir_ciface_buddy_poisson_f32_r0_1(StridedMemRefType<float, 0> *out,
                                         StridedMemRefType<float, 0> *rate) {
  poissonF32<0>(out, rate);
}

void _mlir_ciface_buddy_poisson_f32_r1_1(StridedMemRefType<float, 1> *out,
                                         StridedMemRefType<float, 1> *rate) {
  poissonF32<1>(out, rate);
}

void _mlir_ciface_buddy_poisson_f32_r2_1(StridedMemRefType<float, 2> *out,
                                         StridedMemRefType<float, 2> *rate) {
  poissonF32<2>(out, rate);
}

void _mlir_ciface_buddy_poisson_f32_r3_1(StridedMemRefType<float, 3> *out,
                                         StridedMemRefType<float, 3> *rate) {
  poissonF32<3>(out, rate);
}

void _mlir_ciface_buddy_poisson_f32_r4_1(StridedMemRefType<float, 4> *out,
                                         StridedMemRefType<float, 4> *rate) {
  poissonF32<4>(out, rate);
}

void _mlir_ciface_buddy_multinomial_f32_i64_r1_2(
    StridedMemRefType<int64_t, 1> *out, StridedMemRefType<float, 1> *probs,
    StridedMemRefType<int64_t, 0> *num_samples) {
  multinomialF32I64<1>(out, probs, num_samples);
}

void _mlir_ciface_buddy_multinomial_f32_i64_r2_2(
    StridedMemRefType<int64_t, 2> *out, StridedMemRefType<float, 2> *probs,
    StridedMemRefType<int64_t, 0> *num_samples) {
  multinomialF32I64<2>(out, probs, num_samples);
}

void _mlir_ciface_buddy_multinomial_f32_i64_r3_2(
    StridedMemRefType<int64_t, 3> *out, StridedMemRefType<float, 3> *probs,
    StridedMemRefType<int64_t, 0> *num_samples) {
  multinomialF32I64<3>(out, probs, num_samples);
}

void _mlir_ciface_buddy_multinomial_f32_i64_r4_2(
    StridedMemRefType<int64_t, 4> *out, StridedMemRefType<float, 4> *probs,
    StridedMemRefType<int64_t, 0> *num_samples) {
  multinomialF32I64<4>(out, probs, num_samples);
}

void _mlir_ciface_buddy_randint_like_i64_r0_3(
    StridedMemRefType<int64_t, 0> *out, StridedMemRefType<int64_t, 0> *like,
    StridedMemRefType<int64_t, 0> *low, StridedMemRefType<int64_t, 0> *high) {
  randintLikeI64<0>(out, like, low, high);
}

void _mlir_ciface_buddy_randint_like_i64_r1_3(
    StridedMemRefType<int64_t, 1> *out, StridedMemRefType<int64_t, 1> *like,
    StridedMemRefType<int64_t, 0> *low, StridedMemRefType<int64_t, 0> *high) {
  randintLikeI64<1>(out, like, low, high);
}

void _mlir_ciface_buddy_randint_like_i64_r2_3(
    StridedMemRefType<int64_t, 2> *out, StridedMemRefType<int64_t, 2> *like,
    StridedMemRefType<int64_t, 0> *low, StridedMemRefType<int64_t, 0> *high) {
  randintLikeI64<2>(out, like, low, high);
}

void _mlir_ciface_buddy_randint_like_i64_r3_3(
    StridedMemRefType<int64_t, 3> *out, StridedMemRefType<int64_t, 3> *like,
    StridedMemRefType<int64_t, 0> *low, StridedMemRefType<int64_t, 0> *high) {
  randintLikeI64<3>(out, like, low, high);
}

void _mlir_ciface_buddy_randint_like_i64_r4_3(
    StridedMemRefType<int64_t, 4> *out, StridedMemRefType<int64_t, 4> *like,
    StridedMemRefType<int64_t, 0> *low, StridedMemRefType<int64_t, 0> *high) {
  randintLikeI64<4>(out, like, low, high);
}

void _mlir_ciface_buddy_rrelu_with_noise_f32_r0_5(
    StridedMemRefType<float, 0> *out, StridedMemRefType<float, 0> *self,
    StridedMemRefType<float, 0> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseF32<0>(out, self, noise, lower, upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_f32_r1_5(
    StridedMemRefType<float, 1> *out, StridedMemRefType<float, 1> *self,
    StridedMemRefType<float, 1> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseF32<1>(out, self, noise, lower, upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_f32_r2_5(
    StridedMemRefType<float, 2> *out, StridedMemRefType<float, 2> *self,
    StridedMemRefType<float, 2> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseF32<2>(out, self, noise, lower, upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_f32_r3_5(
    StridedMemRefType<float, 3> *out, StridedMemRefType<float, 3> *self,
    StridedMemRefType<float, 3> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseF32<3>(out, self, noise, lower, upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_f32_r4_5(
    StridedMemRefType<float, 4> *out, StridedMemRefType<float, 4> *self,
    StridedMemRefType<float, 4> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseF32<4>(out, self, noise, lower, upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_functional_f32_r0_5(
    PairMemRefF32<0> *res, StridedMemRefType<float, 0> *self,
    StridedMemRefType<float, 0> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseFunctionalF32<0>(&res->first, &res->second, self, noise, lower,
                                 upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_functional_f32_r1_5(
    PairMemRefF32<1> *res, StridedMemRefType<float, 1> *self,
    StridedMemRefType<float, 1> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseFunctionalF32<1>(&res->first, &res->second, self, noise, lower,
                                 upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_functional_f32_r2_5(
    PairMemRefF32<2> *res, StridedMemRefType<float, 2> *self,
    StridedMemRefType<float, 2> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseFunctionalF32<2>(&res->first, &res->second, self, noise, lower,
                                 upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_functional_f32_r3_5(
    PairMemRefF32<3> *res, StridedMemRefType<float, 3> *self,
    StridedMemRefType<float, 3> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseFunctionalF32<3>(&res->first, &res->second, self, noise, lower,
                                 upper, training);
}

void _mlir_ciface_buddy_rrelu_with_noise_functional_f32_r4_5(
    PairMemRefF32<4> *res, StridedMemRefType<float, 4> *self,
    StridedMemRefType<float, 4> *noise, StridedMemRefType<float, 0> *lower,
    StridedMemRefType<float, 0> *upper,
    StridedMemRefType<int64_t, 0> *training) {
  rreluWithNoiseFunctionalF32<4>(&res->first, &res->second, self, noise, lower,
                                 upper, training);
}

} // extern "C"
