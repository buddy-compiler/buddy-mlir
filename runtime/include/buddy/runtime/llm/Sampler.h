//===- Sampler.h
//-----------------------------------------------------------===//
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
// Token sampler for LLM inference.
//
// Supports greedy (argmax), temperature scaling, top-K, top-P (nucleus),
// min-P filtering, and repetition penalty.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_SAMPLER_H
#define BUDDY_RUNTIME_LLM_SAMPLER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace buddy {

struct SamplerConfig {
  float temperature = 0.0f;   // 0.0 = greedy (argmax)
  int topK = 0;               // 0 = disabled
  float topP = 1.0f;          // 1.0 = disabled
  float minP = 0.0f;          // 0.0 = disabled
  float repeatPenalty = 1.0f; // 1.0 = disabled
  int repeatLastN = 64;       // window size for repeat penalty
  uint64_t seed = 0;          // 0 = use random device
};

class Sampler {
public:
  explicit Sampler(SamplerConfig config) : config_(config) {
    if (config_.seed != 0) {
      rng_.seed(config_.seed);
    } else {
      std::random_device rd;
      rng_.seed(rd());
    }
  }

  /// Select next token from a logits array of the given vocabSize.
  /// recentTokens provides a window of recently generated token IDs
  /// used for repetition penalty.
  int sample(const float *logits, size_t vocabSize,
             const std::vector<int> &recentTokens) {
    assert(logits && "logits pointer must not be null");
    assert(vocabSize > 0 && "vocabSize must be positive");

    const bool applyPenalty =
        config_.repeatPenalty != 1.0f && !recentTokens.empty();

    // Fast greedy path when logits do not need mutation.
    if (config_.temperature == 0.0f && !applyPenalty) {
      return greedySample(logits, vocabSize);
    }

    // Copy logits for in-place mutation.
    std::vector<float> work(logits, logits + vocabSize);

    // Step 1: Repetition penalty.
    if (applyPenalty) {
      applyRepeatPenalty(work, recentTokens);
    }

    // Greedy path after penalties have been applied.
    if (config_.temperature == 0.0f) {
      return greedySample(work.data(), vocabSize);
    }

    // Step 2: Temperature scaling.
    applyTemperature(work);

    // Step 3-5: Filtering (top-K, top-P, min-P) + softmax + sampling.
    return filteredSample(work);
  }

  const SamplerConfig &config() const { return config_; }

private:
  SamplerConfig config_;
  std::mt19937 rng_;

  /// Index of the largest logit: the first one if several are equal, and 0
  /// if logits[0] is NaN (a NaN never compares greater). This is the index
  /// std::max_element returns.
  ///
  /// Greedy decoding runs this over the whole vocabulary (~150K logits) for
  /// every token. std::max_element is one dependent compare chain; here the
  /// maximum is kept as kLanes independent running maxima, and its first
  /// index is searched kLanes logits at a time. Compilers vectorize both
  /// loops.
  int greedySample(const float *logits, size_t vocabSize) {
    constexpr size_t kLanes = 64;
    if (vocabSize == 0)
      return 0;

    // The maximum.
    float lane[kLanes];
    std::fill(lane, lane + kLanes, logits[0]);
    size_t i = 0;
    for (; i + kLanes <= vocabSize; i += kLanes)
      for (size_t j = 0; j < kLanes; ++j)
        lane[j] = logits[i + j] > lane[j] ? logits[i + j] : lane[j];
    float best = lane[0];
    for (size_t j = 1; j < kLanes; ++j)
      best = lane[j] > best ? lane[j] : best;
    for (; i < vocabSize; ++i)
      best = logits[i] > best ? logits[i] : best;

    // Its first index: skip the blocks of kLanes logits that do not hold it.
    i = 0;
    for (; i + kLanes <= vocabSize; i += kLanes) {
      bool found = false;
      for (size_t j = 0; j < kLanes; ++j)
        found |= logits[i + j] == best;
      if (found)
        break;
    }
    for (; i < vocabSize; ++i)
      if (logits[i] == best)
        return static_cast<int>(i);
    // logits[0] is NaN: so is best, and no logit compares equal to it.
    return 0;
  }

  void applyRepeatPenalty(std::vector<float> &logits,
                          const std::vector<int> &recentTokens) {
    int windowStart =
        static_cast<int>(recentTokens.size()) - config_.repeatLastN;
    if (windowStart < 0)
      windowStart = 0;

    for (size_t i = windowStart; i < recentTokens.size(); ++i) {
      int tokenId = recentTokens[i];
      if (tokenId < 0 || tokenId >= static_cast<int>(logits.size()))
        continue;
      if (logits[tokenId] > 0.0f) {
        logits[tokenId] /= config_.repeatPenalty;
      } else {
        logits[tokenId] *= config_.repeatPenalty;
      }
    }
  }

  void applyTemperature(std::vector<float> &logits) {
    float invTemp = 1.0f / config_.temperature;
    for (float &val : logits) {
      val *= invTemp;
    }
  }

  /// Combined top-K, top-P, min-P filtering followed by softmax and sampling.
  int filteredSample(std::vector<float> &logits) {
    size_t vocabSize = logits.size();

    // Build index-value pairs for sorting.
    std::vector<std::pair<int, float>> candidates(vocabSize);
    for (size_t i = 0; i < vocabSize; ++i) {
      candidates[i] = {static_cast<int>(i), logits[i]};
    }

    // Sort descending by logit value.
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Top-K: keep only the top K candidates.
    size_t limit = candidates.size();
    if (config_.topK > 0 && static_cast<size_t>(config_.topK) < limit) {
      limit = static_cast<size_t>(config_.topK);
    }

    // Softmax over the kept candidates for probability computation.
    // Subtract max for numerical stability (max is candidates[0] after sort).
    float maxLogit = candidates[0].second;
    std::vector<float> probs(limit);
    float sumExp = 0.0f;
    for (size_t i = 0; i < limit; ++i) {
      probs[i] = std::exp(candidates[i].second - maxLogit);
      sumExp += probs[i];
    }
    for (size_t i = 0; i < limit; ++i) {
      probs[i] /= sumExp;
    }

    // Min-P: discard tokens with prob < minP * max_prob.
    if (config_.minP > 0.0f) {
      float threshold = config_.minP * probs[0]; // probs[0] is the max
      size_t newLimit = limit;
      for (size_t i = 0; i < limit; ++i) {
        if (probs[i] < threshold) {
          newLimit = i;
          break;
        }
      }
      if (newLimit > 0)
        limit = newLimit;
    }

    // Top-P (nucleus): keep until cumulative probability >= topP.
    if (config_.topP < 1.0f) {
      float cumProb = 0.0f;
      size_t newLimit = limit;
      for (size_t i = 0; i < limit; ++i) {
        cumProb += probs[i];
        if (cumProb >= config_.topP) {
          newLimit = i + 1;
          break;
        }
      }
      limit = newLimit;
    }

    // Re-normalize probabilities over the final candidate set.
    float finalSum = 0.0f;
    for (size_t i = 0; i < limit; ++i) {
      finalSum += probs[i];
    }
    for (size_t i = 0; i < limit; ++i) {
      probs[i] /= finalSum;
    }

    // Weighted random sampling.
    std::discrete_distribution<int> dist(probs.begin(), probs.begin() + limit);
    int chosen = dist(rng_);
    return candidates[chosen].first;
  }
};

} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_SAMPLER_H
