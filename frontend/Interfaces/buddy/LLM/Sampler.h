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

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_SAMPLER
#define FRONTEND_INTERFACES_BUDDY_LLM_SAMPLER

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

    // Greedy path: temperature == 0 or all sampling disabled.
    if (config_.temperature == 0.0f) {
      return greedySample(logits, vocabSize);
    }

    // Copy logits for in-place mutation.
    std::vector<float> work(logits, logits + vocabSize);

    // Step 1: Repetition penalty.
    if (config_.repeatPenalty != 1.0f && !recentTokens.empty()) {
      applyRepeatPenalty(work, recentTokens);
    }

    // Step 2: Temperature scaling.
    applyTemperature(work);

    // Step 3-5: Filtering (top-K, top-P, min-P) + softmax + sampling.
    return filteredSample(work);
  }

private:
  SamplerConfig config_;
  std::mt19937 rng_;

  int greedySample(const float *logits, size_t vocabSize) {
    return static_cast<int>(
        std::distance(logits, std::max_element(logits, logits + vocabSize)));
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

#endif // FRONTEND_INTERFACES_BUDDY_LLM_SAMPLER
