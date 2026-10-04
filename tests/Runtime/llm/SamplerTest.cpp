//===- SamplerTest.cpp ----------------------------------------------------===//
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
// This is the LLM sampler test file: greedy sampling must pick the index
// std::max_element picks (the first maximum; 0 if logits[0] is NaN).
//
//===----------------------------------------------------------------------===//

// RUN: buddy-sampler-test 2>&1 | FileCheck %s

#include <buddy/runtime/llm/Sampler.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using buddy::Sampler;
using buddy::SamplerConfig;

static int greedy(const std::vector<float> &logits) {
  Sampler sampler(SamplerConfig{}); // temperature 0: greedy
  return sampler.sample(logits.data(), logits.size(), {});
}

static int reference(const std::vector<float> &logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

int main() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  //===--------------------------------------------------------------------===//
  // Hand-written cases.
  //===--------------------------------------------------------------------===//
  // CHECK: single: 0
  fprintf(stderr, "single: %d\n", greedy({-1.0f}));
  // CHECK: ties: 2
  fprintf(stderr, "ties: %d\n", greedy({0.0f, 1.0f, 3.0f, 3.0f, 2.0f, 3.0f}));
  // CHECK: signed zeros: 1
  fprintf(stderr, "signed zeros: %d\n", greedy({-1.0f, -0.0f, 0.0f}));
  // CHECK: all -inf: 0
  fprintf(stderr, "all -inf: %d\n", greedy({-inf, -inf, -inf}));
  // CHECK: nan first: 0
  fprintf(stderr, "nan first: %d\n", greedy({nan, 1.0f, 2.0f}));
  // CHECK: nan later: 2
  fprintf(stderr, "nan later: %d\n", greedy({1.0f, nan, 2.0f, nan}));

  // A vocabulary-sized input, the maximum in the tail (not a multiple of the
  // block size) and, separately, a tie between a block and the tail.
  std::vector<float> vocab(151936, -1.0f);
  vocab[151935] = 5.0f;
  // CHECK: vocab, max last: 151935
  fprintf(stderr, "vocab, max last: %d\n", greedy(vocab));
  vocab[1000] = 5.0f;
  // CHECK: vocab, tie: 1000
  fprintf(stderr, "vocab, tie: %d\n", greedy(vocab));

  //===--------------------------------------------------------------------===//
  // Random inputs: sizes around the block size, few distinct values (many
  // ties), signed zeros, infinities and NaNs.
  //===--------------------------------------------------------------------===//
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> size(1, 300), value(-3, 3), kind(0, 9);
  int mismatches = 0;
  for (int t = 0; t < 20000; ++t) {
    std::vector<float> logits(size(rng));
    for (float &x : logits) {
      int k = kind(rng);
      x = k == 0   ? nan
          : k == 1 ? -0.0f
          : k == 2 ? inf
          : k == 3 ? -inf
                   : static_cast<float>(value(rng));
    }
    if (greedy(logits) != reference(logits))
      ++mismatches;
  }
  // CHECK: random: 0 mismatches
  fprintf(stderr, "random: %d mismatches\n", mismatches);

  //===--------------------------------------------------------------------===//
  // Greedy sampling after a repetition penalty (the copied-logits path).
  //===--------------------------------------------------------------------===//
  SamplerConfig penalized;
  penalized.repeatPenalty = 2.0f;
  Sampler sampler(penalized);
  std::vector<float> logits = {1.0f, 4.0f, 3.0f};
  // 4.0 at index 1 was generated recently: 4.0 / 2 = 2.0 < 3.0.
  // CHECK: penalized: 2
  fprintf(stderr, "penalized: %d\n",
          sampler.sample(logits.data(), logits.size(), {1}));

  return 0;
}
