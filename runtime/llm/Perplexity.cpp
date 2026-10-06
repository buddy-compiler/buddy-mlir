//===- Perplexity.cpp - Perplexity of an LLM over a token stream ----------===//
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

#include "buddy/runtime/llm/Perplexity.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace buddy {
namespace runtime {

std::vector<int> readTokenIds(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read token ids: " + path);
  std::vector<int> ids;
  long long value = 0;
  bool inNumber = false;
  for (std::istreambuf_iterator<char> it(in), end; it != end; ++it) {
    const char c = *it;
    if (c >= '0' && c <= '9') {
      value = value * 10 + (c - '0');
      if (value > 0x7fffffff)
        throw std::runtime_error("token id out of range in " + path);
      inNumber = true;
    } else if (inNumber) {
      ids.push_back(static_cast<int>(value));
      value = 0;
      inNumber = false;
    }
  }
  if (inNumber)
    ids.push_back(static_cast<int>(value));
  return ids;
}

// -log softmax(logits)[target]
static double negativeLogLikelihood(const float *logits, int vocab,
                                    int target) {
  double max = logits[0];
  for (int i = 1; i < vocab; ++i)
    if (logits[i] > max)
      max = logits[i];
  double sum = 0;
  for (int i = 0; i < vocab; ++i)
    sum += std::exp(static_cast<double>(logits[i]) - max);
  return std::log(sum) + max - static_cast<double>(logits[target]);
}

PerplexityResult runPerplexity(LLMSession &session, const std::vector<int> &ids,
                               const PerplexityOptions &opts) {
  const int n = opts.context;
  if (n < 4 || n % 2)
    throw std::runtime_error("perplexity context must be even and >= 4, got " +
                             std::to_string(n));
  int chunks = static_cast<int>(ids.size() / n);
  if (opts.maxChunks > 0 && opts.maxChunks < chunks)
    chunks = opts.maxChunks;
  if (chunks == 0)
    throw std::runtime_error("perplexity: fewer token ids (" +
                             std::to_string(ids.size()) + ") than one chunk (" +
                             std::to_string(n) + ")");
  const int vocab = session.vocabSize();
  for (size_t i = 0; i < static_cast<size_t>(chunks) * n; ++i)
    if (ids[i] < 0 || ids[i] >= vocab)
      throw std::runtime_error("token id " + std::to_string(ids[i]) +
                               " outside the vocabulary (" +
                               std::to_string(vocab) + ")");

  std::printf("perplexity: %zu token ids, %d chunks of %d, scoring the "
              "second half of each\n",
              ids.size(), chunks, n);
  std::fflush(stdout);
  const auto start = std::chrono::steady_clock::now();
  double total = 0;
  long count = 0;
  for (int c = 0; c < chunks; ++c) {
    const int *tok = ids.data() + static_cast<size_t>(c) * n;
    Text<size_t, 2> first;
    first.appendTokenIdx(static_cast<size_t>(tok[0]));
    session.prefill(first); // empty KV cache; token 0 at position 0
    for (int j = 0; j < n - 1; ++j) {
      if (j > 0)
        session.decode(tok[j]);
      if (j >= n / 2) {
        total += negativeLogLikelihood(session.logitsData(), vocab, tok[j + 1]);
        ++count;
      }
    }
    if (!opts.quiet) {
      std::printf("[%d]%.4f%s", c + 1, std::exp(total / count),
                  (c + 1) % 10 == 0 || c + 1 == chunks ? "\n" : ",");
      std::fflush(stdout);
    }
  }
  PerplexityResult result;
  result.perplexity = std::exp(total / count);
  result.scoredTokens = count;
  result.chunks = chunks;
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::printf("Final estimate: PPL = %.4f over %ld tokens (%.1f s)\n",
              result.perplexity, result.scoredTokens, result.seconds);
  std::fflush(stdout);
  return result;
}

} // namespace runtime
} // namespace buddy
