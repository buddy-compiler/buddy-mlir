//===- PerplexityTest.cpp -------------------------------------------------===//
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
// runPerplexity on a fake session whose logits depend on the last token and
// the position: the scored positions, the chunks, the resets and the value
// against a direct computation; readTokenIds on the formats it accepts.
//
//===----------------------------------------------------------------------===//

// RUN: buddy-perplexity-test %t 2>&1 | FileCheck %s

#include <buddy/runtime/llm/Perplexity.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace buddy;
using namespace buddy::runtime;

namespace {

constexpr int kVocab = 7;

// The logits after token t at position p: token (t + 1) % V is the most
// likely, by a margin that grows with p.
void fakeLogits(int t, int p, float *out) {
  for (int k = 0; k < kVocab; ++k)
    out[k] = k == (t + 1) % kVocab ? 1.0f + 0.25f * p : 0.1f * k;
}

class FakeSession : public LLMSession {
public:
  int prefills = 0, decodes = 0;
  void loadWeights(const std::vector<std::string> &) override {}
  void prefill(Text<size_t, 2> &tokens) override {
    ++prefills;
    pos_ = static_cast<int>(tokens.getTokenCnt()) - 1;
    fakeLogits(static_cast<int>(tokens.getData()[pos_]), pos_, logits_);
    ++pos_;
  }
  void decode(int tokenId) override {
    ++decodes;
    fakeLogits(tokenId, pos_, logits_);
    ++pos_;
  }
  void resetPosition() override { pos_ = 0; }
  int position() const override { return pos_; }
  const float *logitsData(int) const override { return logits_; }
  int vocabSize() const override { return kVocab; }
  bool handleKVCacheOverflow(int, float) override { return false; }

private:
  int pos_ = 0;
  float logits_[kVocab];
};

// The perplexity by definition, chunk by chunk.
double direct(const std::vector<int> &ids, int n, int chunks) {
  double total = 0;
  long count = 0;
  float logits[kVocab];
  for (int c = 0; c < chunks; ++c)
    for (int j = n / 2; j < n - 1; ++j) {
      const int *tok = ids.data() + c * n;
      fakeLogits(tok[j], j, logits);
      double z = 0;
      for (float l : logits)
        z += std::exp(static_cast<double>(l));
      total += -std::log(std::exp(static_cast<double>(logits[tok[j + 1]])) / z);
      ++count;
    }
  return std::exp(total / count);
}

} // namespace

int main(int argc, char **argv) {
  std::vector<int> ids;
  for (int i = 0; i < 50; ++i)
    ids.push_back((i * i + 3 * i) % kVocab);

  // 50 ids, chunks of 8: 6 whole chunks, positions 4 .. 6 of each scored
  FakeSession s;
  PerplexityOptions opts;
  opts.context = 8;
  opts.quiet = true;
  PerplexityResult r = runPerplexity(s, ids, opts);
  // CHECK: perplexity: 50 token ids, 6 chunks of 8, scoring the second half of
  // each CHECK: Final estimate: PPL =
  std::printf("chunks %d, scored %ld, prefills %d, decodes %d\n", r.chunks,
              r.scoredTokens, s.prefills, s.decodes);
  // CHECK: chunks 6, scored 18, prefills 6, decodes 36
  std::printf("matches the definition: %d\n",
              std::fabs(r.perplexity - direct(ids, 8, 6)) < 1e-9);
  // CHECK: matches the definition: 1

  // at most 2 chunks
  FakeSession s2;
  opts.maxChunks = 2;
  r = runPerplexity(s2, ids, opts);
  std::printf("max 2: chunks %d, scored %ld, definition %d\n", r.chunks,
              r.scoredTokens,
              std::fabs(r.perplexity - direct(ids, 8, 2)) < 1e-9);
  // CHECK: max 2: chunks 2, scored 6, definition 1

  // rejected: an odd context, a token outside the vocabulary
  for (auto [context, bad] : {std::pair{7, 0}, std::pair{8, 1}}) {
    std::vector<int> v = ids;
    if (bad)
      v[3] = kVocab;
    PerplexityOptions o;
    o.context = context;
    o.quiet = true;
    try {
      FakeSession f;
      runPerplexity(f, v, o);
    } catch (const std::exception &e) {
      std::printf("error: %s\n", e.what());
    }
  }
  // CHECK: error: perplexity context must be even and >= 4, got 7
  // CHECK: error: token id 7 outside the vocabulary (7)

  // readTokenIds: `llama-tokenize --ids` output and one id per line
  const std::string path = argc > 1 ? std::string(argv[1]) : "ids.txt";
  for (const char *text : {"[715, 284, 8397]\n", "715\n284\r\n8397"}) {
    std::ofstream(path) << text;
    std::vector<int> got = readTokenIds(path);
    std::printf("ids:");
    for (int id : got)
      std::printf(" %d", id);
    std::printf("\n");
  }
  // CHECK: ids: 715 284 8397
  // CHECK-NEXT: ids: 715 284 8397
  return 0;
}
