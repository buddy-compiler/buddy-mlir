//===- BenchLoop.cpp - Model-agnostic benchmark loop ----------------------===//
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

#include "buddy/runtime/llm/BenchLoop.h"

#include <algorithm>
#include <chrono>

namespace buddy {

using buddy::Text;

namespace runtime {

BenchmarkResult runBenchPass(LLMSession &session, const std::string &vocabPath,
                             const RunConfig &cfg, buddy::Sampler &sampler,
                             const std::vector<long long> &stopTokenIds,
                             const TextCodec &codec, int padTokenId,
                             const PrefillWorkLenFn &prefillWorkLen) {
  BenchmarkResult r;

  // ── Build the prefill token sequence via the codec so the memref shape
  //    (sizes[1] = codec.maxTokenLen) matches what the generated prefill
  //    kernel expects. ──────────────────────────────────────────────────────
  Text<size_t, 2> inputTokens(cfg.prompt);
  codec.tokenize(inputTokens, vocabPath);
  int nPrompt = static_cast<int>(inputTokens.getTokenCnt());

  if (cfg.promptLength == -1) {
    // Decode-only: keep a single seed token (the leading BOS) and report the
    // pass as having no prompt.
    inputTokens.setTokenCnt(1);
    nPrompt = 0;
  } else if (cfg.promptLength > 0) {
    const int target = cfg.promptLength;
    if (target < nPrompt) {
      inputTokens.setTokenCnt(target);
    } else {
      for (int i = nPrompt; i < target; ++i)
        inputTokens.appendTokenIdx(padTokenId);
    }
    nPrompt = target;
  }

  const int actualPromptTokens = static_cast<int>(inputTokens.getTokenCnt());

  // ── Prefill ─────────────────────────────────────────────────────────────
  session.resetPosition();
  const auto t0 = std::chrono::high_resolution_clock::now();
  session.prefill(inputTokens);
  r.prefillSecs = std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
  r.nPrompt = nPrompt;

  // ── Sample the first token from the last prefill position ───────────────
  std::vector<int> recentTokens;
  const int tokenIndex = actualPromptTokens - 1;
  const float *prefillLogits = session.logitsData(tokenIndex);
  int curToken =
      sampler.sample(prefillLogits, session.vocabSize(), recentTokens);
  recentTokens.push_back(curToken);

  auto isStopToken = [&](int id) {
    return std::find(stopTokenIds.begin(), stopTokenIds.end(), id) !=
           stopTokenIds.end();
  };

  // ── Decode loop (decodeBudget steps, mirroring runGeneration) ────────────
  const int decodeBudget = cfg.maxNewTokens;
  const int keepTokenNum = codec.maxTokenLen / 4;
  static constexpr float kRopeTheta = 10000.0f;

  // --ignore-eos keeps decoding for the whole budget even after a stop token,
  // so every repetition generates the same number of tokens.
  const bool stopOnEos = !cfg.ignoreEos;
  double decodeAccumMs = 0.0;
  int decodeCount = 0;
  if (decodeBudget > 0 && (!stopOnEos || !isStopToken(curToken))) {
    for (int step = 1; step <= decodeBudget; ++step) {
      if (g_interrupted.load(std::memory_order_relaxed)) {
        g_interrupted = false;
        break;
      }
      if (session.position() >= codec.maxTokenLen)
        session.handleKVCacheOverflow(keepTokenNum, kRopeTheta);

      const auto ds = std::chrono::high_resolution_clock::now();
      session.decode(curToken);
      decodeAccumMs += std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - ds)
                           .count();
      decodeCount += 1;

      int nextToken = sampler.sample(session.logitsData(), session.vocabSize(),
                                     recentTokens);
      recentTokens.push_back(nextToken);
      if (static_cast<int>(recentTokens.size()) >
          sampler.config().repeatLastN * 2)
        recentTokens.erase(recentTokens.begin(),
                           recentTokens.end() - sampler.config().repeatLastN);

      if (stopOnEos && isStopToken(nextToken))
        break;
      curToken = nextToken;
    }
  }

  r.nGen = decodeCount;
  // A session whose prefill rounds the prompt up to a cache tier reports the
  // positions actually computed rather than the requested prompt length.
  if (prefillWorkLen)
    r.prefillWorkLen = prefillWorkLen();
  r.decodeSecs = decodeAccumMs / 1000.0;
  return r;
}

void runBenchLoop(LLMSession &session, const std::string &vocabPath,
                  const RunConfig &cfg, Sampler &sampler,
                  const std::vector<long long> &stopTokenIds,
                  const TextCodec &codec, int padTokenId,
                  const PrefillWorkLenFn &prefillWorkLen, int warmupReps,
                  int reps, std::vector<BenchmarkResult> &out) {
  // Warmup passes run on the same session as the measured passes, so they
  // warm the allocator, thread pool and caches the measurement actually uses.
  for (int i = 0; i < warmupReps; ++i)
    runBenchPass(session, vocabPath, cfg, sampler, stopTokenIds, codec,
                 padTokenId, prefillWorkLen);

  out.clear();
  out.reserve(reps);
  for (int i = 0; i < reps; ++i)
    out.push_back(runBenchPass(session, vocabPath, cfg, sampler, stopTokenIds,
                               codec, padTokenId, prefillWorkLen));
}

} // namespace runtime
} // namespace buddy
