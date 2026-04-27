//===- TextGeneration.cpp - Model-agnostic text generation ----------------===//
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

#include "buddy/runtime/llm/TextGeneration.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace buddy {

using buddy::Text;

namespace runtime {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log]\033[0m " << msg << "\n";
}

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

std::atomic<bool> g_interrupted{false};

void interruptHandler(int) { g_interrupted = true; }

//===----------------------------------------------------------------------===//
// Generation internals
//===----------------------------------------------------------------------===//

namespace {

/// Stream decoded text incrementally to stdout.
void streamNewText(Text<size_t, 2> &outputContainer, std::string &lastPrinted,
                   const TextCodec &codec) {
  std::string current = codec.detokenize(outputContainer);
  if (current.size() > lastPrinted.size()) {
    std::cout.write(current.data() + lastPrinted.size(),
                    current.size() - lastPrinted.size());
    std::cout.flush();
  }
  lastPrinted = std::move(current);
}

} // namespace

//===----------------------------------------------------------------------===//
// printStats
//===----------------------------------------------------------------------===//

void printStats(const GenerationResult &result, bool verbose) {
  const double decodeTPS =
      result.decodeSecs > 0 ? (double)result.generatedTokens / result.decodeSecs
                            : 0;
  if (verbose) {
    std::cerr << "\n";
    std::cerr << "\033[33;1m[Total time]\033[0m "
              << (result.prefillSecs + result.decodeSecs) << "s\n";
  }
  std::cerr << "\033[33;1m[Prefill]\033[0m " << result.prefillSecs << "s";
  if (verbose)
    std::cerr << "\n";
  else
    std::cerr << "  ";
  std::cerr << "\033[33;1m[Decode]\033[0m " << decodeTPS << " tokens/s ("
            << result.generatedTokens << " tokens)\n";
}

//===----------------------------------------------------------------------===//
// runGeneration
//===----------------------------------------------------------------------===//

GenerationResult runGeneration(const std::string &prompt, LLMSession &session,
                               const std::string &vocabPath, int maxNewTokens,
                               const std::vector<long long> &stopTokenIds,
                               buddy::Sampler &sampler, const TextCodec &codec,
                               bool suppress) {
  GenerationResult result;

  const int keepTokenNum = codec.maxTokenLen / 4;
  static constexpr float kRopeTheta = 10000.0f;

  auto isStopToken = [&](int tokenId) -> bool {
    return std::find(stopTokenIds.begin(), stopTokenIds.end(), tokenId) !=
           stopTokenIds.end();
  };

  // Reset session for this generation pass.
  session.resetPosition();

  Text<size_t, 2> inputTokens(prompt);
  codec.tokenize(inputTokens, vocabPath);

  Text<size_t, 2> outputTokens;
  outputTokens.loadVocab(vocabPath);

  std::vector<int> recentTokens;

  // ── Prefill ─────────────────────────────────────────────────────────────
  const auto t0 = std::chrono::high_resolution_clock::now();
  session.prefill(inputTokens);
  result.prefillSecs = std::chrono::duration<double>(
                           std::chrono::high_resolution_clock::now() - t0)
                           .count();

  // Sample first token from prefill logits.
  const int tokenIndex = static_cast<int>(inputTokens.getTokenCnt()) - 1;
  const float *prefillLogits = session.logitsData(tokenIndex);
  int firstToken =
      sampler.sample(prefillLogits, session.vocabSize(), recentTokens);
  recentTokens.push_back(firstToken);

  if (isStopToken(firstToken)) {
    std::cout << std::endl;
    return result;
  }

  outputTokens.appendTokenIdx(firstToken);
  std::string lastPrinted;
  streamNewText(outputTokens, lastPrinted, codec);

  // ── Decode loop ─────────────────────────────────────────────────────────
  int curToken = firstToken;
  const int maxSteps = (maxNewTokens <= 0)
                           ? std::numeric_limits<int>::max()
                           : maxNewTokens - (int)inputTokens.getTokenCnt();
  double decodeAccumMs = 0.0;
  int decodeCount = 0;

  for (int step = 1; step <= maxSteps; ++step) {
    // Check for user interrupt (Ctrl+C).
    if (g_interrupted.load(std::memory_order_relaxed)) {
      g_interrupted = false;
      std::cerr << "\n[Generation interrupted]\n";
      break;
    }

    // Handle KV cache overflow before decode.
    if (session.position() >= codec.maxTokenLen) {
      printLog("KV cache overflow at position " +
                   std::to_string(session.position()) + ", discarding...",
               suppress);
      session.handleKVCacheOverflow(keepTokenNum, kRopeTheta);
      printLog("New position: " + std::to_string(session.position()), suppress);
    }

    const auto ds = std::chrono::high_resolution_clock::now();
    session.decode(curToken);
    const double stepMs = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - ds)
                              .count();
    decodeAccumMs += stepMs;
    decodeCount += 1;

    int nextToken =
        sampler.sample(session.logitsData(), session.vocabSize(), recentTokens);
    recentTokens.push_back(nextToken);
    // Cap to sampler's repeat penalty window to avoid unbounded growth.
    if ((int)recentTokens.size() > sampler.config().repeatLastN * 2)
      recentTokens.erase(recentTokens.begin(),
                         recentTokens.end() - sampler.config().repeatLastN);

    if (isStopToken(nextToken))
      break;

    outputTokens.appendTokenIdx(nextToken);
    streamNewText(outputTokens, lastPrinted, codec);
    curToken = nextToken;
  }

  std::cout << std::endl;

  result.generatedTokens = decodeCount;
  result.decodeSecs = decodeAccumMs / 1000.0;
  result.text = codec.detokenize(outputTokens);
  return result;
}

} // namespace runtime
} // namespace buddy
