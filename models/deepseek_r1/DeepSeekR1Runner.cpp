//===- DeepSeekR1Runner.cpp - DeepSeekR1 full inference loop --------------===//
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
// Implements DeepSeekR1Runner::run().  This file owns everything that is
// model-specific for text generation:
//   - weight file loading
//   - tokenization (buddy::Text / TextContainer)
//   - prefill + decode loop
//   - performance counters and result printing
//
// buddy-cli calls this through the generic InferenceRunner interface.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1Runner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"
#include "buddy/LLM/TextContainer.h"

using buddy::Text;

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS
#define BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS 1024
#endif

namespace buddy {
namespace runtime {

namespace {

static constexpr int kEosToken = 151643; // <|end▁of▁sentence|>

void printBold(const std::string &label, const std::string &value) {
  std::cout << "\033[33;1m[" << label << "]\033[0m " << value << "\n";
}
void printLog(const std::string &msg) {
  std::cout << "\033[34;1m[Log]\033[0m " << msg << "\n";
}

void loadWeights(const std::string &path, MemRef<float, 1> &weights) {
  printLog("Weights: " + std::filesystem::canonical(path).string());
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open weights: " + path);
  f.read(reinterpret_cast<char *>(weights.getData()),
         sizeof(float) * weights.getSize());
  if (f.fail())
    throw std::runtime_error("Read failed: " + path);

  const double secs = std::chrono::duration<double>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
  printLog("Weights loaded in " + std::to_string(secs) + "s");
}

} // namespace

void DeepSeekR1Runner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;
  if (cfg.maxNewTokens <= 0)
    cfg.maxNewTokens = BUDDY_DSR1_DEFAULT_MAX_NEW_TOKENS;

  std::cout
      << "\033[33;1mDeepSeekR1 Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  std::unique_ptr<ModelSession> session;
  std::string weightsPath;
  std::string vocabPath;

  if (!cfg.raxPath.empty()) {
    // ---- Mode A: load everything from .rax manifest ---------------------
    printLog("Manifest: " + cfg.raxPath);
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);

    weightsPath = manifest.weightsPath;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;

    printLog("  .so     = " + manifest.soPath);
    printLog("  weights = " + manifest.weightsPath);
    printLog("  vocab   = " + vocabPath);

  } else {
    // ---- Mode B: explicit paths -----------------------------------------
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath (--model-so).");
    if (cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires weightsPath (--weights).");

    weightsPath = cfg.weightsPath;
    vocabPath = cfg.vocabPath.empty()
                    ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : cfg.vocabPath;

    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    printLog("Loading model: " + cfg.modelSoPath);
    session = ModelSession::create(mcfg);
  }

  // --- Load weights -------------------------------------------------------
  intptr_t weightsShape[1] = {BUDDY_DSR1_PARAMS_SIZE};
  MemRef<float, 1> weights(weightsShape);
  loadWeights(weightsPath, weights);

  // --- Tokenize -----------------------------------------------------------
  printLog("Vocab: " + vocabPath);
  Text<size_t, 2> inputTokens(cfg.prompt);
  {
    const auto t0 = std::chrono::high_resolution_clock::now();
    inputTokens.tokenizeDeepSeekR1(vocabPath, BUDDY_DSR1_MAX_TOKEN_LEN);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
    printLog("Tokenized in " + std::to_string(ms) + "ms" +
             ", count=" + std::to_string(inputTokens.getTokenCnt()));
  }
  Text<size_t, 2> outputTokens;
  outputTokens.loadVocab(vocabPath);

  printLog("KV cache: " + std::to_string(BUDDY_DSR1_KV_LAYERS) + " x {1," +
           std::to_string(BUDDY_DSR1_HEAD_NUM) + "," +
           std::to_string(BUDDY_DSR1_MAX_TOKEN_LEN) + "," +
           std::to_string(BUDDY_DSR1_HIDDEN_SIZE) + "} f32");

  // --- Prefill -------------------------------------------------------------
  printLog("Prefill...");
  const auto t0 = std::chrono::high_resolution_clock::now();
  int firstToken = session->prefill(weights, inputTokens);
  const double prefillSecs = std::chrono::duration<double>(
                                 std::chrono::high_resolution_clock::now() - t0)
                                 .count();
  const double prefillTPS =
      prefillSecs > 0 ? (double)inputTokens.getTokenCnt() / prefillSecs : 0;

  std::cout << "\033[32;1m[Iter 0]\033[0m Token: "
            << inputTokens.getStr(firstToken) << " | " << prefillSecs << "s\n";
  outputTokens.appendTokenIdx(firstToken);

  // --- Decode loop ---------------------------------------------------------
  int curToken = firstToken;
  const int maxSteps = cfg.maxNewTokens - (int)inputTokens.getTokenCnt();
  double decodeAccumMs = 0.0;
  int decodeCount = 0;

  for (int step = 1; step <= maxSteps; ++step) {
    if (curToken == kEosToken)
      break;

    const auto ds = std::chrono::high_resolution_clock::now();
    int nextToken = session->decode(weights, curToken);
    const double stepMs = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - ds)
                              .count();

    decodeAccumMs += stepMs;
    decodeCount += 1;

    std::cout << "\033[32;1m[Iter " << step
              << "]\033[0m Token: " << inputTokens.getStr(nextToken) << " | "
              << stepMs / 1000.0 << "s\n";

    if (nextToken == kEosToken)
      break;
    outputTokens.appendTokenIdx(nextToken);
    curToken = nextToken;
  }

  // --- Results -------------------------------------------------------------
  const double decodeSecs = decodeAccumMs / 1000.0;
  const double decodeTPS =
      decodeSecs > 0 ? (double)decodeCount / decodeSecs : 0;

  std::cout << "\n";
  printBold("Total time", std::to_string(prefillSecs + decodeSecs) + "s");
  printBold("Prefilling", std::to_string(prefillTPS) + " tokens/s");
  printBold("Decoding", std::to_string(decodeTPS) + " tokens/s");
  printBold("Input", cfg.prompt);
  printBold("Output", outputTokens.revertDeepSeekR1());
}

} // namespace runtime
} // namespace buddy
