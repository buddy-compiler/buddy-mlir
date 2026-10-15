//===- LegacyGenerationTest.cpp ------------------------------------------===//
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

// RUN: buddy-legacy-generation-test \
// RUN:   %S/../../../examples/BuddyDeepSeekR1/vocab.txt \
// RUN:   %S/Inputs/legacy-generation 2>&1 | FileCheck %s

#include <buddy/runtime/llm/TextGeneration.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

using buddy::Sampler;
using buddy::SamplerConfig;
using buddy::Text;
using namespace buddy::runtime;

namespace {

constexpr int kA = 32;
constexpr int kB = 33;
constexpr int kC = 34;
constexpr int kEos = 151643;
constexpr int kContextLength = 64;
using Frame = std::vector<float>;
using Script = std::vector<Frame>;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

Frame logits(size_t vocabSize,
             std::initializer_list<std::pair<int, float>> values) {
  Frame frame(vocabSize, -1000.0f);
  for (const auto &entry : values)
    frame.at(entry.first) = entry.second;
  return frame;
}

// Only model execution is scripted. The test calls the production generation
// loop, DeepSeek tokenizer/revert implementation, vocabulary, and sampler.
class FakeSession final : public LLMSession {
public:
  explicit FakeSession(Script script) : script_(std::move(script)) {}

  void loadWeights(const std::vector<std::string> &) override {
    throw std::runtime_error("golden test must not load weights");
  }

  void resetPosition() override {
    ++resets;
    position_ = 0;
    step_ = 0;
    promptTokens.clear();
    decodedTokens.clear();
    logitOffsets.clear();
  }

  void prefill(Text<size_t, 2> &tokens) override {
    require(position_ == 0, "prefill must follow resetPosition");
    ++prefills;
    promptTokens.assign(tokens.getData(),
                        tokens.getData() + tokens.getTokenCnt());
    position_ = static_cast<int>(tokens.getTokenCnt());
  }

  void decode(int tokenId) override {
    ++decodes;
    ++position_;
    decodedTokens.push_back(tokenId);
    require(++step_ < script_.size(), "unexpected decode past scripted EOS");
  }

  int position() const override { return position_; }

  const float *logitsData(int tokenOffset = 0) const override {
    logitOffsets.push_back(tokenOffset);
    return script_.at(step_).data();
  }

  int vocabSize() const override { return script_.front().size(); }

  bool handleKVCacheOverflow(int, float) override {
    throw std::runtime_error("unexpected context overflow in golden test");
  }

  int resets = 0;
  int prefills = 0;
  int decodes = 0;
  std::vector<size_t> promptTokens;
  std::vector<int> decodedTokens;
  mutable std::vector<int> logitOffsets;

private:
  Script script_;
  size_t step_ = 0;
  // A nonzero initial position also detects a missing reset on the first call.
  int position_ = 17;
};

class CaptureStdout {
public:
  CaptureStdout() : saved_(std::cout.rdbuf(output_.rdbuf())) {}
  ~CaptureStdout() { std::cout.rdbuf(saved_); }
  std::string str() const { return output_.str(); }

private:
  std::ostringstream output_;
  std::streambuf *saved_;
};

std::string readFile(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot read fixture: " + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

struct Golden {
  const char *fixture;
  std::string text;
  std::vector<int> decodedTokens;
  bool hitTokenLimit = false;
  std::string jsonl{};
};

void replacePlaceholder(std::string &text, const std::string &placeholder,
                        const std::string &value) {
  const size_t offset = text.find(placeholder);
  require(offset != std::string::npos,
          "missing random JSONL placeholder: " + placeholder);
  text.replace(offset, placeholder.size(), value);
}

Golden randomReference(std::mt19937 &rng, const std::string &fixtureDir) {
  // C++ does not specify discrete_distribution's mapping from engine output
  // to samples. Use the platform standard library with three independently
  // specified relative weights, rather than freezing libstdc++'s sequence for
  // libc++. No production sampler, filtering, or softmax implementation is used
  // to derive the expectation. The observed libstdc++ sequence at seed 42 was
  // BABBAA followed by AABABC.
  std::discrete_distribution<int> distribution(
      {1.0, std::exp(-0.4 / 0.75), std::exp(-0.9 / 0.75)});
  Golden expected{"seed-template",
                  "",
                  {},
                  false,
                  readFile(fixtureDir + "/seed-template.jsonl.in")};
  for (int index = 0; index < 6; ++index) {
    const int choice = distribution(rng);
    const std::string token(1, 'A' + choice);
    expected.text += token;
    expected.decodedTokens.push_back(kA + choice);
    replacePlaceholder(expected.jsonl, "@ID" + std::to_string(index) + "@",
                       std::to_string(kA + choice));
    replacePlaceholder(expected.jsonl, "@TOKEN" + std::to_string(index) + "@",
                       token);
  }
  replacePlaceholder(expected.jsonl, "@TEXT@", expected.text);
  // The terminating EOS sample also consumes RNG state before the next call.
  std::discrete_distribution<int> stopDistribution({1.0, 0.0, 0.0});
  require(stopDistribution(rng) == 0, "invalid reference EOS sample");
  return expected;
}

void checkRun(FakeSession &session, Sampler &sampler, const TextCodec &codec,
              const std::string &vocabPath, const std::string &fixtureDir,
              int budget, bool jsonl, const Golden &golden) {
  const int resetsBefore = session.resets;
  const int prefillsBefore = session.prefills;
  const int decodesBefore = session.decodes;
  GenerationResult result;
  std::string output;
  {
    CaptureStdout capture;
    // Keep the final argument a bool: the callback overload has a different
    // transport contract and cannot stand in for this CLI regression test.
    result = runGeneration("A", session, vocabPath, budget, {kEos}, sampler,
                           codec, true, jsonl);
    output = capture.str();
  }

  const std::string label =
      std::string(golden.fixture) + (jsonl ? " JSONL" : " stdout");
  const std::string expectedOutput =
      jsonl ? (golden.jsonl.empty()
                   ? readFile(fixtureDir + "/" + golden.fixture + ".jsonl")
                   : golden.jsonl)
            : golden.text + "\n";
  require(output == expectedOutput, label + " output mismatch\nexpected:\n" +
                                        expectedOutput + "actual:\n" + output);
  require(result.text == golden.text, label + " result text mismatch");
  require(result.generatedTokens ==
              static_cast<int>(golden.decodedTokens.size()),
          label + " legacy generatedTokens mismatch");
  require(result.hitTokenLimit == golden.hitTokenLimit,
          label + " token limit marker mismatch");
  require(!result.cancelled, label + " unexpectedly cancelled");
  require(session.resets == resetsBefore + 1 &&
              session.prefills == prefillsBefore + 1,
          label + " reset/prefill count mismatch");
  require(session.decodes ==
              decodesBefore + static_cast<int>(golden.decodedTokens.size()),
          label + " decode call count mismatch");
  require(session.decodedTokens == golden.decodedTokens,
          label + " decode token order mismatch");
  require(session.promptTokens ==
              std::vector<size_t>(
                  {151646, 151646, 151644, kA, 151645, 151648, 198}),
          label + " real DeepSeek prompt encoding changed");
  std::vector<int> expectedOffsets(golden.decodedTokens.size() + 1, 0);
  expectedOffsets.front() = 6;
  require(session.logitOffsets == expectedOffsets,
          label + " prefill/decode logits offset mismatch");
  require(session.position() ==
              7 + static_cast<int>(golden.decodedTokens.size()),
          label + " session position mismatch");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "expected vocabulary path and golden fixture directory");
    const std::string vocabPath = argv[1];
    const std::string fixtureDir = argv[2];
    std::istringstream vocabulary(readFile(vocabPath));
    size_t vocabSize = 0;
    for (std::string token; std::getline(vocabulary, token);)
      ++vocabSize;
    require(vocabSize > kEos, "production vocabulary is incomplete");

    TextCodec codec;
    codec.maxTokenLen = kContextLength;
    codec.tokenize = [](Text<size_t, 2> &tokens, const std::string &path) {
      tokens.tokenizeDeepSeekR1(path, kContextLength);
    };
    codec.detokenize = [](Text<size_t, 2> &tokens) {
      return tokens.revertDeepSeekR1();
    };

    const Script greedyScript = {logits(vocabSize, {{kA, 10.0f}}),
                                 logits(vocabSize, {{kB, 10.0f}}),
                                 logits(vocabSize, {{kEos, 10.0f}})};
    const Golden greedy = {"greedy", "AB", {kA, kB}};
    // Both modes reuse one session. Resetting must restore the scripted model
    // position without changing the old output or counting conventions.
    FakeSession session(greedyScript);
    Sampler sampler(SamplerConfig{});
    for (bool jsonl : {false, true})
      checkRun(session, sampler, codec, vocabPath, fixtureDir, 32, jsonl,
               greedy);
    // CHECK: greedy stdout and JSONL: AB, 2 decode calls, 2 returned tokens
    std::cout
        << "greedy stdout and JSONL: AB, 2 decode calls, 2 returned tokens\n";

    // These are historical CLI semantics: the budget includes prompt tokens,
    // the prefill sample is unconditional, and only decode calls are counted.
    for (bool jsonl : {false, true}) {
      for (int budget : {1, 7})
        checkRun(session, sampler, codec, vocabPath, fixtureDir, budget, jsonl,
                 {"prefill-limit", "A", {}, true});
      checkRun(session, sampler, codec, vocabPath, fixtureDir, 8, jsonl,
               {"decode-limit", "AB", {kA}, true});
      for (int budget : {9, 0, -1})
        checkRun(session, sampler, codec, vocabPath, fixtureDir, budget, jsonl,
                 greedy);
      FakeSession eosSession({logits(vocabSize, {{kEos, 10.0f}})});
      checkRun(eosSession, sampler, codec, vocabPath, fixtureDir, 1, jsonl,
               {"first-eos", "", {}});
    }
    // CHECK: legacy budgets 1/7/8/9/0/-1 and first-token EOS: passed
    std::cout << "legacy budgets 1/7/8/9/0/-1 and first-token EOS: passed\n";

    // The prompt's A is absent from repetition history. Two generated As apply
    // the penalty twice: 8 / 2 / 2 < 3. Deduplicating history would produce A.
    const Script repeatScript = {logits(vocabSize, {{kA, 10.0f}, {kB, 8.0f}}),
                                 logits(vocabSize, {{kA, 10.0f}, {kB, 2.0f}}),
                                 logits(vocabSize, {{kA, 8.0f}, {kB, 3.0f}}),
                                 logits(vocabSize, {{kEos, 10.0f}})};
    SamplerConfig repeatedConfig;
    repeatedConfig.repeatPenalty = 2.0f;
    FakeSession repeatedSession(repeatScript);
    Sampler repeatedSampler(repeatedConfig);
    for (bool jsonl : {false, true})
      checkRun(repeatedSession, repeatedSampler, codec, vocabPath, fixtureDir,
               32, jsonl, {"repetition", "AAB", {kA, kA, kB}});
    // CHECK: duplicate generated IDs retain repeated penalties: AAB
    std::cout << "duplicate generated IDs retain repeated penalties: AAB\n";

    Script randomScript(
        6, logits(vocabSize, {{kA, 0.7f}, {kB, 0.3f}, {kC, -0.2f}}));
    randomScript.push_back(logits(vocabSize, {{kEos, 10.0f}}));
    SamplerConfig randomConfig;
    randomConfig.temperature = 0.75f;
    randomConfig.topK = 3;
    randomConfig.seed = 42;
    FakeSession randomSession(randomScript);
    for (bool jsonl : {false, true}) {
      Sampler randomSampler(randomConfig);
      std::mt19937 referenceRng(42);
      checkRun(randomSession, randomSampler, codec, vocabPath, fixtureDir, 32,
               jsonl, randomReference(referenceRng, fixtureDir));
      // Session reset preserves the sampler RNG across requests; constructing
      // another sampler with the same fixed seed reproduces the first result.
      checkRun(randomSession, randomSampler, codec, vocabPath, fixtureDir, 32,
               jsonl, randomReference(referenceRng, fixtureDir));
    }
    // CHECK: seed 42 and repeated session reset: platform reference matched
    std::cout
        << "seed 42 and repeated session reset: platform reference matched\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
