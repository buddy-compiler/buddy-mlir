//===- ServingGeneration.cpp - Compatible serving generation -------------===//
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

#include "buddy/runtime/llm/ServingGeneration.h"

#include "buddy/runtime/llm/TextGeneration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace buddy {
namespace runtime {
namespace {

// Withhold the longest suffix that can begin a stop string. Matching raw
// bytes before UTF-8 decoding also handles stops split inside a code point.
class StopMatcher {
public:
  explicit StopMatcher(const std::vector<std::string> &stops) : stops(stops) {}

  std::string append(std::string_view bytes) {
    pending.append(bytes.data(), bytes.size());
    std::size_t match = std::string::npos;
    for (const auto &stop : stops)
      match = std::min(match, pending.find(stop));
    if (match != std::string::npos) {
      stopped = true;
      std::string output = pending.substr(0, match);
      pending.clear();
      return output;
    }

    std::size_t keep = 0;
    for (const auto &stop : stops) {
      for (std::size_t length = std::min(pending.size(), stop.size() - 1);
           length > keep; --length) {
        if (pending.compare(pending.size() - length, length, stop, 0, length) ==
            0) {
          keep = length;
          break;
        }
      }
    }
    std::string output = pending.substr(0, pending.size() - keep);
    pending.erase(0, pending.size() - keep);
    return output;
  }

  std::string finish() {
    std::string output;
    output.swap(pending);
    return output;
  }

  bool stopped = false;

private:
  const std::vector<std::string> &stops;
  std::string pending;
};

class ResetOnFailure {
public:
  explicit ResetOnFailure(LLMSession &session) : session(session) {}
  ~ResetOnFailure() {
    if (!completed) {
      // Preserve an exception from execution/transport if reset also fails.
      try {
        session.resetPosition();
      } catch (...) {
      }
    }
  }
  bool completed = false;

private:
  LLMSession &session;
};

std::vector<int> repetitionHistory(const std::vector<int> &history,
                                   int window) {
  const std::size_t count =
      std::min(history.size(), static_cast<std::size_t>(window));
  std::vector<int> unique;
  std::unordered_set<int> seen;
  for (auto it = history.end() - count; it != history.end(); ++it)
    if (seen.insert(*it).second)
      unique.push_back(*it);
  return unique;
}

void validateSampling(const SamplingParams &sampling) {
  if (sampling.maxNewTokens < -1)
    throw std::invalid_argument(
        "max_tokens: expected -1 or a nonnegative limit");
  const auto &config = sampling.samplerConfig;
  if (!std::isfinite(config.temperature))
    throw std::invalid_argument("temperature: expected a finite number");
  if (!std::isfinite(config.topP) || config.topP <= 0 || config.topP > 1)
    throw std::invalid_argument("top_p: expected a number in (0, 1]");
  if (!std::isfinite(config.minP) || config.minP < 0 || config.minP > 1)
    throw std::invalid_argument("min_p: expected a number in [0, 1]");
  if (!std::isfinite(config.repeatPenalty) || config.repeatPenalty <= 0)
    throw std::invalid_argument("repeat_penalty: expected a positive number");
  if (config.repeatLastN < 0)
    throw std::invalid_argument("repeat_last_n: expected a nonnegative window");
  if (sampling.seedPolicy == SamplingSeedPolicy::Fixed &&
      sampling.fixedSeed == std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "seed: the random sentinel is not a fixed seed");
  for (const auto &stop : sampling.stop)
    if (stop.empty())
      throw std::invalid_argument("stop: empty strings are not supported");
}

} // namespace

CompletionResult
runServingGeneration(const std::string &prompt, LLMSession &session,
                     const SamplingParams &sampling,
                     const ServingTextCodec &codec,
                     const CompletionStreamCallback &callback) {
  session.resetPosition();
  ResetOnFailure cleanup(session);
  validateSampling(sampling);
  if (!codec.encode || !codec.tokenBytes || codec.contextLength <= 0)
    throw std::invalid_argument(
        "serving codec: incomplete encoding configuration");
  const int sessionVocabSize = session.vocabSize();
  if (sessionVocabSize <= 0)
    throw std::runtime_error(
        "serving session: vocabulary size must be positive");
  if (codec.vocabSize < 0 || codec.vocabSize > sessionVocabSize)
    throw std::invalid_argument(
        "serving codec: vocabulary size must be a decodable session prefix");
  const int vocabSize =
      codec.vocabSize == 0 ? sessionVocabSize : codec.vocabSize;
  for (long long token : sampling.stopTokenIds)
    if (token < 0 || token >= vocabSize)
      throw std::invalid_argument("stop_token_ids: token ID is out of range");

  const std::vector<int> promptTokens = codec.encode(prompt);
  if (promptTokens.empty())
    throw std::invalid_argument("prompt: encoding produced no tokens");
  if (promptTokens.size() > static_cast<std::size_t>(codec.contextLength))
    throw std::invalid_argument(
        "prompt: token count exceeds the context length");
  for (int token : promptTokens)
    if (token < 0 || token >= vocabSize)
      throw std::invalid_argument("prompt: encoded token ID is out of range");

  const int promptCount = static_cast<int>(promptTokens.size());
  const int remaining = codec.contextLength - promptCount;
  const int limit = sampling.maxNewTokens < 0
                        ? remaining
                        : std::min(remaining, sampling.maxNewTokens);
  Text<std::size_t, 2> input;
  // Generated sessions pass padded prefill shapes to their kernels. Reserve
  // initialized storage for the full context while preserving the real count.
  static_cast<MemRef<std::size_t, 2> &>(input) = MemRef<std::size_t, 2>(
      std::vector<std::size_t>{1,
                               static_cast<std::size_t>(codec.contextLength)},
      std::size_t{0});
  std::copy(promptTokens.begin(), promptTokens.end(), input.getData());
  input.setTokenCnt(promptCount);

  buddy::SamplerConfig config = sampling.samplerConfig;
  if (config.temperature <= 0)
    config.temperature = 0;
  config.seed = 0;
  buddy::Sampler sampler = sampling.seedPolicy == SamplingSeedPolicy::Fixed
                               ? buddy::Sampler(config, sampling.fixedSeed)
                               : buddy::Sampler(config);
  CompletionResult result;
  result.usage.promptTokens = promptCount;
  result.finishReason = FinishReason::Length;
  const auto prefillStart = std::chrono::steady_clock::now();
  session.prefill(input);
  result.timings.prefillMs =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - prefillStart)
          .count();
  if (session.position() != promptCount)
    throw std::runtime_error(
        "serving session: prefill position is inconsistent");

  std::vector<int> history = promptTokens;
  StopMatcher matcher(sampling.stop);
  Utf8StreamDecoder utf8;
  auto emit = [&](std::string delta, int tokenId) {
    result.content += delta;
    if (!callback)
      return true;
    CompletionChunk chunk;
    chunk.delta = std::move(delta);
    chunk.tokenId = tokenId;
    if (callback(chunk))
      return true;
    result.finishReason = FinishReason::Cancelled;
    return false;
  };

  int previous = -1;
  for (int index = 0; index < limit; ++index) {
    if (index != 0) {
      if (session.position() >= codec.contextLength)
        throw std::runtime_error("serving session: decode exceeds the context");
      const auto decodeStart = std::chrono::steady_clock::now();
      session.decode(previous);
      result.timings.decodeMs +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - decodeStart)
              .count();
      if (session.position() != promptCount + index)
        throw std::runtime_error(
            "serving session: decode position is inconsistent");
    }
    if (session.vocabSize() != sessionVocabSize)
      throw std::runtime_error("serving session: vocabulary size changed");
    const float *logits = session.logitsData(index == 0 ? promptCount - 1 : 0);
    if (!logits)
      throw std::runtime_error("serving session: logits pointer is null");
    const auto recent = repetitionHistory(history, config.repeatLastN);
    const int token = sampler.sample(logits, vocabSize, recent);
    ++result.usage.completionTokens;
    history.push_back(token);
    if (std::find(sampling.stopTokenIds.begin(), sampling.stopTokenIds.end(),
                  token) != sampling.stopTokenIds.end()) {
      result.finishReason = FinishReason::Stop;
      break;
    }
    const std::string delta =
        utf8.append(matcher.append(codec.tokenBytes(token)));
    if (!emit(delta, token))
      break;
    if (matcher.stopped) {
      result.finishReason = FinishReason::Stop;
      break;
    }
    previous = token;
  }

  if (result.finishReason != FinishReason::Cancelled) {
    std::string tail = utf8.append(matcher.finish());
    tail += utf8.finish();
    if (!tail.empty())
      emit(std::move(tail), -1);
  }
  result.usage.totalTokens = promptCount + result.usage.completionTokens;
  if (result.timings.decodeMs > 0)
    result.timings.tokensPerSecond =
        result.usage.completionTokens * 1000.0 / result.timings.decodeMs;
  if (callback && result.finishReason != FinishReason::Cancelled) {
    CompletionChunk done;
    done.done = true;
    done.finishReason = result.finishReason;
    done.usage = result.usage;
    done.timings = result.timings;
    if (!callback(done))
      result.finishReason = FinishReason::Cancelled;
  }
  cleanup.completed = result.finishReason != FinishReason::Cancelled;
  return result;
}

} // namespace runtime
} // namespace buddy
