//===- Qwen3VLTestKernels.cpp - Deterministic Qwen3-VL kernel ABI ---------===//
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

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr long kContext = 512;
constexpr long kHidden = 2048;
constexpr long kImages = 98;
std::vector<int> expectedPrompt;
std::vector<int> selected;
long imageOffset = 7;
int visionCalls = 0;
int decoderCalls = 0;
int failCall = 0;
int checkedPrevious = 0;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("Qwen test kernel: " + message);
}

} // namespace

// Test control lives in the same loaded library as the production ABI symbols.
// Selection depends on the logits row, so a new request needs no hidden step
// reset in the runtime. The counters only observe execution and inject errors.
extern "C" void qwen3vl_test_configure(const int *prompt, long promptCount,
                                       const int *script, long scriptCount,
                                       long imagePosition, int failureCall) {
  expectedPrompt.assign(prompt, prompt + promptCount);
  selected.assign(script, script + scriptCount);
  imageOffset = imagePosition;
  visionCalls = decoderCalls = checkedPrevious = 0;
  failCall = failureCall;
}

extern "C" int qwen3vl_test_stat(int index) {
  switch (index) {
  case 0:
    return visionCalls;
  case 1:
    return decoderCalls;
  case 2:
    return checkedPrevious;
  default:
    return -1;
  }
}

extern "C" void qwen3vl_vision(const float *weights, long weightCount,
                               const float *pixels, float *deep0, float *deep1,
                               float *deep2, float *pooled) {
  ++visionCalls;
  check(weightCount == 1 && weights[0] == 0.25f && pixels,
        "vision resource arguments changed");
  for (long row = 0; row < kImages; ++row) {
    std::fill_n(pooled + row * kHidden, kHidden, 9000.0f + row);
    std::fill_n(deep0 + row * kHidden, kHidden, 10000.0f + row);
    std::fill_n(deep1 + row * kHidden, kHidden, 11000.0f + row);
    std::fill_n(deep2 + row * kHidden, kHidden, 12000.0f + row);
  }
}

extern "C" void qwen3vl_decoder(const float *weights, long weightCount,
                                const float *inputs, const float *cos,
                                const float *sin, const float *mask,
                                const float *deep0, const float *deep1,
                                const float *deep2, float *logits, long context,
                                long vocab, long hidden, long head) {
  ++decoderCalls;
  if (decoderCalls == failCall)
    throw std::runtime_error("scripted Qwen decoder failure");
  check(weightCount == 1 && weights[0] == 0.25f && context == kContext &&
            hidden == kHidden && head == 128 && vocab == 264,
        "decoder dimensions or weights changed");
  check(cos[0] == 1 && cos[context * head - 1] == 1 && sin[0] == 0 &&
            mask[context * context - 1] == -42,
        "positional resources did not reach the decoder");
  for (long row = 0; row < static_cast<long>(expectedPrompt.size()); ++row) {
    const long image = row - imageOffset;
    const bool isImage = image >= 0 && image < kImages;
    const float embedding = isImage ? 9000.0f + image : expectedPrompt[row];
    for (long column : {0L, hidden - 1}) {
      const long offset = row * hidden + column;
      check(inputs[offset] == embedding,
            "prompt embedding/image placement at row " + std::to_string(row));
      check(deep0[offset] == (isImage ? 10000.0f + image : 0) &&
                deep1[offset] == (isImage ? 11000.0f + image : 0) &&
                deep2[offset] == (isImage ? 12000.0f + image : 0),
            "deepstack image placement at row " + std::to_string(row));
    }
  }
  for (long index = 0;
       index < static_cast<long>(selected.size()) &&
       static_cast<long>(expectedPrompt.size()) + index < context;
       ++index) {
    const long offset = (expectedPrompt.size() + index) * hidden;
    if (inputs[offset] == 0)
      break;
    check(inputs[offset] == selected[index] &&
              inputs[offset + hidden - 1] == selected[index],
          "previous selected token was not embedded at its decode position");
    ++checkedPrevious;
  }
  std::fill_n(logits, context * vocab, -1000.0f);
  for (long row = 0; row < context; ++row) {
    const long index = row - static_cast<long>(expectedPrompt.size()) + 1;
    const int token = index >= 0 && index < static_cast<long>(selected.size())
                          ? selected[index]
                          : '!';
    check(token >= 0 && token < vocab, "script contains an invalid token");
    logits[row * vocab + token] = 1000.0f;
    // Compiled LM heads may be padded beyond the raw vocabulary. Even these
    // larger logits must not compete with a decodable token ID.
    logits[row * vocab + 262] = 2000.0f;
    logits[row * vocab + 263] = 3000.0f;
  }
}
