//===- DeepSeekR1ResidentModelTest.cpp ------------------------------------===//
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

// RUN: buddy-deepseek-resident-model-test \
// RUN:   %S/../../../examples/BuddyDeepSeekR1/vocab.txt \
// RUN:   %S/../../../examples/BuddyDeepSeekR1/deepseek-r1.json default \
// RUN:   2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: buddy-deepseek-resident-model-test \
// RUN:   %S/../../../examples/BuddyDeepSeekR1/vocab.txt \
// RUN:   %S/../../../examples/BuddyDeepSeekR1/deepseek-r1.json custom \
// RUN:   2>&1 | FileCheck %s --check-prefix=CUSTOM

#include "buddy/runtime/models/DeepSeekR1ResidentModel.h"
#include "buddy/runtime/models/ModelSession.h"

#include <iostream>
#include <stdexcept>

using namespace buddy::runtime;

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4, "expected vocabulary, chat template and mode");
    const bool custom = std::string(argv[3]) == "custom";
    require(custom || std::string(argv[3]) == "default", "unknown mode");
    ResidentModelConfig config;
    config.modelSoPath = "unused-model.so";
    config.weightPaths = {"unused-weights.data"};
    config.vocabPath = argv[1];
    if (custom)
      config.chatTemplatePath = argv[2];
    DeepSeekR1ResidentModel model;
    model.load(config);

    ChatCompletionRequest chat;
    chat.messages.push_back({"user", "A"});
    const std::string prompt = custom ? model.renderChat(chat) : "A";
    const std::vector<int> raw =
        custom ? std::vector<int>{151646, 151644, 32, 151643, 151645}
               : std::vector<int>{32};
    const std::vector<int> generated =
        custom ? raw : std::vector<int>{151646, 151646, 151644, 32,
                                        151645, 151648, 198};

    for (bool addSpecial : {false, true}) {
      for (bool countOnly : {false, true}) {
        TokenizeRequest request;
        request.content = prompt;
        request.addSpecial = addSpecial;
        request.countOnly = countOnly;
        const auto result = model.tokenize(request);
        const auto &expected = addSpecial ? generated : raw;
        require(result.count == expected.size(), "tokenize count mismatch");
        require(result.tokens == (countOnly ? std::vector<int>{} : expected),
                "tokenize sequence mismatch");
      }
    }

    CompletionRequest completion;
    completion.prompt = prompt;
    const auto verifyCompletion = [&](const CompletionResult &result) {
      require(result.usage.promptTokens == static_cast<int>(generated.size()),
              "usage disagrees with tokenize");
      require(ModelSession::promptTokens ==
                  std::vector<size_t>(generated.begin(), generated.end()),
              "prefill disagrees with tokenize");
    };
    verifyCompletion(model.complete(completion));
    bool receivedUsage = false;
    verifyCompletion(
        model.completeStream(completion, [&](const CompletionChunk &chunk) {
          if (chunk.done) {
            require(chunk.usage.promptTokens ==
                        static_cast<int>(generated.size()),
                    "stream usage disagrees with tokenize");
            receivedUsage = true;
          }
          return true;
        }));
    require(receivedUsage, "stream did not report final usage");
    if (custom)
      verifyCompletion(model.chat(chat));

    // DEFAULT: default template: token IDs, count_only and usage agree
    // CUSTOM: custom template: token IDs, count_only and usage agree
    std::cout << (custom ? "custom" : "default")
              << " template: token IDs, count_only and usage agree\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
