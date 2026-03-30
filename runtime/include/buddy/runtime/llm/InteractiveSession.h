//===- InteractiveSession.h - REPL interactive mode -----------------------===//
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
// Declares the interactive REPL session for multi-turn conversation.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_INTERACTIVESESSION_H
#define BUDDY_RUNTIME_LLM_INTERACTIVESESSION_H

#include "buddy/Core/Container.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/Sampler.h"
#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/llm/TextGeneration.h"

#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Run an interactive REPL session for multi-turn conversation.
///
/// Commands:
///   :exit / :quit   Exit
///   :clear          Clear conversation history
///   /history        Show conversation history
///   /system <text>  Set system prompt
///   Line ending \   Multi-line continuation
void runInteractiveSession(LLMSession &session, MemRef<float, 1> &weights,
                           const std::string &vocabPath, const RunConfig &cfg,
                           const std::vector<long long> &stopTokenIds,
                           buddy::ConversationManager &conv,
                           const TextCodec &codec, buddy::Sampler &sampler);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_INTERACTIVESESSION_H
