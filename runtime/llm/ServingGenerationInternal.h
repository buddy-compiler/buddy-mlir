//===- ServingGenerationInternal.h - Serving implementation details -----===//
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

#ifndef BUDDY_RUNTIME_LLM_SERVINGGENERATIONINTERNAL_H
#define BUDDY_RUNTIME_LLM_SERVINGGENERATIONINTERNAL_H

#include "buddy/runtime/llm/ServingGeneration.h"

#include <chrono>

namespace buddy {
namespace runtime {
namespace detail {

using ServingNow = std::function<std::chrono::steady_clock::time_point()>;

// Internal clock injection keeps timing tests independent of machine speed.
CompletionResult runServingGenerationImpl(
    const std::string &prompt, const ServingTokenSource &source,
    const SamplingParams &sampling, const ServingTextCodec &codec,
    const CompletionStreamCallback &callback, const ServingNow &now,
    const std::function<void()> &prepare = {},
    const std::function<void(const std::vector<int> &)> &preparePrefill = {});

} // namespace detail
} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_SERVINGGENERATIONINTERNAL_H
