//===- DefaultChatTemplates.h -------------------------------------------===//
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

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_DEFAULTCHATTEMPLATES
#define FRONTEND_INTERFACES_BUDDY_LLM_DEFAULTCHATTEMPLATES

#include "buddy/LLM/TextContainer.h"
#include <algorithm>
#include <stdexcept>

namespace buddy {

/// Add the historical single-turn template to already tokenized user text.
/// Call once after tokenizeDeepSeekR1(), with capacity for six extra tokens.
/// The two BOS tokens and Assistant/think/newline suffix preserve the original
/// default format. Padding beyond the resulting token count is unchanged.
/// Throws std::length_error without changing tokens if capacity is
/// insufficient.
template <typename T, size_t N>
void applyDefaultDeepSeekR1Template(Text<T, N> &tokens) {
  const size_t count = tokens.getTokenCnt();
  const size_t capacity = tokens.getSize();
  if (capacity < 6 || count > capacity - 6)
    throw std::length_error(
        "DeepSeek R1 default template exceeds token capacity");
  T *data = tokens.getData();
  std::move_backward(data, data + count, data + count + 3);
  data[0] = 151646;         // BOS
  data[1] = 151646;         // BOS
  data[2] = 151644;         // User
  data[count + 3] = 151645; // Assistant
  data[count + 4] = 151648; // Think
  data[count + 5] = 198;    // Newline
  tokens.setTokenCnt(count + 6);
}

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_DEFAULTCHATTEMPLATES
