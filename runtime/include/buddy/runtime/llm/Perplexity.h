//===- Perplexity.h - Perplexity of an LLM over a token stream ------------===//
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
// Perplexity of an LLMSession over a stream of token ids, scored like
// llama.cpp's llama-perplexity so that the two can be compared on the same
// tokens: the stream is cut into chunks of `context` tokens, each chunk
// starts from an empty KV cache, and only the second half of a chunk is
// scored (position j predicts token j + 1 for j in [context / 2,
// context - 1)).
//
// A chunk's first token runs as a one-token prompt (prefill, which empties
// the KV cache), the others one decode step each: this measures the
// numerics of the decode path.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_LLM_PERPLEXITY_H
#define BUDDY_RUNTIME_LLM_PERPLEXITY_H

#include "buddy/runtime/llm/LLMSession.h"

#include <string>
#include <vector>

namespace buddy {
namespace runtime {

struct PerplexityOptions {
  /// Tokens per chunk (even, at least 4, at most the session's KV cache).
  int context = 512;
  /// At most this many chunks; 0: all the whole chunks of the stream.
  int maxChunks = 0;
  /// No per-chunk progress lines.
  bool quiet = false;
};

struct PerplexityResult {
  double perplexity = 0;
  long scoredTokens = 0;
  int chunks = 0;
  double seconds = 0;
};

/// The token ids of a file: its unsigned integers, in order, whatever
/// separates them ("[1, 2, 3]" as `llama-tokenize --ids` prints, or one per
/// line).
std::vector<int> readTokenIds(const std::string &path);

/// Runs `session` over `ids` (see the file comment). Prints the running
/// perplexity after each chunk (unless opts.quiet) and the final estimate.
PerplexityResult runPerplexity(LLMSession &session, const std::vector<int> &ids,
                               const PerplexityOptions &opts);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_LLM_PERPLEXITY_H
