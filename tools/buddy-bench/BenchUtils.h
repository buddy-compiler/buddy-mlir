//===- BenchUtils.h - Config parsing and aggregation for buddy-bench ------===//
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
// Expand -n / -p / -pg specs into configurations; aggregate samples into rows.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_BENCHUTILS_H
#define BUDDY_BENCH_BENCHUTILS_H

#include "Reporter.h"

#include "buddy/runtime/core/InferenceRunner.h"

#include <string>
#include <vector>

#include <CLI11.hpp>

namespace buddy {
namespace bench {

/// One test: promptLen (0 = --prompt text, -1 = decode-only, >0 = fixed), nGen.
struct BenchConfig {
  int promptLen;
  int nGen;
};

/// Expand "N", "N..M" or "N..M:S"; throws CLI::ValidationError on bad input.
std::vector<int> parseIntRange(const std::string &spec);

/// Expand a list of specs (already comma-split by CLI11) into ints.
std::vector<int> expandIntList(const std::vector<std::string> &specs);

/// Build the tests to run; -p, -n, --pg are concatenated (llama-bench order).
std::vector<BenchConfig>
buildConfigs(bool forwardMode, const std::vector<std::string> &nGenSpecs,
             const std::vector<std::string> &nPromptSpecs,
             const std::vector<std::string> &pgPairs);

/// Aggregate one configuration's samples into a report row.
BenchmarkRecord
summarize(const std::string &model,
          const std::vector<buddy::runtime::BenchmarkResult> &samples,
          bool forwardMode);

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_BENCHUTILS_H
