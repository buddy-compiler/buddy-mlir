//===- Stats.h - Statistics helpers for buddy-bench -----------------------===//
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
// Mean, standard deviation and per-run throughput helpers; no runtime dep.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_STATS_H
#define BUDDY_BENCH_STATS_H

#include <vector>

namespace buddy {
namespace bench {

/// Arithmetic mean of a sample set. Empty input returns 0.
double avg(const std::vector<double> &samples);

/// Sample standard deviation (n-1 denominator). <=1 sample returns 0.
double stdev(const std::vector<double> &samples);

/// Per-run tokens/second (0 for non-positive time); per run, not averaged.
std::vector<double> tokensPerSecPerSample(const std::vector<double> &secs,
                                          const std::vector<int> &nTokens);

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_STATS_H
