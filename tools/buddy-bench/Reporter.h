//===- Reporter.h - Benchmark result output for buddy-bench --------------===//
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
// Render benchmark records as Markdown / CSV / JSON / JSON Lines.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_REPORTER_H
#define BUDDY_BENCH_REPORTER_H

#include <string>
#include <vector>

namespace buddy {
namespace bench {

struct EnvInfo;

/// A model-specific metric aggregated across reps: {name, mean, stddev}.
struct ExtraMetric {
  std::string name;
  double avg = 0;
  double stddev = 0;
};

/// One output row, decoupled from the runtime-level BenchmarkResult.
struct BenchmarkRecord {
  std::string model;
  std::string test;            // "pp" (prefill) / "tg" (decode) / "pg" (both)
  int nPrompt = 0;             // -p
  int nGen = 0;                // -n
  double prefillTps = 0;       // prefill tokens/s (avg)
  double decodeTps = 0;        // decode tokens/s (avg)
  double prefillTpsStddev = 0; // prefill tokens/s (sample stddev)
  double decodeTpsStddev = 0;  // decode tokens/s (sample stddev)
  double avgSecs = 0;          // total time (avg)
  double stddevSecs = 0;       // total time (sample stddev)
  int reps = 0;                // number of repetitions
  /// Model-specific metrics, aggregated by name across reps. Empty for LLM.
  std::vector<ExtraMetric> extraMetrics;
};

/// Render all records as a Markdown table.
std::string renderMarkdown(const std::vector<BenchmarkRecord> &records);

/// Render all records as CSV (header row + one row per record).
std::string renderCsv(const std::vector<BenchmarkRecord> &records);

/// Render as {"env": {...}, "results": [...]}.
std::string renderJson(const std::vector<BenchmarkRecord> &records,
                       const EnvInfo &env);

/// Render JSON Lines: one env line, then one object per record.
std::string renderJsonl(const std::vector<BenchmarkRecord> &records,
                        const EnvInfo &env);

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_REPORTER_H
