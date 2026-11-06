//===- BenchUtilsTest.cpp - buddy-bench helper tests ---------------------===//
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
// Tests the pure buddy-bench helpers: range expansion, configuration building
// and per-sample aggregation (per-run tokens/s, reps, test classification).
//
//===----------------------------------------------------------------------===//

// RUN: buddy-bench-utils-test 2>&1 | FileCheck %s

#include "BenchUtils.h"
#include "BenchBackend.h"
#include "Reporter.h"
#include "Stats.h"

#include <buddy/runtime/core/InferenceRunner.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using buddy::bench::avg;
using buddy::bench::BenchConfig;
using buddy::bench::BenchmarkRecord;
using buddy::bench::buildConfigs;
using buddy::bench::EnvInfo;
using buddy::bench::expandIntList;
using buddy::bench::parseIntRange;
using buddy::bench::renderCsv;
using buddy::bench::renderJson;
using buddy::bench::renderJsonl;
using buddy::bench::renderMarkdown;
using buddy::bench::stdev;
using buddy::bench::summarize;
using buddy::bench::tokensPerSecPerSample;
using buddy::runtime::BenchmarkResult;

namespace {

#define CHECK_TRUE(cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      return false;                                                            \
    }                                                                          \
  } while (0)

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

template <typename Fn> bool throwsValidationError(Fn fn) {
  try {
    fn();
  } catch (const CLI::ValidationError &) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool testParseIntRange() {
  CHECK_TRUE(parseIntRange("512") == std::vector<int>{512});
  CHECK_TRUE((parseIntRange("1..4") == std::vector<int>{1, 2, 3, 4}));
  CHECK_TRUE((parseIntRange("1..9:3") == std::vector<int>{1, 4, 7}));
  CHECK_TRUE((parseIntRange("0") == std::vector<int>{0}));

  CHECK_TRUE(throwsValidationError([] { parseIntRange("5..1"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("1..5:0"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("abc"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("-3"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("1..5:"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("1x..5"); }));
  // Range expansion is bounded: a stray wide range must not allocate ~8 GiB
  // and abort with bad_alloc (and the loop must not overflow near INT_MAX).
  CHECK_TRUE(throwsValidationError([] { parseIntRange("0..2000000000"); }));
  CHECK_TRUE(throwsValidationError([] { parseIntRange("0..2147483647"); }));

  CHECK_TRUE((expandIntList({"1..3", "8"}) == std::vector<int>{1, 2, 3, 8}));
  return true;
}

bool testBuildConfigs() {
  // llama-bench defaults: -p 512 and -n 128 are independent tests, so the
  // default run is pp512 followed by tg128 (not one combined row).
  {
    auto c = buildConfigs(false, {}, {}, {});
    CHECK_TRUE(c.size() == 2);
    CHECK_TRUE(c[0].promptLen == 512 && c[0].nGen == 0);
    CHECK_TRUE(c[1].promptLen == -1 && c[1].nGen == 128);
  }
  // The issue's pp / tg spellings.
  {
    auto c = buildConfigs(false, {"0"}, {"512"}, {});
    CHECK_TRUE(c.size() == 1);
    CHECK_TRUE(c[0].promptLen == 512 && c[0].nGen == 0);
  }
  {
    auto c = buildConfigs(false, {"128"}, {"0"}, {});
    CHECK_TRUE(c.size() == 1);
    CHECK_TRUE(c[0].promptLen == -1 && c[0].nGen == 128);
  }
  // Both given: pp tests first, then tg tests (concatenated, not paired).
  {
    auto c = buildConfigs(false, {"1..2"}, {"128", "256"}, {});
    CHECK_TRUE(c.size() == 4);
    CHECK_TRUE(c[0].promptLen == 128 && c[0].nGen == 0);
    CHECK_TRUE(c[1].promptLen == 256 && c[1].nGen == 0);
    CHECK_TRUE(c[2].promptLen == -1 && c[2].nGen == 1);
    CHECK_TRUE(c[3].promptLen == -1 && c[3].nGen == 2);
  }
  // -pg adds its pairs on top of the default pp/tg tests; prompt 0 is
  // decode-only and the (0, 0) pair selects nothing.
  {
    auto c = buildConfigs(false, {}, {}, {"512", "128", "0", "64", "0", "0"});
    CHECK_TRUE(c.size() == 4);
    CHECK_TRUE(c[0].promptLen == 512 && c[0].nGen == 0);
    CHECK_TRUE(c[1].promptLen == -1 && c[1].nGen == 128);
    CHECK_TRUE(c[2].promptLen == 512 && c[2].nGen == 128);
    CHECK_TRUE(c[3].promptLen == -1 && c[3].nGen == 64);
  }
  // Forward mode: explicit -p 0 means "use --prompt text", not decode-only.
  {
    auto c = buildConfigs(true, {"128"}, {"0"}, {});
    CHECK_TRUE(c.size() == 1);
    CHECK_TRUE(c[0].promptLen == 0 && c[0].nGen == 0);
  }
  // Forward mode without -p: one config on the prompt text.
  {
    auto c = buildConfigs(true, {}, {}, {});
    CHECK_TRUE(c.size() == 1);
    CHECK_TRUE(c[0].promptLen == 0 && c[0].nGen == 0);
  }
  // Odd number of -pg values.
  CHECK_TRUE(
      throwsValidationError([] { buildConfigs(false, {"128"}, {}, {"512"}); }));
  // Nothing selected: -p 0 and -n 0 leave no test.
  CHECK_TRUE(
      throwsValidationError([] { buildConfigs(false, {"0"}, {"0"}, {}); }));
  // A (0, 0) -pg pair alone also selects nothing.
  CHECK_TRUE(throwsValidationError(
      [] { buildConfigs(false, {"0"}, {"0"}, {"0", "0"}); }));
  // Malformed -pg value.
  CHECK_TRUE(throwsValidationError(
      [] { buildConfigs(false, {"128"}, {}, {"512", "x"}); }));
  return true;
}

bool testStats() {
  CHECK_TRUE(near(avg({1.0, 2.0, 3.0}), 2.0));
  CHECK_TRUE(near(avg({}), 0.0));
  CHECK_TRUE(near(stdev({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}),
                  2.1380899352993947));
  CHECK_TRUE(near(stdev({3.0}), 0.0));

  auto tps = tokensPerSecPerSample({1.0, 2.0, 0.0}, {10, 5, 7});
  CHECK_TRUE(tps.size() == 3);
  CHECK_TRUE(near(tps[0], 10.0));
  CHECK_TRUE(near(tps[1], 2.5));
  CHECK_TRUE(near(tps[2], 0.0));
  return true;
}

bool testSummarize() {
  // Two repetitions with different token counts (EOS early stop) and a padded
  // prefill (tiered cache), so per-sample throughput is the only correct way.
  std::vector<BenchmarkResult> samples(2);
  samples[0].nPrompt = 100;
  samples[0].prefillWorkLen = 128;
  samples[0].prefillSecs = 1.0;
  samples[0].nGen = 10;
  samples[0].decodeSecs = 1.0;
  samples[1].nPrompt = 100;
  samples[1].prefillWorkLen = 128;
  samples[1].prefillSecs = 2.0;
  samples[1].nGen = 5;
  samples[1].decodeSecs = 2.0;

  auto rec = summarize("m", samples, false);
  CHECK_TRUE(rec.model == "m");
  CHECK_TRUE(rec.test == "pg");
  CHECK_TRUE(rec.nPrompt == 100);
  CHECK_TRUE(rec.nGen == 10);
  CHECK_TRUE(rec.reps == 2);
  // prefill: 128/1 and 128/2 -> mean 96.
  CHECK_TRUE(near(rec.prefillTps, 96.0));
  // decode: 10/1 and 5/2 -> mean 6.25.
  CHECK_TRUE(near(rec.decodeTps, 6.25));
  CHECK_TRUE(near(rec.avgSecs, 3.0));
  CHECK_TRUE(near(rec.prefillTpsStddev, stdev({128.0, 64.0})));
  CHECK_TRUE(near(rec.decodeTpsStddev, stdev({10.0, 2.5})));

  // Without an explicit work length, the requested prompt length is used.
  std::vector<BenchmarkResult> plain(1);
  plain[0].nPrompt = 64;
  plain[0].prefillSecs = 2.0;
  plain[0].nGen = 0;
  plain[0].decodeSecs = 0.0;
  auto pp = summarize("m", plain, false);
  CHECK_TRUE(pp.test == "pp");
  CHECK_TRUE(near(pp.prefillTps, 32.0));

  // Decode-only classification. The runner's seed prefill is setup, not part
  // of the measured decode, so a tg row reports no prefill throughput and
  // times the row over decode alone (llama-bench semantics).
  std::vector<BenchmarkResult> tg(2);
  tg[0].nPrompt = 0;
  tg[0].nGen = 8;
  tg[0].prefillWorkLen = 64;
  tg[0].prefillSecs = 5.0;
  tg[0].decodeSecs = 1.0;
  tg[1].nPrompt = 0;
  tg[1].nGen = 8;
  tg[1].prefillWorkLen = 64;
  tg[1].prefillSecs = 7.0;
  tg[1].decodeSecs = 2.0;
  auto tgRec = summarize("m", tg, false);
  CHECK_TRUE(tgRec.test == "tg");
  CHECK_TRUE(near(tgRec.decodeTps, 6.0)); // (8/1 + 8/2) / 2
  CHECK_TRUE(near(tgRec.prefillTps, 0.0));
  CHECK_TRUE(near(tgRec.prefillTpsStddev, 0.0));
  CHECK_TRUE(near(tgRec.avgSecs, 1.5));
  CHECK_TRUE(near(tgRec.stddevSecs, stdev({1.0, 2.0})));

  // Forward mode classification and extra-metric aggregation.
  std::vector<BenchmarkResult> fwd(2);
  fwd[0].prefillSecs = 1.0;
  fwd[0].extraMetrics = {{"images_per_sec", 2.0}};
  fwd[1].prefillSecs = 1.0;
  fwd[1].extraMetrics = {{"images_per_sec", 4.0}};
  auto fwdRec = summarize("m", fwd, true);
  CHECK_TRUE(fwdRec.test == "forward");
  CHECK_TRUE(fwdRec.extraMetrics.size() == 1);
  CHECK_TRUE(fwdRec.extraMetrics[0].name == "images_per_sec");
  CHECK_TRUE(near(fwdRec.extraMetrics[0].avg, 3.0));
  return true;
}

bool testReporter() {
  EnvInfo env;
  env.cpuModel = "cpu \"name\"\nline"; // quote + newline must be escaped
  env.usedCores = 8;
  env.version = "0.0.0";
  env.modelPath = "/tmp/m.rax";
  env.modelSizeBytes = 42;
  env.modelMtime = "2026-01-02T03:04:05Z";
  env.precision = "f32";
  env.backend = "cpu";
  env.device = "cpu";

  BenchmarkRecord r;
  r.model = "m|delim,\"q\""; // pipe breaks md, comma+quote break csv
  r.test = "pg";
  r.nPrompt = 8;
  r.nGen = 4;
  r.prefillTps = std::nan("");
  r.decodeTps = INFINITY;
  r.prefillTpsStddev = 0.5;
  r.decodeTpsStddev = 0;
  r.avgSecs = 1.25;
  r.stddevSecs = 0;
  r.reps = 1;
  r.extraMetrics = {{"images_per_sec", 12.5, 0.5}};

  const std::string json = renderJson({r}, env);
  // JSON has no NaN / Infinity: both become null.
  CHECK_TRUE(json.find("\"prefill_tps\":null") != std::string::npos);
  CHECK_TRUE(json.find("\"decode_tps\":null") != std::string::npos);
  CHECK_TRUE(json.find("nan") == std::string::npos);
  CHECK_TRUE(json.find("inf") == std::string::npos);
  // Quotes are escaped, including in env strings, and newlines become \n.
  CHECK_TRUE(json.find("\"model\":\"m|delim,\\\"q\\\"\"") != std::string::npos);
  CHECK_TRUE(json.find("\"cpu_model\":\"cpu \\\"name\\\"\\nline\"") !=
             std::string::npos);
  // The .rax identity is part of the envelope.
  CHECK_TRUE(json.find("\"model_size\":42") != std::string::npos);
  CHECK_TRUE(json.find("\"model_mtime\":\"2026-01-02T03:04:05Z\"") !=
             std::string::npos);
  CHECK_TRUE(json.find("\"images_per_sec\":{\"avg\":12.5") !=
             std::string::npos);

  // JSON Lines: one env line, then one record line.
  const std::string jsonl = renderJsonl({r}, env);
  CHECK_TRUE(jsonl.find("{\"env\":{") == 0);
  CHECK_TRUE(jsonl.find('\n') != std::string::npos);
  CHECK_TRUE(jsonl.find("\"test\":\"pg\"") != std::string::npos);

  // CSV: a field with a comma or quote is quoted, inner quotes doubled.
  const std::string csv = renderCsv({r});
  CHECK_TRUE(csv.find("\n\"m|delim,\"\"q\"\"\",pg,") != std::string::npos);

  // Markdown: a pipe is escaped so the table keeps its columns.
  const std::string md = renderMarkdown({r});
  CHECK_TRUE(md.find("| m\\|delim,\"q\" | pg |") != std::string::npos);
  return true;
}

} // namespace

int main() {
  if (!testParseIntRange())
    return 1;
  if (!testBuildConfigs())
    return 1;
  if (!testStats())
    return 1;
  if (!testSummarize())
    return 1;
  if (!testReporter())
    return 1;

  // CHECK: All bench utils tests passed
  std::printf("All bench utils tests passed\n");
  return 0;
}
