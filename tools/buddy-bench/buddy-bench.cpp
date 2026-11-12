//===- buddy-bench.cpp - RAX model benchmarking tool ----------------------===//
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
// Modeled after llama-bench: run prefill / decode / forward passes on a .rax.
//
//===----------------------------------------------------------------------===//

#include "Affinity.h"
#include "BenchBackend.h"
#include "BenchUtils.h"
#include "Reporter.h"

#include "buddy/runtime/core/InferenceRunner.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <CLI11.hpp>

namespace buddy {
namespace bench {

int run(int argc, char **argv) {
  std::string modelPath;
  std::string prompt = "Hello, my name is buddy.";
  std::vector<std::string> nGenRaw{"128"};
  std::vector<std::string> nPromptRaw;
  std::vector<std::string> pgPairs;
  int reps = 5;
  bool noWarmup = false;
  bool ignoreEos = false;
  std::string outputFormat = "md";
  std::string mode = "generation"; // generation | forward
  int batchSize = 1;
  std::string cpuSpec;
  std::string numaNodes;
  std::string numaCpuBind;
  std::string numaInterleave;

  CLI::App app{"buddy-bench: benchmark RAX model packages"};
  app.add_option("--model", modelPath, "Path to a .rax model package")
      ->required();
  app.add_option("--prompt", prompt, "Input prompt text");
  app.add_option("-n,--n-gen", nGenRaw,
                 "Decode budget in tokens (default: 128). Each value adds a "
                 "token-generation test. Comma-separated values or ranges, "
                 "e.g. 32,64 or 16..256:16")
      ->delimiter(',');
  app.add_option("-p,--n-prompt", nPromptRaw,
                 "Fixed prompt length in tokens (default: 512; 0 = no "
                 "prompt). Each positive value adds a prompt-processing test. "
                 "Comma-separated values or ranges. --prompt supplies the "
                 "text, truncated or padded to this length.")
      ->delimiter(',');
  app.add_option("--pg,--prompt-gen", pgPairs,
                 "Explicit prompt,gen pairs, added on top of the -p and -n "
                 "tests, e.g. 512,128 or 512,128,256,64")
      ->delimiter(',');
  app.add_option("-r,--repetitions", reps, "Number of repetitions (>= 1)")
      ->check(CLI::Range(1, std::numeric_limits<int>::max(),
                         "repetitions must be at least 1"));
  app.add_flag("--no-warmup", noWarmup, "Disable warmup");
  app.add_flag("--ignore-eos", ignoreEos,
               "Keep decoding for the full -n budget even after an end-of-"
               "sequence token");
  app.add_option("-o,--output", outputFormat,
                 "Output format: md/csv/json/jsonl")
      ->check(CLI::IsMember({"md", "csv", "json", "jsonl"}));
  app.add_option("--mode", mode,
                 "Benchmark mode: generation (prefill/decode) or forward "
                 "(single forward pass, no decode)")
      ->check(CLI::IsMember({"generation", "forward"}));
  app.add_option("--batch-size", batchSize,
                 "Runtime batch size (only 1 is supported)")
      ->check(CLI::Range(1, 1,
                         "batch size must be 1: no runner implements batched "
                         "benchmarking yet"));
  app.add_option("--cpus", cpuSpec,
                 "CPU affinity, e.g. 0-47 or 0-15,32-47 (taskset -c <spec>)")
      ->group("NUMA / affinity");
  app.add_option("--numa", numaNodes,
                 "Shortcut: sets both cpubind AND interleave, e.g. 0,1,2,3")
      ->group("NUMA / affinity");
  app.add_option("--numa-cpubind", numaCpuBind,
                 "Bind to CPUs of these NUMA nodes")
      ->group("NUMA / affinity");
  app.add_option("--numa-interleave", numaInterleave,
                 "Interleave memory allocation across nodes")
      ->group("NUMA / affinity");

  // CLI11 has no multi-letter short options, so rewrite `-pg` to `--pg`.
  std::vector<std::string> argStore;
  argStore.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    argStore.emplace_back(i > 0 && std::string(argv[i]) == "-pg" ? "--pg"
                                                                 : argv[i]);
  }
  std::vector<char *> args;
  args.reserve(argStore.size());
  for (auto &a : argStore)
    args.push_back(a.data());

  try {
    app.parse(argc, args.data());
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Apply NUMA / affinity before loading the model; failures abort the run.
  try {
    if (!numaNodes.empty()) {
      applyNumaInterleave(numaNodes);
      applyNumaCpuBind(numaNodes);
    }
    if (!numaInterleave.empty())
      applyNumaInterleave(numaInterleave);
    if (!numaCpuBind.empty())
      applyNumaCpuBind(numaCpuBind);
    if (!cpuSpec.empty())
      applyCpuAffinity(cpuSpec);
  } catch (const CLI::Error &e) {
    return app.exit(e);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "buddy-bench: %s\n", e.what());
    return 1;
  }

  // ── Build benchmark configurations ────────────────────────────────────
  const bool forwardMode = (mode == "forward");
  std::vector<BenchConfig> configs;
  try {
    configs = buildConfigs(forwardMode, nGenRaw, nPromptRaw, pgPairs);
  } catch (const CLI::ValidationError &e) {
    return app.exit(e);
  }

  // ── Unified backend: everything below talks only to this facade ───────
  BenchBackend backend;
  if (!backend.load(modelPath)) {
    std::fprintf(stderr, "buddy-bench: failed to load model: %s\n",
                 modelPath.c_str());
    return 1;
  }
  backend.setBatchSize(batchSize);

  if (!backend.supportsBench()) {
    std::fprintf(stderr,
                 "buddy-bench: model %s does not implement runBench(); nothing "
                 "to benchmark\n",
                 backend.modelName().c_str());
    return 1;
  }

  EnvInfo env = backend.getEnvInfo();
  std::fprintf(stderr,
               "buddy-bench: model=%s precision=%s backend=%s device=%s "
               "cpu=%s cores=%d used_cores=%d affinity=%s numa=%s isa=%s "
               "version=%s batch=%d\n",
               backend.modelName().c_str(), env.precision.c_str(),
               env.backend.c_str(), env.device.c_str(), env.cpuModel.c_str(),
               env.numCores, env.usedCores, env.cpuAffinity.c_str(),
               env.numaConfig.c_str(), env.isa.c_str(), env.version.c_str(),
               env.batchSize);

  // ── Benchmark loop: one row per configuration ────────────────────────
  std::vector<BenchmarkRecord> records;
  for (const auto &cfgSpec : configs) {
    std::fprintf(stderr, "buddy-bench: running promptLen=%d nGen=%d\n",
                 cfgSpec.promptLen, cfgSpec.nGen);

    buddy::runtime::RunConfig cfg;
    cfg.raxPath = modelPath;
    cfg.prompt = prompt;
    cfg.suppressStats = true;
    cfg.promptLength =
        cfgSpec.promptLen; // 0 = text, -1 = decode-only, >0 = fixed
    cfg.maxNewTokens = cfgSpec.nGen;
    cfg.batchSize = batchSize;
    cfg.ignoreEos = ignoreEos;

    // Warm up in runBench() so it shares the session with the measurements.
    const int warmupReps = noWarmup ? 0 : 1;

    std::vector<buddy::runtime::BenchmarkResult> samples;
    if (!backend.runBench(cfg, warmupReps, reps, samples)) {
      std::fprintf(stderr,
                   "buddy-bench: runBench failed for promptLen=%d nGen=%d\n",
                   cfgSpec.promptLen, cfgSpec.nGen);
      return 1;
    }
    if (samples.empty()) {
      std::fprintf(stderr, "buddy-bench: no samples collected\n");
      return 1;
    }

    records.push_back(summarize(backend.modelName(), samples, forwardMode));
  }

  // ── Output ────────────────────────────────────────────────────────────
  if (outputFormat == "csv")
    std::cout << renderCsv(records);
  else if (outputFormat == "json")
    std::cout << renderJson(records, env);
  else if (outputFormat == "jsonl")
    std::cout << renderJsonl(records, env);
  else
    std::cout << renderMarkdown(records);

  return 0;
}

} // namespace bench
} // namespace buddy

int main(int argc, char **argv) {
  try {
    return buddy::bench::run(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "buddy-bench: fatal: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "buddy-bench: fatal: unknown exception\n");
    return 1;
  }
}
