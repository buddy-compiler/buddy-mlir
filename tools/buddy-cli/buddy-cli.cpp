// buddy-cli/buddy-cli.cpp — Generic BuddyRuntime inference CLI -----------===//
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
// Usage:
//   buddy-cli --model <path.rax> [--prompt <text>] [--max-tokens N]
//             [--cpus 0-47] [--numa 0,1,2,3]
//             [--numa-interleave 0,1,2,3] [--numa-cpubind 0,1,2,3]
//
// NUMA / affinity flags mirror numactl + taskset:
//   --cpus           0-47          taskset -c 0-47
//   --numa           0,1,2,3       numactl --cpunodebind=... --interleave=...
//   --numa-cpubind   0,1,2,3       numactl --cpunodebind=...
//   --numa-interleave 0,1,2,3      numactl --interleave=...
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/DeepSeekR1Runner.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef BUDDY_CLI_HAVE_NUMA
#include <numa.h>
#endif

//===----------------------------------------------------------------------===//
// Affinity helpers
//===----------------------------------------------------------------------===//

// Parse "0-47" or "0-15,32-47,64" into a cpu_set_t.
static cpu_set_t parseCpuSet(const std::string &spec) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  std::istringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    // strip whitespace
    while (!tok.empty() && tok.front() == ' ')
      tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == ' ')
      tok.pop_back();
    if (tok.empty())
      continue;
    auto dash = tok.find('-');
    if (dash != std::string::npos) {
      int lo = std::stoi(tok.substr(0, dash));
      int hi = std::stoi(tok.substr(dash + 1));
      for (int c = lo; c <= hi; ++c)
        CPU_SET(c, &mask);
    } else {
      CPU_SET(std::stoi(tok), &mask);
    }
  }
  return mask;
}

// Apply CPU affinity via sched_setaffinity.
static void applyCpuAffinity(const std::string &spec) {
  auto mask = parseCpuSet(spec);
  if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
    std::cerr << "[buddy-cli] sched_setaffinity failed: " << strerror(errno)
              << "\n";
  else
    std::cout << "[buddy-cli] CPU affinity set: " << spec << "\n";
}

#ifdef BUDDY_CLI_HAVE_NUMA
// Parse "0,1,2,3" into a numa bitmask.
static struct bitmask *parseNodeMask(const std::string &spec) {
  struct bitmask *mask = numa_allocate_nodemask();
  std::istringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    while (!tok.empty() && tok.front() == ' ')
      tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == ' ')
      tok.pop_back();
    if (!tok.empty())
      numa_bitmask_setbit(mask, std::stoi(tok));
  }
  return mask;
}

// Set NUMA memory interleave policy (like --interleave=nodes).
static void applyNumaInterleave(const std::string &spec) {
  if (numa_available() < 0) {
    std::cerr << "[buddy-cli] NUMA not available on this system\n";
    return;
  }
  auto *mask = parseNodeMask(spec);
  numa_set_interleave_mask(mask);
  numa_free_nodemask(mask);
  std::cout << "[buddy-cli] NUMA interleave set: nodes " << spec << "\n";
}

// Bind execution to CPUs of the given NUMA nodes (like --cpunodebind=nodes).
static void applyNumaCpuBind(const std::string &spec) {
  if (numa_available() < 0) {
    std::cerr << "[buddy-cli] NUMA not available on this system\n";
    return;
  }
  auto *mask = parseNodeMask(spec);
  numa_run_on_node_mask(mask);
  numa_free_nodemask(mask);
  std::cout << "[buddy-cli] NUMA cpubind set: nodes " << spec << "\n";
}
#else
static void applyNumaInterleave(const std::string &) {
  std::cerr << "[buddy-cli] Built without libnuma; --numa-interleave ignored\n";
}
static void applyNumaCpuBind(const std::string &) {
  std::cerr << "[buddy-cli] Built without libnuma; --numa-cpubind ignored\n";
}
#endif

//===----------------------------------------------------------------------===//
// Model dispatch
//===----------------------------------------------------------------------===//

static std::unique_ptr<buddy::runtime::InferenceRunner>
makeRunner(const std::string &modelName) {
  if (modelName.rfind("deepseek_r1", 0) == 0)
    return std::make_unique<buddy::runtime::DeepSeekR1Runner>();

  throw std::runtime_error(
      "buddy-cli: unknown model '" + modelName +
      "'.\n"
      "  Supported models: deepseek_r1\n"
      "  To add a new model, implement InferenceRunner and register it here.");
}

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

static void usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "\n"
      << "Model source (one required):\n"
      << "  --model      <path.rax>  Model manifest (recommended)\n"
      << "  --model-so   <path.so>   Model shared library  (legacy mode)\n"
      << "  --weights    <path>      Weights file           (legacy mode)\n"
      << "  --vocab      <path>      Vocabulary file        (legacy mode)\n"
      << "\n"
      << "Inference:\n"
      << "  --prompt     <text>      Input prompt (interactive if omitted)\n"
      << "  --max-tokens <N>         Max total tokens incl. prompt (default "
         "1024)\n"
      << "\n"
      << "NUMA / affinity (applied before model load):\n"
      << "  --cpus       <spec>      CPU affinity, e.g. 0-47 or 0-15,32-47\n"
      << "                           (equivalent to: taskset -c <spec>)\n"
      << "  --numa       <nodes>     Shortcut: sets both cpubind AND "
         "interleave\n"
      << "                           e.g. 0,1,2,3\n"
      << "                           (equivalent to: numactl "
         "--cpunodebind=<nodes>\n"
      << "                                                    "
         "--interleave=<nodes>)\n"
      << "  --numa-cpubind   <nodes> Bind to CPUs of these NUMA nodes\n"
      << "  --numa-interleave <nodes> Interleave memory allocation across "
         "nodes\n"
      << "\n"
      << "Other:\n"
      << "  --help / -h\n"
      << "\n"
      << "Examples:\n"
      << "  # Equivalent to: numactl --cpunodebind=0,1,2,3 "
         "--interleave=0,1,2,3 \\\n"
      << "  #                taskset -c 0-47\n"
      << "  buddy-cli --numa 0,1,2,3 --cpus 0-47 \\\n"
      << "            --model deepseek_r1.rax --prompt \"Hello\"\n"
      << "\n"
      << "  buddy-cli --model-so model.so --weights arg0.data --prompt "
         "\"Hi\"\n";
}

int main(int argc, char **argv) {
  std::string raxPath;
  std::string modelSoPath;
  std::string weightsPath;
  std::string vocabPath;
  std::string prompt;
  int maxTokens = 1024;

  // NUMA / affinity args (applied before model load)
  std::string cpuSpec;
  std::string numaNodes;      // --numa  (shortcut for both)
  std::string numaCpuBind;    // --numa-cpubind
  std::string numaInterleave; // --numa-interleave

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc)
      raxPath = argv[++i];
    else if (a == "--model-so" && i + 1 < argc)
      modelSoPath = argv[++i];
    else if (a == "--weights" && i + 1 < argc)
      weightsPath = argv[++i];
    else if (a == "--vocab" && i + 1 < argc)
      vocabPath = argv[++i];
    else if (a == "--prompt" && i + 1 < argc)
      prompt = argv[++i];
    else if (a == "--max-tokens" && i + 1 < argc)
      maxTokens = std::stoi(argv[++i]);
    else if (a == "--cpus" && i + 1 < argc)
      cpuSpec = argv[++i];
    else if (a == "--numa" && i + 1 < argc)
      numaNodes = argv[++i];
    else if (a == "--numa-cpubind" && i + 1 < argc)
      numaCpuBind = argv[++i];
    else if (a == "--numa-interleave" && i + 1 < argc)
      numaInterleave = argv[++i];
    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  // ── Apply NUMA / affinity settings BEFORE any memory allocation ──────────
  // Order: interleave first (affects future allocations), then cpubind /
  // taskset.
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

  // ── Validate ─────────────────────────────────────────────────────────────
  if (raxPath.empty() && modelSoPath.empty()) {
    std::cerr << "\033[31;1m[Error]\033[0m "
                 "Provide --model <path.rax> or --model-so <path.so>.\n\n";
    usage(argv[0]);
    return 2;
  }

  if (prompt.empty()) {
    std::cout << "Prompt: ";
    std::getline(std::cin, prompt);
    std::cout << "\n";
  }

  // ── Determine model type ─────────────────────────────────────────────────
  std::string modelName;
  if (!raxPath.empty()) {
    try {
      auto manifest = buddy::runtime::ModelManifest::loadFromRax(raxPath);
      modelName = manifest.modelName;
    } catch (const std::exception &e) {
      std::cerr << "\033[31;1m[Error]\033[0m reading manifest: " << e.what()
                << "\n";
      return 1;
    }
    if (modelName.empty()) {
      std::cerr
          << "\033[33;1m[Warn]\033[0m "
             "model_name not set in .rax attrs; defaulting to 'deepseek_r1'.\n";
      modelName = "deepseek_r1";
    }
  } else {
    modelName = "deepseek_r1";
  }

  // ── Run ──────────────────────────────────────────────────────────────────
  buddy::runtime::RunConfig cfg;
  cfg.raxPath = raxPath;
  cfg.modelSoPath = modelSoPath;
  cfg.weightsPath = weightsPath;
  cfg.vocabPath = vocabPath;
  cfg.prompt = prompt;
  cfg.maxNewTokens = maxTokens;

  try {
    auto runner = makeRunner(modelName);
    runner->run(cfg);
  } catch (const std::exception &e) {
    std::cerr << "\033[31;1m[Error]\033[0m " << e.what() << "\n";
    return 1;
  }

  return 0;
}
