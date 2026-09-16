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
#ifdef BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL
#include "buddy/runtime/models/DeepSeekR1Runner.h"
#endif
#ifdef BUDDY_CLI_HAVE_LLAMA31_TT_MODEL
#include "buddy/runtime/models/Llama31TTRunner.h"
#endif

#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI11.hpp>

#ifdef BUDDY_CLI_HAVE_NUMA
#include <numa.h>
#endif

//===----------------------------------------------------------------------===//
// Affinity helpers
//===----------------------------------------------------------------------===//

#ifdef __APPLE__
static void applyCpuAffinity(const std::string &spec) {
  std::cerr << "[buddy-cli] Doesn't support applyCpuAffinity; --cpus ignored\n";
}
#else
#include <sched.h>

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
#endif

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
makeBuiltinRunner(const std::string &modelName) {
#ifdef BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL
  if (modelName.rfind("deepseek_r1", 0) == 0)
    return std::make_unique<buddy::runtime::DeepSeekR1Runner>();
#endif
#ifdef BUDDY_CLI_HAVE_LLAMA31_TT_MODEL
  if (modelName.rfind("llama31_tt", 0) == 0 ||
      modelName.rfind("llama3.1_tt", 0) == 0 ||
      modelName.rfind("llama32_tt", 0) == 0 ||
      modelName.rfind("llama3.2_tt", 0) == 0)
    return std::make_unique<buddy::runtime::Llama31TTRunner>();
#endif

#if defined(BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL) ||                               \
    defined(BUDDY_CLI_HAVE_LLAMA31_TT_MODEL)
  const char *unknownHint =
      "  Supported models: "
#ifdef BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL
      "deepseek_r1 "
#endif
#ifdef BUDDY_CLI_HAVE_LLAMA31_TT_MODEL
      "llama31_tt llama3.1_tt llama32_tt llama3.2_tt "
#endif
      "\n"
      "  To add a new model, implement InferenceRunner and register it here.";
#else
  const char *unknownHint =
      "  This buddy-cli was built without DeepSeek R1 (no model runner "
      "linked).\n"
      "  Re-configure with -DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON and rebuild, or "
      "run:\n"
      "    python3 tools/buddy-codegen/build_model.py --spec "
      "models/deepseek_r1/specs/<variant>.json";
#endif
  throw std::runtime_error(std::string("buddy-cli: unknown model '") +
                           modelName + "'.\n" + unknownHint);
}

class RunnerHandle {
public:
  RunnerHandle(const std::string &runnerPath, const std::string &modelName) {
    if (runnerPath.empty())
      throw std::runtime_error(
          "buddy-cli: runner plugin not specified. Use a .rax with "
          "runner_library or pass --runner-so <path>.");

    handle = dlopen(runnerPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(
          "buddy-cli: dlopen runner failed: " + runnerPath + ": " + dlerror());

    dlerror();
    auto *createSym = dlsym(handle, "buddy_create_inference_runner_v1");
    const char *createErr = dlerror();
    if (createErr)
      throw std::runtime_error("buddy-cli: runner plugin missing "
                               "buddy_create_inference_runner_v1: " +
                               std::string(createErr));

    dlerror();
    auto *destroySym = dlsym(handle, "buddy_destroy_inference_runner_v1");
    const char *destroyErr = dlerror();
    if (destroyErr)
      throw std::runtime_error("buddy-cli: runner plugin missing "
                               "buddy_destroy_inference_runner_v1: " +
                               std::string(destroyErr));

    create =
        reinterpret_cast<buddy::runtime::CreateInferenceRunnerFn>(createSym);
    destroy =
        reinterpret_cast<buddy::runtime::DestroyInferenceRunnerFn>(destroySym);
    runner = create();
    if (!runner)
      throw std::runtime_error("buddy-cli: runner plugin returned null for " +
                               modelName);
  }

  ~RunnerHandle() {
    if (runner && destroy)
      destroy(runner);
    if (handle)
      dlclose(handle);
  }

  RunnerHandle(const RunnerHandle &) = delete;
  RunnerHandle &operator=(const RunnerHandle &) = delete;

  buddy::runtime::InferenceRunner &get() { return *runner; }

private:
  void *handle = nullptr;
  buddy::runtime::CreateInferenceRunnerFn create = nullptr;
  buddy::runtime::DestroyInferenceRunnerFn destroy = nullptr;
  buddy::runtime::InferenceRunner *runner = nullptr;
};

static std::string resolvePathRelativeToRax(const std::string &path,
                                            const std::string &raxPath) {
  namespace fs = std::filesystem;
  if (path.empty())
    return "";
  fs::path p(path);
  if (p.is_absolute() || raxPath.empty())
    return p.string();
  return (fs::absolute(fs::path(raxPath)).parent_path() / p).string();
}

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  std::string raxPath;
  std::string modelSoPath;
  std::string weightsPath;
  std::string vocabPath;
  std::string runnerSoPath;
  std::string prompt;
  std::string promptFile;
  int promptLength = 0;
  std::string audioPath;
  std::string imagePath;
  int maxTokens = 4096;
  int batchSize = 0;

  // Sampling args
  float temperature = 0.0f;
  int topK = 0;
  float topP = 1.0f;
  float minP = 0.0f;
  float repeatPenalty = 1.0f;
  int repeatLastN = 64;
  uint64_t seed = 0;

  // Chat template & output
  std::string chatTemplatePath;
  bool suppressStats = false;
  bool printAllBatchOutputs = false;
  bool deferDecodeTokenReadback = false;
  bool streamJsonl = false;
  bool interactive = false;

  // NUMA / affinity args (applied before model load)
  std::string cpuSpec;
  std::string numaNodes;      // --numa  (shortcut for both)
  std::string numaCpuBind;    // --numa-cpubind
  std::string numaInterleave; // --numa-interleave

  CLI::App app{"buddy-cli: run Buddy MLIR .rax model packages"};
  app.set_version_flag("--version", BUDDY_VERSION);
  app.footer(
      "Examples:\n"
      "  # Equivalent to: numactl --cpunodebind=0,1,2,3 --interleave=0,1,2,3 "
      "taskset -c 0-47\n"
      "  buddy-cli --numa 0,1,2,3 --cpus 0-47 --model deepseek_r1.rax "
      "--prompt \"Hello\"\n"
      "  buddy-cli --model-so model.so --weights arg0.data --prompt \"Hi\"");

  app.add_option("--model", raxPath, "Model manifest (recommended)")
      ->group("Model source");
  app.add_option("--model-so", modelSoPath,
                 "Model shared library (legacy mode)")
      ->group("Model source");
  app.add_option("--weights", weightsPath, "Weights file (legacy mode)")
      ->group("Model source");
  app.add_option("--vocab", vocabPath, "Vocabulary file (legacy mode)")
      ->group("Model source");
  app.add_option("--runner-so", runnerSoPath, "Runner plugin (legacy mode)")
      ->group("Model source");

  app.add_option("--prompt", prompt, "Input prompt (interactive if omitted)")
      ->group("Inference");
  app.add_option("--prompt-file", promptFile,
                 "One prompt per line for fixed-batch runs")
      ->group("Inference");
  app.add_option("--prompt-length", promptLength,
                 "Fixed prompt/prefill length in tokens")
      ->group("Inference");
  app.add_option("--audio", audioPath,
                 "Audio file for speech models (e.g. Whisper)")
      ->group("Inference");
  app.add_option("--image", imagePath, "Image file for vision-language models")
      ->group("Inference");
  app.add_option("--max-tokens", maxTokens, "Max generated tokens")
      ->group("Inference")
      ->capture_default_str();
  app.add_option("--batch-size", batchSize,
                 "Batch size override for fixed-batch packages")
      ->group("Inference");

  app.add_option("--temperature", temperature,
                 "Sampling temperature (0.0 = greedy, default)")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--top-k", topK, "Top-K candidates (0 = disabled)")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--top-p", topP, "Nucleus sampling threshold (1.0 = disabled)")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--min-p", minP, "Min-P threshold (0.0 = disabled)")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--repeat-penalty", repeatPenalty,
                 "Repetition penalty (1.0 = disabled)")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--repeat-last-n", repeatLastN, "Repeat penalty window")
      ->group("Sampling")
      ->capture_default_str();
  app.add_option("--seed", seed, "Random seed (0 = random)")
      ->group("Sampling")
      ->capture_default_str();

  app.add_option("--chat-template", chatTemplatePath,
                 "Path to chat template JSON config")
      ->group("Chat");
  app.add_flag("--interactive", interactive,
               "Start REPL-style interactive mode (--prompt becomes system "
               "prompt)")
      ->group("Chat");

  app.add_flag("--no-stats", suppressStats, "Suppress performance statistics")
      ->group("Output");
  app.add_flag("--defer-decode-token-readback", deferDecodeTokenReadback,
               "Defer device token-id readback until after fixed-step decode "
               "when supported")
      ->group("Output");
  app.add_flag("--stream-jsonl", streamJsonl, "Emit token events as JSON Lines")
      ->group("Output");
  app.add_flag("--print-all-batch", printAllBatchOutputs,
               "Print every user in batch runs")
      ->group("Output");

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

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
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
    std::cerr << app.help() << "\n";
    return 2;
  }
  if (streamJsonl && deferDecodeTokenReadback) {
    std::cerr << "\033[31;1m[Error]\033[0m "
                 "--stream-jsonl requires per-step token readback; do not "
                 "combine it with --defer-decode-token-readback.\n";
    return 2;
  }

  std::vector<std::string> prompts;
  if (!promptFile.empty()) {
    std::ifstream input(promptFile);
    if (!input) {
      std::cerr << "\033[31;1m[Error]\033[0m cannot read --prompt-file "
                << promptFile << "\n";
      return 1;
    }
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      prompts.push_back(line);
    }
    if (prompts.empty()) {
      std::cerr << "\033[31;1m[Error]\033[0m --prompt-file is empty: "
                << promptFile << "\n";
      return 1;
    }
  }

  // Speech and vision-language runs are driven by media inputs.
  if (prompt.empty() && prompts.empty() && audioPath.empty() &&
      imagePath.empty() && !interactive) {
    std::cout << "Prompt: ";
    std::getline(std::cin, prompt);
    std::cout << "\n";
  }

  // ── Determine model type ─────────────────────────────────────────────────
  std::string modelName;
  std::string manifestRunnerSoPath;
  if (!raxPath.empty()) {
    try {
      auto manifest = buddy::runtime::ModelManifest::loadFromRax(raxPath);
      modelName = manifest.modelName;
      manifestRunnerSoPath = manifest.runnerLibraryPath;
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
    modelName = "legacy";
  }
  if (runnerSoPath.empty())
    runnerSoPath = manifestRunnerSoPath;
  runnerSoPath = resolvePathRelativeToRax(runnerSoPath, raxPath);

  // ── Run ──────────────────────────────────────────────────────────────────
  buddy::runtime::RunConfig cfg;
  cfg.raxPath = raxPath;
  cfg.modelSoPath = modelSoPath;
  cfg.weightsPath = weightsPath;
  cfg.vocabPath = vocabPath;
  cfg.prompt = prompt;
  cfg.prompts = std::move(prompts);
  cfg.audioPath = audioPath;
  cfg.promptLength = promptLength;
  cfg.imagePath = imagePath;
  cfg.maxNewTokens = maxTokens;
  cfg.batchSize = batchSize;
  cfg.samplerConfig.temperature = temperature;
  cfg.samplerConfig.topK = topK;
  cfg.samplerConfig.topP = topP;
  cfg.samplerConfig.minP = minP;
  cfg.samplerConfig.repeatPenalty = repeatPenalty;
  cfg.samplerConfig.repeatLastN = repeatLastN;
  cfg.samplerConfig.seed = seed;
  cfg.chatTemplatePath = chatTemplatePath;
  cfg.suppressStats = suppressStats;
  cfg.printAllBatchOutputs = printAllBatchOutputs;
  cfg.deferDecodeTokenReadback = deferDecodeTokenReadback;
  cfg.streamJsonl = streamJsonl;
  cfg.interactive = interactive;

  try {
    if (!runnerSoPath.empty()) {
      RunnerHandle runner(runnerSoPath, modelName);
      runner.get().run(cfg);
    } else {
      auto runner = makeBuiltinRunner(modelName);
      runner->run(cfg);
    }
  } catch (const std::exception &e) {
    std::cerr << "\033[31;1m[Error]\033[0m " << e.what() << "\n";
    return 1;
  }

  return 0;
}
