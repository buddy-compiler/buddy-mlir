//===- BenchBackend.h - Unified facade for buddy-bench --------------------===//
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
// Facade over the buddy runtime: load .rax, run benches, read env info.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_BENCHBACKEND_H
#define BUDDY_BENCH_BENCHBACKEND_H

#include "buddy/runtime/core/InferenceRunner.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace bench {

/// Host environment info recorded alongside results for reproducibility.
struct EnvInfo {
  std::string cpuModel;    // e.g. "12th Gen Intel(R) Core(TM) i7-12700"
  int numCores = 0;        // available logical cores
  int usedCores = 0;       // logical cores in the effective affinity mask
  std::string cpuAffinity; // effective affinity, e.g. "0-31" ("" = unknown)
  std::string numaConfig;  // e.g. "1 node" / "4 nodes" / "n/a"
  std::string isa;       // CPU ISA / feature flags (RISC-V "isa" / x86 "flags")
  std::string version;   // BUDDY_VERSION
  std::string modelPath; // source .rax
  long long modelSizeBytes = 0; // .rax size in bytes
  std::string modelMtime;       // .rax mtime, UTC ISO 8601
  std::string precision;        // e.g. f16 / f32 / w8a16
  std::string backend;          // e.g. cpu / rvv / ttnn
  std::string device;           // e.g. cpu
  std::string deviceArch;       // cpu model, or accelerator architecture
  int deviceCount = 1;
  int batchSize = 1;
};

/// Unified facade over the buddy runtime. buddy-bench.cpp talks only to this.
class BenchBackend {
public:
  BenchBackend() = default;
  ~BenchBackend();

  BenchBackend(const BenchBackend &) = delete;
  BenchBackend &operator=(const BenchBackend &) = delete;

  /// Read the .rax manifest and dlopen the runner plugin; false on failure.
  bool load(const std::string &raxPath);

  /// Model name from the manifest (valid after a successful load()).
  const std::string &modelName() const;

  /// True if the runner advertises kRunnerCapBench (false if load() failed).
  bool supportsBench() const {
    return (capabilities_ & buddy::runtime::kRunnerCapBench) != 0;
  }

  /// Run warmupReps untimed + reps measured passes; append one result per rep.
  bool runBench(const buddy::runtime::RunConfig &cfg, int warmupReps, int reps,
                std::vector<buddy::runtime::BenchmarkResult> &out);

  /// Read host environment info (CPU model, core count).
  EnvInfo getEnvInfo() const;

  /// Set the batch size reported in getEnvInfo().
  void setBatchSize(int batchSize) { batchSize_ = batchSize; }

private:
  std::string modelName_;
  std::string raxPath_;
  long long raxSizeBytes_ = 0;
  std::string raxMtime_;
  std::string precision_;
  std::string backend_;
  std::string device_ = "cpu";
  int batchSize_ = 1;
  uint64_t capabilities_ = 0;
  void *runnerHandle_ = nullptr;
  buddy::runtime::InferenceRunner *runner_ = nullptr;
  buddy::runtime::DestroyInferenceRunnerFn destroy_ = nullptr;
  // Keeps the model's RAX dependencies dlopen'd while the runner is alive.
  std::unique_ptr<buddy::runtime::RaxDependenciesHandle> raxDeps_;
};

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_BENCHBACKEND_H
