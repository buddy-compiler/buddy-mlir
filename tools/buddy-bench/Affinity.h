//===- Affinity.h - CPU affinity / NUMA helpers for buddy-bench -----------===//
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
// CPU affinity / NUMA helpers, mirroring buddy-cli's --cpus / --numa-* flags.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_AFFINITY_H
#define BUDDY_BENCH_AFFINITY_H

#include <string>

namespace buddy {
namespace bench {

/// Apply a CPU-affinity mask, e.g. "0-47" (taskset -c); no-op off Linux.
void applyCpuAffinity(const std::string &spec);

/// Set NUMA memory interleave policy (like --interleave=nodes).
void applyNumaInterleave(const std::string &spec);

/// Bind execution to CPUs of the given NUMA nodes (like --cpunodebind=nodes).
void applyNumaCpuBind(const std::string &spec);

/// Effective CPU affinity as a compact "0-31" string ("" if unknown).
std::string effectiveCpuAffinity();

/// Number of logical CPUs in the effective affinity mask (0 if unknown).
int effectiveCpuCount();

/// Human-readable NUMA configuration, e.g. "1 node" / "4 nodes" / "n/a".
std::string numaConfig();

/// ISA / feature flags from /proc/cpuinfo ("isa" or "flags"), "" if unknown.
std::string cpuIsa();

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_AFFINITY_H
