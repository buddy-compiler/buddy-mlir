//===- Affinity.cpp - CPU affinity / NUMA helpers for buddy-bench ---------===//
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

#include "Affinity.h"

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <CLI11.hpp>

#ifndef _WIN32
#include <sched.h>
#endif

#ifdef BUDDY_BENCH_HAVE_NUMA
#include <numa.h>
#endif

namespace buddy {
namespace bench {

namespace {

std::string trim(const std::string &s) {
  auto begin = s.find_first_not_of(" \t");
  if (begin == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t");
  return s.substr(begin, end - begin + 1);
}

/// Parse one integer, capped at `max`; rejects trailing junk ("4:8").
int parseIndex(const std::string &text, int max, const std::string &option) {
  int value = 0;
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (ec != std::errc() || ptr != end || value < 0 || value > max)
    throw CLI::ValidationError("invalid " + option + " value '" + text +
                               "' (allowed: 0.." + std::to_string(max) + ")");
  return value;
}

/// Parse "0-3,8" into inclusive {lo, hi} ranges; empty entries are rejected.
std::vector<std::pair<int, int>> parseRanges(const std::string &spec, int max,
                                             const std::string &option) {
  std::vector<std::pair<int, int>> ranges;
  for (size_t start = 0;;) {
    const size_t comma = spec.find(',', start);
    const std::string tok = trim(spec.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start));
    if (tok.empty())
      throw CLI::ValidationError("empty value in " + option + " '" + spec +
                                 "'");
    const size_t dash = tok.find('-');
    if (dash != std::string::npos && (dash == 0 || dash + 1 == tok.size()))
      throw CLI::ValidationError("invalid " + option + " value '" + tok + "'");
    const int lo = parseIndex(tok.substr(0, dash), max, option);
    const int hi = dash == std::string::npos
                       ? lo
                       : parseIndex(tok.substr(dash + 1), max, option);
    if (lo > hi)
      throw CLI::ValidationError(option + " range start exceeds end: '" + tok +
                                 "'");
    ranges.emplace_back(lo, hi);
    if (comma == std::string::npos)
      return ranges;
    start = comma + 1;
  }
}

#ifndef _WIN32
// Parse "0-47" or "0-15,32-47,64" into a cpu_set_t.
cpu_set_t parseCpuSet(const std::string &spec) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto [lo, hi] : parseRanges(spec, CPU_SETSIZE - 1, "--cpus"))
    for (int c = lo; c <= hi; ++c)
      CPU_SET(c, &mask);
  return mask;
}
#endif

#ifdef BUDDY_BENCH_HAVE_NUMA
// Parse "0-1,3" into a NUMA nodemask.
struct bitmask *parseNodeMask(const std::string &spec) {
  struct bitmask *mask = numa_allocate_nodemask();
  const int maxNode = mask->size > 0 ? static_cast<int>(mask->size) - 1 : -1;
  for (auto [lo, hi] : parseRanges(spec, maxNode, "--numa"))
    for (int n = lo; n <= hi; ++n)
      numa_bitmask_setbit(mask, n);
  return mask;
}
#endif

// First "isa" (RISC-V) or "flags" (x86) line of /proc/cpuinfo.
std::string readCpuInfoField() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo)
    return "";
  std::string line;
  std::string flags;
  while (std::getline(cpuinfo, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    const std::string key = trim(line.substr(0, colon));
    // Prefer "isa"; keep "flags" as the fallback.
    if (key == "isa")
      return trim(line.substr(colon + 1));
    if (key == "flags")
      flags = trim(line.substr(colon + 1));
  }
  return flags;
}

} // namespace

void applyCpuAffinity(const std::string &spec) {
#ifndef _WIN32
  auto mask = parseCpuSet(spec);
  // Fail rather than record a placement that never took effect.
  if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
    throw std::runtime_error("sched_setaffinity(" + spec +
                             ") failed: " + std::strerror(errno));
#else
  (void)spec;
  std::fprintf(stderr, "buddy-bench: CPU affinity unsupported on this "
                       "platform; --cpus ignored\n");
#endif
}

void applyNumaInterleave(const std::string &spec) {
#ifdef BUDDY_BENCH_HAVE_NUMA
  if (numa_available() < 0)
    throw std::runtime_error("NUMA is not available on this system");
  auto *mask = parseNodeMask(spec);
  const int rc = numa_set_interleave_mask(mask);
  numa_free_nodemask(mask);
  if (rc < 0)
    throw std::runtime_error("--numa-interleave " + spec +
                             " failed (see /sys/devices/system/node)");
#else
  throw std::runtime_error("--numa-interleave " + spec +
                           " needs a libnuma build; reconfigure with "
                           "BUDDY_BENCH_ENABLE_NUMA=ON");
#endif
}

void applyNumaCpuBind(const std::string &spec) {
#ifdef BUDDY_BENCH_HAVE_NUMA
  if (numa_available() < 0)
    throw std::runtime_error("NUMA is not available on this system");
  auto *mask = parseNodeMask(spec);
  const int rc = numa_run_on_node_mask(mask);
  numa_free_nodemask(mask);
  if (rc != 0)
    throw std::runtime_error("--numa-cpubind " + spec +
                             " failed (see /sys/devices/system/node)");
#else
  throw std::runtime_error("--numa-cpubind " + spec +
                           " needs a libnuma build; reconfigure with "
                           "BUDDY_BENCH_ENABLE_NUMA=ON");
#endif
}

std::string effectiveCpuAffinity() {
#ifndef _WIN32
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return "";
  std::ostringstream os;
  bool first = true;
  for (int c = 0; c < CPU_SETSIZE;) {
    if (!CPU_ISSET(c, &mask)) {
      ++c;
      continue;
    }
    int lo = c;
    while (c < CPU_SETSIZE && CPU_ISSET(c, &mask))
      ++c;
    int hi = c - 1;
    if (!first)
      os << ',';
    first = false;
    if (lo == hi)
      os << lo;
    else
      os << lo << '-' << hi;
  }
  return os.str();
#else
  return "";
#endif
}

int effectiveCpuCount() {
#ifndef _WIN32
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
    return 0;
  return CPU_COUNT(&mask);
#else
  return 0;
#endif
}

std::string numaConfig() {
#ifdef BUDDY_BENCH_HAVE_NUMA
  if (numa_available() < 0)
    return "n/a";
  int nodes = numa_num_configured_nodes();
  if (nodes <= 0)
    return "n/a";
  return std::to_string(nodes) + (nodes == 1 ? " node" : " nodes");
#else
  return "n/a";
#endif
}

std::string cpuIsa() { return trim(readCpuInfoField()); }

} // namespace bench
} // namespace buddy
