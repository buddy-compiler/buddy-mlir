// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BUDDY_EXAMPLES_TRITONSOFTMAX_HOSTPREPARATION_H
#define BUDDY_EXAMPLES_TRITONSOFTMAX_HOSTPREPARATION_H

#include "RiscvCapabilities.h"

#include <charconv>
#include <climits>
#include <fstream>
#include <iterator>
#include <pthread.h>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <vector>

namespace buddy::examples::softmax {

// Linux sysfs CPU lists, including non-contiguous CPU IDs. Reject malformed
// input rather than silently verifying only a prefix of the execution domain.
inline bool parseCpuList(const std::string &text, std::set<int> &cpus) {
  cpus.clear();
  const auto end = text.find_last_not_of(" \t\r\n");
  if (end == std::string::npos)
    return false;
  const char *p = text.data(), *limit = p + end + 1;
  while (p < limit) {
    int first = -1, last = -1;
    auto result = std::from_chars(p, limit, first);
    if (result.ec != std::errc{} || first < 0 || first == INT_MAX)
      return false;
    p = result.ptr;
    last = first;
    if (p < limit && *p == '-') {
      result = std::from_chars(p + 1, limit, last);
      if (result.ec != std::errc{} || last < first || last == INT_MAX)
        return false;
      p = result.ptr;
    }
    for (int cpu = first; cpu <= last; ++cpu)
      if (!cpus.insert(cpu).second)
        return false;
    if (p == limit)
      return true;
    if (*p++ != ',' || p == limit)
      return false;
  }
  return false;
}

inline bool onlineCpus(std::set<int> &cpus) {
  std::ifstream file("/sys/devices/system/cpu/online");
  std::string text;
  return std::getline(file, text) && parseCpuList(text, cpus);
}

inline bool queryVectorPermission() {
#if defined(PR_RISCV_V_GET_CONTROL) &&                                         \
    defined(PR_RISCV_V_VSTATE_CTRL_CUR_MASK) &&                                \
    defined(PR_RISCV_V_VSTATE_CTRL_ON)
  const int state = prctl(PR_RISCV_V_GET_CONTROL, 0UL, 0UL, 0UL, 0UL);
  return state >= 0 &&
         (state & PR_RISCV_V_VSTATE_CTRL_CUR_MASK) == PR_RISCV_V_VSTATE_CTRL_ON;
#else
  return false;
#endif
}

inline bool threadVectorPermission() {
  // Linux does not allow this thread to revoke an established current ON.
  // No destructor, TID key, cross-thread permission or permanent negative
  // cache.
  thread_local bool allowed = false;
  if (!allowed)
    allowed = queryVectorPermission();
  return allowed;
}

inline unsigned long vectorBits() {
  unsigned long bytes;
  // Caller must establish hardware V and current-thread permission first.
  asm volatile("csrr %0, 0xc22" : "=r"(bytes));
  return bytes * 8;
}

struct DomainProbe {
  const std::set<int> &cpus;
  cpu_set_t *mask;
  size_t bytes;
  unsigned long requiredBits;
  bool compatible = false;
};

inline void *probeCpuDomain(void *argument) noexcept {
  auto &probe = *static_cast<DomainProbe *>(argument);
  if (!queryVectorPermission())
    return nullptr;
  for (int cpu : probe.cpus) {
    CPU_ZERO_S(probe.bytes, probe.mask);
    CPU_SET_S(cpu, probe.bytes, probe.mask);
    if (sched_setaffinity(0, probe.bytes, probe.mask) != 0 ||
        vectorBits() != probe.requiredBits)
      return nullptr;
  }
  probe.compatible = true;
  return nullptr;
}

inline bool prepareCpuDomain(const std::string &requiredIsa,
                             unsigned long requiredBits) {
  std::set<int> cpus, after;
  if (!onlineCpus(cpus))
    return false;
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string text((std::istreambuf_iterator<char>(cpuinfo)), {});
  std::istringstream features(text);
  if (!allHartsSupport(features, requiredIsa))
    return false;
  // Match CPU identities as well as extension records/counts, so a restricted
  // /proc mount cannot stand in for unobserved online CPUs.
  std::set<int> reported;
  std::istringstream records(text);
  std::string line;
  while (std::getline(records, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    const auto key = line.substr(0, colon);
    const auto first = key.find_first_not_of(" \t");
    if (first == std::string::npos ||
        key.substr(first, key.find_last_not_of(" \t") - first + 1) !=
            "processor")
      continue;
    const auto value = line.substr(colon + 1);
    const auto start = value.find_first_not_of(" \t");
    std::set<int> id;
    if (start == std::string::npos || !parseCpuList(value.substr(start), id) ||
        id.size() != 1 || !reported.insert(*id.begin()).second)
      return false;
  }
  if (reported != cpus)
    return false;

  const size_t count = static_cast<size_t>(*cpus.rbegin()) + 1;
  cpu_set_t *mask = CPU_ALLOC(count);
  if (!mask)
    return false;
  DomainProbe probe{cpus, mask, CPU_ALLOC_SIZE(count), requiredBits};
  pthread_t worker;
  // Only this short-lived preparation thread changes affinity. It cannot
  // change the caller's mask; failure to reach any online CPU rejects the
  // candidate. No application worker pool or steady-state handoff is added.
  const int created = pthread_create(&worker, nullptr, probeCpuDomain, &probe);
  if (created == 0)
    pthread_join(worker, nullptr);
  CPU_FREE(mask);
  // Deployment contract: CPU capabilities/topology remain stable until the
  // library closes. This comparison detects changes during preparation, not
  // arbitrary later hotplug. Such changes require restart/re-preparation.
  return created == 0 && probe.compatible && onlineCpus(after) && cpus == after;
}
} // namespace buddy::examples::softmax
#endif
