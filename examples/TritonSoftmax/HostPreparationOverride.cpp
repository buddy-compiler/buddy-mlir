// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Test-only LD_PRELOAD instrumentation, never part of a RAX package.
#include <atomic>
#include <cerrno>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
thread_local int permissionMode = 0;
thread_local unsigned queries = 0;
std::atomic<unsigned> affinityReads{0}, affinityWrites{0};
std::atomic<bool> denyPin{false};
} // namespace
extern "C" void host_test_mode(int mode) { permissionMode = mode; }
extern "C" void host_test_deny_pin() { denyPin = true; }
extern "C" unsigned host_test_queries() { return queries; }
extern "C" unsigned host_test_affinity_reads() { return affinityReads; }
extern "C" unsigned host_test_affinity_writes() { return affinityWrites; }

extern "C" int prctl(int option, ...) noexcept {
  if (option != PR_RISCV_V_GET_CONTROL) {
    errno = EINVAL;
    return -1;
  }
  ++queries;
  if (permissionMode == 1)
    return PR_RISCV_V_VSTATE_CTRL_OFF;
  if (permissionMode == 2) {
    errno = EINVAL;
    return -1;
  }
  return syscall(SYS_prctl, option, 0UL, 0UL, 0UL, 0UL);
}
extern "C" int sched_getaffinity(pid_t pid, size_t bytes,
                                 cpu_set_t *mask) noexcept {
  ++affinityReads;
  const long result = syscall(SYS_sched_getaffinity, pid, bytes, mask);
  return result < 0 ? -1 : 0;
}
extern "C" int sched_setaffinity(pid_t pid, size_t bytes,
                                 const cpu_set_t *mask) noexcept {
  ++affinityWrites;
  if (denyPin) {
    errno = EPERM;
    return -1;
  }
  return syscall(SYS_sched_setaffinity, pid, bytes, mask);
}
