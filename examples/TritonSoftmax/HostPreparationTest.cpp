// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Cross-compile on the build machine; execute only with board-test approval.
#include "HostPreparation.h"
#include "buddy/runtime/core/ExternalKernel.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace buddy::runtime;
static_assert(PR_RISCV_V_GET_CONTROL == 70 &&
                  PR_RISCV_V_VSTATE_CTRL_CUR_MASK == 3 &&
                  PR_RISCV_V_VSTATE_CTRL_OFF == 1,
              "Update VectorPolicyExec.S to match the target UAPI");
namespace {
std::atomic<int> fallbackCalls{0};
// Test-only callback for dispatch/cache mechanics. Real compiled Buddy
// correctness is covered independently by Run.cpp and the RAX replay.
int32_t fallback(const BuddyKernelCallV1 *call) {
  ++fallbackCalls;
  const auto *input = static_cast<const float *>(call->input.data);
  auto *output = static_cast<float *>(call->output.data);
  for (int row = 0; row < 2; ++row) {
    const float maximum =
        *std::max_element(input + row * 4, input + row * 4 + 4);
    float sum = 0;
    for (int col = 0; col < 4; ++col)
      sum += std::exp(input[row * 4 + col] - maximum);
    for (int col = 0; col < 4; ++col)
      output[row * 4 + col] = std::exp(input[row * 4 + col] - maximum) / sum;
  }
  return BUDDY_KERNEL_OK;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  std::set<int> parsed;
  for (const char *bad : {"", "1,", "-1", "2-1", "0,,1", "0,0", "1x"})
    if (buddy::examples::softmax::parseCpuList(bad, parsed))
      return 2;
  if (!buddy::examples::softmax::parseCpuList("0-2,7,1025\n", parsed) ||
      parsed != std::set<int>({0, 1, 2, 7, 1025}))
    return 3;
  auto mode =
      reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "host_test_mode"));
  auto queries = reinterpret_cast<unsigned (*)()>(
      dlsym(RTLD_DEFAULT, "host_test_queries"));
  auto reads = reinterpret_cast<unsigned (*)()>(
      dlsym(RTLD_DEFAULT, "host_test_affinity_reads"));
  auto writes = reinterpret_cast<unsigned (*)()>(
      dlsym(RTLD_DEFAULT, "host_test_affinity_writes"));
  auto deny =
      reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "host_test_deny_pin"));
  if (!mode || !queries || !reads || !writes || !deny)
    return 4;
  if (std::strcmp(argv[2], "concurrent") == 0) {
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    auto query = handle ? reinterpret_cast<BuddyKernelQueryV1>(
                              dlsym(handle, "buddy_external_kernel_query_v1"))
                        : nullptr;
    const auto *descriptor = query ? query() : nullptr;
    if (!descriptor)
      return 19;
    std::atomic<int> ready{0}, failed{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i)
      workers.emplace_back([&] {
        ++ready;
        while (ready != 8)
          std::this_thread::yield();
        if (!descriptor->isCompatibleHost() || queries() != 1)
          ++failed;
        if (!descriptor->isCompatibleHost() || queries() != 1)
          ++failed;
      });
    for (auto &worker : workers)
      worker.join();
    dlclose(handle);
    if (failed)
      return 20;
    std::cout
        << "PASS concurrent first-use workers=8 independent_permission=1\n";
    return 0;
  }
  const bool denied = std::strcmp(argv[2], "deny-domain") == 0;
  const bool rejected = denied;
  if (denied)
    deny();
  // Loading must preserve the application's affinity mask.
  const size_t bytes = CPU_ALLOC_SIZE(65536);
  cpu_set_t *before = CPU_ALLOC(65536), *after = CPU_ALLOC(65536);
  if (!before || !after)
    return 5;
  CPU_ZERO_S(bytes, before);
  CPU_ZERO_S(bytes, after);
  if (sched_getaffinity(0, bytes, before) != 0)
    return 6;
  ExternalKernel kernel(argv[1]);
  if (sched_getaffinity(0, bytes, after) != 0 ||
      std::memcmp(before, after, bytes) != 0)
    return 7;
  CPU_FREE(before);
  CPU_FREE(after);
  float input[8] = {1, 2, 3, 4, 0, 0, 0, 0}, output[8];
  int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
  BuddyKernelCallV1 call{1,
                         sizeof(BuddyKernelCallV1),
                         1,
                         0,
                         {input, 32, 2, 1, 1, 0, shape, strides},
                         {output, 32, 2, 1, 1, 0, shape, strides},
                         &axis,
                         1,
                         nullptr,
                         0};
  const float expected[] = {.032058604f, .087144323f, .236882836f, .643914282f,
                            .25f,        .25f,        .25f,        .25f};
  auto correct = [&] {
    for (int j = 0; j < 8; ++j)
      if (!std::isfinite(output[j]) ||
          std::abs(output[j] - expected[j]) > 1e-6f)
        return false;
    return true;
  };
  const unsigned q = queries(), r = reads(), w = writes();
  for (int i = 0; i < 100; ++i) {
    const auto result = kernel.dispatch(call, fallback);
    if (result.status ||
        result.path != (rejected ? KernelPath::Fallback : KernelPath::External))
      return 8;
    if (!correct())
      return 9;
  }
  if (queries() != q || reads() != r || writes() != w)
    return 10;
  if (rejected) {
    if (fallbackCalls != 100)
      return 11;
    std::cout << "PASS rejected domain fallback=100 output=valid\n";
    return 0;
  }
  int threadStatus = 0;
  std::thread worker([&] {
    // Thread A's success must not authorize thread B. Failures must retry.
    for (int state : {1, 2}) {
      mode(state);
      const unsigned previous = queries();
      auto result = kernel.dispatch(call, fallback);
      if (result.path != KernelPath::Fallback || queries() != previous + 1)
        threadStatus = 12;
    }
    mode(0);
    auto result = kernel.dispatch(call, fallback);
    const unsigned ready = queries();
    if (result.path != KernelPath::External || ready != 3)
      threadStatus = 13;
    result = kernel.dispatch(call, fallback);
    if (result.path != KernelPath::External || queries() != ready)
      threadStatus = 14;
  });
  worker.join();
  if (threadStatus)
    return threadStatus;
  // Reuse the prepared candidate after changing this thread's CPU mask.
  std::set<int> online;
  if (!buddy::examples::softmax::onlineCpus(online) || online.size() < 2)
    return 15;
  const size_t count = static_cast<size_t>(*online.rbegin()) + 1;
  cpu_set_t *mask = CPU_ALLOC(count);
  if (!mask)
    return 16;
  const size_t maskBytes = CPU_ALLOC_SIZE(count);
  for (int cpu : online) {
    CPU_ZERO_S(maskBytes, mask);
    CPU_SET_S(cpu, maskBytes, mask);
    if (sched_setaffinity(0, maskBytes, mask) != 0 ||
        kernel.dispatch(call, fallback).path != KernelPath::External ||
        !correct())
      return 17;
  }
  CPU_ZERO_S(maskBytes, mask);
  for (int cpu : online)
    CPU_SET_S(cpu, maskBytes, mask);
  if (sched_setaffinity(0, maskBytes, mask) != 0 ||
      kernel.dispatch(call, fallback).path != KernelPath::External ||
      !correct())
    return 18;
  CPU_FREE(mask);
  errno = 0;
  if (syscall(SYS_prctl, PR_RISCV_V_SET_CONTROL, PR_RISCV_V_VSTATE_CTRL_OFF,
              0UL, 0UL, 0UL) != -1 ||
      errno != EPERM)
    return 21;
  // At this point no other test thread is alive. Check inherited permission,
  // then exec with NEXT=OFF to check that old image/TLS success cannot leak.
  const std::string executable(argv[0]);
  const auto slash = executable.find_last_of('/');
  const std::string policyProbe =
      (slash == std::string::npos ? "./" : executable.substr(0, slash + 1)) +
      "vector-policy-exec";
  const pid_t child = fork();
  if (child < 0)
    return 22;
  if (child == 0) {
    const long control =
        syscall(SYS_prctl, PR_RISCV_V_GET_CONTROL, 0UL, 0UL, 0UL, 0UL);
    if (control < 0 || (control & PR_RISCV_V_VSTATE_CTRL_CUR_MASK) !=
                           PR_RISCV_V_VSTATE_CTRL_ON)
      _exit(23);
    if (syscall(SYS_prctl, PR_RISCV_V_SET_CONTROL,
                PR_RISCV_V_VSTATE_CTRL_OFF << 2, 0UL, 0UL, 0UL) != 0)
      _exit(24);
    execl(policyProbe.c_str(), policyProbe.c_str(),
          static_cast<char *>(nullptr));
    _exit(25);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    std::cerr << "fork/exec child wait_status=" << status << '\n';
    return 26;
  }
  std::cout << "PASS preparation queries_cached=1 worker_isolated=1 retry=1 "
               "caller_affinity_preserved=1 migration_cpus="
            << online.size()
            << " steady_syscalls=0 fork_exec=1 on_to_off_eperm=1\n";
}
