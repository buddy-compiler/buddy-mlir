//===- Adapter.cpp - Triton-RISCV ABI to Buddy tensor ABI -----------------===//
//--------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "HostPreparation.h"
#include "SoftmaxTarget.h"
#include "buddy/runtime/core/ExternalKernelValidation.h"

#include <cstddef>
#include <cstdint>

#if !(defined(__riscv) && __riscv_xlen == 64 &&                                \
      defined(__riscv_float_abi_double))
#error "Adapter.cpp requires riscv64 Linux lp64d"
#endif

#include <cstdlib>
#include <fstream>
#include <sched.h>
#include <string>
#include <sys/auxv.h>
#include <sys/prctl.h>

#define TRITON_SOFTMAX_TARGET "riscv64-unknown-linux-gnu"

// This declaration is checked against softmax.llir by export.py.
extern "C" void softmax_kernel(int64_t, void *, int64_t, void *, int32_t,
                               int32_t, int32_t, int32_t, int32_t, int32_t,
                               int32_t, int32_t, int32_t);

namespace {
struct PointerDescriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
};
static_assert(sizeof(PointerDescriptor) == 24);
static_assert(offsetof(PointerDescriptor, aligned) == 8);

bool probeHostStrict() {
  // Baseline rv64gc code until Linux confirms vector availability. Never change
  // the application's vector policy. Historical strict comparison path only.
  constexpr unsigned long vectorHwcap = 1UL << ('V' - 'A');
  if (!(getauxval(AT_HWCAP) & vectorHwcap))
    return false;
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0 ||
      CPU_COUNT(&affinity) != 1)
    return false;
#if defined(PR_RISCV_V_GET_CONTROL) &&                                         \
    defined(PR_RISCV_V_VSTATE_CTRL_CUR_MASK) &&                                \
    defined(PR_RISCV_V_VSTATE_CTRL_ON)
  int state = prctl(PR_RISCV_V_GET_CONTROL);
  if (state < 0 ||
      (state & PR_RISCV_V_VSTATE_CTRL_CUR_MASK) != PR_RISCV_V_VSTATE_CTRL_ON)
    return false;
#else
  return false;
#endif
  static const bool isaCompatible = [] {
    std::ifstream cpuinfo("/proc/cpuinfo");
    return buddy::examples::softmax::allHartsSupport(cpuinfo,
                                                     TRITON_SOFTMAX_CPU_ISA);
  }();
  if (!isaCompatible)
    return false;
  unsigned long vectorBytes = 0;
  // vlenb CSR, reached only after full V and OS permission have been checked.
  asm volatile("csrr %0, 0xc22" : "=r"(vectorBytes));
  return vectorBytes * 8 == TRITON_SOFTMAX_VLEN_BITS;
}

int32_t compatibleHost() noexcept {
  try {
    static const bool strict = [] {
      const char *mode = std::getenv("BUDDY_EXTERNAL_KERNEL_HOSTCHECK");
      return mode && std::string(mode) == "strict";
    }();
    if (strict)
      return probeHostStrict();
    constexpr unsigned long vectorHwcap = 1UL << ('V' - 'A');
    static const bool hasVector = (getauxval(AT_HWCAP) & vectorHwcap) != 0;
    // Permission is thread-local. A loader thread cannot authorize workers.
    if (!hasVector || !buddy::examples::softmax::threadVectorPermission())
      return false;
    static const bool domainCompatible =
        buddy::examples::softmax::prepareCpuDomain(TRITON_SOFTMAX_CPU_ISA,
                                                   TRITON_SOFTMAX_VLEN_BITS);
    return domainCompatible;
  } catch (...) {
    // Preparation can allocate/parse; exceptions must not cross the C ABI.
    return false;
  }
}

int32_t invoke(const BuddyKernelCallV1 *call) {
  if (!call || !buddy::runtime::detail::validSoftmaxCall(*call))
    return BUDDY_KERNEL_INVALID;
  if (call->input.shape[0] != 2 || call->input.shape[1] != 4 ||
      !compatibleHost())
    return BUDDY_KERNEL_UNSUPPORTED;
  PointerDescriptor output{call->output.data, call->output.data, 0};
  PointerDescriptor input{call->input.data, call->input.data, 0};
  for (int32_t row = 0; row < 2; ++row)
    softmax_kernel(0, &output, 0, &input, 4, 4, 4, 2, 1, 1, row, 0, 0);
  return BUDDY_KERNEL_OK;
}

const BuddyKernelDescriptorV1 descriptor{BUDDY_KERNEL_ABI_V1,
                                         sizeof(BuddyKernelDescriptorV1),
                                         BUDDY_KERNEL_SOFTMAX,
                                         BUDDY_KERNEL_F32,
                                         2,
                                         BUDDY_KERNEL_ROW_MAJOR,
                                         2,
                                         4,
                                         1,
                                         TRITON_SOFTMAX_TARGET,
                                         TRITON_SOFTMAX_REQUIRED_FEATURES,
                                         compatibleHost,
                                         invoke};
} // namespace

extern "C" const BuddyKernelDescriptorV1 *buddy_external_kernel_query_v1() {
  return &descriptor;
}
