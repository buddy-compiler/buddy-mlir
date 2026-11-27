// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "buddy/runtime/core/ExternalKernelABI.h"

static int32_t compatible() { return TEST_MODE != 3; }
static int32_t invoke(const BuddyKernelCallV1 *call) {
  if (TEST_MODE == 12)
    return BUDDY_KERNEL_UNSUPPORTED;
  if (TEST_MODE == 13)
    return BUDDY_KERNEL_INVALID;
  if (TEST_MODE == 4) {
    static_cast<float *>(call->output.data)[0] = -100;
    return BUDDY_KERNEL_FAILED;
  }
  static_cast<float *>(call->output.data)[0] = 42;
  return BUDDY_KERNEL_OK;
}
#if defined(__x86_64__)
#define TEST_TARGET "x86_64-unknown-linux-gnu"
#elif defined(__riscv) && __riscv_xlen == 64
#define TEST_TARGET "riscv64-unknown-linux-gnu"
#else
#define TEST_TARGET "unsupported"
#endif

#if TEST_MODE == 5
#define buddy_external_kernel_query_v1 buddy_intentionally_wrong_query_name
#endif
#if TEST_MODE == 6
// Canonical requirement tokens for the metadata cross-check fixture.
#define TEST_REQUIRED_FEATURES "avx2,bmi,fma"
#else
#define TEST_REQUIRED_FEATURES "baseline"
#endif
extern "C" const BuddyKernelDescriptorV1 *buddy_external_kernel_query_v1() {
  if (TEST_MODE == 7)
    return nullptr;
  static const BuddyKernelDescriptorV1 descriptor{
      TEST_MODE == 1 ? 999U : BUDDY_KERNEL_ABI_V1,
      TEST_MODE == 8 ? 0U : sizeof(BuddyKernelDescriptorV1),
      BUDDY_KERNEL_SOFTMAX,
      BUDDY_KERNEL_F32,
      2,
      BUDDY_KERNEL_ROW_MAJOR,
      TEST_MODE == 11 ? 1 : 2,
      4,
      1,
      TEST_MODE == 2 ? "incompatible-target" : TEST_TARGET,
      TEST_REQUIRED_FEATURES,
      TEST_MODE == 9 ? nullptr : compatible,
      TEST_MODE == 10 ? nullptr : invoke};
  return &descriptor;
}
