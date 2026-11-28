// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "buddy/runtime/core/ExternalKernel.h"

#include <algorithm>
#include <cmath>
#include <iostream>

static_assert(__cplusplus >= 201703L,
              "Consumer must inherit C++17 from the target");

namespace {
int fallbackCalls = 0;
// A status-only test callback. Actual compiled Buddy numerics are independently
// covered by the Softmax/RAX suite; do not count this as that computation.
int32_t fallback(const BuddyKernelCallV1 *) {
  ++fallbackCalls;
  return BUDDY_KERNEL_OK;
}
} // namespace

extern "C" int buddy_external_consumer_check(const char *library) {
  using namespace buddy::runtime;
  ExternalKernel kernel(library), missing("");
  float input[] = {1, 2, 3, 4, 0, 0, 0, 0}, output[8];
  const float expected[] = {.032058604f, .087144323f, .236882836f, .643914282f,
                            .25f,        .25f,        .25f,        .25f};
  int64_t shape[] = {2, 4}, strides[] = {4, 1}, axis = -1;
  BuddyKernelCallV1 call{BUDDY_KERNEL_ABI_V1,
                         sizeof(BuddyKernelCallV1),
                         BUDDY_KERNEL_SOFTMAX,
                         0,
                         {input, sizeof(input), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         {output, sizeof(output), 2, BUDDY_KERNEL_F32,
                          BUDDY_KERNEL_ROW_MAJOR, 0, shape, strides},
                         &axis,
                         1,
                         nullptr,
                         0};
  fallbackCalls = 0;
  auto result = kernel.dispatch(call, fallback);
  if (result.status || result.path != KernelPath::External || fallbackCalls)
    return 1;
  for (int i = 0; i < 8; ++i)
    if (!std::isfinite(output[i]) || std::abs(output[i] - expected[i]) > 1e-6f)
      return 2;
  result = missing.dispatch(call, fallback);
  if (result.status || result.path != KernelPath::Fallback ||
      fallbackCalls != 1)
    return 3;
  std::fill_n(output, 8, -77.f);
  axis = 0;
  result = kernel.dispatch(call, fallback);
  if (result.status != BUDDY_KERNEL_INVALID || fallbackCalls != 1 ||
      !std::all_of(output, output + 8, [](float v) { return v == -77.f; }))
    return 4;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 5;
  const int status = buddy_external_consumer_check(argv[1]);
  if (!status)
    std::cout << "PASS CMake consumer external=1 fallback=1 invalid=1\n";
  return status;
}
