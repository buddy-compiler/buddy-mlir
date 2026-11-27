//===- Standalone.cpp - No Python or Buddy compiler at execution
//-----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "buddy/runtime/core/ExternalKernelABI.h"

#include <cmath>
#include <iostream>

int main() {
  float input[8] = {1, 2, 3, 4, 0, 0, 0, 0};
  float output[8] = {};
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
  const auto *kernel = buddy_external_kernel_query_v1();
  if (!kernel->isCompatibleHost() || kernel->invoke(&call) != BUDDY_KERNEL_OK)
    return 1;
  const float expected[] = {0.032058604f, 0.087144323f, 0.236882836f,
                            0.643914282f, 0.25f,        0.25f,
                            0.25f,        0.25f};
  for (int i = 0; i < 8; ++i) {
    std::cout << output[i] << (i % 4 == 3 ? '\n' : ' ');
    if (!std::isfinite(output[i]) ||
        std::abs(output[i] - expected[i]) > 1e-6f + 1e-5f * expected[i])
      return 2;
  }
  std::cout << "PASS standalone C++ Triton ABI\n";
}
