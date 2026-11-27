// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "buddy/runtime/core/ExternalKernelABI.h"
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return 2;
  auto query = reinterpret_cast<BuddyKernelQueryV1>(
      dlsym(handle, "buddy_external_kernel_query_v1"));
  if (!query || !query())
    return 3;
  const auto *descriptor = query();
  bool expected = std::atoi(argv[2]);
  bool compatible = descriptor->isCompatibleHost();
  if (compatible != expected)
    return 4;
  float input[8] = {1, 2, 3, 4}, output[8];
  std::fill_n(output, 8, -77.f);
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
  int status = descriptor->invoke(&call);
  if (expected ? status != BUDDY_KERNEL_OK
               : (status != BUDDY_KERNEL_UNSUPPORTED ||
                  !std::all_of(output, output + 8,
                               [](float v) { return v == -77.f; })))
    return 5;
  std::cout << "PASS capability compatible=" << compatible
            << " status=" << status << '\n';
  dlclose(handle);
}
