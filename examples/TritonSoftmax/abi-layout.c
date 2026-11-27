// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "buddy/runtime/core/ExternalKernelABI.h"
#include <stddef.h>
#include <stdio.h>
_Static_assert(sizeof(BuddyKernelTensorV1) == 48, "tensor ABI");
_Static_assert(sizeof(BuddyKernelCallV1) == 144, "call ABI");
_Static_assert(sizeof(BuddyKernelDescriptorV1) == 80, "descriptor ABI");
_Static_assert(offsetof(BuddyKernelCallV1, scalars) == 112, "scalar offset");
_Static_assert(offsetof(BuddyKernelDescriptorV1, invoke) == 72,
               "invoke offset");
int main(void) {
  puts("PASS C ABI layout tensor=48 call=144 descriptor=80");
  return 0;
}
