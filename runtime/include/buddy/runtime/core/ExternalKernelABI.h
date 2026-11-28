//===- ExternalKernelABI.h - Experimental synchronous host kernel ABI -----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_RUNTIME_CORE_EXTERNALKERNELABI_H
#define BUDDY_RUNTIME_CORE_EXTERNALKERNELABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  BUDDY_KERNEL_ABI_V1 = 1,
  BUDDY_KERNEL_F32 = 1,
  BUDDY_KERNEL_ROW_MAJOR = 1,
  BUDDY_KERNEL_SOFTMAX = 1,
  BUDDY_KERNEL_OK = 0,
  BUDDY_KERNEL_UNSUPPORTED = 1,
  BUDDY_KERNEL_INVALID = 2,
  BUDDY_KERNEL_FAILED = 3
};

// Shape and strides are borrowed arrays of rank elements; strides are in
// elements. data is the first logical element, aligned to the element type.
// byteSize is the accessible span starting at data. V1 uses host memory only.
typedef struct BuddyKernelTensorV1 {
  void *data;
  uint64_t byteSize;
  uint32_t rank;
  uint32_t dtype;
  uint32_t layout;
  uint32_t reserved;
  const int64_t *shape;
  const int64_t *strides;
} BuddyKernelTensorV1;

typedef struct BuddyKernelCallV1 {
  uint32_t abiVersion;
  uint32_t structSize;
  uint32_t operation;
  uint32_t reserved;
  BuddyKernelTensorV1 input;
  BuddyKernelTensorV1 output;
  const int64_t *scalars;
  uint64_t scalarCount;
  void *workspace;
  uint64_t workspaceBytes;
} BuddyKernelCallV1;

typedef int32_t (*BuddyKernelInvokeV1)(const BuddyKernelCallV1 *);

// Query and descriptor code must use the baseline host ISA. Target checking
// happens before invoke. All pointers remain valid until the library closes.
typedef struct BuddyKernelDescriptorV1 {
  uint32_t abiVersion;
  uint32_t structSize;
  uint32_t operation;
  uint32_t dtype;
  uint32_t rank;
  uint32_t layout;
  int64_t rows;
  int64_t cols;
  int64_t axis;
  const char *target;
  const char *requiredFeatures;
  int32_t (*isCompatibleHost)(void);
  BuddyKernelInvokeV1 invoke;
} BuddyKernelDescriptorV1;

typedef const BuddyKernelDescriptorV1 *(*BuddyKernelQueryV1)(void);

// invoke is synchronous, borrows every buffer, and must not throw across ABI.
// V1 is the contiguous 2D FP32 Softmax contract: malformed descriptors,
// unsupported dtype/layout/rank/operation, invalid axis/scalars/workspace and
// overlapping buffers return INVALID, consistently at runtime and invoke.
// A valid call outside the kernel's shape specialization, or an incompatible
// host, returns UNSUPPORTED. This does not imply support for arbitrary shapes.
// UNSUPPORTED/INVALID leave output untouched. FAILED makes output undefined;
// runtime may overwrite it with the fallback (input/output must not overlap).
const BuddyKernelDescriptorV1 *buddy_external_kernel_query_v1(void);

#ifdef __cplusplus
}
#endif
#endif
