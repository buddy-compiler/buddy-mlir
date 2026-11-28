//===- ExternalKernelValidation.h - Shared v1 Softmax call checks
//-----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_RUNTIME_CORE_EXTERNALKERNELVALIDATION_H
#define BUDDY_RUNTIME_CORE_EXTERNALKERNELVALIDATION_H

#include "buddy/runtime/core/ExternalKernelABI.h"

#include <cstdint>
#include <limits>

namespace buddy::runtime::detail {

// This validates the v1 contiguous 2D FP32 Softmax contract, independently of
// any fixed-shape specialization or host ISA. No allocation or tensor scan.
inline bool validSoftmaxTensor(const BuddyKernelTensorV1 &tensor) {
  if (!tensor.data || reinterpret_cast<uintptr_t>(tensor.data) % 4 != 0 ||
      tensor.rank != 2 || tensor.dtype != BUDDY_KERNEL_F32 ||
      tensor.layout != BUDDY_KERNEL_ROW_MAJOR || tensor.reserved ||
      !tensor.shape || !tensor.strides)
    return false;
  const int64_t rows = tensor.shape[0], cols = tensor.shape[1];
  if (rows <= 0 || cols <= 0 || tensor.strides[0] != cols ||
      tensor.strides[1] != 1)
    return false;
  constexpr uint64_t maxElements = std::numeric_limits<uint64_t>::max() / 4;
  return static_cast<uint64_t>(rows) <= maxElements / cols &&
         tensor.byteSize >= static_cast<uint64_t>(rows) * cols * 4;
}

inline bool validSoftmaxCall(const BuddyKernelCallV1 &call) {
  if (call.abiVersion != BUDDY_KERNEL_ABI_V1 ||
      call.structSize != sizeof(call) || call.reserved ||
      call.operation != BUDDY_KERNEL_SOFTMAX ||
      !validSoftmaxTensor(call.input) || !validSoftmaxTensor(call.output) ||
      call.scalarCount != 1 || !call.scalars ||
      (call.scalars[0] != 1 && call.scalars[0] != -1) || call.workspace ||
      call.workspaceBytes)
    return false;
  if (call.input.shape[0] != call.output.shape[0] ||
      call.input.shape[1] != call.output.shape[1])
    return false;
  const auto input = reinterpret_cast<uintptr_t>(call.input.data);
  const auto output = reinterpret_cast<uintptr_t>(call.output.data);
  const uint64_t bytes =
      static_cast<uint64_t>(call.input.shape[0]) * call.input.shape[1] * 4;
  return (input <= output ? output - input : input - output) >= bytes;
}

} // namespace buddy::runtime::detail
#endif
