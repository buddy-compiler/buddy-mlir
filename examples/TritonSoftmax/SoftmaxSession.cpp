// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "SoftmaxSession.h"

#include <cstdlib>
#include <cstring>

namespace buddy::examples::softmax {
using runtime::KernelPath;

extern "C" void _mlir_ciface_buddy_softmax(MemRef2 *, MemRef2 *);
extern "C" int32_t _mlir_ciface_buddy_computation(MemRef2 *, MemRef2 *, int64_t,
                                                  void *);

uint64_t fallbackCalls = 0;
static int32_t fallback(const BuddyKernelCallV1 *call) {
  if (call->input.shape[0] != 2 || call->input.shape[1] != 4)
    return BUDDY_KERNEL_UNSUPPORTED;
  MemRef2 input{static_cast<float *>(call->input.data),
                static_cast<float *>(call->input.data),
                0,
                {2, 4},
                {4, 1}};
  MemRef2 result{};
  _mlir_ciface_buddy_softmax(&result, &input);
  std::memcpy(call->output.data, result.aligned + result.offset, 32);
  std::free(result.allocated);
  ++fallbackCalls;
  return BUDDY_KERNEL_OK;
}

// This is the C interface called by compiled computation.mlir. Its metadata
// comes from MLIR memrefs; the session supplies the optional kernel and
// fallback.
extern "C" int32_t _mlir_ciface_buddy_softmax_dispatch(MemRef2 *input,
                                                       MemRef2 *output,
                                                       int64_t axis,
                                                       void *opaque) {
  auto &session = *static_cast<Session *>(opaque);
  BuddyKernelCallV1 call{
      BUDDY_KERNEL_ABI_V1,
      sizeof(BuddyKernelCallV1),
      BUDDY_KERNEL_SOFTMAX,
      0,
      {input->aligned + input->offset, 32, 2, BUDDY_KERNEL_F32,
       BUDDY_KERNEL_ROW_MAJOR, 0, input->sizes, input->strides},
      {output->aligned + output->offset, 32, 2, BUDDY_KERNEL_F32,
       BUDDY_KERNEL_ROW_MAJOR, 0, output->sizes, output->strides},
      &axis,
      1,
      nullptr,
      0};
  session.last = session.kernel.dispatch(call, fallback);
  return session.last.status;
}

buddy::runtime::KernelResult compute(Session &session,
                                     const BuddyKernelCallV1 &call) {
  // A static memref graph cannot represent arbitrary metadata. Validate the
  // original descriptor before lowering it, rather than silently replacing it.
  if (!runtime::ExternalKernel::isValidCall(call))
    return {BUDDY_KERNEL_INVALID, KernelPath::None, BUDDY_KERNEL_INVALID};
  if (call.input.shape[0] != 2 || call.input.shape[1] != 4)
    return {BUDDY_KERNEL_UNSUPPORTED, KernelPath::None,
            BUDDY_KERNEL_UNSUPPORTED};
  MemRef2 input{static_cast<float *>(call.input.data),
                static_cast<float *>(call.input.data),
                0,
                {call.input.shape[0], call.input.shape[1]},
                {call.input.strides[0], call.input.strides[1]}};
  MemRef2 output{static_cast<float *>(call.output.data),
                 static_cast<float *>(call.output.data),
                 0,
                 {call.output.shape[0], call.output.shape[1]},
                 {call.output.strides[0], call.output.strides[1]}};
  session.last = {BUDDY_KERNEL_FAILED, KernelPath::None, BUDDY_KERNEL_FAILED};
  int32_t status = _mlir_ciface_buddy_computation(&input, &output,
                                                  call.scalars[0], &session);
  if (status != session.last.status)
    return {BUDDY_KERNEL_FAILED, KernelPath::None, BUDDY_KERNEL_FAILED};
  return session.last;
}

} // namespace buddy::examples::softmax
