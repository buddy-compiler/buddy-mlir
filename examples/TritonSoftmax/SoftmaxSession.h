// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BUDDY_EXAMPLES_TRITONSOFTMAX_SESSION_H
#define BUDDY_EXAMPLES_TRITONSOFTMAX_SESSION_H

#include "buddy/runtime/core/ExternalKernel.h"

namespace buddy::examples::softmax {

struct MemRef2 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

struct Session {
  runtime::ExternalKernel kernel;
  runtime::KernelResult last{};
  explicit Session(const std::string &path) : kernel(path) {}
};

// Process-wide instrumentation for this single-threaded example only. Kernel
// selection and execution context are owned by each Session, not by this
// counter.
extern uint64_t fallbackCalls;
// Validate the complete call, then execute the fixed 2x4 graph. Axis is passed
// explicitly through the compiled graph; no metadata is silently defaulted.
runtime::KernelResult compute(Session &session, const BuddyKernelCallV1 &call);

} // namespace buddy::examples::softmax
#endif
