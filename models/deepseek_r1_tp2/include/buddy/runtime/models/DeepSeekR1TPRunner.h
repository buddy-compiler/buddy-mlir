//===- DeepSeekR1TPRunner.h - DeepSeek R1 TP=2 runner ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_DEEPSEEKR1TPRUNNER_H
#define BUDDY_RUNTIME_MODELS_DEEPSEEKR1TPRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// Runs rank-local DeepSeek R1 RAX schedules through the MPI communicator.
class DeepSeekR1TPRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DEEPSEEKR1TPRUNNER_H
