//===- DeepSeekR1TPRunnerPlugin.cpp - DeepSeek TP runner plugin -----------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1TPRunner.h"

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new buddy::runtime::DeepSeekR1TPRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}
