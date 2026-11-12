//===- ProteinGLMRunnerPlugin.cpp - ProteinGLM runner plugin -------------===//
//
// Licensed under the Apache License, Version 2.0.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/ProteinGLMRunner.h"

extern "C" buddy::runtime::InferenceRunner *buddy_create_inference_runner_v1() {
  return new buddy::runtime::ProteinGLMRunner();
}

extern "C" void
buddy_destroy_inference_runner_v1(buddy::runtime::InferenceRunner *runner) {
  delete runner;
}

// This model does not implement runBench().
extern "C" uint64_t buddy_inference_runner_capabilities_v1() { return 0; }
