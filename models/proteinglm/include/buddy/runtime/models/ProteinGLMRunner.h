//===- ProteinGLMRunner.h - ProteinGLM MLM inference runner --------------===//
//
// Licensed under the Apache License, Version 2.0.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_PROTEINGLMRUNNER_H
#define BUDDY_RUNTIME_MODELS_PROTEINGLMRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

class ProteinGLMRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_PROTEINGLMRUNNER_H
