//===- Llama31TTRunner.h - Tenstorrent Llama 3.1 runner --------*- C++ -*-===//
//
// Licensed under the Apache License, Version 2.0.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_LLAMA31TTRUNNER_H
#define BUDDY_RUNTIME_MODELS_LLAMA31TTRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

class Llama31TTRunner final : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_LLAMA31TTRUNNER_H
