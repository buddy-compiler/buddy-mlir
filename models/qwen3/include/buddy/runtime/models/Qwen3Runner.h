//===- Qwen3Runner.h - Qwen3 inference runner -----------------------------===//
#ifndef BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H
#define BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

class Qwen3Runner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_QWEN3RUNNER_H
