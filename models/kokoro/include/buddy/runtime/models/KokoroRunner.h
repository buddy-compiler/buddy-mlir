#ifndef BUDDY_RUNTIME_MODELS_KOKORO_RUNNER_H
#define BUDDY_RUNTIME_MODELS_KOKORO_RUNNER_H
#include "buddy/runtime/core/InferenceRunner.h"
namespace buddy { namespace runtime {
class KokoroRunner : public InferenceRunner { public: void run(const RunConfig &cfg) override; };
} }
#endif
