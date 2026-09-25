//===- TimesfmRunner.h - TimesFM 2.5 time-series runner ------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_TIMESFMRUNNER_H
#define BUDDY_RUNTIME_MODELS_TIMESFMRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

namespace buddy {
namespace runtime {

/// TimesFM 2.5 (200M) time-series forecasting runner.
///
/// The compiled kernel consumes a fixed-length context window
/// (num_patches x patch_length time points) and produces the model's point
/// forecast. The runner owns context-window construction, weight loading,
/// kernel invocation, and emission of the forecast.
class TimesfmRunner : public InferenceRunner {
public:
  void run(const RunConfig &cfg) override;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_TIMESFMRUNNER_H
