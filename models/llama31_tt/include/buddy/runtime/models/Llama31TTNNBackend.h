//===- Llama31TTNNBackend.h - Native TTNN serving backend ----------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_LLAMA31TTNNBACKEND_H
#define BUDDY_RUNTIME_MODELS_LLAMA31TTNNBACKEND_H

#include "buddy/runtime/models/Llama31TTExecution.h"

namespace buddy {
namespace runtime {

/// Construct the production TTNN backend used by the resident plugin.
///
/// The implementation keeps TTNN types private to the backend translation
/// unit. It opens the device and loads both phase programs while the factory
/// is called from Llama31TTExecution::load(). An empty result is never valid;
/// initialization failures are reported as exceptions by the implementation.
Llama31TTExecution::BackendHooks
createLlama31TTNNBackend(const Llama31TTExecution::Metadata &metadata);

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_LLAMA31TTNNBACKEND_H
