//===- RVProfInstrumentation.h - RVProf Instrumentation Pass ----*- C++ -*-===//
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
//
// This file declares the RVProf instrumentation pass.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_RVPROFINSTRUMENTATION_RVPROFINSTRUMENTATION_H
#define CONVERSION_RVPROFINSTRUMENTATION_RVPROFINSTRUMENTATION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace buddy {
void registerRVProfInstrumentationPass();
} // namespace buddy
} // namespace mlir

#endif // CONVERSION_RVPROFINSTRUMENTATION_RVPROFINSTRUMENTATION_H
