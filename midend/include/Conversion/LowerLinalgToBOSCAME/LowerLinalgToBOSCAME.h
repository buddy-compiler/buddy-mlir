//===- LowerLinalgToBOSCAME.h - Linalg BOSCAME lowering -------===//
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
// Pass entry points for the shared Kernel Contract -> plan -> emit pipeline
// that lowers bufferized Linalg into `bosc_ame` SPM/DMA/tile IR.
//
// `createLowerLinalgToBOSCAMEPass` is the historical name;
// `createLinalgToBOSCAMEPass` is the preferred alias used by buddy-opt
// registrations. Both construct the same pipeline (normalize buffers, plan
// placement/streams, emit BOSCAME ops). Target selection and hardware contracts
// live in BOSCAMETargetInfo.h / plan attributes; this header only exposes the
// MLIR Pass factories.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H

#include <memory>

namespace mlir {
class Pass;
namespace buddy {

// Full Linalg -> BOSCAME lowering pass (contract/plan/emit).
std::unique_ptr<Pass> createLowerLinalgToBOSCAMEPass();

// Preferred alias of createLowerLinalgToBOSCAMEPass for buddy-opt registration.
std::unique_ptr<Pass> createLinalgToBOSCAMEPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_H
