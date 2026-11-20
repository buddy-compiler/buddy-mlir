//===- LowerLinalgToAME.h - Linalg to AME lowering -----------------*- C++
//-*-===//
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

#ifndef BUDDY_CONVERSION_LOWERLINALGTOAME_H
#define BUDDY_CONVERSION_LOWERLINALGTOAME_H

#include <memory>

namespace mlir {
class Pass;
namespace buddy {

/// Lower bufferized i32 matmuls (named or standard generic contraction) and
/// rank-1/2 all-parallel generic regions to ZTT v0.6 AME operations. Pointwise
/// regions support projected-permutation inputs, permuted outputs, wrapping
/// integer arithmetic, shifts, bitwise operations, signed/unsigned min/max,
/// constants and captured scalars. These maps also cover copy, transpose and
/// row/column/scalar broadcast. Unsupported Linalg remains for later passes.
/// The generated code manages ownership; no caller-owned AME state may be live
/// across a converted operation. target=gem5 must be paired with +xame-gem5 in
/// llc; target=generic (the default) uses the default 8x8 XAME profile.
std::unique_ptr<Pass> createLowerLinalgToAMEPass();
void registerLowerLinalgToAMEPass();

} // namespace buddy
} // namespace mlir

#endif // BUDDY_CONVERSION_LOWERLINALGTOAME_H
