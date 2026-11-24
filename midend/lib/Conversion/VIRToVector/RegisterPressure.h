//===- RegisterPressure.h - Dynamic vectors counted for registers -*- C++ -*-=//
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
// The dynamic vectors that the register pressure of a vir.set_vl region
// counts, shared by -lower-vir-to-vector (Req_Group, from which it sizes the
// vectors) and -vir-schedule (MaxLive, which it orders the ops for).
//
//===----------------------------------------------------------------------===//

#ifndef VIRTOVECTOR_REGISTERPRESSURE_H
#define VIRTOVECTOR_REGISTERPRESSURE_H

#include "mlir/IR/Value.h"

namespace buddy {
namespace vir {

using mlir::Value;

/// Whether `v` is a broadcast scalar that only feeds the scalar operand of
/// arithmetic on RVV (vfmul.vf, vfmacc.vf, ...): it stays in a scalar
/// register (LLVM sinks a splat to its users, CodeGenPrepare) and takes no
/// vector register. Such an instruction has one scalar operand: if the other
/// operand (the other multiplicand of an fma) is a broadcast too, neither is.
bool isRVVScalarOperand(Value v);

/// Whether `v` takes vector registers: a dynamic vector, but on RVV (`rvv`)
/// not a scalar operand (isRVVScalarOperand).
bool takesVectorRegisters(Value v, bool rvv);

} // namespace vir
} // namespace buddy

#endif // VIRTOVECTOR_REGISTERPRESSURE_H
