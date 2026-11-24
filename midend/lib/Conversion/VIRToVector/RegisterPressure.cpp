//===- RegisterPressure.cpp - Dynamic vectors counted for registers -------===//
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

#include "RegisterPressure.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include "VIR/VIROps.h"
#include "VIR/VIRTypes.h"

using namespace mlir;

namespace buddy {
namespace vir {

bool isRVVScalarOperand(Value v) {
  if (!v.getDefiningOp<vir::BroadcastOp>())
    return false;
  return llvm::all_of(v.getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    unsigned n = use.getOperandNumber();
    // a multiplicand of an fma, not the addend, or an operand of a binary op
    bool binary =
        isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
            arith::MinNumFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp>(
            user);
    if (!(isa<vir::FMAOp>(user) && n < 2) && !binary)
      return false;
    Value other = user->getOperand(1 - n);
    return !other.getDefiningOp<vir::BroadcastOp>();
  });
}

bool takesVectorRegisters(Value v, bool rvv) {
  return isa<vir::DynamicVectorType>(v.getType()) &&
         !(rvv && isRVVScalarOperand(v));
}

} // namespace vir
} // namespace buddy
