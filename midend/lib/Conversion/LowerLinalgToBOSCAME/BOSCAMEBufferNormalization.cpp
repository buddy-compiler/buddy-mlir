//===- BOSCAMEBufferNormalization.cpp - Compatibility buffer forms --------===//
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
// Rewrite supported handwritten buffer generics into ordinary copies, explicit
// scalar broadcasts, and one arithmetic operation per generic for planning.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEBufferNormalization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace {
Value storageRoot(Value value) {
  while (auto view =
             dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
    value = view.getViewSource();
  return value;
}

bool independentLanes(linalg::GenericOp generic) {
  Value output = generic.getOutputs()[0], outputRoot = storageRoot(output);
  for (Value input : generic.getInputs()) {
    if (!isa<MemRefType>(input.getType()) || input == output)
      continue;
    Value inputRoot = storageRoot(input);
    bool privateStorage = isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
                              inputRoot.getDefiningOp()) ||
                          isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
                              outputRoot.getDefiningOp());
    if (inputRoot == outputRoot || !privateStorage)
      return false;
  }
  return true;
}

bool binaryArithmetic(Operation &op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AddFOp,
             arith::SubFOp, arith::MulFOp, arith::DivFOp>(op);
}

void sourceFacts(Operation *source, Operation *target) {
  for (NamedAttribute attr : source->getAttrs())
    if ((attr.getName().strref().starts_with("contract.") &&
         attr.getName().strref() != "contract.task") ||
        attr.getName().strref().starts_with("bosc_ame."))
      target->setAttr(attr.getName(), attr.getValue());
}

bool parallelShape(linalg::GenericOp generic) {
  if (!generic.hasPureBufferSemantics() || generic.getNumDpsInits() != 1 ||
      generic.getNumLoops() < 1 || generic.getNumLoops() > 2 ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType i) {
        return i == utils::IteratorType::parallel;
      }))
    return false;
  auto output = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
  if (!output || !output.hasStaticShape() ||
      output.getRank() != int64_t(generic.getNumLoops()) ||
      !output.getElementType().isIntOrFloat())
    return false;
  for (auto [value, map] :
       llvm::zip(generic.getOperands(), generic.getIndexingMapsArray())) {
    if (auto type = dyn_cast<MemRefType>(value.getType())) {
      if (type.getShape() != output.getShape() || !map.isIdentity() ||
          type.getElementType() != output.getElementType())
        return false;
    } else if (value.getType() != output.getElementType() ||
               map.getNumResults()) {
      return false;
    }
  }
  return true;
}

// Preserve a lane select by computing its original arithmetic, then assigning
// the false value to the static tail. No predicate multiplication is used;
// NaN/Inf and the sign/bits of the false constant retain their semantics.
linalg::GenericOp normalizeTailSelect(linalg::GenericOp generic) {
  auto output = cast<MemRefType>(generic.getOutputs()[0].getType());
  if (output.getRank() != 1)
    return generic;
  Block &body = generic.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto select = yield.getValues()[0].getDefiningOp<arith::SelectOp>();
  if (!select || select->getBlock() != &body)
    return generic;
  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare->getBlock() != &body ||
      (compare.getPredicate() != arith::CmpIPredicate::slt &&
       compare.getPredicate() != arith::CmpIPredicate::ult))
    return generic;
  auto lane = compare.getLhs().getDefiningOp<linalg::IndexOp>();
  APInt limit;
  if (!lane || lane->getBlock() != &body || lane.getDim() != 0 ||
      !matchPattern(compare.getRhs(), m_ConstantInt(&limit)) ||
      limit.getSExtValue() <= 0 || limit.getSExtValue() >= output.getDimSize(0))
    return generic;
  Value falseValue = select.getFalseValue();
  if (auto argument = dyn_cast<BlockArgument>(falseValue);
      argument && argument.getOwner() == &body &&
      argument.getArgNumber() < generic.getNumDpsInputs())
    falseValue = generic.getInputs()[argument.getArgNumber()];
  Attribute constant;
  if (!falseValue.getType().isIntOrFloat() ||
      !matchPattern(falseValue, m_Constant(&constant)))
    return generic;
  for (Operation &op : body.without_terminator())
    if (&op != select && &op != compare && &op != lane && !binaryArithmetic(op))
      return generic;
  auto arithmetic = select.getTrueValue().getDefiningOp();
  if (!arithmetic || arithmetic->getBlock() != &body ||
      !binaryArithmetic(*arithmetic))
    return generic;
  if (!independentLanes(generic))
    return generic;
  OpBuilder builder(generic);
  auto replacement = linalg::GenericOp::create(
      builder, generic.getLoc(), TypeRange{}, generic.getInputs(),
      generic.getOutputs(), generic.getIndexingMapsArray(),
      generic.getIteratorTypesArray(),
      [&](OpBuilder &nested, Location loc, ValueRange args) {
        IRMapping mapping;
        mapping.map(body.getArguments(), args);
        for (Operation &op : body.without_terminator())
          if (binaryArithmetic(op))
            nested.clone(op, mapping);
        linalg::YieldOp::create(nested, loc,
                                mapping.lookup(select.getTrueValue()));
      });
  sourceFacts(generic, replacement);
  int64_t prefix = limit.getSExtValue();
  Value tail = memref::SubViewOp::create(
      builder, generic.getLoc(), generic.getOutputs()[0],
      ArrayRef<OpFoldResult>{builder.getIndexAttr(prefix)},
      ArrayRef<OpFoldResult>{
          builder.getIndexAttr(output.getDimSize(0) - prefix)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
  auto fill = linalg::FillOp::create(builder, generic.getLoc(),
                                     ValueRange{falseValue}, ValueRange{tail});
  sourceFacts(generic, fill);
  generic.erase();
  return replacement;
}

void normalizeArithmetic(linalg::GenericOp generic) {
  Block &body = generic.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  // Dynamic scalar inputs have explicit content dependencies and are
  // broadcast by the selected vector recipe. Materializing one full buffer
  // per scalar would recreate the storage eliminated by tensor fusion.
  if (llvm::any_of(generic.getInputs(),
                   [](Value input) { return input.getType().isIntOrFloat(); }))
    return;
  if (generic.getNumDpsInputs() == 1 && body.getOperations().size() == 1 &&
      yield.getValues()[0] == body.getArgument(0) &&
      isa<MemRefType>(generic.getInputs()[0].getType())) {
    OpBuilder builder(generic);
    auto copy =
        memref::CopyOp::create(builder, generic.getLoc(),
                               generic.getInputs()[0], generic.getOutputs()[0]);
    sourceFacts(generic, copy);
    generic.erase();
    return;
  }
  auto outputType = cast<MemRefType>(generic.getOutputs()[0].getType());
  for (Operation &op : body.without_terminator())
    if (!binaryArithmetic(op) ||
        op.getResult(0).getType() != outputType.getElementType() ||
        llvm::any_of(op.getOperandTypes(), [&](Type type) {
          return type != outputType.getElementType();
        }))
      return;
  Operation *last = yield.getValues()[0].getDefiningOp();
  if (!last || last->getBlock() != &body)
    return;
  bool scalar = llvm::any_of(generic.getInputs(), [](Value input) {
    return !isa<MemRefType>(input.getType());
  });
  bool captures = llvm::any_of(body.without_terminator(), [&](Operation &op) {
    return llvm::any_of(op.getOperands(), [&](Value value) {
      auto arg = dyn_cast<BlockArgument>(value);
      return (!arg || arg.getOwner() != &body) && value.getDefiningOp() != last;
    });
  });
  if (body.getOperations().size() == 2 && !scalar) {
    bool constantCaptures = llvm::all_of(last->getOperands(), [&](Value value) {
      auto argument = dyn_cast<BlockArgument>(value);
      if (argument && argument.getOwner() == &body)
        return true;
      auto constant = value.getDefiningOp<arith::ConstantOp>();
      return constant && constant->getBlock() != &body &&
             isa<IntegerAttr, FloatAttr>(constant.getValue());
    });
    // Constants can be broadcast directly into vector registers. In
    // particular, a strict +0 left after tensor storage normalization must
    // not recreate an entire zero matrix and its SPM window here.
    if (!captures || constantCaptures)
      return;
  }
  if (!independentLanes(generic))
    return;
  OpBuilder builder(generic);
  auto temporaryType =
      MemRefType::get(outputType.getShape(), outputType.getElementType());
  DenseMap<Value, Value> arrays;
  for (auto [argument, input] :
       llvm::zip(body.getArguments(), generic.getOperands()))
    arrays[argument] = input;
  auto materialize = [&](Value value) -> Value {
    auto found = arrays.find(value);
    Value input = found == arrays.end() ? value : found->second;
    if (isa<MemRefType>(input.getType()))
      return input;
    auto allocation =
        memref::AllocOp::create(builder, generic.getLoc(), temporaryType,
                                builder.getI64IntegerAttr(64));
    auto fill = linalg::FillOp::create(
        builder, generic.getLoc(), ValueRange{input}, ValueRange{allocation});
    sourceFacts(generic, fill);
    arrays[value] = allocation;
    return allocation;
  };
  auto identity = AffineMap::getMultiDimIdentityMap(outputType.getRank(),
                                                    builder.getContext());
  SmallVector<AffineMap> maps(3, identity);
  for (Operation &op : body.without_terminator()) {
    SmallVector<Value> inputs;
    for (Value value : op.getOperands())
      inputs.push_back(materialize(value));
    Value output =
        &op == last
            ? generic.getOutputs()[0]
            : memref::AllocOp::create(builder, generic.getLoc(), temporaryType,
                                      builder.getI64IntegerAttr(64))
                  .getResult();
    auto arithmetic = linalg::GenericOp::create(
        builder, generic.getLoc(), TypeRange{}, inputs, ValueRange{output},
        maps, generic.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location loc, ValueRange args) {
          IRMapping mapping;
          for (auto [old, replacement] :
               llvm::zip(op.getOperands(), args.take_front(2)))
            mapping.map(old, replacement);
          auto cloned = nested.clone(op, mapping);
          linalg::YieldOp::create(nested, loc, cloned->getResult(0));
        });
    sourceFacts(generic, arithmetic);
    arrays[op.getResult(0)] = output;
  }
  generic.erase();
}
} // namespace

void buddy::boscame::normalizeBOSCAMEBufferInput(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp generic) { generics.push_back(generic); });
  for (auto generic : generics)
    if (parallelShape(generic))
      normalizeArithmetic(normalizeTailSelect(generic));
}
