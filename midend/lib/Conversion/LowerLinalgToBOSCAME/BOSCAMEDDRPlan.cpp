//===- BOSCAMEDDRPlan.cpp - Exact CPU recipes for materialized DDR --------===//
//----------===//
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
// Derive, verify, and emit exact CPU recipes for operations that remain on
// materialized DDR. Recipes never authorize changing arithmetic or touching
// SPM.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEDDRPlan.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEMathApprox.h"
#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEPointwise.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/MathExtras.h"
#include <functional>

using namespace mlir;
namespace buddy::boscame {
namespace {
Value root(Value value) {
  while (auto view =
             dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp()))
    value = view.getViewSource();
  return value;
}
bool inDDR(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getMemorySpace())
    return false;
  unsigned width = spmdma::getSpmElementByteWidth(type.getElementType());
  if (!width || !spmdma::isEndpointPointerAlignmentProven(value, width))
    return false;
  if (auto op = root(value).getDefiningOp()) {
    auto placement = op->getAttrOfType<DictionaryAttr>("plan.buffer");
    if (placement) {
      auto level = placement.getAs<StringAttr>("level");
      if (!level || level.getValue() != "ddr")
        return false;
    }
  }
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         llvm::all_of(type.getShape(), [](int64_t n) { return n > 0; }) &&
         llvm::all_of(strides, [](int64_t s) { return s > 0; });
}
Value idx(OpBuilder &b, Location loc, int64_t n) {
  return arith::ConstantIndexOp::create(b, loc, n);
}
// A scope supplies an actual completion even for rank-zero operations.
Operation *loopNest(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                    function_ref<void(ValueRange)> body) {
  auto scope = scf::ExecuteRegionOp::create(b, loc, TypeRange{});
  Block *block = new Block;
  scope.getRegion().push_back(block);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(block);
  SmallVector<Value> indices;
  std::function<void(unsigned)> descend = [&](unsigned d) {
    if (d == shape.size()) {
      body(indices);
      return;
    }
    auto loop = scf::ForOp::create(b, loc, idx(b, loc, 0),
                                   idx(b, loc, shape[d]), idx(b, loc, 1));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    descend(d + 1);
    indices.pop_back();
  };
  descend(0);
  b.setInsertionPointToEnd(block);
  scf::YieldOp::create(b, loc);
  return scope;
}
Value expression(OpBuilder &b, Block &body, ValueRange args) {
  IRMapping mapping;
  for (auto [from, to] : llvm::zip(body.getArguments(), args))
    if (to)
      mapping.map(from, to);
  for (Operation &op : body.without_terminator()) {
    if (isSupportedMathApprox(op)) {
      mapping.map(
          op.getResult(0),
          emitMathApprox(b, op, mapping.lookupOrDefault(op.getOperand(0))));
    } else if (isa<math::AbsFOp>(op) && op.getResult(0).getType().isF32()) {
      Location loc = op.getLoc();
      Value bits = arith::BitcastOp::create(
          b, loc, b.getI32Type(), mapping.lookupOrDefault(op.getOperand(0)));
      Value mask = arith::ConstantIntOp::create(b, loc, 0x7fffffff, 32);
      bits = arith::AndIOp::create(b, loc, bits, mask);
      mapping.map(op.getResult(0),
                  arith::BitcastOp::create(b, loc, b.getF32Type(), bits));
    } else {
      b.clone(op, mapping);
    }
  }
  return mapping.lookupOrDefault(
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0]);
}
} // namespace

FailureOr<DictionaryAttr> getDDRRecipe(Operation *op) {
  Builder b(op->getContext());
  StringRef kind;
  SmallVector<int64_t> shape;
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    auto source = dyn_cast<MemRefType>(copy.getSource().getType());
    auto target = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (!source || !target || source.getShape() != target.getShape() ||
        source.getElementType() != target.getElementType())
      return failure();
    kind = "copy";
    shape.assign(target.getShape().begin(), target.getShape().end());
  } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
    auto type = dyn_cast<MemRefType>(fill.getOutputs()[0].getType());
    if (!type)
      return failure();
    kind = "fill";
    shape.assign(type.getShape().begin(), type.getShape().end());
  } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (generic.getNumDpsInits() != 1 ||
        generic.getNumLoops() != generic.getNumParallelLoops())
      return failure();
    auto type = dyn_cast<MemRefType>(generic.getOutputs()[0].getType());
    if (!type || generic.getNumLoops() != type.getRank())
      return failure();
    for (auto [operand, map] :
         llvm::zip(generic.getOperands(), generic.getIndexingMapsArray())) {
      auto memref = dyn_cast<MemRefType>(operand.getType());
      if (map.getNumResults() == 0 && (!memref || memref.getRank() == 0))
        continue;
      if (!memref || !map.isIdentity() || memref.getShape() != type.getShape())
        return failure();
    }
    for (Operation &payload : generic.getBody()->without_terminator())
      if (!supportedPointwiseOperation(payload))
        return failure();
    kind = "pointwise";
    shape.assign(type.getShape().begin(), type.getShape().end());
  } else if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
    if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1 ||
        reduce.getDimensions() != ArrayRef<int64_t>{0})
      return failure();
    auto input = dyn_cast<MemRefType>(reduce.getInputs()[0].getType());
    auto output = dyn_cast<MemRefType>(reduce.getInits()[0].getType());
    if (!input || !output || input.getRank() != 1 || output.getRank() != 0 ||
        !input.getElementType().isF32() ||
        input.getElementType() != output.getElementType())
      return failure();
    Block &body = reduce.getRegion().front();
    if (std::distance(body.begin(), body.end()) != 2 ||
        !isa<arith::AddFOp, arith::MaxNumFOp>(body.front()))
      return failure();
    kind = "ordered-reduce";
    shape.assign(input.getShape().begin(), input.getShape().end());
  } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    kind = isa<memref::LoadOp>(op) ? "load" : "store";
  } else {
    return failure();
  }
  for (Value operand : op->getOperands())
    if (isa<BaseMemRefType>(operand.getType()) && !inDDR(operand))
      return failure();
  return b.getDictionaryAttr(
      {b.getNamedAttr("kind", b.getStringAttr(kind)),
       b.getNamedAttr("engine", b.getStringAttr("scalar-ddr")),
       b.getNamedAttr("shape", b.getDenseI64ArrayAttr(shape)),
       b.getNamedAttr("fp_policy", b.getStringAttr("preserve"))});
}

LogicalResult verifyDDRRecipe(Operation *op) {
  auto expected = getDDRRecipe(op);
  auto recorded = op->getAttrOfType<DictionaryAttr>("plan.cpu");
  if (failed(expected) || !recorded || *expected != recorded)
    return op->emitError("invalid exact DDR recipe for actual operation");
  return success();
}

Operation *emitDDRRecipe(Operation *op) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    Operation *copy = b.clone(*op);
    op->replaceAllUsesWith(copy->getResults());
    return copy;
  }
  auto recipe = op->getAttrOfType<DictionaryAttr>("plan.cpu");
  auto shape = recipe.getAs<DenseI64ArrayAttr>("shape").asArrayRef();
  if (auto reduce = dyn_cast<linalg::ReduceOp>(op)) {
    Value output = reduce.getInits()[0];
    Value initial = memref::LoadOp::create(b, loc, output, ValueRange{});
    auto loop =
        scf::ForOp::create(b, loc, idx(b, loc, 0), idx(b, loc, shape[0]),
                           idx(b, loc, 1), ValueRange{initial});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value input = memref::LoadOp::create(b, loc, reduce.getInputs()[0],
                                           ValueRange{loop.getInductionVar()});
      Value next = expression(b, reduce.getRegion().front(),
                              ValueRange{input, loop.getRegionIterArgs()[0]});
      scf::YieldOp::create(b, loc, next);
    }
    return memref::StoreOp::create(b, loc, loop.getResult(0), output,
                                   ValueRange{});
  }
  return loopNest(b, loc, shape, [&](ValueRange indices) {
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      Value value = memref::LoadOp::create(b, loc, copy.getSource(), indices);
      memref::StoreOp::create(b, loc, value, copy.getTarget(), indices);
    } else if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      memref::StoreOp::create(b, loc, fill.getInputs()[0], fill.getOutputs()[0],
                              indices);
    } else {
      auto generic = cast<linalg::GenericOp>(op);
      SmallVector<Value> args;
      for (auto [value, map, argument] :
           llvm::zip(generic.getOperands(), generic.getIndexingMapsArray(),
                     generic.getBody()->getArguments())) {
        if (argument.use_empty()) {
          args.push_back(Value());
        } else if (isa<MemRefType>(value.getType())) {
          args.push_back(memref::LoadOp::create(
              b, loc, value, map.getNumResults() ? indices : ValueRange{}));
        } else {
          args.push_back(value);
        }
      }
      Value result = expression(b, *generic.getBody(), args);
      memref::StoreOp::create(b, loc, result, generic.getOutputs()[0], indices);
    }
  });
}
} // namespace buddy::boscame
