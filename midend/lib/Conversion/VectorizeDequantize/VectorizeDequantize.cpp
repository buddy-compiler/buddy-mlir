//===- VectorizeDequantize.cpp - Vectorize contiguous dequantization
//--------===//
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
// Vectorize bufferized rank-one linalg.generic operations containing signed
// i32-to-f32 conversion and/or f32 multiplication. This pass deliberately has
// no knowledge of kernel names: maps, strides, scalar operations and aliases
// determine eligibility. Tensor elementwise fusion can run before bufferization
// to make conversion followed by two multiplies a single vector loop.
//
// Keep both multiplies in their original order. Do not introduce fast-math
// flags, combine scales, speculate tail loads or assume dynamic strides are 1.
// Fixed 16-lane i32/f32 vectors use e32/m1 on the 512-bit NR RVV target; this
// opt-in pass is also executable on a host target for numerical comparison.
// The separate vectorize-quantize pass also handles fused abs/divide/round/
// clamp/narrow and proven abs-max reductions. It preserves FP32 division and
// rounding operations; it never substitutes reciprocal multiplication.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

constexpr int64_t vectorWidth = 16;

static bool isElementType(Type type, bool quantize = false) {
  return type.isInteger(32) || type.isF32() ||
         (quantize && (type.isInteger(8) || type.isInteger(1)));
}

static bool isContiguousVector(MemRefType type, bool quantize = false) {
  SmallVector<int64_t> strides;
  int64_t offset;
  // i1 is allowed for temporary comparison masks, not memory: vector<i1>
  // packing must not be confused with scalar memref element addressing.
  return type.getRank() == 1 && !type.getElementType().isInteger(1) &&
         isElementType(type.getElementType(), quantize) &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides[0] == 1;
}

static bool canVectorize(linalg::GenericOp op, AliasAnalysis &aliases,
                         bool quantize = false) {
  if (!op.hasPureBufferSemantics() || op.getNumLoops() != 1 ||
      op.getNumParallelLoops() != 1 || op.getNumDpsInits() != 1)
    return false;
  Value output = op.getDpsInits()[0];
  auto outputType = dyn_cast<MemRefType>(output.getType());
  if (!outputType || !isContiguousVector(outputType, quantize) ||
      !op.getIndexingMapsArray().back().isIdentity())
    return false;

  auto maps = op.getIndexingMapsArray();
  for (auto [index, input] : llvm::enumerate(op.getDpsInputs())) {
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType) {
      if (!isElementType(input.getType(), quantize) || maps[index].getNumResults() != 0)
        return false;
      continue;
    }
    // Scalar memrefs are broadcast. Vector inputs require identity indexing
    // and a statically known unit stride; dynamic offsets remain supported.
    if (inputType.getRank() == 0) {
      if (!isElementType(inputType.getElementType(), quantize) ||
          maps[index].getNumResults() != 0)
        return false;
    } else if (!isContiguousVector(inputType, quantize) || !maps[index].isIdentity()) {
      return false;
    }
    // Exact in-place identity access is safe. Partial/unknown aliasing may
    // change loop-carried memory dependencies and is intentionally left alone.
    if (input != output && !aliases.alias(input, output).isNo())
      return false;
  }

  bool hasArithmetic = false;
  for (Operation &nested : op.getRegion().front().without_terminator()) {
    if (auto convert = dyn_cast<arith::SIToFPOp>(nested)) {
      if (!convert.getIn().getType().isInteger(32) ||
          !convert.getOut().getType().isF32())
        return false;
    } else if (auto multiply = dyn_cast<arith::MulFOp>(nested)) {
      if (!multiply.getType().isF32())
        return false;
    } else if (quantize && isa<math::AbsFOp, arith::DivFOp, arith::AddFOp,
                                arith::CmpFOp, arith::SelectOp, arith::MaxNumFOp,
                                arith::MinNumFOp, arith::FPToSIOp,
                                arith::TruncIOp>(nested)) {
      for (Type type : nested.getOperandTypes())
        if (!isElementType(type, true))
          return false;
      for (Type type : nested.getResultTypes())
        if (!isElementType(type, true))
          return false;
    } else {
      return false;
    }
    hasArithmetic = true;
  }
  return hasArithmetic;
}

static void vectorize(linalg::GenericOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  builder.setInsertionPoint(op);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value width = arith::ConstantIndexOp::create(builder, loc, vectorWidth);
  Value output = op.getDpsInits()[0];
  Value extent = memref::DimOp::create(builder, loc, output, zero);
  Value remainder = arith::RemUIOp::create(builder, loc, extent, width);
  Value vectorEnd = arith::SubIOp::create(builder, loc, extent, remainder);
  Block &body = op.getRegion().front();

  auto emitBody = [&](OpBuilder &b, Location loopLoc, Value index, ValueRange,
                      bool useVector) {
    IRMapping mapping;
    auto broadcast = [&](Value value) -> Value {
      if (!useVector)
        return value;
      return vector::BroadcastOp::create(
          b, loopLoc, VectorType::get({vectorWidth}, value.getType()), value);
    };
    for (auto [operand, arg] :
         llvm::zip(op->getOperands(), body.getArguments())) {
      if (arg.use_empty())
        continue;
      auto type = dyn_cast<MemRefType>(operand.getType());
      Value loaded;
      if (!type) {
        loaded = broadcast(operand);
      } else if (type.getRank() == 0) {
        loaded = broadcast(memref::LoadOp::create(b, loopLoc, operand));
      } else if (useVector) {
        loaded = vector::LoadOp::create(
            b, loopLoc, VectorType::get({vectorWidth}, type.getElementType()),
            operand, ValueRange{index});
      } else {
        loaded = memref::LoadOp::create(b, loopLoc, operand, ValueRange{index});
      }
      mapping.map(arg, loaded);
    }

    for (Operation &nested : body.without_terminator()) {
      // Scalar captures (e.g. one activation scale per row) are loop invariant
      // values. Broadcast them without materializing a temporary scale buffer.
      if (useVector) {
        for (Value operand : nested.getOperands()) {
          if (!mapping.contains(operand))
            mapping.map(operand, broadcast(operand));
        }
      }
      Operation *copy = b.clone(nested, mapping);
      if (useVector)
        copy->getResult(0).setType(
            VectorType::get({vectorWidth}, nested.getResult(0).getType()));
    }

    Value result = mapping.lookupOrDefault(
        cast<linalg::YieldOp>(body.getTerminator()).getValues()[0]);
    if (useVector && !isa<VectorType>(result.getType()))
      result = broadcast(result);
    if (useVector)
      vector::StoreOp::create(b, loopLoc, result, output, ValueRange{index});
    else
      memref::StoreOp::create(b, loopLoc, result, output, ValueRange{index});
    scf::YieldOp::create(b, loopLoc);
  };
  scf::ForOp::create(builder, loc, zero, vectorEnd, width, ValueRange{},
                     [&](OpBuilder &b, Location l, Value iv, ValueRange args) {
                       emitBody(b, l, iv, args, true);
                     });
  scf::ForOp::create(builder, loc, vectorEnd, extent, one, ValueRange{},
                     [&](OpBuilder &b, Location l, Value iv, ValueRange args) {
                       emitBody(b, l, iv, args, false);
                     });
  op.erase();
}

// Max(abs(x)) has no negative-zero ordering ambiguity. Restrict reduction
// vectorization to an explicitly recognized abs producer and -infinity seed.
// Other reductions (especially floating-point sums) keep their original order.
static bool isAbsMax(linalg::ReduceOp op, AliasAnalysis &aliases) {
  if (!op.hasPureBufferSemantics() || op.getInputs().size() != 1 ||
      op.getInits().size() != 1 || op.getDimensions() != ArrayRef<int64_t>{0})
    return false;
  Value input = op.getInputs()[0], output = op.getInits()[0];
  auto inType = dyn_cast<MemRefType>(input.getType());
  auto outType = dyn_cast<MemRefType>(output.getType());
  if (!inType || !outType || !isContiguousVector(inType) ||
      !inType.getElementType().isF32() || outType.getRank() != 0 ||
      !outType.getElementType().isF32() || !aliases.alias(input, output).isNo())
    return false;
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  auto max = dyn_cast<arith::MaxNumFOp>(body.front());
  if (!max || max.getLhs() != body.getArgument(0) ||
      max.getRhs() != body.getArgument(1) ||
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] != max.getResult())
    return false;
  bool seed = false;
  for (Operation *prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
    if (auto fill = dyn_cast<linalg::FillOp>(prev)) {
      if (fill.getOutputs()[0] != output || seed)
        return false;
      auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
      if (!value || !value.getValue().isInfinity() || !value.getValue().isNegative())
        return false;
      seed = true;
      continue;
    }
    if (isa<memref::AllocOp, memref::AllocaOp, arith::ConstantOp>(prev))
      continue;
    auto abs = dyn_cast<linalg::GenericOp>(prev);
    if (!seed || !abs || abs.getNumDpsInputs() != 1 ||
        abs.getNumDpsInits() != 1 || abs.getDpsInits()[0] != input ||
        !canVectorize(abs, aliases, true))
      return false;
    Block &ab = abs.getRegion().front();
    auto operation = dyn_cast<math::AbsFOp>(ab.front());
    return ab.getOperations().size() == 2 && operation &&
           operation.getOperand() == ab.getArgument(0) &&
           cast<linalg::YieldOp>(ab.getTerminator()).getValues()[0] == operation.getResult();
  }
  return false;
}

static void vectorizeAbsMax(linalg::ReduceOp op, OpBuilder &b) {
  b.setInsertionPoint(op);
  Location l = op.getLoc();
  Value input = op.getInputs()[0], output = op.getInits()[0];
  Value zero = arith::ConstantIndexOp::create(b, l, 0);
  Value one = arith::ConstantIndexOp::create(b, l, 1);
  Value width = arith::ConstantIndexOp::create(b, l, vectorWidth);
  Value size = memref::DimOp::create(b, l, input, zero);
  Value rem = arith::RemUIOp::create(b, l, size, width);
  Value end = arith::SubIOp::create(b, l, size, rem);
  Value init = memref::LoadOp::create(b, l, output);
  auto vecType = VectorType::get({vectorWidth}, b.getF32Type());
  Value initVector = vector::BroadcastOp::create(b, l, vecType, init);
  auto loop = scf::ForOp::create(b, l, zero, end, width, ValueRange{initVector},
      [&](OpBuilder &nb, Location nl, Value iv, ValueRange args) {
        Value data = vector::LoadOp::create(nb, nl, vecType, input, ValueRange{iv});
        Value maximum = arith::MaxNumFOp::create(nb, nl, data, args[0]);
        scf::YieldOp::create(nb, nl, maximum);
      });
  Value maximum = vector::ReductionOp::create(
      b, l, vector::CombiningKind::MAXNUMF, loop.getResult(0), init);
  auto tail = scf::ForOp::create(b, l, end, size, one, ValueRange{maximum},
      [&](OpBuilder &nb, Location nl, Value iv, ValueRange args) {
        Value data = memref::LoadOp::create(nb, nl, input, ValueRange{iv});
        Value next = arith::MaxNumFOp::create(nb, nl, data, args[0]);
        scf::YieldOp::create(nb, nl, next);
      });
  memref::StoreOp::create(b, l, tail.getResult(0), output);
  op.erase();
}

struct VectorizeQuantizePass
    : PassWrapper<VectorizeQuantizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeQuantizePass)
  StringRef getArgument() const final { return "vectorize-quantize"; }
  StringRef getDescription() const final {
    return "Vectorize fused contiguous quantization and abs-max reduction without reciprocal approximation";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, math::MathDialect, memref::MemRefDialect,
             scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    AliasAnalysis aliases(getOperation());
    SmallVector<linalg::ReduceOp> reductions;
    getOperation().walk([&](linalg::ReduceOp op) {
      if (isAbsMax(op, aliases)) reductions.push_back(op);
    });
    OpBuilder b(&getContext());
    for (auto op : reductions) vectorizeAbsMax(op, b);
    SmallVector<linalg::GenericOp> generics;
    getOperation().walk([&](linalg::GenericOp op) {
      if (canVectorize(op, aliases, true)) generics.push_back(op);
    });
    for (auto op : generics) vectorize(op, b);
  }
};

struct VectorizeDequantizePass
    : PassWrapper<VectorizeDequantizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeDequantizePass)

  StringRef getArgument() const final { return "vectorize-dequantize"; }
  StringRef getDescription() const final {
    return "Vectorize contiguous i32-to-f32/multiply linalg generics in 16 "
           "lanes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    AliasAnalysis aliases(getOperation());
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp op) {
      if (canVectorize(op, aliases))
        candidates.push_back(op);
    });
    OpBuilder builder(&getContext());
    for (linalg::GenericOp op : candidates)
      vectorize(op, builder);
  }
};

} // namespace

namespace mlir::buddy {
void registerVectorizeQuantizePass() {
  PassRegistration<VectorizeQuantizePass>();
}
void registerVectorizeDequantizePass() {
  PassRegistration<VectorizeDequantizePass>();
}
} // namespace mlir::buddy
