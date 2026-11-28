//===- LowerLinalgToMagia.cpp - Linalg matrix products to MAGIA ----------===//
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
// The FP32 matrix products of a bufferized graph become general matrix
// products of the MAGIA mesh. RedMulE computes in FP16, so the operands are
// converted on the way in and the result on the way out. The rest runs on
// the host, Loom, whose erf and exp are polynomials in line: a call into the
// C library for each element of a GELU or a softmax costs it hundreds of
// cycles.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Magia/MagiaDialect.h"
#include "Magia/MagiaOps.h"

using namespace mlir;
using namespace buddy;

namespace {

/// The elements of `source` converted into `target` (FP32 to FP16 or back);
/// `transposed` reads a matrix `source` with its two dimensions exchanged.
void convert(PatternRewriter &rewriter, Location loc, Value source,
             Value target, bool transposed = false) {
  unsigned rank = cast<MemRefType>(source.getType()).getRank();
  AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  AffineMap exchanged = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0},
                                                     rewriter.getContext());
  SmallVector<AffineMap> maps = {transposed ? exchanged : identity, identity};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  linalg::GenericOp::create(
      rewriter, loc, ValueRange{source}, ValueRange{target}, maps, iterators,
      [](OpBuilder &builder, Location loc, ValueRange elements) {
        Type to = elements[1].getType();
        Value value =
            to.isF16()
                ? Value(arith::TruncFOp::create(builder, loc, to, elements[0]))
                : Value(arith::ExtFOp::create(builder, loc, to, elements[0]));
        linalg::YieldOp::create(builder, loc, value);
      });
}

/// A contiguous FP16 copy of the FP32 matrices `source`.
Value half(PatternRewriter &rewriter, Location loc, Value source,
           bool transposed = false) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<int64_t> shape(type.getShape());
  SmallVector<Value> dynamic;
  if (transposed)
    std::swap(shape[0], shape[1]);
  for (unsigned i = 0; i < shape.size(); ++i)
    if (ShapedType::isDynamic(shape[i]))
      dynamic.push_back(
          memref::DimOp::create(rewriter, loc, source, transposed ? 1 - i : i));
  Value target = memref::AllocOp::create(
      rewriter, loc, MemRefType::get(shape, rewriter.getF16Type()), dynamic);
  convert(rewriter, loc, source, target, transposed);
  return target;
}

/// The fill of zeros that last wrote `c` before `op`, in the same block, when
/// it is one: `c` itself, or the buffer `c` is a subview of. Nothing else
/// touches that buffer between the fill and `op` but subviews.
linalg::FillOp zeroFill(Operation *op, Value c) {
  Value buffer = c;
  if (auto view = c.getDefiningOp<memref::SubViewOp>())
    buffer = view.getSource();
  linalg::FillOp fill;
  for (Operation *user : buffer.getUsers()) {
    if (user->getBlock() != op->getBlock() || !user->isBeforeInBlock(op))
      continue;
    if (auto found = dyn_cast<linalg::FillOp>(user)) {
      if (fill && !fill->isBeforeInBlock(found))
        continue;
      fill = found;
    }
  }
  if (!fill || fill.getOutputs()[0] != buffer)
    return {};
  auto value = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto zero = value ? dyn_cast<FloatAttr>(value.getValue()) : FloatAttr();
  if (!zero || !zero.getValue().isZero())
    return {};
  for (Operation *user : buffer.getUsers())
    if (user != fill && user != op && !isa<memref::SubViewOp>(user) &&
        user->getBlock() == op->getBlock() && fill->isBeforeInBlock(user) &&
        user->isBeforeInBlock(op))
      return {};
  return fill;
}

/// Z of the product into C in FP16: zeros when a fill of zeros wrote C (the
/// fill is dead then if it is of C itself: the product writes all of C),
/// else C converted.
Value start(PatternRewriter &rewriter, Operation *op, Value c) {
  Location loc = op->getLoc();
  linalg::FillOp fill = zeroFill(op, c);
  if (!fill)
    return half(rewriter, loc, c);
  auto type = cast<MemRefType>(c.getType());
  SmallVector<Value> dynamic;
  for (unsigned i = 0; i < type.getRank(); ++i)
    if (type.isDynamicDim(i))
      dynamic.push_back(memref::DimOp::create(rewriter, loc, c, i));
  Value z = memref::AllocOp::create(
      rewriter, loc, MemRefType::get(type.getShape(), rewriter.getF16Type()),
      dynamic);
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getF16FloatAttr(0.0));
  linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{z});
  if (fill.getOutputs()[0] == c)
    rewriter.eraseOp(fill);
  return z;
}

/// C += A * B in FP32 becomes Z = Z + X * W in FP16 on the mesh. B is read as
/// [K, N], or as [N, K] the way a Linear keeps its weight. A B already in
/// FP16 is W itself: the mesh reads it in place. A C a fill of zeros wrote
/// starts Z at zero in FP16: converting it would read and write every element
/// to the same result; a fill of C itself is dead then, the product writes
/// all of C.
class MatmulLowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
    for (Value operand : {a, c}) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getElementType().isF32())
        return failure();
    }
    auto weight = dyn_cast<MemRefType>(b.getType());
    if (!weight)
      return failure();
    bool converted = weight.getElementType().isF16();
    if (!converted && !weight.getElementType().isF32())
      return failure();
    if (converted && op.hasUserDefinedMaps())
      return failure();
    bool transposed = false;
    if (op.hasUserDefinedMaps()) {
      AffineExpr m, n, k;
      bindDims(op.getContext(), m, n, k);
      SmallVector<AffineMap> weight = {
          AffineMap::get(3, 0, {m, k}, op.getContext()),
          AffineMap::get(3, 0, {n, k}, op.getContext()),
          AffineMap::get(3, 0, {m, n}, op.getContext())};
      if (op.getIndexingMapsArray() != weight)
        return failure();
      transposed = true;
    }
    Location loc = op.getLoc();
    Value x = half(rewriter, loc, a);
    Value w = converted ? b : half(rewriter, loc, b, transposed);
    Value z = start(rewriter, op, c);
    magia::GemmOp::create(rewriter, loc, x, w, z);
    convert(rewriter, loc, z, c);
    memref::DeallocOp::create(rewriter, loc, x);
    if (!converted)
      memref::DeallocOp::create(rewriter, loc, w);
    memref::DeallocOp::create(rewriter, loc, z);
    rewriter.eraseOp(op);
    return success();
  }
};

/// The products of a batch, one launch of the mesh for all of them: each tile
/// takes whole products. The matrices may have dynamic sizes.
class BatchMatmulLowering : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasUserDefinedMaps())
      return failure();
    Value a = op.getInputs()[0], b = op.getInputs()[1], c = op.getOutputs()[0];
    for (Value operand : {a, b, c}) {
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.getElementType().isF32())
        return failure();
    }
    Location loc = op.getLoc();
    Value x = half(rewriter, loc, a);
    Value w = half(rewriter, loc, b);
    Value z = start(rewriter, op, c);
    magia::GemmOp::create(rewriter, loc, x, w, z);
    convert(rewriter, loc, z, c);
    memref::DeallocOp::create(rewriter, loc, x);
    memref::DeallocOp::create(rewriter, loc, w);
    memref::DeallocOp::create(rewriter, loc, z);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerLinalgToMagiaPass
    : public PassWrapper<LowerLinalgToMagiaPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToMagiaPass);
  LowerLinalgToMagiaPass() = default;
  LowerLinalgToMagiaPass(const LowerLinalgToMagiaPass &) {}
  StringRef getArgument() const final { return "convert-linalg-to-magia"; }
  StringRef getDescription() const final {
    return "convert linalg matrix products to the magia dialect";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulLowering, BatchMatmulLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    RewritePatternSet approximations(&getContext());
    populateMathPolynomialApproximationPatterns(
        approximations, [](StringRef name) {
          return name == "math.erf" || name == "math.exp";
        });
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(approximations))))
      signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<magia::MagiaDialect, memref::MemRefDialect,
                    linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerLowerLinalgToMagiaPass() {
  PassRegistration<LowerLinalgToMagiaPass>();
}
} // namespace buddy
} // namespace mlir
