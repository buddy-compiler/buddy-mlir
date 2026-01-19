//====- ConvNhwcFhwcOptimize.cpp----------------------------===//
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
// This file implements the Conv2DNhwcFhwcOp optimize.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

static std::pair<int64_t, int64_t>
getConvStrides(linalg::Conv2DNhwcFhwcOp convOp) {
  int64_t strideH, strideW;
  if (!convOp.getStrides()) {
    strideH = 1;
    strideW = 1;
  } else {
    SmallVector<int64_t> strides =
        llvm::to_vector(convOp.getStrides().getValues<int64_t>());
    strideH = strides.front();
    strideW = strides.back();
  }
  return {strideH, strideW};
}

static std::pair<int64_t, int64_t>
getConvDilations(linalg::Conv2DNhwcFhwcOp convOp) {
  int64_t dilHeight, dilWidth;
  if (!convOp.getDilations()) {
    dilHeight = 1;
    dilWidth = 1;
  } else {
    SmallVector<int64_t> dilations =
        llvm::to_vector(convOp.getDilations().getValues<int64_t>());
    dilHeight = dilations.front();
    dilWidth = dilations.back();
  }
  return {dilHeight, dilWidth};
}

static Value getDimValue(OpBuilder &b, Location loc, Value input,
                         ShapedType type, int64_t dim) {
  if (type.isDynamicDim(dim)) {
    return b.create<memref::DimOp>(loc, input, dim);
  }
  return b.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
}

static Value fmaOrMulAdd(OpBuilder &b, Location loc, Value lhs, Value rhs,
                         Value acc, Type elemTy) {
  if (isa<IntegerType>(elemTy)) {
    return b.create<arith::AddIOp>(loc, acc,
                                   b.create<arith::MulIOp>(loc, lhs, rhs));
  }
  if (isa<FloatType>(elemTy)) {
    return b.create<vector::FMAOp>(loc, lhs, rhs, acc);
  }
  llvm_unreachable("unsupported element type in fmaOrMulAdd");
}

namespace {

class ConvNhwcFhwcOptimizePattern : public ConversionPattern {
public:
  explicit ConvNhwcFhwcOptimizePattern(MLIRContext *context,
                                       int64_t vecSizeParam)
      : ConversionPattern(linalg::Conv2DNhwcFhwcOp::getOperationName(), 1,
                          context) {
    vecSize = vecSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto convOp = dyn_cast_or_null<mlir::linalg::Conv2DNhwcFhwcOp>(op);
    if (!convOp || !convOp.hasPureBufferSemantics()) {
      return failure();
    }
    Location loc = convOp->getLoc();
    Value input = convOp.getDpsInputs()[0];
    Value kernel = convOp.getDpsInputs()[1];
    Value output = convOp.getDpsInits()[0];

    ShapedType outputTy = cast<ShapedType>(output.getType());
    ShapedType kernelTy = cast<ShapedType>(kernel.getType());
    Type elemTy = kernelTy.getElementType();
    VectorType compVecTy = VectorType::get({vecSize}, elemTy);
    VectorType maskVecTy = VectorType::get({vecSize}, rewriter.getI1Type());

    const Value allZero = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(compVecTy, rewriter.getZeroAttr(elemTy)));
    const Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    const Value vecStep = rewriter.create<arith::ConstantIndexOp>(loc, vecSize);

    auto [strideH, strideW] = getConvStrides(convOp);
    auto [dilHeight, dilWidth] = getConvDilations(convOp);

    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);

    // Create map for vector mask generation and input indexing.
    AffineMap vectorMaskMap =
        AffineMap::get(2, 1, {-d0 + d1, s0}, rewriter.getContext());
    AffineMap inputMap = AffineMap::get(2, 0, {d0 * strideH + d1 * dilHeight},
                                        rewriter.getContext());

    const Value N = getDimValue(rewriter, loc, output, outputTy, 0);
    const Value OH = getDimValue(rewriter, loc, output, outputTy, 1);
    const Value OW = getDimValue(rewriter, loc, output, outputTy, 2);
    const Value OC = getDimValue(rewriter, loc, output, outputTy, 3);
    const Value FH = getDimValue(rewriter, loc, kernel, kernelTy, 1);
    const Value FW = getDimValue(rewriter, loc, kernel, kernelTy, 2);
    const Value IC = getDimValue(rewriter, loc, kernel, kernelTy, 3);

    auto forallOp = rewriter.create<scf::ForallOp>(
        loc, ArrayRef<OpFoldResult>({N, OH, OW, OC}), ValueRange{},
        std::nullopt);
    Value ivN = forallOp.getInductionVar(0);
    Value ivOH = forallOp.getInductionVar(1);
    Value ivOW = forallOp.getInductionVar(2);
    Value ivOC = forallOp.getInductionVar(3);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallOp.getBody());

    // clang-format off
    //
    // Insert three nested loops for FH, FW, IC:
    // %FHRes = scf.for %ivFH = 0 to FH step 1 iter_args(%accFH = allZero) {
    //  %FWRes = scf.for %ivFW = 0 to FW step 1 iter_args(%accFW = %accFH) {
    //    %ICRes = scf.for %ivIC = 0 to IC step vecSize iter_args(%accIC = %accFW) {
    //      %mask = vector.create_mask ...
    //      %inputData = vector.masked_load %input[...] {mask} , %zero
    //      %kernelData = vector.masked_load %kernel[...] {mask} , %zero
    //      %acc = fma/mul_add(%inputData, %kernelData, %accIC)
    //      scf.yield %acc
    //    }
    //    scf.yield %ICRes
    //  }
    //  scf.yield %FWRes
    // }
    // vector.reduce_add %FHRes
    // Load output data, add and store.
    //
    // clang-format on

    auto FHForOp =
        rewriter.create<scf::ForOp>(loc, zero, FH, one, ValueRange{allZero});
    rewriter.setInsertionPointToStart(FHForOp.getBody());
    auto FWForOp = rewriter.create<scf::ForOp>(
        loc, zero, FW, one, ValueRange{FHForOp.getRegionIterArgs()});
    rewriter.setInsertionPointToStart(FWForOp.getBody());
    auto ICForOp = rewriter.create<scf::ForOp>(
        loc, zero, IC, vecStep, ValueRange{FWForOp.getRegionIterArgs()});
    rewriter.setInsertionPointToStart(ICForOp.getBody());

    Value ivFH = FHForOp.getInductionVar();
    Value ivFW = FWForOp.getInductionVar();
    Value ivIC = ICForOp.getInductionVar();
    Value actualLoadLength = rewriter.create<affine::AffineMinOp>(
        loc, vectorMaskMap, ValueRange{ivIC, IC, vecStep});
    Value vectorMask = rewriter.create<vector::CreateMaskOp>(
        loc, maskVecTy, ValueShapeRange{actualLoadLength});

    // VectorLoad input and kernel.
    Value inputData = rewriter.create<vector::MaskedLoadOp>(
        loc, compVecTy, input,
        ValueRange{ivN,
                   rewriter.create<affine::AffineApplyOp>(
                       loc, inputMap, ValueRange{ivOH, ivFH}),
                   rewriter.create<affine::AffineApplyOp>(
                       loc, inputMap, ValueRange{ivOW, ivFW}),
                   ivIC},
        vectorMask, allZero);
    Value kernelData = rewriter.create<vector::MaskedLoadOp>(
        loc, compVecTy, kernel, ValueRange{ivOC, ivFH, ivFW, ivIC}, vectorMask,
        allZero);
    Value acc = fmaOrMulAdd(rewriter, loc, inputData, kernelData,
                            ICForOp.getRegionIterArgs()[0], elemTy);

    // Yield acc of each loop.
    rewriter.create<scf::YieldOp>(loc, acc);
    rewriter.setInsertionPointAfter(ICForOp);
    rewriter.create<scf::YieldOp>(loc, ICForOp->getResult(0));
    rewriter.setInsertionPointAfter(FWForOp);
    rewriter.create<scf::YieldOp>(loc, FWForOp->getResult(0));
    rewriter.setInsertionPointAfter(FHForOp);

    // Compute reduction add, load output, add and store.
    Value reduceAdd = rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::ADD, FHForOp->getResult(0));
    Value outputData = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{ivN, ivOH, ivOW, ivOC});
    if (isa<IntegerType>(elemTy)) {
      outputData = rewriter.create<arith::AddIOp>(loc, outputData, reduceAdd);
    } else {
      outputData = rewriter.create<arith::AddFOp>(loc, outputData, reduceAdd);
    }
    rewriter.create<memref::StoreOp>(loc, outputData, output,
                                     ValueRange{ivN, ivOH, ivOW, ivOC});
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t vecSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvNhwcFhwcOptimizePass
//===----------------------------------------------------------------------===//

namespace {
class ConvNhwcFhwcOptimizePass
    : public PassWrapper<ConvNhwcFhwcOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvNhwcFhwcOptimizePass)
  StringRef getArgument() const final { return "conv-nhwc-fhwc-optimize"; }
  StringRef getDescription() const final {
    return "Conv2d NHWC FHWC optimize.";
  }
  ConvNhwcFhwcOptimizePass() = default;
  ConvNhwcFhwcOptimizePass(const ConvNhwcFhwcOptimizePass &) {}
  explicit ConvNhwcFhwcOptimizePass(int64_t vecSizeParam) {
    vecSize = vecSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    affine::AffineDialect, VectorDialect>();
  }

  Option<int64_t> vecSize{*this, "vec-size", llvm::cl::desc("Vector size."),
                          llvm::cl::init(16)};
};
} // end anonymous namespace.

void ConvNhwcFhwcOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvNhwcFhwcOptimizePattern>(context, vecSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvNhwcFhwcOptimizePass() {
  PassRegistration<ConvNhwcFhwcOptimizePass>();
}
} // namespace buddy
} // namespace mlir
