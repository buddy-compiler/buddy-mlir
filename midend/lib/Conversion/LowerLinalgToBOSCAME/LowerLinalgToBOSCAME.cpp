//=== LowerLinalgToBOSCAME.cpp - Linalg to BOSCAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to BOSCAME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.hasPureBufferSemantics())
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return failure();

    Type elemTypeA = AType.getElementType();
    Type elemTypeB = BType.getElementType();
    Type elemTypeC = CType.getElementType();

    if (elemTypeA != elemTypeB) {
      return rewriter.notifyMatchFailure(
          op, "Operand A and B must have the same type.");
    }

    int64_t tileM = 4, tileN = 4, tileK = 4;
    int64_t msetTypeImm = 32;

    // [1] (i8 * i8 -> i32)
    if (elemTypeA.isInteger(8) && elemTypeC.isInteger(32)) {
      tileK = 16;
      msetTypeImm = 8;
    }
    // [2] (f16/bf16 * f16/bf16 -> f32)
    else if ((elemTypeA.isF16() || elemTypeA.isBF16()) && elemTypeC.isF32()) {
      tileK = 8;
      msetTypeImm = 16;
    }
    // [3] (i16 * i16 -> i32)
    else if (elemTypeA.isInteger(16) && elemTypeC.isInteger(32)) {
      tileK = 8;
      msetTypeImm = 16;
    }
    // [4] (f32 * f32 -> f32)
    else if (elemTypeA.isF32() && elemTypeC.isF32()) {
      tileK = 4;
      msetTypeImm = 32;
    }
    // [5] (i32 * i32 -> i32)
    else if (elemTypeA.isInteger(32) && elemTypeC.isInteger(32)) {
      tileK = 4;
      msetTypeImm = 32;
    }
    // [6] (f64 * f64 -> f64)
    else if (elemTypeA.isF64() && elemTypeC.isF64()) {
      tileK = 2;
      msetTypeImm = 64;
    }
    // [7] (i4 * i4 -> i32)
    else if (elemTypeA.isInteger(4) && elemTypeC.isInteger(32)) {
      tileK = 32;
      msetTypeImm = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported mixed-precision combination.");
    }

    Value dimM = memref::DimOp::create(rewriter, loc, A, 0);
    Value dimK = memref::DimOp::create(rewriter, loc, A, 1);
    Value dimN = memref::DimOp::create(rewriter, loc, B, 1);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    auto calcCurrentSize = [&](Value bound, Value iv, int64_t step) {
      Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
      Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
      Value cmp = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, remain, stepVal);
      return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
    };

    Value currM = calcCurrentSize(dimM, ivM, tileM);
    Value currN = calcCurrentSize(dimN, ivN, tileN);
    Value currK = calcCurrentSize(dimK, ivK, tileK);

    Value currMI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currM);
    Value currNI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currN);
    Value currKI64 =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), currK);

    SmallVector<OpFoldResult> stridesAttr = {rewriter.getIndexAttr(1),
                                             rewriter.getIndexAttr(1)};
    Value subA = memref::SubViewOp::create(
        rewriter, loc, A, ArrayRef<OpFoldResult>{ivM, ivK},
        ArrayRef<OpFoldResult>{currM, currK}, stridesAttr);
    Value subB = memref::SubViewOp::create(
        rewriter, loc, B, ArrayRef<OpFoldResult>{ivK, ivN},
        ArrayRef<OpFoldResult>{currK, currN}, stridesAttr);
    Value subC = memref::SubViewOp::create(
        rewriter, loc, C, ArrayRef<OpFoldResult>{ivM, ivN},
        ArrayRef<OpFoldResult>{currM, currN}, stridesAttr);

    auto getRowStride = [&](Value subview) -> Value {
      auto meta =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, subview);
      Value strideElem = meta.getResult(4);

      auto memrefType = cast<MemRefType>(subview.getType());
      unsigned bytesPerElem = memrefType.getElementTypeBitWidth() / 8;
      Value bytesVal =
          arith::ConstantIndexOp::create(rewriter, loc, bytesPerElem);

      Value strideBytes =
          arith::MulIOp::create(rewriter, loc, strideElem, bytesVal);

      return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                        strideBytes);
    };
    Value strideA = getRowStride(subA);
    Value strideB = getRowStride(subB);
    Value strideC = getRowStride(subC);

    MSettypeiOp::create(rewriter, loc, rewriter.getI64Type(), msetTypeImm);
    MSettilemOp::create(rewriter, loc, rewriter.getI64Type(), currMI64);
    MSettilenOp::create(rewriter, loc, rewriter.getI64Type(), currNI64);
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(), currKI64);

    if (elemTypeC.isInteger(32)) {
      MsubWMmOp::create(rewriter, loc, 0, 0, 0);
    } else if (elemTypeC.isInteger(16)) {
      MsubHMmOp::create(rewriter, loc, 0, 0, 0);
    } else if (elemTypeC.isInteger(64)) {
      MsubDwMmOp::create(rewriter, loc, 0, 0, 0);
    }

    if (elemTypeA.isInteger(8)) {
      Mlae8mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe8mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isF16() || elemTypeA.isBF16() ||
               elemTypeA.isInteger(16)) {
      Mlae16mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe16mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isInteger(32) || elemTypeA.isF32()) {
      Mlae32mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe32mOp::create(rewriter, loc, 1, subB, strideB);
    } else if (elemTypeA.isInteger(64) || elemTypeA.isF64()) {
      Mlae64mOp::create(rewriter, loc, 0, subA, strideA);
      Mlbe64mOp::create(rewriter, loc, 1, subB, strideB);
    }

    if (elemTypeC.isInteger(32) && elemTypeA.isInteger(32)) {
      MmaWmmOp::create(rewriter, loc, 0, 0, 1);
    } else if (elemTypeC.isInteger(16) && elemTypeA.isInteger(16)) {
      MmaHmmOp::create(rewriter, loc, 0, 0, 1);
    } else if (elemTypeC.isInteger(64) && elemTypeA.isInteger(64)) {
      MmaDwmmOp::create(rewriter, loc, 0, 0, 1);
    }

    if (elemTypeC.isInteger(32) || elemTypeC.isF32()) {
      Msce32mOp::create(rewriter, loc, 0, subC, strideC);
    } else if (elemTypeC.isInteger(64) || elemTypeC.isF64()) {
      Msce64mOp::create(rewriter, loc, 0, subC, strideC);
    } else if (elemTypeC.isInteger(16) || elemTypeC.isF16()) {
      Msce16mOp::create(rewriter, loc, 0, subC, strideC);
    }

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to BOSCAME dialect operations.";
  }

  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<BOSCAMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToBOSCAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<MatmulToBOSCAMELowering>(context);

  ConversionTarget target(*context);
  target.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                         memref::MemRefDialect, scf::SCFDialect>();
  target.addIllegalOp<linalg::MatmulOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass>();
}
} // namespace buddy
} // namespace mlir
