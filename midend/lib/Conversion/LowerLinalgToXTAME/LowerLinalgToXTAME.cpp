//====- LowerLinalgToXTAME.cpp - Linalg to XTAME Dialect Lowering Pass ----===//
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
// This file defines Linalg dialect lowering pass to XTAME dialect.
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

#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"

using namespace mlir;
using namespace buddy::xtame;

namespace {

class MatmulToXTAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return rewriter.notifyMatchFailure(matmulOp, "Operands must be memrefs");

    Type elemA = AType.getElementType();
    Type elemB = BType.getElementType();
    Type elemC = CType.getElementType();

    if (!elemA.isInteger(8) || !elemB.isInteger(8) || !elemC.isInteger(32)) {
      return rewriter.notifyMatchFailure(
          matmulOp,
          "Only supports 8-bit to 32-bit integer matrix multiplication");
    }

    bool isUnsignedA = elemA.isUnsignedInteger(8);
    bool isUnsignedB = elemB.isUnsignedInteger(8);

    int64_t dimM = AType.getShape()[0];
    int64_t dimK = AType.getShape()[1];
    int64_t dimN = BType.getShape()[1];

    if (ShapedType::isDynamic(dimM) || ShapedType::isDynamic(dimK) ||
        ShapedType::isDynamic(dimN)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Requires static dimensions for XTAME configuration");
    }

    int64_t bytesI8 = 1;
    int64_t bytesI32 = 4;
    int64_t strideA_val = dimK * bytesI8;
    int64_t strideB_val = dimN * bytesI8;
    int64_t strideC_val = dimN * bytesI32;

    Value byteStrideA =
        rewriter.create<arith::ConstantIntOp>(loc, strideA_val, 64);
    Value byteStrideB =
        rewriter.create<arith::ConstantIntOp>(loc, strideB_val, 64);
    Value byteStrideC =
        rewriter.create<arith::ConstantIntOp>(loc, strideC_val, 64);

    rewriter.create<ThMcfgmiOp>(loc, (uint64_t)dimM);
    rewriter.create<ThMcfgniOp>(loc, (uint64_t)dimN);
    rewriter.create<ThMcfgkiOp>(loc, (uint64_t)(dimK * bytesI8));
    rewriter.create<ThMzeroOp>(loc, 0);
    rewriter.create<ThMlde8Op>(loc, /*tile_reg=*/1, byteStrideA, A);
    rewriter.create<ThMldte8Op>(loc, /*tile_reg=*/2, byteStrideB, B);

    if (isUnsignedA && isUnsignedB) {
      rewriter.create<ThMmaccuWBOp>(loc, /*acc_reg=*/0, /*b_reg=*/2,
                                    /*a_reg=*/1);
    } else if (isUnsignedA && !isUnsignedB) {
      rewriter.create<ThMmaccusWBOp>(loc, /*acc_reg=*/0, /*b_reg=*/2,
                                     /*a_reg=*/1);
    } else if (!isUnsignedA && isUnsignedB) {
      rewriter.create<ThMmaccsuWBOp>(loc, /*acc_reg=*/0, /*b_reg=*/2,
                                     /*a_reg=*/1);
    } else {
      rewriter.create<ThMmaccWBOp>(loc, /*acc_reg=*/0, /*b_reg=*/2,
                                   /*a_reg=*/1);
    }

    rewriter.create<ThMcfgkiOp>(loc, (uint64_t)(dimN * bytesI32));
    rewriter.create<ThMste32Op>(loc, /*tile_reg=*/0, byteStrideC, C);

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToXTAMEPass
    : public PassWrapper<LowerLinalgToXTAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToXTAMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-xtame"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to XTAME dialect operations.";
  }

  LowerLinalgToXTAMEPass() = default;
  LowerLinalgToXTAMEPass(const LowerLinalgToXTAMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<XTAMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToXTAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);

  patterns.add<MatmulToXTAMELowering>(context);

  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToXTAMEPass() {
  PassRegistration<LowerLinalgToXTAMEPass>();
}
} // namespace buddy
} // namespace mlir
