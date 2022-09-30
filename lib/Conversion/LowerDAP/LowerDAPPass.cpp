//====- LowerDAPPass.cpp - DAP Dialect Lowering Pass  ---------------------===//
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
// This file defines DAP dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class DAPFirLowering : public OpRewritePattern<dap::FirOp> {
public:
  using OpRewritePattern<dap::FirOp>::OpRewritePattern;

  explicit DAPFirLowering(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::FirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // auto ctx = op->getContext();
    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    rewriter.create<linalg::Conv1DOp>(loc, ValueRange{input, kernel},
                                      ValueRange{output});

    rewriter.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void populateLowerDAPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DAPFirLowering>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerDAPPass
//===----------------------------------------------------------------------===//

namespace {
class LowerDAPPass : public PassWrapper<LowerDAPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDAPPass)
  LowerDAPPass() = default;
  LowerDAPPass(const LowerDAPPass &) {}

  StringRef getArgument() const final { return "lower-dap"; }
  StringRef getDescription() const final { return "Lower DAP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dap::DAPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    AffineDialect, arith::ArithDialect>();
  }
};
} // end anonymous namespace.

void LowerDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, VectorDialect,
                         arith::ArithDialect, linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerDAPConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerDAPPass() { PassRegistration<LowerDAPPass>(); }
} // namespace buddy
} // namespace mlir
