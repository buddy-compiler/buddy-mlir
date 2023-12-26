//====- DAPVectorization.cpp - DAP Dialect Vectorization Pass  ------------===//
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
// This file defines DAP dialect vectorization pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "DAP/DAPDialect.h"
#include "DAP/DAPOps.h"
#include "Utils/DAPUtils.h"
#include <optional>

using namespace mlir;
using namespace buddy;
using namespace vector;
using namespace mlir::arith;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class DAPIirVectorization : public OpRewritePattern<dap::IirOp> {
public:
  using OpRewritePattern<dap::IirOp>::OpRewritePattern;

  explicit DAPIirVectorization(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(dap::IirOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    Value input = op->getOperand(0);
    Value kernel = op->getOperand(1);
    Value output = op->getOperand(2);

    Value c0 = rewriter.create<ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<ConstantIndexOp>(loc, 2);
    Value c4 = rewriter.create<ConstantIndexOp>(loc, 4);
    Value c5 = rewriter.create<ConstantIndexOp>(loc, 5);
    Value c8 = rewriter.create<ConstantIndexOp>(loc, 8);
    Value c16 = rewriter.create<ConstantIndexOp>(loc, 16);
    Value c32 = rewriter.create<ConstantIndexOp>(loc, 32);

    Value N = rewriter.create<memref::DimOp>(loc, input, c0);
    Value filterSize = rewriter.create<memref::DimOp>(loc, kernel, c0);

    FloatType f32 = FloatType::getF32(ctx);
    Value f0 = rewriter.create<ConstantFloatOp>(loc, APFloat(0.0f), f32);
    Value f1 = rewriter.create<ConstantFloatOp>(loc, APFloat(1.0f), f32);

    Value cond4 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c4);
    Value cond8 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c8);
    Value cond16 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c16);
    Value cond32 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::ule, filterSize, c32);

    // clang-format off
    rewriter.create<scf::IfOp>(loc, cond4,
    /*thenBuilder=*/
    [&](OpBuilder &builder, Location loc) {
        dap::iirVectorizationProcess(builder, loc, 4, f32, f0, f1, c0, c1, c2, c4, c5,
                                     filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2},
                                     N, input, output);
        
        builder.create<scf::YieldOp>(loc);
    },
    /*elseBuilder=*/
    [&](OpBuilder &builder, Location loc) {
        builder.create<scf::IfOp>(loc, cond8,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc){
            dap::iirVectorizationProcess(builder, loc, 8, f32, f0, f1, c0, c1, c2, c4, c5,
                                         filterSize, kernel, 
                                         ArrayRef<int64_t>{0, 0, 1, 2, 3, 4, 5, 6}, N, 
                                         input, output);
            
            builder.create<scf::YieldOp>(loc);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
            builder.create<scf::IfOp>(loc, cond16,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc){
                dap::iirVectorizationProcess(builder, loc, 16, f32, f0, f1, c0, c1, c2, c4, c5,
                                             filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2, 
                                             3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, N, 
                                             input, output);
                
                builder.create<scf::YieldOp>(loc);
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
                builder.create<scf::IfOp>(loc, cond32,
                /*thenBuilder=*/
                [&](OpBuilder &builder, Location loc){
                    dap::iirVectorizationProcess(builder, loc, 32, f32, f0, f1, c0, c1, c2, c4, c5,
                                                 filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2, 
                                                 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                 30}, N, input, output);
                    
                    builder.create<scf::YieldOp>(loc);
                },
                /*elseBuilder=*/
                [&](OpBuilder &builder, Location loc) {
                    dap::iirVectorizationProcess(builder, loc, 64, f32, f0, f1, c0, c1, c2, c4, c5,
                                                 filterSize, kernel, ArrayRef<int64_t>{0, 0, 1, 2, 3, 4,
                                                 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
                                                 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
                                                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                                 62}, N, input, output);
                    
                    builder.create<scf::YieldOp>(loc);
                }
                );
                builder.create<scf::YieldOp>(loc);
            });

            builder.create<scf::YieldOp>(loc);
        });

        builder.create<scf::YieldOp>(loc);
    });
    // clang-format on

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVectorizeDAPConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DAPIirVectorization>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// VectorizeDAPPass
//===----------------------------------------------------------------------===//

namespace {
class VectorizeDAPPass
    : public PassWrapper<VectorizeDAPPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeDAPPass)
  VectorizeDAPPass() = default;
  VectorizeDAPPass(const VectorizeDAPPass &) {}

  StringRef getArgument() const final { return "vectorize-dap"; }
  StringRef getDescription() const final { return "Vectorize DAP Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<buddy::dap::DAPDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, VectorDialect,
                    affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect>();
  }
};
} // end anonymous namespace.

void VectorizeDAPPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      affine::AffineDialect,
      scf::SCFDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      VectorDialect,
      arith::ArithDialect,
      linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  // clang-format on

  RewritePatternSet patterns(context);
  populateVectorizeDAPConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerDAPVectorizePass() { PassRegistration<VectorizeDAPPass>(); }
} // namespace buddy
} // namespace mlir
