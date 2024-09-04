//===- MatMulTransposeBVec.cpp --------------------------------------------===//
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
// This file implements the Matmul_TransposeB vectorization.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#include "Utils/Utils.h"

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class MatMul_TransposeB_VecPattern : public ConversionPattern{
public:
    explicit MatMul_TransposeB_VecPattern(MLIRContext *context,int64_t veSizeparam)
        : ConversionPattern(linalg::MatmulTransposeBOp::getOperationName(),1,context){
            veSize = veSizeparam;
        }
    
    LogicalResult
    matchAndRewrite(Operation *op,ArrayRef<Value> /*operands*/,
                ConversionPatternRewriter &rewriter) const override{
        auto loc = op->getLoc();
        auto ctx = op->getContext();
        //Get tensor input
        Value A = op->getOperand(0);
        Value B = op->getOperand(1);
        Value C = op->getOperand(2);
        ShapedType ATy = A.getType().cast<ShapedType>();
        Type eleTy = ATy.getElementType();
        VectorType vectorTy = mlir::VectorType::get({veSize}, eleTy);

        const Value c0=
            rewriter.create<arith::ConstantOp>(loc,rewriter.getIndexAttr(0));
        const Value c1=
            rewriter.create<arith::ConstantOp>(loc,rewriter.getIndexAttr(1));
        const Value step = 
            rewriter.create<arith::ConstantIndexOp>(loc,veSize);
        
        const Value aRow = rewriter.create<memref::DimOp>(loc,A,c0);
        const Value bRow = rewriter.create<memref::DimOp>(loc,B,c0);
        const Value bCol = rewriter.create<memref::DimOp>(loc,B,c1);
        
        SmallVector<Value,8> lowerBounds(2,c0);
        SmallVector<Value,8> uperBounds {aRow,bRow/*bCol*/};
        SmallVector<int64_t,8> steps{1,1};
        
        affine::buildAffineLoopNest(
            rewriter,loc,lowerBounds,uperBounds,steps,
            [&](OpBuilder &builder,Location loc,ValueRange ivs){
                Value sum_0 = buddy::insertZeroConstantOp(ctx, rewriter, loc, eleTy);
                // Create loop based on vector size.
                auto lbMap = mlir::AffineMap::get(0, 0, ctx);
                auto ubMap = mlir::AffineMap::get(0, 0, ctx);
                auto sum= builder.create<affine::AffineForOp>(
                loc,ValueRange{c0}, builder.getDimIdentityMap(),ValueRange{bCol}, builder.getDimIdentityMap()
                    ,veSize,ValueRange{sum_0},
                    [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
                    ValueRange itrArgs) {
                        Value aVec = builder.create<mlir::vector::LoadOp>(
                            loc,vectorTy,A,ValueRange{ivs[0],iv}
                        );
                        Value bVec = builder.create<mlir::vector::LoadOp>(
                            loc,vectorTy,B,ValueRange{ivs[1],iv}
                        );
                        Value resvec = builder.create<arith::MulFOp>(loc,aVec,bVec);
                        Value ans = builder.create<mlir::vector::ReductionOp>(
                            loc,mlir::vector::CombiningKind::ADD,resvec
                        );
                        Value sum = builder.create<arith::AddFOp>(loc,itrArgs[0],ans);
                        builder.create<affine::AffineYieldOp>(loc,sum);
                    }
                );
                builder.create<affine::AffineStoreOp>(loc,sum.getResult(0),C,ValueRange{ivs[0],ivs[1]});
            }
        );
        // clang-format on
        rewriter.eraseOp(op);
        return success();   
    }
private:
    int64_t veSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatMulVectorizationPass
//===----------------------------------------------------------------------===//

namespace{
    class MatMulTransposeBVecPass
        :public PassWrapper<MatMulTransposeBVecPass,OperationPass<ModuleOp>>{
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulTransposeBVecPass)
    StringRef getArgument() const final{ return "transpose_matmul_bvectorization"; }
    StringRef getDescription() const final { return "MatMul Vectorization second version.MatMul receive tensortype oprands."; }
    MatMulTransposeBVecPass() = default;
    MatMulTransposeBVecPass(const MatMulTransposeBVecPass &) {}
    void runOnOperation()   override;
    void getDependentDialects(DialectRegistry &registry) const override{
        registry.insert<linalg::LinalgDialect,scf::SCFDialect,
            affine::AffineDialect,VectorDialect>();
    }
    Option<int64_t> veSize{*this,"vec-size",
                            llvm::cl::desc("The size of vectorization"),
                            llvm::cl::init(32)};
                            
};
}

void MatMulTransposeBVecPass::runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                       scf::SCFDialect, memref::MemRefDialect, VectorDialect>();
    target.addLegalOp<ModuleOp,func::FuncOp,func::ReturnOp>();
    target.addLegalOp<linalg::FillOp>();

    RewritePatternSet patterns(context);
    patterns.add<MatMul_TransposeB_VecPattern>(context,veSize);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatMulTransposeBVecPass() {
  PassRegistration<MatMulTransposeBVecPass>();
}
} // namespace buddy
} // namespace mlir
