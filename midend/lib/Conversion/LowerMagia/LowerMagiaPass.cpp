//===- LowerMagiaPass.cpp - MAGIA dialect to runtime calls ---------------===//
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
// The operations of the MAGIA dialect become calls into the host runtime,
// which hands the job to the mesh: magia_gemm(x, w, z, batch, m, n, k) with
// the matrices as addresses of contiguous FP16 rows, the matrices of a batch
// one after another.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Magia/MagiaDialect.h"
#include "Magia/MagiaOps.h"

using namespace mlir;
using namespace buddy;

namespace {

/// The address of the first element of FP16 matrices whose rows, and the
/// matrices of a batch, lie one after another, from any offset: a matrix of
/// its own or one of a pack of parameters. Strides the type does not give are
/// checked when the program runs.
Value address(PatternRewriter &rewriter, Location loc, Value matrix) {
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, matrix);
  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, matrix);
  int64_t rank = cast<MemRefType>(matrix.getType()).getRank();
  // Each stride is the size of what follows it.
  Value contiguous = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  Value size = arith::ConstantIndexOp::create(rewriter, loc, 1);
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    Value equal = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        metadata.getStrides()[dimension], size);
    contiguous = arith::AndIOp::create(rewriter, loc, contiguous, equal);
    size = arith::MulIOp::create(
        rewriter, loc, size,
        memref::DimOp::create(rewriter, loc, matrix, dimension));
  }
  cf::AssertOp::create(rewriter, loc, contiguous,
                       "the rows of a matrix of the mesh are not contiguous");
  Value bytes =
      arith::MulIOp::create(rewriter, loc, metadata.getOffset(),
                            arith::ConstantIndexOp::create(rewriter, loc, 2));
  Value first = arith::AddIOp::create(rewriter, loc, base, bytes);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    first);
}

Value size(PatternRewriter &rewriter, Location loc, Value matrix,
           unsigned dimension) {
  Value index = memref::DimOp::create(rewriter, loc, matrix, dimension);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    index);
}

class GemmLowering : public OpRewritePattern<magia::GemmOp> {
public:
  using OpRewritePattern<magia::GemmOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(magia::GemmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The matrices are the last two dimensions; a leading one is the batch.
    unsigned rows = cast<MemRefType>(op.getX().getType()).getRank() - 2;
    Value batch = rows ? size(rewriter, loc, op.getX(), 0)
                       : arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    SmallVector<Value> arguments = {address(rewriter, loc, op.getX()),
                                    address(rewriter, loc, op.getW()),
                                    address(rewriter, loc, op.getZ()),
                                    batch,
                                    size(rewriter, loc, op.getX(), rows),
                                    size(rewriter, loc, op.getX(), rows + 1),
                                    size(rewriter, loc, op.getW(), rows + 1)};
    func::CallOp::create(rewriter, loc, "magia_gemm", TypeRange{}, arguments);
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerMagiaPass
    : public PassWrapper<LowerMagiaPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMagiaPass);
  LowerMagiaPass() = default;
  LowerMagiaPass(const LowerMagiaPass &) {}
  StringRef getArgument() const final { return "lower-magia"; }
  StringRef getDescription() const final {
    return "lower the magia dialect to calls into the host runtime";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getBody(), module.getBody()->begin());
    Type word = builder.getI64Type();
    auto runtime = func::FuncOp::create(
        builder, module.getLoc(), "magia_gemm",
        builder.getFunctionType({word, word, word, word, word, word, word},
                                {}));
    runtime.setPrivate();
    RewritePatternSet patterns(&getContext());
    patterns.add<GemmLowering>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, cf::ControlFlowDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerLowerMagiaPass() { PassRegistration<LowerMagiaPass>(); }
} // namespace buddy
} // namespace mlir
