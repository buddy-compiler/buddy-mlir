//===- LinalgTransposeTile.cpp --------------------------------------------===//
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
// This file implements a tiled lowering of memref linalg.transpose ops.
//
// The naive lowering of linalg.transpose performs element-wise memref
// load/store with the two swapped dimensions strided by full row length,
// which thrashes caches for large matrices. This pass instead rewrites a
// single-swap permutation into an scf.parallel tile loop over T x T tiles,
// so that all loads and stores of a tile stay within T cache lines.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <functional>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class LinalgTransposeTilePattern : public ConversionPattern {
public:
  explicit LinalgTransposeTilePattern(MLIRContext *context,
                                      int64_t tileSizeParam)
      : ConversionPattern(linalg::TransposeOp::getOperationName(), 1, context) {
    tileSize = tileSizeParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto permutationArrayAttr =
        mlir::cast<DenseI64ArrayAttr>(
            op->getAttr(rewriter.getStringAttr("permutation")))
            .asArrayRef();

    Value in = op->getOperand(0);
    Value out = op->getOperand(1);

    auto inType = dyn_cast<MemRefType>(in.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inType || !outType)
      return failure();
    int64_t rank = inType.getRank();
    if (rank != static_cast<int64_t>(permutationArrayAttr.size()))
      return failure();

    // Locate the two swapped dimensions. Only single swaps are supported
    // (e.g. [1, 0] and [0, 2, 1, 3]).
    int64_t a = -1;
    int64_t b = -1;
    for (int64_t i = 0; i < rank; ++i) {
      if (permutationArrayAttr[i] == i)
        continue;
      if (a == -1) {
        a = i;
      } else if (b == -1) {
        b = i;
      } else {
        return failure();
      }
    }
    if (a == -1 || b == -1 || permutationArrayAttr[a] != b ||
        permutationArrayAttr[b] != a)
      return failure();

    auto loc = op->getLoc();

    Value c0 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    Value c1 =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
    Value cT = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(tileSize));

    // Runtime dimensions of the input.
    SmallVector<Value> dims;
    for (int64_t i = 0; i < rank; ++i)
      dims.push_back(memref::DimOp::create(rewriter, loc, in, i));

    // Parallel loop over the tiles of the second swapped dimension (b).
    auto parallelLoop = scf::ParallelOp::create(
        rewriter, loc, ValueRange{c0}, ValueRange{dims[b]}, ValueRange{cT});
    rewriter.setInsertionPointToStart(parallelLoop.getBody());
    Value ivBTile = parallelLoop.getInductionVars()[0];
    Value ubB = createMinBound(rewriter, loc, ivBTile, dims[b], cT);

    // Sequential tile loop over the first swapped dimension (a).
    auto aTileLoop = scf::ForOp::create(rewriter, loc, c0, dims[a], cT);
    rewriter.setInsertionPointToStart(aTileLoop.getBody());
    Value ivATile = aTileLoop.getInductionVar();
    Value ubA = createMinBound(rewriter, loc, ivATile, dims[a], cT);

    // Inner loop over the first swapped dimension.
    auto aLoop = scf::ForOp::create(rewriter, loc, ivATile, ubA, c1);
    rewriter.setInsertionPointToStart(aLoop.getBody());
    Value ivA = aLoop.getInductionVar();

    // Inner loop over the second swapped dimension.
    auto bLoop = scf::ForOp::create(rewriter, loc, ivBTile, ubB, c1);
    rewriter.setInsertionPointToStart(bLoop.getBody());
    Value ivB = bLoop.getInductionVar();

    // Loops over the non-swapped dimensions, innermost for locality.
    SmallVector<Value> srcIdx(rank);
    std::function<void(int64_t)> emitNonSwappedLoops =
        [&](int64_t dim) -> void {
      if (dim == rank) {
        // Innermost body: assemble destination indices and copy.
        SmallVector<Value> dstIdx(rank);
        for (int64_t i = 0; i < rank; ++i)
          dstIdx[permutationArrayAttr[i]] = srcIdx[i];
        Value loaded =
            memref::LoadOp::create(rewriter, loc, in, ValueRange{srcIdx});
        memref::StoreOp::create(rewriter, loc, loaded, out, ValueRange{dstIdx});
        return;
      }
      if (dim == a || dim == b) {
        srcIdx[dim] = (dim == a) ? ivA : ivB;
        emitNonSwappedLoops(dim + 1);
        return;
      }
      auto loop = scf::ForOp::create(rewriter, loc, c0, dims[dim], c1);
      rewriter.setInsertionPointToStart(loop.getBody());
      srcIdx[dim] = loop.getInductionVar();
      emitNonSwappedLoops(dim + 1);
      rewriter.setInsertionPointAfter(loop);
    };
    emitNonSwappedLoops(0);

    rewriter.setInsertionPointAfter(parallelLoop);
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Returns (start + min(dim - start, tileSize)).
  Value createMinBound(OpBuilder &builder, Location loc, Value start, Value dim,
                       Value cT) const {
    Value diff = arith::SubIOp::create(builder, loc, dim, start);
    Value clipped = arith::MinSIOp::create(builder, loc, diff, cT);
    Value ub = arith::AddIOp::create(builder, loc, start, clipped);
    return ub;
  }

  int64_t tileSize;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LinalgTransposeTilePass
//===----------------------------------------------------------------------===//

namespace {
class LinalgTransposeTilePass
    : public PassWrapper<LinalgTransposeTilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgTransposeTilePass)
  StringRef getArgument() const final { return "linalg-transpose-tile"; }
  StringRef getDescription() const final {
    return "Tiled lowering of memref linalg.transpose with a single swap.";
  }
  LinalgTransposeTilePass() = default;
  LinalgTransposeTilePass(const LinalgTransposeTilePass &) {}
  explicit LinalgTransposeTilePass(int64_t tileSizeParam) {
    tileSize = tileSizeParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    memref::MemRefDialect, arith::ArithDialect>();
  }

  Option<int64_t> tileSize{*this, "tile-size",
                           llvm::cl::desc("Transpose tile size."),
                           llvm::cl::init(8)};
};
} // end anonymous namespace.

void LinalgTransposeTilePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         memref::MemRefDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  // Transposes that the pattern cannot handle (non-memref operands or
  // permutations that are not a single swap) stay untouched.
  target.addDynamicallyLegalOp<linalg::TransposeOp>([](linalg::TransposeOp op) {
    auto permutationArrayAttr =
        mlir::cast<DenseI64ArrayAttr>(op->getAttr("permutation")).asArrayRef();
    if (!dyn_cast<MemRefType>(op.getOperand(0).getType()) ||
        !dyn_cast<MemRefType>(op.getOperand(1).getType()))
      return true;
    if (static_cast<int64_t>(permutationArrayAttr.size()) !=
        dyn_cast<MemRefType>(op.getOperand(0).getType()).getRank())
      return true;
    int64_t a = -1;
    int64_t b = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(permutationArrayAttr.size());
         ++i) {
      if (permutationArrayAttr[i] == i)
        continue;
      if (a == -1) {
        a = i;
      } else if (b == -1) {
        b = i;
      } else {
        return true;
      }
    }
    if (a == -1 || b == -1 || permutationArrayAttr[a] != b ||
        permutationArrayAttr[b] != a)
      return true;
    // Single-swap memref transpose: illegal here, handled by the pattern.
    return false;
  });

  RewritePatternSet patterns(context);
  patterns.add<LinalgTransposeTilePattern>(context, tileSize);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLinalgTransposeTilePass() {
  PassRegistration<LinalgTransposeTilePass>();
}
} // namespace buddy
} // namespace mlir
