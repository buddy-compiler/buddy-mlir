//===- MatmulAMX.cpp -----------------------------------------------------===//
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
// This file implements the conversion from linalg.matmul to AMX operations.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/AMX/AMXDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace mlir::amx;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Check if the matrix dimensions are compatible with AMX requirements
/// M, N must be multiples of 16; K must be a multiple of 32
/// For static shapes, check directly; for dynamic shapes, assume compatibility
bool isAMXCompatible(Value A, Value B, Value C) {
  auto AType = mlir::dyn_cast<MemRefType>(A.getType());
  auto BType = mlir::dyn_cast<MemRefType>(B.getType());
  auto CType = mlir::dyn_cast<MemRefType>(C.getType());

  if (!AType || !BType || !CType)
    return false;

  auto AShape = AType.getShape();
  auto BShape = BType.getShape();
  auto CShape = CType.getShape();

  if (AShape.size() != 2 || BShape.size() != 2 || CShape.size() != 2)
    return false;

  // For static shapes, check AMX constraints
  if (AType.hasStaticShape() && BType.hasStaticShape() &&
      CType.hasStaticShape()) {
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // Check AMX constraints: M, N multiples of 16; K multiple of 32
    bool dimensionsValid = (M % 16 == 0) && (N % 16 == 0) && (K % 32 == 0);
    bool shapesMatch = (BShape[0] == K) && (CShape[0] == M) && (CShape[1] == N);

    return dimensionsValid && shapesMatch;
  }

  // For dynamic shapes, assume they will be compatible at runtime
  // TODO: Add runtime checks
  return true;
}

/// Check if data types are compatible with AMX BF16 requirements
/// A, B should be bf16; C should be f32
bool isAMXDataTypeCompatible(Value A, Value B, Value C) {
  auto AType = mlir::dyn_cast<MemRefType>(A.getType());
  auto BType = mlir::dyn_cast<MemRefType>(B.getType());
  auto CType = mlir::dyn_cast<MemRefType>(C.getType());

  if (!AType || !BType || !CType)
    return false;

  Type AElemType = AType.getElementType();
  Type BElemType = BType.getElementType();
  Type CElemType = CType.getElementType();

  return AElemType.isBF16() && BElemType.isBF16() && CElemType.isF32();
}

/// Create a pre-packed version of matrix B for AMX-friendly tile loads
/// This function reorganizes B matrix so that each 16x32 block can be loaded
/// efficiently by amx.tile_load operations
Value createPackedBMatrix(OpBuilder &builder, Location loc, Value B, int64_t K,
                          int64_t N) {
  auto BType = mlir::cast<MemRefType>(B.getType());
  auto elementType = BType.getElementType();

  // Handle dynamic dimensions
  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticShape;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // For K dimension
  if (K == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    dynamicSizes.push_back(builder.create<memref::DimOp>(loc, B, c0));
  } else {
    staticShape.push_back(K);
  }

  // For N dimension
  if (N == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    dynamicSizes.push_back(builder.create<memref::DimOp>(loc, B, c1));
  } else {
    staticShape.push_back(N);
  }

  // Create a new memref for the packed B matrix with the same dimensions
  // The packing is done logically - the physical layout optimization
  // will be handled by later passes
  auto packedBType = MemRefType::get(staticShape, elementType);
  Value packedB =
      builder.create<memref::AllocOp>(loc, packedBType, dynamicSizes);

  // Get actual dimension sizes (handle both static and dynamic cases)
  Value cK, cN;
  if (K == ShapedType::kDynamic) {
    cK = builder.create<memref::DimOp>(loc, B, c0).getResult();
  } else {
    cK = builder.create<arith::ConstantIndexOp>(loc, K).getResult();
  }

  if (N == ShapedType::kDynamic) {
    cN = builder.create<memref::DimOp>(loc, B, c1).getResult();
  } else {
    cN = builder.create<arith::ConstantIndexOp>(loc, N).getResult();
  }

  // Simple copy for now - in a real implementation, this would be
  // a more sophisticated packing operation
  builder.create<scf::ForOp>(
      loc, c0, cK, c1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
        builder.create<scf::ForOp>(
            loc, c0, cN, c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
              Value val =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{i, j});
              builder.create<memref::StoreOp>(loc, val, packedB,
                                              ValueRange{i, j});
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });

  return packedB;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class MatmulAMXPattern : public RewritePattern {
public:
  explicit MatmulAMXPattern(MLIRContext *context)
      : RewritePattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto matmulOp = cast<linalg::MatmulOp>(op);

    // Get input operands A, B, C
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    // Check AMX compatibility
    if (!isAMXCompatible(A, B, C)) {
      return rewriter.notifyMatchFailure(
          op, "Matrix dimensions not compatible with AMX: M,N must be "
              "multiples of 16, K must be multiple of 32");
    }

    if (!isAMXDataTypeCompatible(A, B, C)) {
      return rewriter.notifyMatchFailure(
          op, "Data types not compatible with AMX BF16: A,B must be bf16, C "
              "must be f32");
    }

    // Additional validation for memref types
    auto AMemRefType = mlir::cast<MemRefType>(A.getType());
    auto BMemRefType = mlir::cast<MemRefType>(B.getType());
    auto CMemRefType = mlir::cast<MemRefType>(C.getType());

    if (AMemRefType.getRank() != 2 || BMemRefType.getRank() != 2 ||
        CMemRefType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Only 2D matrices are supported for AMX conversion");
    }

    // Get matrix dimensions
    auto AShape = AMemRefType.getShape();
    auto BShape = BMemRefType.getShape();
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value c32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);

    // Handle dynamic dimensions
    Value cM, cN, cK;
    if (M == ShapedType::kDynamic) {
      cM = rewriter.create<memref::DimOp>(loc, A, c0);
    } else {
      cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    }

    if (N == ShapedType::kDynamic) {
      cN = rewriter.create<memref::DimOp>(loc, B, c1);
    } else {
      cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    }

    if (K == ShapedType::kDynamic) {
      cK = rewriter.create<memref::DimOp>(loc, A, c1);
    } else {
      cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
    }

    // Create pre-packed B matrix for AMX-friendly tile loads
    Value Bpack = createPackedBMatrix(rewriter, loc, B, K, N);

    // Create AMX tile types
    auto bf16Type = rewriter.getBF16Type();
    auto f32Type = rewriter.getF32Type();
    auto tileTypeBF16 = TileType::get({16, 32}, bf16Type);
    auto tileTypeF32 = TileType::get({16, 16}, f32Type);

    // Generate AMX tile computation loops
    // Outer loops: iterate over M and N dimensions in 16x16 tiles
    rewriter.create<scf::ForOp>(
        loc, c0, cM, c16, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value m, ValueRange) {
          builder.create<scf::ForOp>(
              loc, c0, cN, c16, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
                // Initialize accumulator tile to zero
                Value zeroTile = builder.create<TileZeroOp>(loc, tileTypeF32);
                builder.create<TileStoreOp>(loc, C, ValueRange{m, n}, zeroTile);

                // Inner loop: iterate over K dimension in chunks of 32
                builder.create<scf::ForOp>(
                    loc, c0, cK, c32, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value k, ValueRange) {
                      // Load A tile: 16x32xbf16 from [%m, %k]
                      Value tA = builder.create<TileLoadOp>(
                          loc, tileTypeBF16, A, ValueRange{m, k});

                      // Load B tile (pre-packed): 16x32xbf16 from [%k, %n]
                      Value tB = builder.create<TileLoadOp>(
                          loc, tileTypeBF16, Bpack, ValueRange{k, n});

                      // Load current accumulator from C
                      Value tAcc = builder.create<TileLoadOp>(
                          loc, tileTypeF32, C, ValueRange{m, n});

                      // Perform tile multiplication with accumulation
                      Value tAcc2 = builder.create<TileMulFOp>(loc, tileTypeF32,
                                                               tA, tB, tAcc);

                      // Store result back to C
                      builder.create<TileStoreOp>(loc, C, ValueRange{m, n},
                                                  tAcc2);

                      builder.create<scf::YieldOp>(loc);
                    });
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    // Remove the original linalg.matmul operation
    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatmulAMXPass
//===----------------------------------------------------------------------===//

namespace {
class MatmulAMXPass
    : public PassWrapper<MatmulAMXPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulAMXPass)
  StringRef getArgument() const final { return "matmul-amx"; }
  StringRef getDescription() const final {
    return "Convert linalg.matmul to AMX operations";
  }
  MatmulAMXPass() = default;
  MatmulAMXPass(const MatmulAMXPass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    amx::AMXDialect>();
  }
};
} // end anonymous namespace

void MatmulAMXPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<MatmulAMXPattern>(context);

  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatmulAMXPass() { PassRegistration<MatmulAMXPass>(); }
} // namespace buddy
} // namespace mlir
