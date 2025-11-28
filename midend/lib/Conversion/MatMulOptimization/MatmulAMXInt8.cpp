//===- MatmulAMXInt8.cpp -------------------------------------------------===//
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
// This file implements the conversion from linalg.matmul to AMX INT8
// operations.
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

/// Check if the matrix dimensions are compatible with AMX INT8 requirements
/// M, N must be multiples of 16; K must be a multiple of 64
/// For static shapes, check directly; for dynamic shapes, assume compatibility
bool isAMXInt8Compatible(Value A, Value B, Value C) {
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

  // For static shapes, check AMX INT8 constraints
  if (AType.hasStaticShape() && BType.hasStaticShape() &&
      CType.hasStaticShape()) {
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // Check AMX INT8 constraints: M, N multiples of 16; K multiple of 64
    bool dimensionsValid = (M % 16 == 0) && (N % 16 == 0) && (K % 64 == 0);
    bool shapesMatch = (BShape[0] == K) && (CShape[0] == M) && (CShape[1] == N);

    return dimensionsValid && shapesMatch;
  }

  // For dynamic shapes, assume they will be compatible at runtime
  // TODO: Add runtime checks
  return true;
}

/// Check if data types are compatible with AMX INT8 requirements
/// A, B should be i8; C should be i32
bool isAMXInt8DataTypeCompatible(Value A, Value B, Value C) {
  auto AType = mlir::dyn_cast<MemRefType>(A.getType());
  auto BType = mlir::dyn_cast<MemRefType>(B.getType());
  auto CType = mlir::dyn_cast<MemRefType>(C.getType());

  if (!AType || !BType || !CType)
    return false;

  Type AElemType = AType.getElementType();
  Type BElemType = BType.getElementType();
  Type CElemType = CType.getElementType();

  // A, B must be i8; C must be i32
  return AElemType.isInteger(8) && BElemType.isInteger(8) &&
         CElemType.isInteger(32);
}

/// Create a VNNI-packed version of matrix B for AMX INT8 operations.
/// Input: B[K x N] in standard row-major layout
/// Output: B_packed[K/4 x N*4] in VNNI format
/// VNNI format for INT8: each row contains interleaved elements from 4
/// consecutive rows of B
///   B_packed[row_quad, 4*n+0] = B[4*row_quad+0, n]
///   B_packed[row_quad, 4*n+1] = B[4*row_quad+1, n]
///   B_packed[row_quad, 4*n+2] = B[4*row_quad+2, n]
///   B_packed[row_quad, 4*n+3] = B[4*row_quad+3, n]
Value createPackedBMatrixInt8(OpBuilder &builder, Location loc, Value B,
                              int64_t K, int64_t N) {
  auto BType = mlir::cast<MemRefType>(B.getType());
  auto elementType = BType.getElementType();

  // Handle dynamic dimensions
  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticShape;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = builder.create<arith::ConstantIndexOp>(loc, 2);
  Value c3 = builder.create<arith::ConstantIndexOp>(loc, 3);
  Value c4 = builder.create<arith::ConstantIndexOp>(loc, 4);

  // The packed buffer is [K/4 x N*4] in VNNI format for INT8
  // First dimension: K/4 (number of row quads)
  if (K == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimK = builder.create<memref::DimOp>(loc, B, c0);
    Value dimKDiv4 = builder.create<arith::DivUIOp>(loc, dimK, c4);
    dynamicSizes.push_back(dimKDiv4);
  } else {
    staticShape.push_back(K / 4);
  }

  // Second dimension: N*4 (interleaved columns)
  if (N == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimN = builder.create<memref::DimOp>(loc, B, c1);
    Value dimNMul4 = builder.create<arith::MulIOp>(loc, dimN, c4);
    dynamicSizes.push_back(dimNMul4);
  } else {
    staticShape.push_back(N * 4);
  }

  // Create a new memref for the VNNI-packed B matrix
  auto packedBType = MemRefType::get(staticShape, elementType);
  Value packedB =
      builder.create<memref::AllocOp>(loc, packedBType, dynamicSizes);

  // Get actual dimension sizes (handle both static and dynamic cases)
  Value cK, cN, cKDiv4;
  if (K == ShapedType::kDynamic) {
    cK = builder.create<memref::DimOp>(loc, B, c0);
    cKDiv4 = builder.create<arith::DivUIOp>(loc, cK, c4);
  } else {
    cK = builder.create<arith::ConstantIndexOp>(loc, K);
    cKDiv4 = builder.create<arith::ConstantIndexOp>(loc, K / 4);
  }

  if (N == ShapedType::kDynamic) {
    cN = builder.create<memref::DimOp>(loc, B, c1);
  } else {
    cN = builder.create<arith::ConstantIndexOp>(loc, N);
  }

  // Pack into VNNI format for INT8
  // For each row quad (row_quad = 0, 1, 2, ..., K/4-1):
  //   k0 = row_quad * 4
  //   k1 = row_quad * 4 + 1
  //   k2 = row_quad * 4 + 2
  //   k3 = row_quad * 4 + 3
  //   For each column n:
  //     B_packed[row_quad, 4*n+0] = B[k0, n]
  //     B_packed[row_quad, 4*n+1] = B[k1, n]
  //     B_packed[row_quad, 4*n+2] = B[k2, n]
  //     B_packed[row_quad, 4*n+3] = B[k3, n]
  builder.create<scf::ForOp>(
      loc, c0, cKDiv4, c1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value rowQuad, ValueRange) {
        // Compute k0, k1, k2, k3 = rowQuad * 4, rowQuad * 4 + 1/2/3
        Value k0 = builder.create<arith::MulIOp>(loc, rowQuad, c4);
        Value k1 = builder.create<arith::AddIOp>(loc, k0, c1);
        Value k2 = builder.create<arith::AddIOp>(loc, k0, c2);
        Value k3 = builder.create<arith::AddIOp>(loc, k0, c3);

        builder.create<scf::ForOp>(
            loc, c0, cN, c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
              // Load B[k0, n], B[k1, n], B[k2, n], B[k3, n]
              Value val0 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k0, n});
              Value val1 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k1, n});
              Value val2 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k2, n});
              Value val3 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k3, n});

              // Compute column indices: col0 = n*4, col1 = n*4+1, col2 = n*4+2,
              // col3 = n*4+3
              Value col0 = builder.create<arith::MulIOp>(loc, n, c4);
              Value col1 = builder.create<arith::AddIOp>(loc, col0, c1);
              Value col2 = builder.create<arith::AddIOp>(loc, col0, c2);
              Value col3 = builder.create<arith::AddIOp>(loc, col0, c3);

              // Store in VNNI format: interleaved
              builder.create<memref::StoreOp>(loc, val0, packedB,
                                              ValueRange{rowQuad, col0});
              builder.create<memref::StoreOp>(loc, val1, packedB,
                                              ValueRange{rowQuad, col1});
              builder.create<memref::StoreOp>(loc, val2, packedB,
                                              ValueRange{rowQuad, col2});
              builder.create<memref::StoreOp>(loc, val3, packedB,
                                              ValueRange{rowQuad, col3});

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

class MatmulAMXInt8Pattern : public RewritePattern {
public:
  explicit MatmulAMXInt8Pattern(MLIRContext *context)
      : RewritePattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto matmulOp = cast<linalg::MatmulOp>(op);

    // Get input operands A, B, C
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    // Check AMX INT8 compatibility
    if (!isAMXInt8Compatible(A, B, C)) {
      return rewriter.notifyMatchFailure(
          op, "Matrix dimensions not compatible with AMX INT8: M,N must be "
              "multiples of 16, K must be multiple of 64");
    }

    if (!isAMXInt8DataTypeCompatible(A, B, C)) {
      return rewriter.notifyMatchFailure(
          op, "Data types not compatible with AMX INT8: A,B must be i8, "
              "C must be i32");
    }

    // Additional validation for memref types
    auto AMemRefType = mlir::cast<MemRefType>(A.getType());
    auto BMemRefType = mlir::cast<MemRefType>(B.getType());
    auto CMemRefType = mlir::cast<MemRefType>(C.getType());

    if (AMemRefType.getRank() != 2 || BMemRefType.getRank() != 2 ||
        CMemRefType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Only 2D matrices are supported for AMX INT8 conversion");
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
    Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value c16 = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value c64 = rewriter.create<arith::ConstantIndexOp>(loc, 64);

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

    // Create pre-packed B matrix for AMX-friendly tile loads (INT8 VNNI format)
    Value Bpack = createPackedBMatrixInt8(rewriter, loc, B, K, N);

    // Create AMX tile types for INT8
    auto i8Type = rewriter.getIntegerType(8);
    auto i32Type = rewriter.getIntegerType(32);
    auto tileTypeI8A = TileType::get({16, 64}, i8Type);
    auto tileTypeI8B = TileType::get({16, 64}, i8Type);
    auto tileTypeI32 = TileType::get({16, 16}, i32Type);

    // Generate AMX INT8 tile computation loops
    // Outer loops: iterate over M and N dimensions in 16x16 tiles
    rewriter.create<scf::ForOp>(
        loc, c0, cM, c16, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value m, ValueRange) {
          builder.create<scf::ForOp>(
              loc, c0, cN, c16, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
                // Initialize accumulator tile to zero
                Value zeroTile = builder.create<TileZeroOp>(loc, tileTypeI32);
                builder.create<TileStoreOp>(loc, C, ValueRange{m, n}, zeroTile);

                // Inner loop: iterate over K dimension in chunks of 64
                builder.create<scf::ForOp>(
                    loc, c0, cK, c64, ValueRange{},
                    [&](OpBuilder &builder, Location loc, Value k, ValueRange) {
                      // Load A tile: 16x64xi8 from [m, k]
                      Value tA = builder.create<TileLoadOp>(loc, tileTypeI8A, A,
                                                            ValueRange{m, k});

                      // Load B tile from VNNI-packed format
                      // B_packed is [K/4 x N*4] in VNNI format
                      // For tile at (m, n, k), we need B[k:k+64, n:n+16]
                      // In B_packed, this is at [k/4:k/4+16, n*4:n*4+64]
                      Value kDiv4 = builder.create<arith::DivUIOp>(loc, k, c4);
                      Value nMul4 = builder.create<arith::MulIOp>(loc, n, c4);

                      // Load from B_packed[k/4, n*4] (16x64 tile)
                      Value tB = builder.create<TileLoadOp>(
                          loc, tileTypeI8B, Bpack, ValueRange{kDiv4, nMul4});

                      // Load current accumulator from C (i32)
                      Value tAcc = builder.create<TileLoadOp>(
                          loc, tileTypeI32, C, ValueRange{m, n});

                      // Perform tile multiplication with accumulation (INT8)
                      // Using signed x signed (default, no zext attributes)
                      Value tAcc2 = builder.create<TileMulIOp>(loc, tileTypeI32,
                                                               tA, tB, tAcc);

                      // Store result back to C (i32)
                      builder.create<TileStoreOp>(loc, C, ValueRange{m, n},
                                                  tAcc2);

                      builder.create<scf::YieldOp>(loc);
                    });
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    // Deallocate the packed B matrix
    rewriter.create<memref::DeallocOp>(loc, Bpack);

    // Remove the original linalg.matmul operation
    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// MatmulAMXInt8Pass
//===----------------------------------------------------------------------===//

namespace {
class MatmulAMXInt8Pass
    : public PassWrapper<MatmulAMXInt8Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulAMXInt8Pass)
  StringRef getArgument() const final { return "matmul-amx-int8"; }
  StringRef getDescription() const final {
    return "Convert linalg.matmul to AMX INT8 operations";
  }
  MatmulAMXInt8Pass() = default;
  MatmulAMXInt8Pass(const MatmulAMXInt8Pass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, arith::ArithDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    amx::AMXDialect>();
  }
};
} // end anonymous namespace

void MatmulAMXInt8Pass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<MatmulAMXInt8Pattern>(context);

  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerMatmulAMXInt8Pass() { PassRegistration<MatmulAMXInt8Pass>(); }
} // namespace buddy
} // namespace mlir
