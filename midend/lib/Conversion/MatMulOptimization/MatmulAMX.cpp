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
// Supports both BF16 and INT8 data types with automatic detection.
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
// Data Type Enum and Detection
//===----------------------------------------------------------------------===//

namespace {

/// AMX supported data types
enum class AMXDataType { BF16, INT8, Unsupported };

/// Detect the AMX data type from matrix operands
AMXDataType detectAMXDataType(Value A, Value B, Value C) {
  auto AType = mlir::dyn_cast<MemRefType>(A.getType());
  auto BType = mlir::dyn_cast<MemRefType>(B.getType());
  auto CType = mlir::dyn_cast<MemRefType>(C.getType());

  if (!AType || !BType || !CType)
    return AMXDataType::Unsupported;

  Type AElemType = AType.getElementType();
  Type BElemType = BType.getElementType();
  Type CElemType = CType.getElementType();

  // BF16: A=bf16, B=bf16, C=f32 or bf16
  if (AElemType.isBF16() && BElemType.isBF16() &&
      (CElemType.isF32() || CElemType.isBF16())) {
    return AMXDataType::BF16;
  }

  // INT8: A=i8, B=i8, C=i32
  if (AElemType.isInteger(8) && BElemType.isInteger(8) &&
      CElemType.isInteger(32)) {
    return AMXDataType::INT8;
  }

  return AMXDataType::Unsupported;
}

//===----------------------------------------------------------------------===//
// Dimension Validation
//===----------------------------------------------------------------------===//

/// Check if matrix dimensions are compatible with AMX BF16 requirements
/// M, N must be multiples of 16; K must be a multiple of 32
bool isAMXBF16DimensionsValid(Value A, Value B, Value C) {
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

  // For static shapes, check AMX BF16 constraints
  if (AType.hasStaticShape() && BType.hasStaticShape() &&
      CType.hasStaticShape()) {
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // Check AMX BF16 constraints: M, N multiples of 16; K multiple of 32
    bool dimensionsValid = (M % 16 == 0) && (N % 16 == 0) && (K % 32 == 0);
    bool shapesMatch = (BShape[0] == K) && (CShape[0] == M) && (CShape[1] == N);

    return dimensionsValid && shapesMatch;
  }

  // For dynamic shapes, assume they will be compatible at runtime
  return true;
}

/// Check if matrix dimensions are compatible with AMX INT8 requirements
/// M, N must be multiples of 16; K must be a multiple of 64
bool isAMXInt8DimensionsValid(Value A, Value B, Value C) {
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
  return true;
}

//===----------------------------------------------------------------------===//
// VNNI Packing Functions
//===----------------------------------------------------------------------===//

/// Create a VNNI-packed version of matrix B for AMX BF16 operations.
/// Input: B[K x N] in standard row-major layout
/// Output: B_packed[K/2 x N*2] in VNNI format
/// VNNI format: each row contains interleaved elements from 2 consecutive rows
Value createPackedBMatrixBF16(OpBuilder &builder, Location loc, Value B,
                              int64_t K, int64_t N) {
  auto BType = mlir::cast<MemRefType>(B.getType());
  auto elementType = BType.getElementType();

  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticShape;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = builder.create<arith::ConstantIndexOp>(loc, 2);

  // The packed buffer is [K/2 x N*2] in VNNI format
  if (K == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimK = builder.create<memref::DimOp>(loc, B, c0);
    Value dimKDiv2 = builder.create<arith::DivUIOp>(loc, dimK, c2);
    dynamicSizes.push_back(dimKDiv2);
  } else {
    staticShape.push_back(K / 2);
  }

  if (N == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimN = builder.create<memref::DimOp>(loc, B, c1);
    Value dimNMul2 = builder.create<arith::MulIOp>(loc, dimN, c2);
    dynamicSizes.push_back(dimNMul2);
  } else {
    staticShape.push_back(N * 2);
  }

  auto packedBType = MemRefType::get(staticShape, elementType);
  Value packedB =
      builder.create<memref::AllocOp>(loc, packedBType, dynamicSizes);

  Value cK, cN, cKDiv2;
  if (K == ShapedType::kDynamic) {
    cK = builder.create<memref::DimOp>(loc, B, c0);
    cKDiv2 = builder.create<arith::DivUIOp>(loc, cK, c2);
  } else {
    cK = builder.create<arith::ConstantIndexOp>(loc, K);
    cKDiv2 = builder.create<arith::ConstantIndexOp>(loc, K / 2);
  }

  if (N == ShapedType::kDynamic) {
    cN = builder.create<memref::DimOp>(loc, B, c1);
  } else {
    cN = builder.create<arith::ConstantIndexOp>(loc, N);
  }

  // Pack into VNNI format for BF16
  builder.create<scf::ForOp>(
      loc, c0, cKDiv2, c1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value rowPair, ValueRange) {
        Value k0 = builder.create<arith::MulIOp>(loc, rowPair, c2);
        Value k1 = builder.create<arith::AddIOp>(loc, k0, c1);

        builder.create<scf::ForOp>(
            loc, c0, cN, c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
              Value val0 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k0, n});
              Value val1 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k1, n});

              Value col0 = builder.create<arith::MulIOp>(loc, n, c2);
              Value col1 = builder.create<arith::AddIOp>(loc, col0, c1);

              builder.create<memref::StoreOp>(loc, val0, packedB,
                                              ValueRange{rowPair, col0});
              builder.create<memref::StoreOp>(loc, val1, packedB,
                                              ValueRange{rowPair, col1});

              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });

  return packedB;
}

/// Create a VNNI-packed version of matrix B for AMX INT8 operations.
/// Input: B[K x N] in standard row-major layout
/// Output: B_packed[K/4 x N*4] in VNNI format
Value createPackedBMatrixInt8(OpBuilder &builder, Location loc, Value B,
                              int64_t K, int64_t N) {
  auto BType = mlir::cast<MemRefType>(B.getType());
  auto elementType = BType.getElementType();

  SmallVector<Value> dynamicSizes;
  SmallVector<int64_t> staticShape;

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c2 = builder.create<arith::ConstantIndexOp>(loc, 2);
  Value c3 = builder.create<arith::ConstantIndexOp>(loc, 3);
  Value c4 = builder.create<arith::ConstantIndexOp>(loc, 4);

  // The packed buffer is [K/4 x N*4] in VNNI format for INT8
  if (K == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimK = builder.create<memref::DimOp>(loc, B, c0);
    Value dimKDiv4 = builder.create<arith::DivUIOp>(loc, dimK, c4);
    dynamicSizes.push_back(dimKDiv4);
  } else {
    staticShape.push_back(K / 4);
  }

  if (N == ShapedType::kDynamic) {
    staticShape.push_back(ShapedType::kDynamic);
    Value dimN = builder.create<memref::DimOp>(loc, B, c1);
    Value dimNMul4 = builder.create<arith::MulIOp>(loc, dimN, c4);
    dynamicSizes.push_back(dimNMul4);
  } else {
    staticShape.push_back(N * 4);
  }

  auto packedBType = MemRefType::get(staticShape, elementType);
  Value packedB =
      builder.create<memref::AllocOp>(loc, packedBType, dynamicSizes);

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
  builder.create<scf::ForOp>(
      loc, c0, cKDiv4, c1, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value rowQuad, ValueRange) {
        Value k0 = builder.create<arith::MulIOp>(loc, rowQuad, c4);
        Value k1 = builder.create<arith::AddIOp>(loc, k0, c1);
        Value k2 = builder.create<arith::AddIOp>(loc, k0, c2);
        Value k3 = builder.create<arith::AddIOp>(loc, k0, c3);

        builder.create<scf::ForOp>(
            loc, c0, cN, c1, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
              Value val0 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k0, n});
              Value val1 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k1, n});
              Value val2 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k2, n});
              Value val3 =
                  builder.create<memref::LoadOp>(loc, B, ValueRange{k3, n});

              Value col0 = builder.create<arith::MulIOp>(loc, n, c4);
              Value col1 = builder.create<arith::AddIOp>(loc, col0, c1);
              Value col2 = builder.create<arith::AddIOp>(loc, col0, c2);
              Value col3 = builder.create<arith::AddIOp>(loc, col0, c3);

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

//===----------------------------------------------------------------------===//
// AMX Tile Computation - BF16
//===----------------------------------------------------------------------===//

/// Rewrite linalg.matmul to AMX BF16 operations
LogicalResult rewriteMatmulBF16(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter, Value A, Value B,
                                Value C) {
  auto loc = matmulOp.getLoc();

  auto AMemRefType = mlir::cast<MemRefType>(A.getType());
  auto BMemRefType = mlir::cast<MemRefType>(B.getType());
  auto CMemRefType = mlir::cast<MemRefType>(C.getType());

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
  Value Bpack = createPackedBMatrixBF16(rewriter, loc, B, K, N);

  // Create AMX tile types
  auto bf16Type = rewriter.getBF16Type();
  auto f32Type = rewriter.getF32Type();
  auto tileTypeBF16A = TileType::get({16, 32}, bf16Type);
  auto tileTypeBF16B = TileType::get({16, 32}, bf16Type);
  auto tileTypeF32 = TileType::get({16, 16}, f32Type);

  // Check if output is bf16 (need conversion) or f32 (direct)
  Type CElemType = CMemRefType.getElementType();
  bool needsBF16Conversion = CElemType.isBF16();

  // If output is bf16, create temporary f32 buffer for AMX computation
  Value CTemp = C;
  if (needsBF16Conversion) {
    auto CShape = CMemRefType.getShape();
    auto f32MemRefType = MemRefType::get(CShape, f32Type);
    CTemp = rewriter.create<memref::AllocOp>(loc, f32MemRefType);
  }

  // Generate AMX tile computation loops
  rewriter.create<scf::ForOp>(
      loc, c0, cM, c16, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value m, ValueRange) {
        builder.create<scf::ForOp>(
            loc, c0, cN, c16, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value n, ValueRange) {
              // Initialize accumulator tile to zero
              Value zeroTile = builder.create<TileZeroOp>(loc, tileTypeF32);
              builder.create<TileStoreOp>(loc, CTemp, ValueRange{m, n},
                                          zeroTile);

              // Inner loop: iterate over K dimension in chunks of 32
              builder.create<scf::ForOp>(
                  loc, c0, cK, c32, ValueRange{},
                  [&](OpBuilder &builder, Location loc, Value k, ValueRange) {
                    // Load A tile: 16x32xbf16 from [m, k]
                    Value tA = builder.create<TileLoadOp>(loc, tileTypeBF16A, A,
                                                          ValueRange{m, k});

                    // Load B tile from VNNI-packed format
                    Value c2Val =
                        builder.create<arith::ConstantIndexOp>(loc, 2);
                    Value kDiv2 = builder.create<arith::DivUIOp>(loc, k, c2Val);
                    Value nMul2 = builder.create<arith::MulIOp>(loc, n, c2Val);

                    Value tB = builder.create<TileLoadOp>(
                        loc, tileTypeBF16B, Bpack, ValueRange{kDiv2, nMul2});

                    // Load current accumulator
                    Value tAcc = builder.create<TileLoadOp>(
                        loc, tileTypeF32, CTemp, ValueRange{m, n});

                    // Perform tile multiplication with accumulation
                    Value tAcc2 = builder.create<TileMulFOp>(loc, tileTypeF32,
                                                             tA, tB, tAcc);

                    // Store result back
                    builder.create<TileStoreOp>(loc, CTemp, ValueRange{m, n},
                                                tAcc2);

                    builder.create<scf::YieldOp>(loc);
                  });
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });

  // If output is bf16, convert f32 -> bf16
  if (needsBF16Conversion) {
    rewriter.create<scf::ForOp>(
        loc, c0, cM, c1, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value i, ValueRange) {
          builder.create<scf::ForOp>(
              loc, c0, cN, c1, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value j, ValueRange) {
                Value f32Val = builder.create<memref::LoadOp>(loc, CTemp,
                                                              ValueRange{i, j});
                Value bf16Val =
                    builder.create<arith::TruncFOp>(loc, bf16Type, f32Val);
                builder.create<memref::StoreOp>(loc, bf16Val, C,
                                                ValueRange{i, j});
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });

    rewriter.create<memref::DeallocOp>(loc, CTemp);
  }

  // Deallocate the packed B matrix
  rewriter.create<memref::DeallocOp>(loc, Bpack);

  return success();
}

//===----------------------------------------------------------------------===//
// AMX Tile Computation - INT8
//===----------------------------------------------------------------------===//

/// Rewrite linalg.matmul to AMX INT8 operations
LogicalResult rewriteMatmulInt8(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter, Value A, Value B,
                                Value C) {
  auto loc = matmulOp.getLoc();

  auto AMemRefType = mlir::cast<MemRefType>(A.getType());
  auto BMemRefType = mlir::cast<MemRefType>(B.getType());

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
                    Value kDiv4 = builder.create<arith::DivUIOp>(loc, k, c4);
                    Value nMul4 = builder.create<arith::MulIOp>(loc, n, c4);

                    Value tB = builder.create<TileLoadOp>(
                        loc, tileTypeI8B, Bpack, ValueRange{kDiv4, nMul4});

                    // Load current accumulator from C (i32)
                    Value tAcc = builder.create<TileLoadOp>(loc, tileTypeI32, C,
                                                            ValueRange{m, n});

                    // Perform tile multiplication with accumulation (INT8)
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

  return success();
}

//===----------------------------------------------------------------------===//
// Unified Rewrite Pattern
//===----------------------------------------------------------------------===//

class MatmulAMXPattern : public RewritePattern {
public:
  explicit MatmulAMXPattern(MLIRContext *context)
      : RewritePattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);

    // Get input operands A, B, C
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    // Detect data type
    AMXDataType dtype = detectAMXDataType(A, B, C);

    if (dtype == AMXDataType::Unsupported) {
      return rewriter.notifyMatchFailure(
          op, "Data types not supported by AMX: requires BF16 (bf16*bf16->f32) "
              "or INT8 (i8*i8->i32)");
    }

    // Validate dimensions based on data type
    if (dtype == AMXDataType::BF16) {
      if (!isAMXBF16DimensionsValid(A, B, C)) {
        return rewriter.notifyMatchFailure(
            op, "Matrix dimensions not compatible with AMX BF16: M,N must be "
                "multiples of 16, K must be multiple of 32");
      }
    } else if (dtype == AMXDataType::INT8) {
      if (!isAMXInt8DimensionsValid(A, B, C)) {
        return rewriter.notifyMatchFailure(
            op, "Matrix dimensions not compatible with AMX INT8: M,N must be "
                "multiples of 16, K must be multiple of 64");
      }
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

    // Dispatch to appropriate rewrite function
    LogicalResult result = failure();
    if (dtype == AMXDataType::BF16) {
      result = rewriteMatmulBF16(matmulOp, rewriter, A, B, C);
    } else if (dtype == AMXDataType::INT8) {
      result = rewriteMatmulInt8(matmulOp, rewriter, A, B, C);
    }

    if (failed(result)) {
      return failure();
    }

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
    return "Convert linalg.matmul to AMX operations (supports BF16 and INT8)";
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
