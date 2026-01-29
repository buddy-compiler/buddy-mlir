//====- LowerLinalgToIME.cpp - Linalg to IME Dialect Lowering Pass --------===//
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
// This file defines Linalg dialect lowering pass to IME dialect.
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

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"

using namespace mlir;
using namespace buddy::ime;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get the tile sizes based on element type.
/// For int8: TILE_M=4, TILE_K=8, TILE_N=4
/// For int16: TILE_M=4, TILE_K=4, TILE_N=4
static void getTileSizes(Type elemType, int64_t &tileM, int64_t &tileK,
                         int64_t &tileN) {
  if (elemType.isInteger(8)) {
    tileM = 4;
    tileK = 8;
    tileN = 4;
  } else if (elemType.isInteger(16)) {
    tileM = 4;
    tileK = 4;
    tileN = 4;
  } else {
    // Default to int8 tile sizes
    tileM = 4;
    tileK = 8;
    tileN = 4;
  }
}

/// Check if the element type is supported by IME vmadot.
static bool isSupportedElementType(Type elemType) {
  return elemType.isInteger(8) || elemType.isInteger(16);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Pattern to lower linalg.matmul to ime.vmadot with tiling.
/// linalg.matmul: C[M,N] = A[M,K] * B[K,N]
/// ime.vmadot: performs tiled matrix multiply-accumulate
class MatmulToIMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    // Get input and output operands
    Value A = matmulOp.getInputs()[0]; // M x K
    Value B = matmulOp.getInputs()[1]; // K x N
    Value C = matmulOp.getOutputs()[0]; // M x N

    // Get memref types
    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "operands must be memref types");
    }

    // Check element types
    Type AElemType = AType.getElementType();
    Type BElemType = BType.getElementType();
    Type CElemType = CType.getElementType();

    if (!isSupportedElementType(AElemType) ||
        !isSupportedElementType(BElemType)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "only int8 and int16 element types are supported");
    }

    // Verify that A and B have same element type
    if (AElemType != BElemType) {
      return rewriter.notifyMatchFailure(
          matmulOp, "A and B must have the same element type");
    }

    // Verify that C is int32 for accumulation
    if (!CElemType.isInteger(32)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "output C must be int32 for accumulation");
    }

    // Get matrix dimensions
    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();

    if (AShape.size() != 2 || BShape.size() != 2) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "only 2D matrices are supported");
    }

    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    // For dynamic shapes, we need to get dimensions at runtime
    bool isDynamic = ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                     ShapedType::isDynamic(N);

    // Get tile sizes based on element type
    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN);

    // Create constants for loop bounds and steps
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, tileN);

    Value boundM, boundK, boundN;
    if (isDynamic) {
      boundM = rewriter.create<memref::DimOp>(loc, A, 0);
      boundK = rewriter.create<memref::DimOp>(loc, A, 1);
      boundN = rewriter.create<memref::DimOp>(loc, B, 1);
    } else {
      boundM = rewriter.create<arith::ConstantIndexOp>(loc, M);
      boundK = rewriter.create<arith::ConstantIndexOp>(loc, K);
      boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    }

    // Generate nested loops: for i in [0, M, tileM) for j in [0, N, tileN) for k in [0, K, tileK)
    // Loop over M dimension (rows of A and C)
    auto loopI = rewriter.create<scf::ForOp>(loc, c0, boundM, stepM);
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value ivI = loopI.getInductionVar();

    // Loop over N dimension (columns of B and C)
    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, boundN, stepN);
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value ivJ = loopJ.getInductionVar();

    // Loop over K dimension (reduction dimension)
    auto loopK = rewriter.create<scf::ForOp>(loc, c0, boundK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    // Create subviews for the tiles with static tile sizes
    // The SubViewOp uses static sizes when the tile exactly fits
    // A_tile: A[i:i+tileM, k:k+tileK]
    SmallVector<OpFoldResult> aOffsets = {ivI, ivK};
    SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(tileM), 
                                        rewriter.getIndexAttr(tileK)};
    SmallVector<OpFoldResult> aStrides = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};

    Value ATile = rewriter.create<memref::SubViewOp>(loc, A, aOffsets,
                                                     aSizes, aStrides);

    // B_tile: B[k:k+tileK, j:j+tileN]
    SmallVector<OpFoldResult> bOffsets = {ivK, ivJ};
    SmallVector<OpFoldResult> bSizes = {rewriter.getIndexAttr(tileK), 
                                        rewriter.getIndexAttr(tileN)};
    SmallVector<OpFoldResult> bStrides = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};

    Value BTile = rewriter.create<memref::SubViewOp>(loc, B, bOffsets,
                                                     bSizes, bStrides);

    // C_tile: C[i:i+tileM, j:j+tileN]
    SmallVector<OpFoldResult> cOffsets = {ivI, ivJ};
    SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(tileM), 
                                        rewriter.getIndexAttr(tileN)};
    SmallVector<OpFoldResult> cStrides = {rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(1)};

    Value CTile = rewriter.create<memref::SubViewOp>(loc, C, cOffsets,
                                                     cSizes, cStrides);

    // Create ime.vmadot operation
    // vmadot performs: C_tile += A_tile * B_tile
    rewriter.create<VmadotOp>(loc, CTile, ATile, BTile);

    // Move insertion point after all loops
    rewriter.setInsertionPointAfter(loopI);

    // Erase the original matmul operation
    rewriter.eraseOp(matmulOp);

    return success();
  }
};

/// Pattern to lower linalg.generic (matmul-like) to ime.vmadot.
/// This handles cases where matmul is represented as linalg.generic.
class GenericMatmulToIMELowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Check if this is a matmul-like operation
    // A matmul has: 2 inputs, 1 output, 3 loops (2 parallel + 1 reduction)
    if (genericOp.getInputs().size() != 2 || genericOp.getOutputs().size() != 1)
      return failure();

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    if (iteratorTypes.size() != 3)
      return failure();

    // Check iterator types: parallel, parallel, reduction
    if (iteratorTypes[0] != utils::IteratorType::parallel ||
        iteratorTypes[1] != utils::IteratorType::parallel ||
        iteratorTypes[2] != utils::IteratorType::reduction)
      return failure();

    // Check indexing maps for matmul pattern:
    // A: (m, n, k) -> (m, k)
    // B: (m, n, k) -> (k, n)
    // C: (m, n, k) -> (m, n)
    auto indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 3)
      return failure();

    // Verify the indexing maps match matmul pattern
    AffineMap mapA = indexingMaps[0];
    AffineMap mapB = indexingMaps[1];
    AffineMap mapC = indexingMaps[2];

    // Check A: (d0, d1, d2) -> (d0, d2) i.e., (m, k)
    // Check B: (d0, d1, d2) -> (d2, d1) i.e., (k, n)
    // Check C: (d0, d1, d2) -> (d0, d1) i.e., (m, n)
    auto d0 = rewriter.getAffineDimExpr(0);
    auto d1 = rewriter.getAffineDimExpr(1);
    auto d2 = rewriter.getAffineDimExpr(2);

    auto expectedMapA = AffineMap::get(3, 0, {d0, d2}, rewriter.getContext());
    auto expectedMapB = AffineMap::get(3, 0, {d2, d1}, rewriter.getContext());
    auto expectedMapC = AffineMap::get(3, 0, {d0, d1}, rewriter.getContext());

    if (mapA != expectedMapA || mapB != expectedMapB || mapC != expectedMapC)
      return failure();

    // Check the body is a multiply-accumulate
    Block &body = genericOp.getRegion().front();
    if (body.getOperations().size() != 3) // mul, add, yield
      return failure();

    auto ops = body.getOperations().begin();
    auto *firstOp = &*ops++;
    auto *secondOp = &*ops++;
    auto *yieldOp = &*ops;

    // Check for mul followed by add
    if (!isa<arith::MulIOp, arith::MulFOp>(firstOp))
      return failure();
    if (!isa<arith::AddIOp, arith::AddFOp>(secondOp))
      return failure();
    if (!isa<linalg::YieldOp>(yieldOp))
      return failure();

    // Now we know this is a matmul, delegate to the matmul lowering
    // For simplicity, we convert the generic to an equivalent structure
    // and apply similar tiling logic

    Location loc = genericOp.getLoc();
    Value A = genericOp.getInputs()[0];
    Value B = genericOp.getInputs()[1];
    Value C = genericOp.getOutputs()[0];

    auto AType = dyn_cast<MemRefType>(A.getType());
    auto BType = dyn_cast<MemRefType>(B.getType());
    auto CType = dyn_cast<MemRefType>(C.getType());

    if (!AType || !BType || !CType)
      return failure();

    Type AElemType = AType.getElementType();
    if (!isSupportedElementType(AElemType))
      return failure();

    // Get dimensions and tile sizes
    int64_t tileM, tileK, tileN;
    getTileSizes(AElemType, tileM, tileK, tileN);

    ArrayRef<int64_t> AShape = AType.getShape();
    ArrayRef<int64_t> BShape = BType.getShape();
    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    bool isDynamic = ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                     ShapedType::isDynamic(N);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, tileM);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, tileK);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, tileN);

    Value boundM, boundK, boundN;
    if (isDynamic) {
      boundM = rewriter.create<memref::DimOp>(loc, A, 0);
      boundK = rewriter.create<memref::DimOp>(loc, A, 1);
      boundN = rewriter.create<memref::DimOp>(loc, B, 1);
    } else {
      boundM = rewriter.create<arith::ConstantIndexOp>(loc, M);
      boundK = rewriter.create<arith::ConstantIndexOp>(loc, K);
      boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    }

    // Generate tiled loops
    auto loopI = rewriter.create<scf::ForOp>(loc, c0, boundM, stepM);
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value ivI = loopI.getInductionVar();

    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, boundN, stepN);
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value ivJ = loopJ.getInductionVar();

    auto loopK = rewriter.create<scf::ForOp>(loc, c0, boundK, stepK);
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value ivK = loopK.getInductionVar();

    // Use static tile sizes for subview
    SmallVector<OpFoldResult> aOffsets = {ivI, ivK};
    SmallVector<OpFoldResult> aSizes = {rewriter.getIndexAttr(tileM), 
                                        rewriter.getIndexAttr(tileK)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};

    Value ATile = rewriter.create<memref::SubViewOp>(loc, A, aOffsets,
                                                     aSizes, strides);

    SmallVector<OpFoldResult> bOffsets = {ivK, ivJ};
    SmallVector<OpFoldResult> bSizes = {rewriter.getIndexAttr(tileK), 
                                        rewriter.getIndexAttr(tileN)};
    Value BTile = rewriter.create<memref::SubViewOp>(loc, B, bOffsets,
                                                     bSizes, strides);

    SmallVector<OpFoldResult> cOffsets = {ivI, ivJ};
    SmallVector<OpFoldResult> cSizes = {rewriter.getIndexAttr(tileM), 
                                        rewriter.getIndexAttr(tileN)};
    Value CTile = rewriter.create<memref::SubViewOp>(loc, C, cOffsets,
                                                     cSizes, strides);

    rewriter.create<VmadotOp>(loc, CTile, ATile, BTile);

    rewriter.setInsertionPointAfter(loopI);
    rewriter.eraseOp(genericOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conv2D to IME Lowering Pattern (with Sliding-Window Instructions)
//===----------------------------------------------------------------------===//

/// Pattern to lower linalg.conv_2d_nhwc_hwcf to IME sliding-window operations.
///
/// Conv2D NHWC layout: 
///   input:  [N, H, W, IC]
///   filter: [FH, FW, IC, OC]
///   output: [N, OH, OW, OC]
///
/// This uses sliding-window instructions (vmadot1, vmadot2, vmadot3, vmadotn)
/// to efficiently implement convolution without explicit im2col transformation.
///
/// For strideH=1: Use vmadot1 with 8-row input buffer, slide through rows
/// For strideH=2: Use vmadot2
/// For strideH=3: Use vmadot3
/// For other strides: Use vmadotn with dynamic slide parameter
///
/// The sliding window mechanism:
/// - Input A is loaded as 2M×K (8×8 for int8)
/// - slide=n selects rows [n, n+1, n+2, n+3] from the 8 rows
/// - This allows computing 4 consecutive output rows with data reuse
///
class Conv2DNhwcHwcfToIMELowering
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Get input, filter, and output
    Value input = convOp.getInputs()[0];   // [N, H, W, IC]
    Value filter = convOp.getInputs()[1];  // [FH, FW, IC, OC]
    Value output = convOp.getOutputs()[0]; // [N, OH, OW, OC]

    // Get memref types
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());

    if (!inputType || !filterType || !outputType)
      return rewriter.notifyMatchFailure(convOp, "operands must be memref types");

    // Check element types - only support int8 for now
    Type inputElemType = inputType.getElementType();
    if (!inputElemType.isInteger(8))
      return rewriter.notifyMatchFailure(convOp, "only int8 is supported");

    // Get shapes
    ArrayRef<int64_t> inputShape = inputType.getShape();   // [N, H, W, IC]
    ArrayRef<int64_t> filterShape = filterType.getShape(); // [FH, FW, IC, OC]
    ArrayRef<int64_t> outputShape = outputType.getShape(); // [N, OH, OW, OC]

    if (inputShape.size() != 4 || filterShape.size() != 4 || outputShape.size() != 4)
      return rewriter.notifyMatchFailure(convOp, "only 4D tensors are supported");

    int64_t N = inputShape[0];
    int64_t IC = inputShape[3];
    int64_t FH = filterShape[0];
    int64_t FW = filterShape[1];
    int64_t OC = filterShape[3];
    int64_t OH = outputShape[1];
    int64_t OW = outputShape[2];

    // Get strides
    int64_t strideH = 1, strideW = 1;

    if (auto stridesAttr = convOp.getStrides()) {
      auto strides = stridesAttr.getValues<int64_t>();
      strideH = strides[0];
      strideW = strides[1];
    }

    // IME tile sizes for int8
    // For sliding-window: input is 2M×K (8×8), output is M×N (4×4)
    const int64_t TILE_M = 4;      // Output rows per tile
    const int64_t TILE_2M = 8;     // Input rows for sliding window (2*TILE_M)
    const int64_t TILE_K = 8;      // K dimension
    const int64_t TILE_N = 4;      // Output columns per tile

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Create bounds
    Value boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value boundOH = rewriter.create<arith::ConstantIndexOp>(loc, OH);
    Value boundOW = rewriter.create<arith::ConstantIndexOp>(loc, OW);
    Value boundOC = rewriter.create<arith::ConstantIndexOp>(loc, OC);
    Value boundFH = rewriter.create<arith::ConstantIndexOp>(loc, FH);
    Value boundFW = rewriter.create<arith::ConstantIndexOp>(loc, FW);
    Value boundIC = rewriter.create<arith::ConstantIndexOp>(loc, IC);

    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    Value strideHVal = rewriter.create<arith::ConstantIndexOp>(loc, strideH);
    Value strideWVal = rewriter.create<arith::ConstantIndexOp>(loc, strideW);

    // Allocate tile buffers outside the loops for reuse
    // For sliding window: input is 2M×K (8×8)
    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType = MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    // Loop over batch
    auto loopN = rewriter.create<scf::ForOp>(loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    // Loop over output height (tiled by TILE_M)
    auto loopOH = rewriter.create<scf::ForOp>(loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    // Loop over output channels (tiled by TILE_N)
    auto loopOC = rewriter.create<scf::ForOp>(loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    // Loop over output width
    auto loopOW = rewriter.create<scf::ForOp>(loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    // Loop over filter height
    auto loopFH = rewriter.create<scf::ForOp>(loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    // Loop over filter width
    auto loopFW = rewriter.create<scf::ForOp>(loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    // Loop over input channels (tiled by TILE_K)
    auto loopIC = rewriter.create<scf::ForOp>(loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    // Allocate tiles inside innermost loop
    Value inputTile = rewriter.create<memref::AllocaOp>(loc, inputTileType);
    Value filterTile = rewriter.create<memref::AllocaOp>(loc, filterTileType);
    Value outputTile = rewriter.create<memref::AllocaOp>(loc, outputTileType);

    // Compute base input position: ih_base = oh * strideH + fh
    Value ihBase = rewriter.create<arith::MulIOp>(loc, ivOH, strideHVal);
    ihBase = rewriter.create<arith::AddIOp>(loc, ihBase, ivFH);
    Value iw = rewriter.create<arith::MulIOp>(loc, ivOW, strideWVal);
    iw = rewriter.create<arith::AddIOp>(loc, iw, ivFW);

    Value tileMBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value tile2MBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_2M);
    Value tileKBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value tileNBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    // Fill input tile: 8 rows for sliding window
    // inputTile[m, k] = input[n, ih_base + m*strideH, iw, ic+k]
    // For 8 rows: m = 0..7, each row corresponds to a different output row's input
    auto fillInputLoop = rewriter.create<scf::ForOp>(loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    // For sliding window: inputRow = ih_base + m * strideH
    Value mTimesStride = rewriter.create<arith::MulIOp>(loc, fillM, strideHVal);
    Value inputIH = rewriter.create<arith::AddIOp>(loc, ihBase, mTimesStride);
    Value inputIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillK);
    Value inputVal = rewriter.create<memref::LoadOp>(
        loc, input, ValueRange{ivN, inputIH, iw, inputIC});
    rewriter.create<memref::StoreOp>(
        loc, inputVal, inputTile, ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    // Fill filter tile: filterTile[k, n] = filter[fh, fw, ic+k, oc+n]
    auto fillFilterLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillFK);
    Value filterOC = rewriter.create<arith::AddIOp>(loc, ivOC, fillFN);
    Value filterVal = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{ivFH, ivFW, filterIC, filterOC});
    rewriter.create<memref::StoreOp>(
        loc, filterVal, filterTile, ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    // Load existing output values
    auto loadOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOH = rewriter.create<arith::AddIOp>(loc, ivOH, loadM);
    Value outOC = rewriter.create<arith::AddIOp>(loc, ivOC, loadN);
    Value outVal = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{ivN, outOH, ivOW, outOC});
    rewriter.create<memref::StoreOp>(
        loc, outVal, outputTile, ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    // Create the appropriate IME sliding-window operation based on stride
    // slide=0 means use rows [0,1,2,3], which is what we want for the first
    // tile of outputs
    if (strideH == 1) {
      rewriter.create<Vmadot1Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      rewriter.create<Vmadot2Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      rewriter.create<Vmadot3Op>(loc, outputTile, inputTile, filterTile);
    } else {
      // For other strides, use vmadotn with dynamic slide
      // slide = 0 since we already computed the correct input positions
      Value slide = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      rewriter.create<VmadotnOp>(loc, outputTile, inputTile, filterTile, slide);
    }

    // Store output values back
    auto storeOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOH = rewriter.create<arith::AddIOp>(loc, ivOH, storeM);
    Value storeOC = rewriter.create<arith::AddIOp>(loc, ivOC, storeN);
    Value storeVal = rewriter.create<memref::LoadOp>(
        loc, outputTile, ValueRange{storeM, storeN});
    rewriter.create<memref::StoreOp>(
        loc, storeVal, output, ValueRange{ivN, storeOH, ivOW, storeOC});

    rewriter.setInsertionPointAfter(storeOutputLoop);

    // Move insertion point after all loops
    rewriter.setInsertionPointAfter(loopN);

    // Erase the original conv operation
    rewriter.eraseOp(convOp);

    return success();
  }
};

/// Pattern to lower linalg.conv_2d_nchw_fchw to IME sliding-window operations.
/// NCHW layout:
///   input:  [N, IC, H, W]
///   filter: [OC, IC, FH, FW]
///   output: [N, OC, OH, OW]
class Conv2DNchwFchwToIMELowering
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern<linalg::Conv2DNchwFchwOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();

    // Get input, filter, and output
    Value input = convOp.getInputs()[0];   // [N, IC, H, W]
    Value filter = convOp.getInputs()[1];  // [OC, IC, FH, FW]
    Value output = convOp.getOutputs()[0]; // [N, OC, OH, OW]

    // Get memref types
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto filterType = dyn_cast<MemRefType>(filter.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());

    if (!inputType || !filterType || !outputType)
      return rewriter.notifyMatchFailure(convOp, "operands must be memref types");

    // Check element types - only support int8 for now
    Type inputElemType = inputType.getElementType();
    if (!inputElemType.isInteger(8))
      return rewriter.notifyMatchFailure(convOp, "only int8 is supported");

    // Get shapes
    ArrayRef<int64_t> inputShape = inputType.getShape();   // [N, IC, H, W]
    ArrayRef<int64_t> filterShape = filterType.getShape(); // [OC, IC, FH, FW]
    ArrayRef<int64_t> outputShape = outputType.getShape(); // [N, OC, OH, OW]

    if (inputShape.size() != 4 || filterShape.size() != 4 || outputShape.size() != 4)
      return rewriter.notifyMatchFailure(convOp, "only 4D tensors are supported");

    int64_t N = inputShape[0];
    int64_t IC = inputShape[1];
    int64_t OC = filterShape[0];
    int64_t FH = filterShape[2];
    int64_t FW = filterShape[3];
    int64_t OH = outputShape[2];
    int64_t OW = outputShape[3];

    // Get strides
    int64_t strideH = 1, strideW = 1;

    if (auto stridesAttr = convOp.getStrides()) {
      auto strides = stridesAttr.getValues<int64_t>();
      strideH = strides[0];
      strideW = strides[1];
    }

    // IME tile sizes for int8, with sliding window
    const int64_t TILE_M = 4;
    const int64_t TILE_2M = 8;   // 2*TILE_M for sliding window input
    const int64_t TILE_K = 8;
    const int64_t TILE_N = 4;

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value boundN = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value boundOC = rewriter.create<arith::ConstantIndexOp>(loc, OC);
    Value boundOH = rewriter.create<arith::ConstantIndexOp>(loc, OH);
    Value boundOW = rewriter.create<arith::ConstantIndexOp>(loc, OW);
    Value boundIC = rewriter.create<arith::ConstantIndexOp>(loc, IC);
    Value boundFH = rewriter.create<arith::ConstantIndexOp>(loc, FH);
    Value boundFW = rewriter.create<arith::ConstantIndexOp>(loc, FW);

    Value stepM = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value stepK = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value stepN = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    Value strideHVal = rewriter.create<arith::ConstantIndexOp>(loc, strideH);
    Value strideWVal = rewriter.create<arith::ConstantIndexOp>(loc, strideW);

    // Loop over batch
    auto loopN = rewriter.create<scf::ForOp>(loc, c0, boundN, c1);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    // Loop over output channels (tiled by TILE_N)
    auto loopOC = rewriter.create<scf::ForOp>(loc, c0, boundOC, stepN);
    rewriter.setInsertionPointToStart(loopOC.getBody());
    Value ivOC = loopOC.getInductionVar();

    // Loop over output height (tiled by TILE_M)
    auto loopOH = rewriter.create<scf::ForOp>(loc, c0, boundOH, stepM);
    rewriter.setInsertionPointToStart(loopOH.getBody());
    Value ivOH = loopOH.getInductionVar();

    // Loop over output width
    auto loopOW = rewriter.create<scf::ForOp>(loc, c0, boundOW, c1);
    rewriter.setInsertionPointToStart(loopOW.getBody());
    Value ivOW = loopOW.getInductionVar();

    // Loop over input channels (tiled by TILE_K)
    auto loopIC = rewriter.create<scf::ForOp>(loc, c0, boundIC, stepK);
    rewriter.setInsertionPointToStart(loopIC.getBody());
    Value ivIC = loopIC.getInductionVar();

    // Loop over filter height
    auto loopFH = rewriter.create<scf::ForOp>(loc, c0, boundFH, c1);
    rewriter.setInsertionPointToStart(loopFH.getBody());
    Value ivFH = loopFH.getInductionVar();

    // Loop over filter width
    auto loopFW = rewriter.create<scf::ForOp>(loc, c0, boundFW, c1);
    rewriter.setInsertionPointToStart(loopFW.getBody());
    Value ivFW = loopFW.getInductionVar();

    // Compute base input position
    Value ihBase = rewriter.create<arith::MulIOp>(loc, ivOH, strideHVal);
    ihBase = rewriter.create<arith::AddIOp>(loc, ihBase, ivFH);
    Value iw = rewriter.create<arith::MulIOp>(loc, ivOW, strideWVal);
    iw = rewriter.create<arith::AddIOp>(loc, iw, ivFW);

    // Allocate temporary tiles (2M×K for sliding window input)
    auto inputTileType = MemRefType::get({TILE_2M, TILE_K}, inputElemType);
    auto filterTileType = MemRefType::get({TILE_K, TILE_N}, inputElemType);
    auto outputTileType = MemRefType::get({TILE_M, TILE_N}, rewriter.getI32Type());

    Value inputTile = rewriter.create<memref::AllocaOp>(loc, inputTileType);
    Value filterTile = rewriter.create<memref::AllocaOp>(loc, filterTileType);
    Value outputTile = rewriter.create<memref::AllocaOp>(loc, outputTileType);

    Value tileMBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_M);
    Value tile2MBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_2M);
    Value tileKBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_K);
    Value tileNBound = rewriter.create<arith::ConstantIndexOp>(loc, TILE_N);

    // Fill input tile with 8 rows for sliding window
    // inputTile[m, k] = input[n, ic+k, ih_base + m*strideH, iw]
    auto fillInputLoop = rewriter.create<scf::ForOp>(loc, c0, tile2MBound, c1);
    rewriter.setInsertionPointToStart(fillInputLoop.getBody());
    Value fillM = fillInputLoop.getInductionVar();

    auto fillInputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillInputInnerLoop.getBody());
    Value fillK = fillInputInnerLoop.getInductionVar();

    Value mTimesStride = rewriter.create<arith::MulIOp>(loc, fillM, strideHVal);
    Value inputIH = rewriter.create<arith::AddIOp>(loc, ihBase, mTimesStride);
    Value inputIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillK);
    Value inputVal = rewriter.create<memref::LoadOp>(
        loc, input, ValueRange{ivN, inputIC, inputIH, iw});
    rewriter.create<memref::StoreOp>(
        loc, inputVal, inputTile, ValueRange{fillM, fillK});

    rewriter.setInsertionPointAfter(fillInputLoop);

    // Fill filter tile: filterTile[k, n] = filter[oc+n, ic+k, fh, fw]
    auto fillFilterLoop = rewriter.create<scf::ForOp>(loc, c0, tileKBound, c1);
    rewriter.setInsertionPointToStart(fillFilterLoop.getBody());
    Value fillFK = fillFilterLoop.getInductionVar();

    auto fillFilterInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(fillFilterInnerLoop.getBody());
    Value fillFN = fillFilterInnerLoop.getInductionVar();

    Value filterOC = rewriter.create<arith::AddIOp>(loc, ivOC, fillFN);
    Value filterIC = rewriter.create<arith::AddIOp>(loc, ivIC, fillFK);
    Value filterVal = rewriter.create<memref::LoadOp>(
        loc, filter, ValueRange{filterOC, filterIC, ivFH, ivFW});
    rewriter.create<memref::StoreOp>(
        loc, filterVal, filterTile, ValueRange{fillFK, fillFN});

    rewriter.setInsertionPointAfter(fillFilterLoop);

    // Load existing output values
    auto loadOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(loadOutputLoop.getBody());
    Value loadM = loadOutputLoop.getInductionVar();

    auto loadOutputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(loadOutputInnerLoop.getBody());
    Value loadN = loadOutputInnerLoop.getInductionVar();

    Value outOC = rewriter.create<arith::AddIOp>(loc, ivOC, loadN);
    Value outOH = rewriter.create<arith::AddIOp>(loc, ivOH, loadM);
    Value outVal = rewriter.create<memref::LoadOp>(
        loc, output, ValueRange{ivN, outOC, outOH, ivOW});
    rewriter.create<memref::StoreOp>(
        loc, outVal, outputTile, ValueRange{loadM, loadN});

    rewriter.setInsertionPointAfter(loadOutputLoop);

    // Create the appropriate IME sliding-window operation based on stride
    if (strideH == 1) {
      rewriter.create<Vmadot1Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 2) {
      rewriter.create<Vmadot2Op>(loc, outputTile, inputTile, filterTile);
    } else if (strideH == 3) {
      rewriter.create<Vmadot3Op>(loc, outputTile, inputTile, filterTile);
    } else {
      Value slide = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      rewriter.create<VmadotnOp>(loc, outputTile, inputTile, filterTile, slide);
    }

    // Store output values back
    auto storeOutputLoop = rewriter.create<scf::ForOp>(loc, c0, tileMBound, c1);
    rewriter.setInsertionPointToStart(storeOutputLoop.getBody());
    Value storeM = storeOutputLoop.getInductionVar();

    auto storeOutputInnerLoop = rewriter.create<scf::ForOp>(loc, c0, tileNBound, c1);
    rewriter.setInsertionPointToStart(storeOutputInnerLoop.getBody());
    Value storeN = storeOutputInnerLoop.getInductionVar();

    Value storeOC = rewriter.create<arith::AddIOp>(loc, ivOC, storeN);
    Value storeOH = rewriter.create<arith::AddIOp>(loc, ivOH, storeM);
    Value storeVal = rewriter.create<memref::LoadOp>(
        loc, outputTile, ValueRange{storeM, storeN});
    rewriter.create<memref::StoreOp>(
        loc, storeVal, output, ValueRange{ivN, storeOC, storeOH, ivOW});

    rewriter.setInsertionPointAfter(storeOutputLoop);

    // Move insertion point after all loops
    rewriter.setInsertionPointAfter(loopN);

    // Erase the original conv operation
    rewriter.eraseOp(convOp);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToIMEPass
    : public PassWrapper<LowerLinalgToIMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToIMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-ime"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to IME dialect operations.";
  }

  LowerLinalgToIMEPass() = default;
  LowerLinalgToIMEPass(const LowerLinalgToIMEPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IMEDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToIMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  RewritePatternSet patterns(context);
  // Matmul patterns
  patterns.add<MatmulToIMELowering>(context);
  patterns.add<GenericMatmulToIMELowering>(context);
  // Convolution patterns
  patterns.add<Conv2DNhwcHwcfToIMELowering>(context);
  patterns.add<Conv2DNchwFchwToIMELowering>(context);

  // Use greedy pattern rewriter
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToIMEPass() {
  PassRegistration<LowerLinalgToIMEPass>();
}
} // namespace buddy
} // namespace mlir
