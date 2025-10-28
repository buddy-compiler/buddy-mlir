//===- FuseFC.cpp - Fully Connected Fusion Pass ------------------------===//
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
// This pass is going to deal with pattern like:
  // %collapsed_29 = tensor.collapse_shape %46 [[0, 1], [2]] : tensor<1x1024x1536xf32> into tensor<1024x1536xf32>
  // %51 = bufferization.alloc_tensor() : tensor<1536x256xf32>
  // %transposed_30 = linalg.transpose ins(%arg7 : tensor<256x1536xf32>) outs(%51 : tensor<1536x256xf32>) permutation = [1, 0] 
  // %52 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%collapsed_29, %transposed_30 : tensor<1024x1536xf32>, tensor<1536x256xf32>) outs(%cst_6 : tensor<1024x256xf32>) -> tensor<1024x256xf32>
  // %expanded_31 = tensor.expand_shape %arg8 [[0, 1]] output_shape [1, 256] : tensor<256xf32> into tensor<1x256xf32>
  // %53 = bufferization.alloc_tensor() : tensor<1024x256xf32>
  // %54 = linalg.generic {indexing_maps = [#map10, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_31, %52 : tensor<1x256xf32>, tensor<1024x256xf32>) outs(%53 : tensor<1024x256xf32>) {
  // ^bb0(%in: f32, %in_1538: f32, %out: f32):
  //   %3044 = arith.addf %in, %in_1538 : f32
  //   linalg.yield %3044 : f32
  // } -> tensor<1024x256xf32>
// This pass will transform it into two linalg.generic Operations. So that they could be fused later in affine level.
// Redundent operations needs to be cleaned with canonicalization pass.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fuse-fc"

using namespace mlir;

namespace {

// Helper function to check if a linalg.generic op is elementwise
static bool isElementwise(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      });
}

class FuseFCPattern : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp addOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "\n=== FuseFCPattern: Checking GenericOp ===\n");
    LLVM_DEBUG(addOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // 1. Check if the current op is a bias-add (elementwise with 2 loops)
    if (!isElementwise(addOp)) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Not elementwise, skipping\n");
      return failure();
    }

    if (addOp.getNumLoops() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Number of loops != 2 (got " 
                              << addOp.getNumLoops() << "), skipping\n");
      return failure();
    }

    if (addOp.getNumDpsInputs() != 2 || addOp.getNumDpsInits() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Wrong number of inputs/outputs (inputs=" 
                              << addOp.getNumDpsInputs() << ", outputs=" 
                              << addOp.getNumDpsInits() << "), skipping\n");
      return failure();
    }

    // Check for add operation in the body
    auto &body = addOp.getRegion().front();
    bool hasAdd = !body.getOps<arith::AddFOp>().empty() ||
                  !body.getOps<arith::AddIOp>().empty();
    if (!hasAdd) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Body doesn't contain add operation, skipping\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  -> Confirmed as elementwise add operation\n");

    // 2. Find which operand is the matmul result and which is the bias
    Value matmulResult, bias;
    linalg::MatmulOp matmulOp;
    int matmulIdx = -1;

    for (int i = 0; i < 2; ++i) {
      auto input = addOp.getDpsInputOperand(i)->get();
      LLVM_DEBUG(llvm::dbgs() << "  -> Checking input " << i << ": ");
      LLVM_DEBUG(input.print(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      if (auto defOp = input.getDefiningOp<linalg::MatmulOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "     Found MatmulOp at index " << i << "\n");
        matmulOp = defOp;
        matmulResult = input;
        matmulIdx = i;
        bias = addOp.getDpsInputOperand(1 - i)->get();
        break;
      }
    }

    if (!matmulOp) {
      LLVM_DEBUG(llvm::dbgs() << "  -> No MatmulOp input found, skipping\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  -> MatmulOp found:\n");
    LLVM_DEBUG(matmulOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // 3. The second operand of the matmul must be the result of a transpose
    auto transposeOp =
        matmulOp.getDpsInputOperand(1)->get().getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Second matmul operand is not TransposeOp, skipping\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  -> TransposeOp found:\n");
    LLVM_DEBUG(transposeOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Check transpose permutation is [1, 0]
    auto perm = transposeOp.getPermutation();
    if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Transpose permutation is not [1, 0], skipping\n");
      return failure();
    }

    // 4. The bias must be broadcastable
    LLVM_DEBUG(llvm::dbgs() << "  -> Checking bias: ");
    LLVM_DEBUG(bias.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Trace through expand_shape if present
    Value originalBias = bias;
    if (auto expandOp = bias.getDefiningOp<tensor::ExpandShapeOp>()) {
      LLVM_DEBUG(llvm::dbgs() << "     Bias comes from expand_shape, using source\n");
      originalBias = expandOp.getSrc();
    }

    auto biasType = mlir::dyn_cast<RankedTensorType>(originalBias.getType());
    if (!biasType) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Bias is not RankedTensorType, skipping\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "     Bias type: ");
    LLVM_DEBUG(biasType.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Bias must be rank 1 or rank 2 with one dimension being 1
    if (biasType.getRank() != 1 && biasType.getRank() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Bias rank is " << biasType.getRank() 
                              << " (expected 1 or 2), skipping\n");
      return failure();
    }

    if (biasType.getRank() == 2 && biasType.getShape()[0] != 1 &&
        biasType.getShape()[1] != 1) {
      LLVM_DEBUG(llvm::dbgs() << "  -> 2D bias doesn't have dimension of size 1, skipping\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\n=== Pattern matched! Starting rewrite ===\n");

    Location loc = addOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Get the operands from the original operations
    Value inputA = matmulOp.getDpsInputOperand(0)->get();
    Value inputB = transposeOp.getDpsInputOperand(0)->get(); // Before transpose
    Value output = addOp.getDpsInitOperand(0)->get();

    auto outputType = mlir::cast<RankedTensorType>(output.getType());
    Type elementTy = outputType.getElementType();

    LLVM_DEBUG(llvm::dbgs() << "  Input A type: " << inputA.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Input B type: " << inputB.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Bias type: " << originalBias.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Output type: " << outputType << "\n");

    // --- Part 1: Create the bias broadcast operation ---
    LLVM_DEBUG(llvm::dbgs() << "\n--- Creating bias broadcast operation ---\n");

    SmallVector<AffineMap> biasMaps;
    if (biasType.getRank() == 1) {
      // from tensor<N> to tensor<M, N>
      biasMaps.push_back(AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, ctx));
      LLVM_DEBUG(llvm::dbgs() << "  Broadcast 1D bias along first dimension\n");
    } else { // Rank 2
      if (biasType.getShape()[0] == 1) {
        // tensor<1, N> -> broadcast along dim 0
        biasMaps.push_back(AffineMap::get(2, 0, 
          {rewriter.getAffineConstantExpr(0), rewriter.getAffineDimExpr(1)}, ctx));
        LLVM_DEBUG(llvm::dbgs() << "  Broadcast 2D bias (1xN) along first dimension\n");
      } else {
        // tensor<N, 1> -> broadcast along dim 1
        biasMaps.push_back(AffineMap::get(2, 0, 
          {rewriter.getAffineDimExpr(0), rewriter.getAffineConstantExpr(0)}, ctx));
        LLVM_DEBUG(llvm::dbgs() << "  Broadcast 2D bias (Nx1) along second dimension\n");
      }
    }
    biasMaps.push_back(rewriter.getMultiDimIdentityMap(2));

    SmallVector<utils::IteratorType> biasIteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel};

    auto broadcastedBiasOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/outputType,
        /*inputs=*/originalBias,
        /*outputs=*/output,
        /*indexingMaps=*/biasMaps,
        /*iteratorTypes=*/biasIteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          b.create<linalg::YieldOp>(nestedLoc, args[0]);
        });
    Value broadcastedBiasResult = broadcastedBiasOp.getResult(0);

    LLVM_DEBUG(llvm::dbgs() << "  Created bias broadcast op:\n");
    LLVM_DEBUG(broadcastedBiasOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // --- Part 2: Create the matmul with folded transpose ---
    LLVM_DEBUG(llvm::dbgs() << "\n--- Creating fused matmul operation ---\n");

    // A[i,k] * B[j,k] -> C[i,j] (with B accessed as transpose)
    AffineMap mapA = AffineMap::get(3, 0, 
      {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(2)}, ctx);
    AffineMap mapB = AffineMap::get(3, 0, 
      {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(2)}, ctx);
    AffineMap mapC = AffineMap::get(3, 0, 
      {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, ctx);

    LLVM_DEBUG(llvm::dbgs() << "  Map A (input): " << mapA << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Map B (transposed): " << mapB << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Map C (output): " << mapC << "\n");

    SmallVector<AffineMap> matmulMaps = {mapA, mapB, mapC};
    SmallVector<utils::IteratorType> matmulIteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};

    auto fusedMatmulOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/outputType,
        /*inputs=*/ValueRange{inputA, inputB},
        /*outputs=*/broadcastedBiasResult,
        /*indexingMaps=*/matmulMaps,
        /*iteratorTypes=*/matmulIteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          // args[0] = A[i,k], args[1] = B[j,k], args[2] = C[i,j] (init with bias)
          Value mulResult;
          if (isa<FloatType>(elementTy)) {
            mulResult = b.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
          } else {
            mulResult = b.create<arith::MulIOp>(nestedLoc, args[0], args[1]);
          }
          Value addResult;
          if (isa<FloatType>(elementTy)) {
            addResult = b.create<arith::AddFOp>(nestedLoc, mulResult, args[2]);
          } else {
            addResult = b.create<arith::AddIOp>(nestedLoc, mulResult, args[2]);
          }
          b.create<linalg::YieldOp>(nestedLoc, addResult);
        });

    LLVM_DEBUG(llvm::dbgs() << "  Created fused matmul op:\n");
    LLVM_DEBUG(fusedMatmulOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    rewriter.replaceOp(addOp, fusedMatmulOp.getResults());

    LLVM_DEBUG(llvm::dbgs() << "=== Rewrite successful ===\n\n");
    return success();
  }
};

struct FuseFCPass : public PassWrapper<FuseFCPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseFCPass)

  StringRef getArgument() const final { return "fuse-fc"; }

  StringRef getDescription() const final {
    return "Fuse linalg.transpose + linalg.matmul + bias-add into a "
           "sequence of linalg.generic ops.";
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n\n====================================\n");
    LLVM_DEBUG(llvm::dbgs() << "Starting FuseFCPass\n");
    LLVM_DEBUG(llvm::dbgs() << "====================================\n\n");

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseFCPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "FuseFCPass: Pattern application failed\n");
      signalPassFailure();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "\n====================================\n");
      LLVM_DEBUG(llvm::dbgs() << "FuseFCPass completed successfully\n");
      LLVM_DEBUG(llvm::dbgs() << "====================================\n\n");
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerFuseFCPass() {
  PassRegistration<FuseFCPass>();
}
} // namespace buddy
} // namespace mlir
