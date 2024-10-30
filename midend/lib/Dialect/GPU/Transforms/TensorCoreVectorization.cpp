//===- TensorCoreVectorization.cpp ----------------------------------------===//
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
// The process in this file references the IREE project,
// which is hereby acknowledged.
// For the license of the IREE project
// please see: https://github.com/iree-org/iree/blob/main/LICENSE
//
//===----------------------------------------------------------------------===//

#include "GPU/Transforms/TensorCoreVectorization.h"
#include "PassDetail.h"
#include "Utils/GemmCodegenUtils.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::buddy;

namespace {

static void vectorizeLinalgOps(scf::ForallOp forallOp) {
  MLIRContext *ctx = forallOp->getContext();
  IRRewriter rewriter(ctx);
  forallOp->walk([&](Operation *op) {
    if (isa<linalg::FillOp, linalg::GenericOp, linalg::ContractionOpInterface>(
            op) &&
        hasMarker(op, getVectorizeMarkerAttrName())) {
      (void)linalg::vectorize(rewriter, op);
    }
    return WalkResult::advance();
  });
}

static std::optional<int64_t> getReadVectorInContractOperandId(Operation *op) {
  if (op->use_empty()) {
    return std::nullopt;
  }
  auto firstLevelUser = *(op->getUsers().begin());
  if (!firstLevelUser) {
    return std::nullopt;
  }
  if (isa<vector::ContractionOp>(firstLevelUser)) {
    vector::ContractionOp contractOp =
        dyn_cast<vector::ContractionOp>(firstLevelUser);
    if (contractOp.getLhs() == op->getResult(0)) {
      return 0;
    } else if (contractOp.getRhs() == op->getResult(0)) {
      return 1;
    } else if (contractOp.getAcc() == op->getResult(0)) {
      return 2;
    } else {
      return std::nullopt;
    }
  }
  if (firstLevelUser->use_empty()) {
    return std::nullopt;
  }

  auto secondLevelUser = *(firstLevelUser->getUsers().begin());
  if (!secondLevelUser) {
    return std::nullopt;
  }
  if (isa<vector::ContractionOp>(secondLevelUser)) {
    vector::ContractionOp contractOp =
        dyn_cast<vector::ContractionOp>(secondLevelUser);
    if (contractOp.getLhs() == firstLevelUser->getResult(0)) {
      return 0;
    } else if (contractOp.getRhs() == firstLevelUser->getResult(0)) {
      return 1;
    } else if (contractOp.getAcc() == firstLevelUser->getResult(0)) {
      return 2;
    } else {
      return std::nullopt;
    }
  }

  return std::nullopt;
}

static std::optional<SmallVector<int64_t>>
getUnrollTraversalOrder(Operation *op) {
  SmallVector<int64_t> order;
  vector::ContractionOp contractOp;
  if (isa<vector::ContractionOp>(op)) {
    contractOp = dyn_cast<vector::ContractionOp>(op);
  } else {
    return std::nullopt;
  }

  for (auto [index, iter] : llvm::enumerate(contractOp.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> lhsDims;

  for (AffineExpr expr : contractOp.getIndexingMapsArray()[0].getResults()) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      lhsDims.insert(dimExpr.getPosition());
    } else {
      return std::nullopt;
    }
  }

  for (auto [index, iter] : llvm::enumerate(contractOp.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && lhsDims.count(index)) {
      order.push_back(index);
    }
  }

  for (auto [index, iter] : llvm::enumerate(contractOp.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !lhsDims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>>
getTensorCoreNativeVectorShape(Operation *op) {
  int64_t mmaShapeM = 16;
  int64_t mmaShapeN = 8;
  int64_t mmaShapeK;

  // 1. Contract Op Vector needs to match mma.sync's size
  if (isa<vector::ContractionOp>(op)) {
    auto contractOp = dyn_cast<vector::ContractionOp>(op);
    Type mmaPrecision = contractOp.getLhsType().getElementType();
    if (mmaPrecision.isInteger(4)) {
      mmaShapeK = 64;
    } else if (mmaPrecision.isInteger(8)) {
      mmaShapeK = 32;
    } else if (mmaPrecision.isF16() || mmaPrecision.isBF16()) {
      mmaShapeK = 16;
    } else if (mmaPrecision.isF32()) {
      mmaShapeK = 8;
    } else {
      return std::nullopt;
    }
    // To handle BMM
    SmallVector<int64_t> mmaShape(contractOp.getIteratorTypes().size() - 3, 1);
    mmaShape.append({mmaShapeM, mmaShapeN, mmaShapeK});
    return mmaShape;
  }

  // 2. TransferWrite Op writes to smem/gmem directly in mma's result size
  //    vector size still in warp level
  if (isa<vector::TransferWriteOp>(op)) {
    auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op);
    if (transferWriteOp.getVectorType().getRank() < 2) {
      return std::nullopt;
    }
    SmallVector<int64_t> vectorWriteShape(
        transferWriteOp.getVectorType().getRank() - 2, 1);
    vectorWriteShape.append({mmaShapeM, mmaShapeN});
    return vectorWriteShape;
  }

  // 3. TranferRead Op
  if (isa<vector::TransferReadOp>(op)) {
    auto transferReadOp = dyn_cast<vector::TransferReadOp>(op);
    auto readVectorType = transferReadOp.getVector().getType();
    Type readVectorElementType = readVectorType.getElementType();
    std::optional<int64_t> optionalOperandId =
        getReadVectorInContractOperandId(op);
    if (!optionalOperandId.has_value()) {
      return std::nullopt;
    }
    int64_t operandId = optionalOperandId.value();
    // Load F16 Values from Shared Mem to Register
    if (readVectorElementType.isF16() || readVectorElementType.isBF16()) {
      // For Matrix A and Matrix B
      // use mma.sync.m16n8k16
      if (operandId == 0 || operandId == 1) {
        // MmaSyncOp input operands: matrixA and matrixB.
        // LDSMx1, x2, x4:
        // - LDSMx1 loads a 1 tile  of 8x8.
        // - LDSMx2 loads a 2 tiles of 8x8.
        // - LDSMx4 loads a 4 tiles of 8x8. (in use)
        // For Matrix B the ldmatrix result needed to be seperated
        // For Matrix A the ldmatrix result can be calculated directly
        SmallVector<int64_t> readVectorShape;
        readVectorShape.append({16, 16});
        return readVectorShape;
      }
      // For Matrix C
      if (operandId == 2) {
        SmallVector<int64_t> readVectorShape;
        readVectorShape.append({mmaShapeM, mmaShapeN});
        return readVectorShape;
      }
    }
    // Load F32 Values from Shared Mem to Register
    if (readVectorElementType.isF32()) {
      mmaShapeK = 8;
      if (operandId == 2) {
        SmallVector<int64_t> readVectorShape;
        readVectorShape.append({mmaShapeM, mmaShapeN});
        return readVectorShape;
      }
      if (operandId == 0) {
        SmallVector<int64_t> readVectorShape;
        readVectorShape.append({mmaShapeM, mmaShapeK});
        return readVectorShape;
      }
      if (operandId == 1) {
        // todo
        VectorType sliceType;
        for (Operation *users : op->getUsers()) {
          auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
          if (!extract)
            return std::nullopt;
          auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
          if (sliceType && sliceType != vecType)
            return std::nullopt;
          sliceType = vecType;
        }
        return llvm::to_vector(sliceType.getShape());
      }
    }
  }
  return std::nullopt;
}

struct TensorCoreVectorizationPass
    : public TensorCoreVectorizationBase<TensorCoreVectorizationPass> {
public:
  TensorCoreVectorizationPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto ctx = funcOp->getContext();
    if (!funcHasGemm(funcOp)) {
      return signalPassFailure();
    }

    auto forallOpOptional = getForallOpMappedToBlock(funcOp);
    if (!forallOpOptional.has_value()) {
      return signalPassFailure();
    }
    auto forallOp = forallOpOptional.value();

    // Use the following three steps to apply Vectorization (from Linalg to
    // Vector) Use vector.transfer_read / write to load / store the warp-level
    // tile Use vector.contract to express calculation
    vectorizeLinalgOps(forallOp);

    RewritePatternSet vectorizePatterns(ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizePatterns);
    vector::populateVectorReductionToContractPatterns(vectorizePatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorizePatterns)))) {
      return signalPassFailure();
    }

    // Prepare vector operations to be lowered to TensorCore operations
    RewritePatternSet vectorMatchTensorCorePatterns(ctx);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
        vectorMatchTensorCorePatterns);
    mlir::populatePrepareVectorToMMAPatterns(vectorMatchTensorCorePatterns,
                                             true);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorMatchTensorCorePatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet vectorContractPatterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
        vectorContractPatterns);
    mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns,
                                             /*useMMASync=*/true);
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(vectorContractPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet vectorUnrollPatterns(ctx);
    vector::UnrollVectorOptions unrollVectorOptions;
    unrollVectorOptions.setNativeShapeFn(getTensorCoreNativeVectorShape);
    unrollVectorOptions.setUnrollTraversalOrderFn(getUnrollTraversalOrder);
    vector::populateVectorUnrollPatterns(vectorUnrollPatterns,
                                         unrollVectorOptions);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorUnrollPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createTensorCoreVectorizationPass() {
  return std::make_unique<TensorCoreVectorizationPass>();
}
