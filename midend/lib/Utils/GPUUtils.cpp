//====- GPUUtils.cpp ------------------------------------------------------===//
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
//
// This file implements GPU dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_GPUUTILS_DEF
#define UTILS_GPUUTILS_DEF

#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include "Utils/GPUUtils.h"

#include <optional>

#define DEBUG_TYPE "buddy-codegen-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

static constexpr unsigned kShuffleBitWidth = 32;

namespace mlir::buddy {
namespace gpu {

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(expr.cast<AffineDimExpr>().getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

//===----------------------------------------------------------------------===//
// Reduction utils
//===----------------------------------------------------------------------===//

/// Packs scalar element to it's vector equivalent.
/// (i.e f16 -> vector<1xf16> and f32 -> vector<1xf32>)
static Value promoteElementToVector(Location loc, OpBuilder &builder,
                                    Value input) {
  VectorType vectorTypeBroadcast = VectorType::get({1}, input.getType());
  Value vectorInput =
      builder.create<vector::BroadcastOp>(loc, vectorTypeBroadcast, input);
  return vectorInput;
}

Value packVectorToSupportedWidth(Location loc, OpBuilder &builder,
                                 Value input) {
  LLVM_DEBUG({
    auto vecType = input.getType().cast<VectorType>();
    Type elementType = vecType.getElementType();
    assert(vecType.getDimSize(0) * elementType.getIntOrFloatBitWidth() ==
               kShuffleBitWidth &&
           "vecSize * vecBitWidth needs to packable into 32-bitwidth.");
    assert(elementType.isIntOrFloat() &&
           "Only int and float packing is supported.");
  });
  VectorType packed32Type = VectorType::get({1}, builder.getI32Type());
  Value packedInputVec =
      builder.create<vector::BitCastOp>(loc, packed32Type, input);
  Value packedInput = builder.create<vector::ExtractOp>(loc, packedInputVec, 0);
  return packedInput;
}

Value unpackToVector(Location loc, OpBuilder &builder, Value packedInput,
                     VectorType targetVecType) {
  LLVM_DEBUG({
    Type packedType = packedInput.getType();
    assert(packedType.isIntOrFloat() && "Only ints and floats are unpackable.");
    Type elementType = targetVecType.getElementType();
    assert(targetVecType.getDimSize(0) * elementType.getIntOrFloatBitWidth() ==
               packedType.getIntOrFloatBitWidth() &&
           "packed width needs to be unpackable to vecSize * vecBitWidth.");
  });
  Value packedVector = promoteElementToVector(loc, builder, packedInput);
  Value unpackedVector =
      builder.create<vector::BitCastOp>(loc, targetVecType, packedVector);
  return unpackedVector;
}

//===----------------------------------------------------------------------===//
// getMmaNativeVectorSize
//===----------------------------------------------------------------------===//
/// Returns vector::ContractionOp operand's index where the result is used.
static std::optional<int>
getVectorContractOpOperandId(vector::ContractionOp contractOp,
                             OpResult result) {
  if (contractOp.getLhs() == result)
    return 0;
  if (contractOp.getRhs() == result)
    return 1;
  if (contractOp.getAcc() == result)
    return 2;
  return std::nullopt;
}

/// Returns vector::ContractionOp operand's index  where the
/// vector::TransferReadOp is consumed either consumed directly or via
/// vector::ExtractStridedSliceOp.
static std::optional<int>
getVectorContractOpOperandIdForVectorReadOp(Operation *op) {
  vector::ContractionOp contractOp;

  // Check if the vector::TransferReadOp is consumed directly by
  // vector::ContractionOp.
  if (op->use_empty())
    return std::nullopt;
  Operation *firstLevelUser = *((op->getUsers()).begin());
  if (!firstLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(firstLevelUser))
    return getVectorContractOpOperandId(contractOp, op->getResult(0));

  // Check if the vector::TransferReadOp is consumed indirectly by
  // vector::ContractionOp. Only check until the second level of use-def chain.
  if (firstLevelUser->use_empty())
    return std::nullopt;
  Operation *secondLevelUser = *((firstLevelUser->getUsers()).begin());
  if (!secondLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(secondLevelUser))
    return getVectorContractOpOperandId(contractOp,
                                        firstLevelUser->getResult(0));
  return std::nullopt;
}

/// Helper function to return native size for MMA.SYNC-based operations.
std::optional<SmallVector<int64_t>> getMmaNativeVectorSize(Operation *op) {
  // Shape of native Tensor Core GPU mma.sync operations.
  int64_t mmaShapeM = 16;
  int64_t mmaShapeN = 8;
  int64_t mmaShapeK;

  // Shape the mma.sync warp-level operation.
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    Type sourceType = contract.getLhsType().getElementType();

    // Set mmaShapeK based on sourceType.
    if (sourceType.isInteger(4))
      mmaShapeK = 64;
    else if (sourceType.isInteger(8))
      mmaShapeK = 32;
    else if (sourceType.isF16() || sourceType.isBF16())
      mmaShapeK = 16;
    else if (sourceType.isF32())
      mmaShapeK = 8;
    else {
      LDBG("unsupported shape for vector.contract: ");
      return std::nullopt;
    }

    // Initialize/set the starting dims of the ranked shape, such as batch,
    // to 1.
    SmallVector<int64_t> mmaShape(contract.getIteratorTypes().size() - 3, 1);
    mmaShape.append({mmaShapeM, mmaShapeN, mmaShapeK});
    LLVM_DEBUG({
      llvm::interleaveComma(mmaShape, DBGS() << "shape for vector.contract: ");
      llvm::dbgs() << "\n";
    });
    return mmaShape;
  }

  // Shape of warp-level vector write operation.
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    if (writeOp.getVectorType().getRank() < 2)
      return std::nullopt;
    SmallVector<int64_t> outputShape(writeOp.getVectorType().getRank() - 2, 1);
    outputShape.append({mmaShapeM, mmaShapeN});
    LLVM_DEBUG({
      llvm::interleaveComma(outputShape,
                            DBGS() << "shape for vector.xfer_write: ");
      llvm::dbgs() << "\n";
    });
    return outputShape;
  }

  // Shape of warp-level vector read (load) operation.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    auto resultVectorType =
        llvm::cast<VectorType>(readOp.getVector().getType());
    Type resultElementType = resultVectorType.getElementType();

    std::optional<int> operandId =
        getVectorContractOpOperandIdForVectorReadOp(op);
    if (!operandId) {
      LLVM_DEBUG({
        DBGS() << "Failed to get operandId for vector::xfer_read: " << *op
               << "\n";
      });
      return std::nullopt;
    }

    // Loading F16 values from Shared Memory to Registers.
    if (resultElementType.isF16() || resultElementType.isBF16()) {
      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }

      // For matrixA and matrixB.
      if (*operandId == 0 || *operandId == 1) {
        // MmaSyncOp input operands: matrixA and matrixB.
        // LDSMx1, x2, x4:
        // - LDSMx1 loads a 1 tile  of 8x8.
        // - LDSMx2 loads a 2 tiles of 8x8.
        // - LDSMx4 loads a 4 tiles of 8x8. (in use)
        // here uses the largest tiled load, i.e., LDSMx4.

        // MmaSyncOp source operand: matrixC.
        // matrixC is also read/written in tiled block of 16x16. In the pass
        // OptimizeVectorTransfer, matrixC reads are moved above the mainloop
        // and writes are moved below the mainloop. Thus, mma.sync read/write
        // accumulator inplace.
        SmallVector<int64_t> readShape;
        readShape.append({16, 16});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
    }

    // Loading F32 values from Shared Memory to Registers.
    if (resultElementType.isF32()) {
      // Set mmaShapeK for F32 datatype mma.sync.f32.tf32.m16n8k8.
      mmaShapeK = 8;

      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixA.
      if (*operandId == 0) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeK});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixB.
      if (*operandId == 1) {
        // Do not use ldmatrix for matrixB.
        // Transfer read ops may need different shapes based on how they are
        // being used. For simplicity just match the shape used by the extract
        // strided op.
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
        LLVM_DEBUG({
          llvm::interleaveComma(sliceType.getShape(),
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return llvm::to_vector(sliceType.getShape());
      }
    }
  }
  LDBG("unsupported shape for " << op->getName().getStringRef());
  return std::nullopt;
}

bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace = llvm::dyn_cast_if_present<mlir::gpu::AddressSpaceAttr>(
      memrefType.getMemorySpace());
  return addrSpace && addrSpace.getValue() ==
                          mlir::gpu::GPUDialect::getWorkgroupAddressSpace();
}

template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  Location loc, MemRefType allocLikeType,
                                  ValueRange dynamicSizes,
                                  std::optional<uint64_t> alignment) {
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;
  // For static case just create a new allocation in the entry block of the same
  // size. No need to insert a subview.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Value allocation =
        builder.create<AllocLikeOpType>(loc, allocLikeType, alignmentAttr);
    if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
      builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
      builder.create<memref::DeallocOp>(loc, allocation);
    }
    return allocation;
  }

  /// For the dynamic but bounded case, insert an allocation of the shape of the
  /// bounds, and a subview of the required size to be used as a replacement.
  SmallVector<int64_t> staticShape;
  SmallVector<OpFoldResult> subviewSizes;
  staticShape.reserve(allocLikeType.getRank());
  subviewSizes.reserve(allocLikeType.getRank());

  int index = 0;
  for (auto dimSize : allocLikeType.getShape()) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticShape.push_back(dimSize);
      subviewSizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dynamicSize = dynamicSizes[index++];
    auto ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, dynamicSize,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    staticShape.push_back(ub.value());
    subviewSizes.push_back(dynamicSize);
  }
  SmallVector<OpFoldResult> offsets(allocLikeType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocLikeType.getRank(),
                                    builder.getIndexAttr(1));

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto allocationType =
        MemRefType::get(staticShape, allocLikeType.getElementType());
    allocation =
        builder.create<AllocLikeOpType>(loc, allocationType, alignmentAttr);
  }

  Value subviewOp = builder.create<memref::SubViewOp>(loc, allocation, offsets,
                                                      subviewSizes, strides);

  if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
    builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
    builder.create<memref::DeallocOp>(loc, allocation);
  }
  return subviewOp;
}

template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  AllocLikeOpType allocLikeOp) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(allocLikeOp);
  return hoistOneStaticallyBoundAllocation<AllocLikeOpType>(
      funcOp, builder, allocLikeOp.getLoc(), allocLikeOp.getType(),
      allocLikeOp.getDynamicSizes(), allocLikeOp.getAlignment());
}

/// Some uses of a AllocLike can be replaced with a `memref.subview`
/// easily. Other uses (like a use in a `scf.yield` or `func.return`) are
/// non-trivial because of compatibility between types of different SSA values.
static bool isUseReplaceableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  return isa<linalg::LinalgOp, memref::DeallocOp, memref::StoreOp,
             memref::SubViewOp>(user);
}

template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(RewriterBase &rewriter,
                                           func::FuncOp funcOp) {
  SmallVector<AllocLikeOpType> allocLikeOps;

  // Collect all allocLikes that are hoistable.
  funcOp.walk([&](AllocLikeOpType allocLikeOp) {
    if (allocLikeOp->getBlock() == &funcOp.getBody().front())
      return;
    if (allocLikeOp.getDynamicSizes().empty()) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
    if (llvm::all_of(allocLikeOp->getUses(), [](OpOperand &use) {
          return isUseReplaceableWithSubview(use);
        })) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
  });

  // Hoist the allocLikes and replace all uses.
  for (auto allocLikeOp : allocLikeOps) {
    // Record potential memref::DeallocOps to clean up after hoisting occurs.
    SmallVector<memref::DeallocOp> deallocOps;
    for (Operation *user : allocLikeOp->getUsers()) {
      auto dealloc = dyn_cast<memref::DeallocOp>(user);
      if (dealloc)
        deallocOps.push_back(dealloc);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Alloca Op : ";
      allocLikeOp->dump();
      int numUses = std::distance(allocLikeOp.getResult().use_begin(),
                                  allocLikeOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement =
        hoistOneStaticallyBoundAllocation(funcOp, rewriter, allocLikeOp);
    if (!replacement)
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    rewriter.replaceOp(allocLikeOp, replacementVal);

    for (memref::DeallocOp deallocOp : deallocOps)
      rewriter.eraseOp(deallocOp);
  }
}

/// Explicit instantiations for `hoistStaticallyBoundAllocationsInFunc` and
/// dependent functions.
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocOp>(
    func::FuncOp funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    func::FuncOp funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocOp>(func::FuncOp funcOp,
                                                   OpBuilder &builder,
                                                   memref::AllocOp allocLikeOp);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    func::FuncOp funcOp, OpBuilder &builder, memref::AllocaOp allocLikeOp);
template void
hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(RewriterBase &rewriter,
                                                       func::FuncOp funcOp);
template void
hoistStaticallyBoundAllocationsInFunc<memref::AllocaOp>(RewriterBase &rewriter,
                                                        func::FuncOp funcOp);

} // namespace gpu
} // namespace mlir::buddy
#endif // UTILS_GPUUTILS_DEF
