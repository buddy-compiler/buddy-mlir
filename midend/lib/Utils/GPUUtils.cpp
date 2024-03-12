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
// This file implements GPU dialect specific utility functions for the buddy
// compiler ecosystem.
//
//===----------------------------------------------------------------------===//

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UTILS_GPUUTILS_DEF
#define UTILS_GPUUTILS_DEF

#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
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
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include "Utils/GPUUtils.h"

#include <optional>

#define DEBUG_TYPE "iree-codegen-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

static constexpr unsigned kShuffleBitWidth = 32;

namespace mlir::buddy {
namespace buddygpu {

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                         unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

// llvm::SmallVector<mlir::linalg::ProcInfo, 2>
// getSubgroupIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
//                         unsigned warpSize, unsigned numDims,
//                         llvm::ArrayRef<int64_t> numSubgroups) {
//   assert(numDims <= kNumGPUDims);
//   llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
//   std::array<gpu::Dimension, kNumGPUDims> dimAttr{
//       gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
//   mlir::Type indexType = builder.getIndexType();
//   for (unsigned i = 0; i < numDims; ++i) {
//     mlir::Value subgroupId =
//         builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
//     if (i == 0) {
//       mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
//       subgroupId = mlir::affine::makeComposedAffineApply(
//           builder, loc, d0.floorDiv(builder.getAffineConstantExpr(warpSize)),
//           {subgroupId});
//     }
//     procInfo[numDims - 1 - i] = {
//         subgroupId,
//         builder.create<mlir::arith::ConstantOp>(
//             loc, builder.getIndexAttr(numSubgroups[i])),
//         linalg::DistributionMethod::Cyclic};
//   }
//   return procInfo;
// }

// std::array<int64_t, 3> getWorkgroupSize(mlir::func::FuncOp funcOp) {
//   std::array<int64_t, 3> workgroupSize;
//   FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
//       mlir::iree_compiler::getEntryPoint(funcOp);
//   std::optional<mlir::ArrayAttr> workgroupSizeAttr =
//       exportOp->getWorkgroupSize();
//   assert(workgroupSizeAttr.has_value());
//   for (auto [index, attr] : llvm::enumerate(workgroupSizeAttr.value())) {
//     workgroupSize[index] =
//         llvm::cast<mlir::IntegerAttr>(attr).getValue().getZExtValue();
//   }
//   return workgroupSize;
// }

//===----------------------------------------------------------------------===//
// GPU vectorization
//===----------------------------------------------------------------------===//

bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize) {
  // Verify that each dimension of the shape can be distributed on the
  // threads
  // For zero dim tensor, consider it's too small to access using all threads.
  if (shape.size() == 0)
    return false;
  int64_t threadsAvailable = threadCount;
  for (const auto &[index, dim] : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = index == 0 ? vectorSize : 1;
    int64_t numThreads = dim / numElementPerThread;
    if (numThreads == 0)
      return false;
    if (numThreads > threadsAvailable) {
      // If there are no enough remaining threads to distribute the current
      // dimension, try to use all remaining threads. But we still need to make
      // sure all work can be distributed to these threads evenly.
      if (numThreads % threadsAvailable != 0)
        return false;
      numThreads = threadsAvailable;
    }
    if (threadsAvailable % numThreads != 0)
      return false;
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1)
      break;
  }
  return threadsAvailable == 1;
}

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
// GPU workgroup memory
//===----------------------------------------------------------------------===//

std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                             memref::SubViewOp subview,
                                             ArrayRef<Value> sizeBounds,
                                             DataLayout &) {
  OpBuilder::InsertionGuard guard(builder);

  func::FuncOp funcOp = subview->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return std::nullopt;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t, 2> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value)))
      return std::nullopt;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  auto type = MemRefType::get(
      shape, subview.getType().getElementType(), MemRefLayoutAttrInterface{},
      gpu::AddressSpaceAttr::get(builder.getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));
  Value buffer = builder.create<memref::AllocOp>(funcOp.getLoc(), type);
  return buffer;
}

// LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
//   return success();
// }

// LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
//   Operation *copyOp = b.create<memref::CopyOp>(src.getLoc(), src, dst);
//   setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
//   return success();
// }

// static bool propagateCopyDestIntoProducerFill(memref::CopyOp copyOp) {
//   // Look for a fill Op writing into the copyOp source.
//   Operation *prevOp = copyOp->getPrevNode();
//   while (prevOp) {
//     if (isMemoryEffectFree(prevOp)) {
//       prevOp = prevOp->getPrevNode();
//       continue;
//     }

//     auto fillOp = dyn_cast<linalg::FillOp>(prevOp);
//     if (!fillOp)
//       break;
//     if (fillOp.output() != copyOp.getSource())
//       break;
//     // Move the fillOp and change the destination to the copy destination.
//     fillOp->moveBefore(copyOp);
//     fillOp.getOutputsMutable().assign(copyOp.getTarget());
//     return true;
//   }
//   return false;
// }

// // Split input/output operand from copy from shared memory into a separate
// // input.
// static void insertInputValueIntoGeneric(Value source, linalg::GenericOp op) {
//   SmallVector<Value> newOperands;
//   SmallVector<AffineMap> maps;
//   for (OpOperand *in : op.getDpsInputOperands()) {
//     newOperands.push_back(in->get());
//     maps.push_back(op.getMatchingIndexingMap(in));
//   }
//   newOperands.push_back(source);
//   assert(op.getNumDpsInits() == 1);
//   OpOperand *outOperand = op.getDpsInitOperand(0);
//   maps.push_back(op.getMatchingIndexingMap(outOperand));
//   maps.push_back(op.getMatchingIndexingMap(outOperand));
//   Location loc = op.getLoc();
//   SmallVector<utils::IteratorType> iterTypes(op.getNumLoops(),
//                                              utils::IteratorType::parallel);
//   OpBuilder builder(op);
//   auto newOp = builder.create<linalg::GenericOp>(
//       loc, newOperands, outOperand->get(), maps, iterTypes);
//   newOp.getRegion().getBlocks().splice(newOp.getRegion().begin(),
//                                        op.getRegion().getBlocks());

//   Block &payload = newOp.getRegion().front();
//   payload.addArgument(payload.getArguments().back().getType(), loc);
//   setMarker(newOp, getCopyToWorkgroupMemoryMarker());
// }

// /// Propagate the shared memory copy into the consumer op if it's a fully
// /// parallel linalg.generic.
// static bool
// propagateCopySourceIntoConsumerGeneric(memref::CopyOp copyOp,
//                                        SmallVector<Operation *> &toDelete) {
//   // Look for a generic Op reading the copyOp target.
//   Operation *nextOp = copyOp->getNextNode();
//   while (nextOp) {
//     if (isMemoryEffectFree(nextOp)) {
//       nextOp = nextOp->getNextNode();
//       continue;
//     }
//     auto consumer = dyn_cast<linalg::GenericOp>(nextOp);
//     if (!consumer || consumer.getNumDpsInits() != 1 ||
//         !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0))
//              .isIdentity())
//       break;
//     if (*consumer.getOutputs().begin() != copyOp.getTarget())
//       break;
//     insertInputValueIntoGeneric(copyOp.getSource(), consumer);
//     toDelete.push_back(consumer);
//     return true;
//   }
//   return false;
// }

// /// This is needed because we are doing promotion to shared memory on buffers.
// /// This is a fragile and temporary solution until we move to be able to do this
// /// kind of transformations on tensors.
// void propagateSharedMemoryCopy(func::FuncOp funcOp) {
//   SmallVector<Operation *> toDelete;
//   funcOp.walk([&toDelete](memref::CopyOp copyOp) {
//     if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
//       if (propagateCopyDestIntoProducerFill(copyOp) ||
//           propagateCopySourceIntoConsumerGeneric(copyOp, toDelete))
//         toDelete.push_back(copyOp.getOperation());
//     }
//   });
//   for (Operation *op : toDelete)
//     op->erase();
// }

// void insertBarriersAroundSharedMemoryCopy(func::FuncOp funcOp) {
//   OpBuilder builder(funcOp.getContext());
//   // Insert barriers before and after copies to workgroup memory and skip
//   // insert barriers between back to back copy to workgroup memory.
//   funcOp.walk([&builder](Operation *copyOp) {
//     if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
//       Operation *prevOp = copyOp->getPrevNode();
//       if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
//         builder.setInsertionPoint(copyOp);
//         builder.create<gpu::BarrierOp>(copyOp->getLoc());
//       }
//       Operation *nextOp = copyOp->getNextNode();
//       if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
//         builder.setInsertionPointAfter(copyOp);
//         builder.create<gpu::BarrierOp>(copyOp->getLoc());
//       }
//     }
//   });
// }

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

/// Emit warp reduction code sequence for a given input.
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           vector::CombiningKind kind, uint32_t warpSize,
                           uint32_t numLaneToReduce) {
  VectorType unpackedType = llvm::dyn_cast<VectorType>(input.getType());
  Value laneVal = input;
  assert(llvm::isPowerOf2_32(numLaneToReduce));
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < numLaneToReduce; i <<= 1) {
    Value shuffleInput = laneVal;
    if (unpackedType) {
      shuffleInput = packVectorToSupportedWidth(loc, builder, laneVal);
    }
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, shuffleInput, i,
                                                 /*width=*/warpSize,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    if (unpackedType) {
      shuffled = unpackToVector(loc, builder, shuffled, unpackedType);
    }
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  // Broadcast the result to all the lanes.
  if (warpSize != numLaneToReduce) {
    if (unpackedType) {
      laneVal = packVectorToSupportedWidth(loc, builder, laneVal);
    }
    laneVal = builder
                  .create<gpu::ShuffleOp>(loc, laneVal, 0,
                                          /*width=*/warpSize,
                                          /*mode=*/gpu::ShuffleMode::IDX)
                  .getShuffleResult();
    if (unpackedType) {
      laneVal = unpackToVector(loc, builder, laneVal, unpackedType);
    }
  }
  return laneVal;
}

// List of identity elements by operation.
// https://en.wikipedia.org/wiki/Identity_element
static TypedAttr getCombiningKindIdentity(OpBuilder &builder,
                                          vector::CombiningKind combiningKind,
                                          Type type) {
  switch (combiningKind) {
  case vector::CombiningKind::ADD:
    return builder.getZeroAttr(type);
  case vector::CombiningKind::MUL: {
    if (type.isIntOrIndex()) {
      return builder.getIntegerAttr(type, 1);
    }
    return builder.getFloatAttr(type, 1);
  }
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
    return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::max());
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
    return builder.getIntegerAttr(type, std::numeric_limits<int64_t>::min());
  case vector::CombiningKind::AND:
    return builder.getIntegerAttr(type, 1);
  case vector::CombiningKind::OR:
  case vector::CombiningKind::XOR:
    return builder.getZeroAttr(type);
  case vector::CombiningKind::MINIMUMF:
  case vector::CombiningKind::MINF: {
    auto posInfApFloat = APFloat::getInf(
        llvm::cast<FloatType>(type).getFloatSemantics(), /*Negative=*/false);
    return builder.getFloatAttr(type, posInfApFloat);
  }
  case vector::CombiningKind::MAXIMUMF:
  case vector::CombiningKind::MAXF: {
    auto negInfApFloat = APFloat::getInf(
        llvm::cast<FloatType>(type).getFloatSemantics(), /*Negative=*/true);
    return builder.getFloatAttr(type, negInfApFloat);
  }
  }
  return TypedAttr();
}

/// Compute the value on a single thread to get per lane reduction value.
/// If bit-width is not supported on shuffle operations, and a lower precision,
/// we represent them as a vector S.T we can pack them into a single 32-bit
/// width for shuffles.
static Value reduceToSupportedWidth(Location loc, OpBuilder &builder,
                                    Value input, vector::CombiningKind kind) {
  auto vecType = llvm::cast<VectorType>(input.getType());
  Type elementType = vecType.getElementType();
  int64_t vecSize = vecType.getDimSize(0);
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  // Simply reduce if it's already 32 bits.
  if (bitWidth == kShuffleBitWidth) {
    return builder.create<vector::ReductionOp>(loc, kind, input);
  }
  assert(kShuffleBitWidth % bitWidth == 0 &&
         "Bitwidth needs to be able to be packed into shuffle-bitwidth.");
  int64_t unrollCount = kShuffleBitWidth / bitWidth;
  // Original size needs to be divisble by or less than unroll count to
  // determine slice size.
  assert(vecSize % unrollCount == 0 || vecSize < unrollCount);
  unsigned sliceSize = vecSize / unrollCount;
  VectorType unrolledLaneValType = VectorType::get({unrollCount}, elementType);
  Value perLaneReduction = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(unrolledLaneValType));
  if (vecSize % unrollCount == 0) {
    // Unroll reductions s.t we can pack into a supported 32-bitWidth format.
    for (int64_t i = 0; i < unrollCount; i++) {
      Value laneValSlice = builder.create<vector::ExtractStridedSliceOp>(
          loc, input,
          /*offsets=*/ArrayRef<int64_t>{sliceSize * i},
          /*sizes=*/ArrayRef<int64_t>{sliceSize},
          /*strides=*/ArrayRef<int64_t>{1});
      Value reductionSlice =
          builder.create<vector::ReductionOp>(loc, kind, laneValSlice);
      SmallVector<int64_t> perLaneUnrollId = {i};
      perLaneReduction = builder.create<vector::InsertOp>(
          loc, reductionSlice, perLaneReduction, perLaneUnrollId);
    }
  } else {
    // In cases where vecSize < unrollCount, we would pad the vector
    // with identity elements until it's total bit size is 32.
    TypedAttr identityAttr =
        getCombiningKindIdentity(builder, kind, elementType);
    identityAttr = DenseElementsAttr::get(unrolledLaneValType, identityAttr);
    Value identity = builder.create<arith::ConstantOp>(loc, unrolledLaneValType,
                                                       identityAttr);
    perLaneReduction = builder.create<vector::InsertStridedSliceOp>(
        loc, input, identity, /*offsets=*/ArrayRef<int64_t>{0},
        /*strides=*/ArrayRef<int64_t>{1});
  }
  return perLaneReduction;
}

/// Emit identity variable.
static Value getCombiningIdentityValue(Location loc, OpBuilder &builder,
                                       vector::CombiningKind kind,
                                       Type identityType) {
  auto vectorType = llvm::dyn_cast<VectorType>(identityType);
  Type elementType = identityType;
  if (vectorType) {
    elementType = vectorType.getElementType();
  }
  TypedAttr identityAttr = getCombiningKindIdentity(builder, kind, elementType);
  if (vectorType) {
    identityAttr = DenseElementsAttr::get(vectorType, identityAttr);
  }
  assert(identityAttr && "Unknown identity value for the reduction");
  Value identity =
      builder.create<arith::ConstantOp>(loc, identityType, identityAttr);
  return identity;
}

/// Return a matching GPU reduction operations.
static std::optional<gpu::AllReduceOperation>
combiningKindToAllReduce(vector::CombiningKind kind) {
  using gpu::AllReduceOperation;
  using vector::CombiningKind;

  switch (kind) {
  case CombiningKind::ADD:
    return AllReduceOperation::ADD;
  case CombiningKind::AND:
    return AllReduceOperation::AND;
  case CombiningKind::MUL:
    return AllReduceOperation::MUL;
  case CombiningKind::OR:
    return AllReduceOperation::OR;
  case CombiningKind::XOR:
    return AllReduceOperation::XOR;
  // Currently, the min/max reductions are not well-defined in the gpu dialect.
  // See https://github.com/llvm/llvm-project/issues/72354.
  default:
    break;
  }
  return std::nullopt;
}

/// Emit reduction across a group for a given input.
Value emitGPUGroupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            unsigned int warpSize, bool expandSubgroupReduce) {
  assert(
      size % warpSize == 0 &&
      "Group reduction only support for sizes aligned on warp size for now.");

  if (!expandSubgroupReduce && size == warpSize) {
    if (auto gpuReduceKind = combiningKindToAllReduce(kind)) {
      // Simple case -- emit `gpu.subgroup_reduce` directly.
      Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
      return builder.create<gpu::SubgroupReduceOp>(loc, laneVal,
                                                   *gpuReduceKind);
    }
  }

  // More-involved case -- generate `gpu.shuffle` ops over i32 values (using the
  // butterfly shuffle algorithm).
  //
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = reduceToSupportedWidth(loc, builder, input, kind);
  laneVal = warpReduction(loc, builder, laneVal, kind, warpSize, warpSize);
  // if we have more than one warp, reduce across warps.
  if (size > warpSize) {
    uint32_t numWarp = size / warpSize;
    assert(numWarp <= warpSize &&
           "Only support 1 level, need to implement recursive/loop for this "
           "case.");
    auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    MemRefType memrefType =
        MemRefType::get(numWarp, laneVal.getType(), MemRefLayoutAttrInterface{},
                        addressSpaceAttr);
    Value alloc = builder.create<memref::AllocOp>(loc, memrefType);
    Value threadX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
                                                    gpu::Dimension::x);
    Value cstWarpSize = builder.create<arith::ConstantIndexOp>(loc, warpSize);
    Value warpId = builder.create<arith::DivUIOp>(loc, threadX, cstWarpSize);
    Value laneId = builder.create<arith::RemUIOp>(loc, threadX, cstWarpSize);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value lane0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                laneId, zero);
    // Store the reduction for each warp.
    SmallVector<Value> indices = {warpId};
    builder.create<scf::IfOp>(loc, lane0, [&](OpBuilder &b, Location l) {
      b.create<memref::StoreOp>(l, laneVal, alloc, indices);
      b.create<scf::YieldOp>(l, std::nullopt);
    });
    builder.create<gpu::BarrierOp>(loc);
    // Further reduce the outputs from each warps with a single warp reduce.
    Value memrefSize = builder.create<arith::ConstantIndexOp>(loc, numWarp - 1);
    Value laneIdInBounds =
        builder.create<arith::MinUIOp>(loc, laneId, memrefSize);
    Value loadVal = builder.create<memref::LoadOp>(loc, alloc, laneIdInBounds);
    Value cstNumWarp = builder.create<arith::ConstantIndexOp>(loc, numWarp);
    if (!llvm::isPowerOf2_32(numWarp)) {
      // Pad with identity element if numel < warpSize for valid warp reduction.
      Value useIdentityElement = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, laneId, cstNumWarp);
      numWarp = llvm::PowerOf2Ceil(numWarp);
      Value identity =
          getCombiningIdentityValue(loc, builder, kind, loadVal.getType());
      loadVal = builder.create<arith::SelectOp>(loc, useIdentityElement,
                                                identity, loadVal);
    }
    laneVal = warpReduction(loc, builder, loadVal, kind, warpSize, numWarp);
  }
  // Handles cases for sub-32bit precision where output is still in vector form.
  if (llvm::isa<VectorType>(laneVal.getType())) {
    laneVal = builder.create<vector::ReductionOp>(loc, kind, laneVal);
  }
  return laneVal;
}

std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation *op) {
  // Currently hardcode the size of wmma operation. When more cases are
  // supported this should be picked based on what the backend supports.
  int64_t m = 16;
  int64_t n = 16;
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t k = contract.getLhsType().getElementType().isF16() ? 16 : 8;
    SmallVector<int64_t> nativeSize(contract.getIteratorTypes().size() - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    if (writeOp.getVectorType().getRank() < 2)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(writeOp.getVectorType().getRank() - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
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
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      // TODO: The condition for unrolling elementwise should be restricted
      // only to operations that need unrolling (connected to the contract).
      if (vecType.getRank() < 2)
        return std::nullopt;

      // First check whether there is a slice to infer the shape from. This is
      // required for cases where the accumulator type differs from the input
      // types, in which case we will see an `arith.ext_` between the contract
      // and transfer_read which needs to be unrolled.
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
      if (sliceType)
        return llvm::to_vector(sliceType.getShape());

      // Else unroll for trailing elementwise.
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
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
        // IREE uses the largest tiled load, i.e., LDSMx4.

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
  auto addrSpace = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(
      memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

//===----------------------------------------------------------------------===//
// GPU CodeGen op filter
//===----------------------------------------------------------------------===//

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
// bool sharedMemTransposeFilter(AffineMap indexMap) {
//   if (!indexMap.isEmpty() && indexMap.isPermutation()) {
//     // Ensure that the fasted moving dimension (the last one) is permuted,
//     // Otherwise shared memory promotion will not benefit the operation.
//     if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
//         indexMap.getNumDims() - 1) {
//       return true;
//     }
//   }
//   return false;
// }

//===----------------------------------------------------------------------===//
// Utility from compiler/src/iree/compiler/Codegen/Transforms/Transforms.h
//===----------------------------------------------------------------------===//

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
        presburger::BoundType::UB, dynamicSize, /*dim=*/std::nullopt,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(ub)) {
      return std::nullopt;
    }
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

//===----------------------------------------------------------------------===//
// Utility from compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.cpp
//===----------------------------------------------------------------------===//

static Value getMemrefOperand(Operation *op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

static bool isContiguousStore(Operation *write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      LDBG("--not a contiguous store op: " << *write);
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  LDBG("--not a store op: " << write->getName().getStringRef());
  return false;
}

static bool isContiguousRead(Operation *read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      LDBG("--not a contiguous load op: " << *read);
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  LDBG("--not a load op: " << read->getName().getStringRef());
  return false;
}

struct MaskResult {
  vector::CreateMaskOp maskOp;
  vector::ExtractOp maybeExtractOp;
};
static MaskResult getMask(Operation *op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask())
    return MaskResult{};
  vector::ExtractOp maybeExtractOp =
      transferRead.getMask().getDefiningOp<vector::ExtractOp>();
  auto maskOp =
      maybeExtractOp
          ? maybeExtractOp.getVector().getDefiningOp<vector::CreateMaskOp>()
          : transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  if (maybeExtractOp) {
    if (maybeExtractOp.getStaticPosition().size() + 1 !=
        llvm::cast<VectorType>(maskOp->getResultTypes().front()).getRank()) {
      LDBG("----mask through extract unexpected position size -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    if (maybeExtractOp.getStaticPosition().size() != 1) {
      LDBG("----only mask through 2-D -> 1-D extract supported atm -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    LDBG("----mask through extract: " << maybeExtractOp);
  }
  return MaskResult{maskOp, maybeExtractOp};
}

static Value getMaskValue(RewriterBase &rewriter, Operation *op) {
  MaskResult maskResult = getMask(op);
  if (!maskResult.maskOp)
    return Value();
  Value count = maskResult.maskOp->getOperands().back();
  vector::ExtractOp maybeExtractOp = maskResult.maybeExtractOp;
  if (maybeExtractOp) {
    assert(maybeExtractOp.getStaticPosition().size() == 1 &&
           "expected single pos");
    int64_t sliceNum = maybeExtractOp.getStaticPosition()[0];
    // TODO: to support >2-D mask + extract, and all the cmp.
    Location loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        rewriter.create<arith::ConstantIndexOp>(loc, sliceNum),
        maskResult.maskOp->getOperands().front());
    count = rewriter.create<arith::SelectOp>(loc, cmp, count, zero);
  }
  return count;
}

static Value getValueStored(Operation *writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation *op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

/// Return `true` if the conversion to async copy is legal.
static bool resultsInSupportedAsyncCopy(MemRefType memrefType,
                                        Operation::operand_range indices,
                                        VectorType vecType) {
  constexpr int64_t kSupportedCpAsyncAlignmentsInBytes[3] = {4, 8, 16};
  // Condition 1: the vectory rank must be supported.
  if (vecType.hasRank() != 1) {
    LDBG("----> cp.async failed, not a 1-D vector: " << vecType);
    return false;
  }

  // Condition 2: the copy size must be supported.
  bool supportedCopySize = false;
  int64_t numElements = vecType.getNumElements();
  Type elementType = vecType.getElementType();
  for (int64_t alignmentInBytes : kSupportedCpAsyncAlignmentsInBytes) {
    if (alignmentInBytes * 8 ==
        numElements * elementType.getIntOrFloatBitWidth()) {
      supportedCopySize = true;
      break;
    }
  }
  if (!supportedCopySize) {
    LDBG("----> cp.async alignment failed, "
         << numElements << " elts * " << elementType.getIntOrFloatBitWidth()
         << "b/elem = " << numElements * elementType.getIntOrFloatBitWidth()
         << "b is not supported by cp.async");
    return false;
  }

  // TODO: Condition 3: the alignments must be supported. For cp.async the
  // NVIDIA doc (section 6.4.1) says: "The address must be naturally aligned to
  // a multiple of the access size. If an address is not properly aligned, the
  // resulting behavior is undefined.".
  return true;
}

void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync) {
  LDBG("Start asyncGroups: useMMASync=" << useMMASync);
  llvm::SmallSetVector<Operation *, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation *writeOp) {
    if (!isContiguousStore(writeOp))
      return WalkResult::advance();
    LDBG("--candidate writeOp: " << *writeOp);
    Value vectorVal = getValueStored(writeOp);
    if (llvm::cast<VectorType>(vectorVal.getType()).getRank() != 1) {
      LDBG("----writeOp is not an inbounds 1-D minor identity -> Skip");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(memrefOperand.getType()))) {
      LDBG("----address space is not workgroup -> Skip");
      return WalkResult::advance();
    }
    Operation *readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LDBG("----no contiguous readOp defining the writeOp -> Skip");
      return WalkResult::advance();
    }

    LDBG("--candidate readOp: " << *readOp);
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (transferRead.getMask()) {
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LDBG("----read padding value is not 0.f -> Skip");
          return WalkResult::advance();
        }
        auto maskResult = getMask(transferRead);
        if (!maskResult.maskOp) {
          LDBG("----read mask is not a vector.create_mask op -> Skip: "
               << transferRead.getMask());
          return WalkResult::advance();
        }
      }
    }

    // Check whether both accesses are supported before we emit: this is
    // necessary to ensure the correctness of DeviceAsyncCopyOp.
    VectorType vecType = llvm::cast<VectorType>(vectorVal.getType());
    Value storeBase = getMemrefOperand(writeOp);
    Value loadBase = getMemrefOperand(readOp);
    if (!resultsInSupportedAsyncCopy(cast<MemRefType>(loadBase.getType()),
                                     getIndices(readOp), vecType) ||
        !resultsInSupportedAsyncCopy(cast<MemRefType>(storeBase.getType()),
                                     getIndices(writeOp), vecType))
      return WalkResult::advance();

    LDBG("--writeOp can be made async -> SUCCESS");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation *> group;
    Operation *writeOp = *copyToSharedMem.begin();
    LDBG("--START a group from: " << *writeOp);
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation *nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation *readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                llvm::cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      LDBG("----> STOP accumulating into group due to: " << *nextNode);
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    for (Operation *writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      auto vectorType = llvm::cast<VectorType>(vectorVal.getType());
      int64_t numElements = vectorType.getNumElements();
      Operation *readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMaskValue(rewriter, readOp);
      auto dstMemref = llvm::cast<MemRefType>(storeBase.getType());
      int64_t sizeInBytes =
          (dstMemref.getElementTypeBitWidth() * numElements) / 8;
      UnitAttr bypassL1 =
          useMMASync && sizeInBytes == 16 ? rewriter.getUnitAttr() : UnitAttr();
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(numElements), mask,
          /*bypassL1=*/bypassL1);
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation *writeOp : group)
      rewriter.eraseOp(writeOp);
  }
}

} // namespace buddygpu
} // namespace mlir::buddy
#endif // UTILS_GPUUTILS_DEF
