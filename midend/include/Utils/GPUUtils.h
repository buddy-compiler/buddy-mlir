// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef INCLUDE_UTILS_GPUUTILS_H
#define INCLUDE_UTILS_GPUUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir{
namespace buddy::buddygpu{
static constexpr int32_t kNumGPUDims = 3;
static constexpr int32_t kWarpSize = 32;


//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//

llvm::SmallVector<linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(OpBuilder &builder, Location loc, unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize);

/// Computes subgroup ID and returns in (X, Y, Z) order.
///
/// Note that CUDA doesn't have a subgroupId equivalent so we are are computing
/// the subgroup ID based on the threadID. When tiling to warp we assume each
/// warp is full and we pick a workgroup size so that `workgroupSize.x %
/// warpSize == 0`. This is why we can have warpId = { threadId.x / warpSize,
/// threadId.y, threadId.z }.
// llvm::SmallVector<linalg::ProcInfo, 2>
// getSubgroupIdsAndCounts(OpBuilder &builder, Location loc, unsigned warpSize,
//                         unsigned numDims, llvm::ArrayRef<int64_t> numSubgroups);

/// Returns the workgroup size associated to the funcOp entry point.
// Temporaily disables this as it requires HAL op. It could be reimplemented
// using GPU dialect ops, but we currently don't need it.
//std::array<int64_t, 3> getWorkgroupSize(func::FuncOp funcOp);

//===----------------------------------------------------------------------===//
// GPU vectorization
//===----------------------------------------------------------------------===//

/// Returns true if we can use all threads to perform vectorized load/store of
/// the given `shape`.
bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize);

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract);

//===----------------------------------------------------------------------===//
// GPU workgroup memory
//===----------------------------------------------------------------------===//

/// Allocates GPU workgroup memory matching the given `subview`. If there are
/// dynamic dimensions, the bounds are in `sizeBounds`.
// std::optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
//                                              memref::SubViewOp subview,
//                                              ArrayRef<Value> sizeBounds,
//                                              DataLayout &);

// /// Deallocates GPU workgroup memory behind `buffer`.
// LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value buffer);

// /// Copies `src` value to `dst` in shared memory.
// LogicalResult copyToWorkgroupMemory(OpBuilder &builder, Value src, Value dst);

// /// Propagates shared memory copy to producer linalg.fill or consumer
// /// linalg.generic when possible.
// void propagateSharedMemoryCopy(func::FuncOp funcOp);

// /// Inserts barriers before and after shared memory copy.
// void insertBarriersAroundSharedMemoryCopy(func::FuncOp funcOp);

/// Emit reduction across a group for a given input. Emits `gpu.shuffle`
/// based reduction only when `expandSubgroupReduce` is set.
Value emitGPUGroupReduction(Location loc, OpBuilder &builder, Value input,
                            vector::CombiningKind kind, uint32_t size,
                            unsigned int warpSize, bool expandSubgroupReduce);

/// Return the native size of an operation used in contraction calculation.
// TODO: Make this take HW specific sizes.
std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation *op);

/// Helper function to return native size for MMA.SYNC-based operations.
std::optional<SmallVector<int64_t>> getMmaNativeVectorSize(Operation *op);

/// Return true if the given memref has workgroup memory space.
bool hasSharedMemoryAddressSpace(MemRefType memrefType);

/// Packs vector of lower precision into a single 32-bit width element.
/// (i.e <2xf16> -> i32 and <4xi8> -> i32)
Value packVectorToSupportedWidth(Location loc, OpBuilder &builder, Value input);

/// Unpack single scalar element into a target vector type.
/// (i.e i32 -> vector<4xi8> or f32 -> vector<2xf16>)
Value unpackToVector(Location loc, OpBuilder &builder, Value packedInput,
                     VectorType targetVecType);

//===----------------------------------------------------------------------===//
// GPU CodeGen op filter
//===----------------------------------------------------------------------===//

// /// Returns true if the index map represents a transpose that benefits from
// /// using shared memory when CodeGen towards the GPU.
// bool sharedMemTransposeFilter(AffineMap indexMap);

//===----------------------------------------------------------------------===//
// Utility from compiler/src/iree/compiler/Codegen/Transforms/Transforms.h
//===----------------------------------------------------------------------===//
/// Creates an allocation in the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. Returns std::nullopt if the method
/// couldnt creat an allocation in the entry block.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  Location loc, MemRefType allocaType,
                                  ValueRange dynamicSizes,
                                  std::optional<uint64_t> alignment);

/// Hoists `allocaOp` to the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. The method returns a value, but
/// does not replace the uses of the `allocaOp`.
template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  AllocLikeOpType allocaOp);

/// Traverse funcOp and try to hoist every AllocaOp to the entry block of the
/// function if the size is statically bounded.
template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(RewriterBase &rewriter,
                                           func::FuncOp funcOp);

//===----------------------------------------------------------------------===//
// Utility from compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h
//===----------------------------------------------------------------------===//
void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync);

//===----------------------------------------------------------------------===//
// Utility from compiler/src/iree/compiler/Codegen/Utils/Utils.h
//===----------------------------------------------------------------------===//

Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

} // namespace buddy::buddygpu
} // namespace mlir

#endif // INCLUDE_UTILS_GPUUTILS_H
