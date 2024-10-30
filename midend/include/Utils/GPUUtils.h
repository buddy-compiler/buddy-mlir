//===- GPUUtils.h ---------------------------------------------------------===//
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
namespace buddy::gpu{
static constexpr int32_t kNumGPUDims = 3;
static constexpr int32_t kWarpSize = 32;

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract);

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

} // namespace buddy::gpu
} // namespace mlir

#endif // INCLUDE_UTILS_GPUUTILS_H
