//===- TransformOps.cpp ---------------------------------------------------===//
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
// This file implements transform ops for GPU targets.
//
//===----------------------------------------------------------------------===//

#include "GPU/TransformOps.h"

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <optional>

#include "Utils/GPUUtils.h"

using namespace mlir;
using namespace mlir::buddy;

using llvm::dbgs;

#define DEBUG_TYPE "transform-llvmgpu-extensions"
#define DEBUG_TYPE_ALIAS "transform-llvmgpu-extensions-alias"
#define DEBUG_VECTOR_TO_MMA "transform-llvmgpu-extensions-vector-to-mma"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(dbgs() << '[' << DEBUG_TYPE << "] " << X)
#define DBGS_ALIAS() (dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")
#define DBGS_VECTOR_TO_MMA() (dbgs() << '[' << DEBUG_VECTOR_TO_MMA << "] ")

buddy::gpu::TransformExtensions::TransformExtensions() {
  // CreateAsyncGroupsOp depends on the following two dialects.
  declareGeneratedDialect<mlir::gpu::GPUDialect>();
  declareGeneratedDialect<mlir::nvgpu::NVGPUDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "GPU/TransformOps.cpp.inc"
      >();
}

void buddy::registerBuddyGPUTransformOps(DialectRegistry &registry) {
  registry.addExtensions<buddy::gpu::TransformExtensions>();
}

//===----------------------------------------------------------------------===//
// HoistStaticAllocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure buddy::gpu::HoistStaticAllocOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, func::FuncOp target,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(rewriter, target);
  return DiagnosedSilenceableFailure::success();
}

void buddy::gpu::HoistStaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getTargetMutable(), effects);
  mlir::transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// ApplyUnrollVectorsGpuMmaSyncPatternsOp
//===---------------------------------------------------------------------===//

static std::optional<SmallVector<int64_t>>
getGPUTensorCoreNativeMmaSyncVectorSize(Operation *op) {
  return buddy::gpu::getMmaNativeVectorSize(op);
}

void buddy::gpu::ApplyUnrollVectorsGpuMmaSyncPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeMmaSyncVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===---------------------------------------------------------------------===//
// VectorToMMAConversionOp
//===---------------------------------------------------------------------===//

void buddy::gpu::VectorToMMAConversionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getTargetMutable(), effects);
  mlir::transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure buddy::gpu::VectorToMMAConversionOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, Operation *target,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    // target->emitOpError(
    //     "applies only to isolated-from-above targets because it "
    //     "needs to apply "
    //     "patterns greedily");
    // return emitDefaultDefiniteFailure(target);
  }

  auto funcOp = dyn_cast<func::FuncOp>(target);
  if (!funcOp) {
    target->emitOpError("Must apply to a func op");
    return emitDefaultDefiniteFailure(target);
  }

  if (!(getUseMmaSync() ^ getUseWmma())) {
    target->emitOpError(
        "Exactly one of use_mma_sync or use_wmma must be specified");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext *ctx = target->getContext();
  mlir::transform::ErrorCheckingTrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;

  // Unrolling to native vector size must have previously occurred.
  // TODO: Add pattern to propagate the extract through the scf.for
  // ops. Convert slice of contract operations to mma_sync/wmma ops.
  RewritePatternSet patterns(ctx);
  mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
  if (failed(
          applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
    target->emitOpError("vector to mma preparation patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  auto diag = DiagnosedSilenceableFailure::success();
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target)))
      return mlir::emitDefiniteFailure(
          target, "vector to wmma patterns failed to apply");
    return listener.checkAndResetError();
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp)))
    return mlir::emitDefiniteFailure(target,
                                     "vector to mma patterns failed to apply");

  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(funcOp.getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns,
                                          nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(f32ToTF32patterns),
                                          config)))
    return mlir::emitDefiniteFailure(
        target, "vector to mma F32ToTF32 patterns failed to apply");

  return listener.checkAndResetError();
}

#define GET_OP_CLASSES
#include "GPU/TransformOps.cpp.inc"
