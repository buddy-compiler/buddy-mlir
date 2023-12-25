// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GPUBufferizePass.cpp - ---------------------------------------------===//
//
// Wrapper pass to use MLIR's One-Shot Bufferize pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;

using namespace mlir;

namespace {

bool hasSharedMemoryAddressSpace(MemRefType memrefType) {
  auto addrSpace = llvm::dyn_cast_if_present<gpu::AddressSpaceAttr>(
      memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(from.getType()))) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  // Operation *copy =
  builder.create<memref::CopyOp>(loc, from, to);
  if (needsBarrier) {
    // setMarker(copy, getCopyToWorkgroupMemoryMarker());
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

/// Pass to convert from tensor based ops to memref based ops.
class BuudyGPUBufferizePass
    : public PassWrapper<BuudyGPUBufferizePass,
                         OperationPass<ModuleOp>> {
public:
  explicit BuudyGPUBufferizePass(
      std::optional<BufferizationOptions::AllocationFn> allocationFn =
          gpuAllocationFn,
      std::optional<BufferizationOptions::MemCpyFn> memCpyFn = gpuCopyFn)
      : allocationFn(allocationFn), memCpyFn(memCpyFn) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuudyGPUBufferizePass)
  StringRef getArgument() const final { return "gpu-bufferize"; }
  StringRef getDescription() const final {
    return "One shot bufferize GPU pass.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                vector::VectorDialect,
                gpu::GPUDialect>();
    // clang-format on
  }

  void runOnOperation() override;

private:
  const std::optional<BufferizationOptions::AllocationFn> allocationFn;
  const std::optional<BufferizationOptions::MemCpyFn> memCpyFn;
};



} // namespace

// The following is copied from bufferization::runOneShotBufferize with
// modifications.
LogicalResult
runBuudyOneShotBufferize(Operation *op,
                        const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  if (failed(analyzeOp(op, state)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  return bufferization::runOneShotBufferize(op, options);
}

/// Run comprehensive bufferize.
void BuudyGPUBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OneShotBufferizationOptions options;
  options.allocationFn = allocationFn;
  options.memCpyFn = memCpyFn;

  if (failed(runBuudyOneShotBufferize(moduleOp, options))) {
    return signalPassFailure();
  }

  // Remove redundant args and unused results.
  {
    RewritePatternSet patterns(&getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createBuudyGPUBufferizePass(
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  if (!allocationFn)
    allocationFn = gpuAllocationFn;
  if (!memCpyFn)
    memCpyFn = gpuCopyFn;
  return std::make_unique<BuudyGPUBufferizePass>(allocationFn,
                                                          memCpyFn);
}

void addBuudyPostBufferizationPasses(OpPassManager &passManager) {
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
  // There are redundant memcpy (with linalg.generic form) ops created, which
  // can be deleted by canonicalizer. We have to run it again because the
  // memrefs are unified in CSE pass, so we can truely remove redundant memcpy.
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void addBuudyGPUBufferizePasses(
    OpPassManager &passManager,
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn) {
  passManager.addPass(bufferization::createEmptyTensorEliminationPass());
  passManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  passManager.addPass(
      createBuudyGPUBufferizePass(allocationFn, memCpyFn));
  addBuudyPostBufferizationPasses(passManager);
}

namespace mlir {
namespace buddy {
void registerBuddyGPUBufferizePass() { PassRegistration<BuudyGPUBufferizePass>(); }
} // namespace buddy
} // namespace mlir