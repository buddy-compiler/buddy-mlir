#include "Pipelines/BufferizeOpt.h"
#include "Utils/GemmCodegenUtils.h"
#include "Utils/PipelineUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

static bool isGPUSharedMem(MemRefType type) {
  if (auto memorySpace = llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
          type.getMemorySpace())) {
    if (memorySpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      return true;
    }
  }
  return false;
}

template <typename AllocT>
static auto createAlloc(OpBuilder &b, Location loc, MemRefType type,
                        ValueRange dynShape, size_t bufferAlignment) {
  if (bufferAlignment != 0)
    return b
        .create<AllocT>(loc, type, dynShape,
                        b.getI64IntegerAttr(bufferAlignment))
        .getResult();
  return b.create<AllocT>(loc, type, dynShape).getResult();
}

void mlir::buddy::createBufferizeOptPipeline(
    OpPassManager &pm, const BuddyBufferizeOptOptions &options) {
  mlir::buddy::invokeOpPassPipelineBuilder(
      [&](OpPassManager &pm) {
        // OneShotBufferization not implement bufferize on funcOp's arguments on
        // default
        bufferization::OneShotBufferizationOptions bufferizeOptions;
        bufferizeOptions.bufferizeFunctionBoundaries = true;
        bufferizeOptions.allowReturnAllocsFromLoops = true;
        bufferizeOptions.bufferAlignment = 0;
        bufferizeOptions.setFunctionBoundaryTypeConversion(
            bufferization::LayoutMapOption::IdentityLayoutMap);
        bufferizeOptions.allocationFn =
            [](OpBuilder &b, Location loc, MemRefType type, ValueRange dynShape,
               unsigned int bufferAlignment) -> FailureOr<Value> {
          if (isGPUSharedMem(type)) {
            return createAlloc<memref::AllocaOp>(b, loc, type, dynShape,
                                                 bufferAlignment);
          }
          return createAlloc<memref::AllocOp>(b, loc, type, dynShape,
                                              bufferAlignment);
        };
        // bufferizeOptions.allowReturnAllocsFromLoops
        pm.addNestedPass<func::FuncOp>(
            bufferization::createEmptyTensorEliminationPass());
        pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
        pm.addNestedPass<func::FuncOp>(memref::createFoldMemRefAliasOpsPass());
        addCleanUpPassPipeline(pm);
      },
      pm);
}

void mlir::buddy::registerBufferizeOptPassPipeline() {
  PassPipelineRegistration<BuddyBufferizeOptOptions>(
      "bufferize-opt", "bufferize opt lowering tensor to memref",
      createBufferizeOptPipeline);
}
