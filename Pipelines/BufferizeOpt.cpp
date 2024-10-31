#include "Pipelines/BufferizeOpt.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "Utils/GemmCodegenUtils.h"
#include "Utils/PipelineUtils.h"

using namespace mlir;

void mlir::buddy::createBufferizeOptPipeline(OpPassManager &pm, 
                                             const BuddyBufferizeOptOptions &options) {
    mlir::buddy::invokeOpPassPipelineBuilder(
        [&](OpPassManager &pm) {
            // OneShotBufferization not implement bufferize on funcOp's arguments on default
            bufferization::OneShotBufferizationOptions bufferizeOptions;
            bufferizeOptions.bufferizeFunctionBoundaries = true;
            // bufferizeOptions.allowReturnAllocsFromLoops
            pm.addNestedPass<func::FuncOp>(bufferization::createEmptyTensorEliminationPass());
            pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
            pm.addNestedPass<func::FuncOp>(memref::createFoldMemRefAliasOpsPass());
            addCleanUpPassPipeline(pm);
        }, pm);
}

void mlir::buddy::registerBufferizeOptPassPipeline() {
    PassPipelineRegistration<BuddyBufferizeOptOptions>(
        "bufferize-opt",
        "bufferize opt lowering tensor to memref",
        createBufferizeOptPipeline
    );
}