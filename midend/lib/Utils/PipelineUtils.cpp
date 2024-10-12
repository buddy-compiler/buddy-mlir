#include "Utils/PipelineUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::buddy::addCleanUpPassPipeline(OpPassManager &pm, bool isModuleOp) {
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (isModuleOp) {
        pm.addPass(createSymbolDCEPass());
    }
}