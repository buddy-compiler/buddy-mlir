#include "Pipelines/LinalgTensorOpt.h"
#include "Pipelines/GPU/Utils.h"
#include "Pipelines/GPU/GemmCodegenTransform.h"

#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "Transform/Transforms/TransformInsertion.h"

#include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;

namespace {

void addGPULinalgOptPasses(OpPassManager &pm) {
    { // Gemm Codegen Linalg Tensor Opt
        SmallVector<int64_t> tileConfig = {32, 32, 16};
        SmallVector<int64_t> workGroup = {32, 2, 1};
        int64_t stages = 3;
        mlir::buddy::GPUGemmCodegenConfigOptions configOption;
        configOption.tileConfig = tileConfig;
        configOption.workGroup = workGroup;
        configOption.stages = stages;
        createGPUGemmTileConfigInsertTransform(pm, configOption);
        pm.addPass(createTransformDialectInterpreter(true));
    }
}

void createLinalgTensorOptPassPipelineimpl(OpPassManager &pm,
                                            const std::string &target,
                                            const std::string &arch) {
    if (target == "gpu") {
        addGPULinalgOptPasses(pm);
    } else {
        /*TODO*/
    }
}

} // namespace

void mlir::buddy::createLinalgTensorOptPassPipeline(OpPassManager &pm,
                                                    const LinalgTensorOptPipelineOptions &options) {
    mlir::buddy::invokeOpPassPipelineBuilder(createLinalgTensorOptPassPipelineimpl, pm, options.target, options.arch);
}

void mlir::buddy::registerLinalgTensorOptPassPipeline() {
    PassPipelineRegistration<LinalgTensorOptPipelineOptions>(
        "linalg-tensor-opt", "Linalg with Tensor Opt Pass Pipeline", mlir::buddy::createLinalgTensorOptPassPipeline);
}
