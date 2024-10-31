//===- LinalgTensorOpt.cpp ------------------------------------------------===//
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

#include "Pipelines/LinalgTensorOpt.h"
#include "Pipelines/GPU/GemmCodegenTransform.h"
#include "Utils/GemmCodegenUtils.h"
#include "Utils/PipelineUtils.h"

#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "Transform/Transforms/TransformInsertion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

void addGPULinalgOptPasses(OpPassManager &pm) {
  { // Gemm Codegen Linalg Tensor Opt
    // TODO : to mark the func that has gemm linalg op
    // now the below option's funcanchor is set to empty
    // which considers that all func has matmul op
    auto funcGemmAnchor = mlir::buddy::getGemmMarkerAttrName().str();
    // TileSizeConfig of Dim (M) & Dim(N) & Dim(K) -> BM & BN & BK
    // blockIdx.y = M / BM
    // blockIdx.x = N / BN
    SmallVector<int64_t> tileConfig = {128, 128, 32};
    // threadIdx.x y z
    SmallVector<int64_t> workGroup = {64, 2, 1};
    int64_t stages = 3;
    mlir::buddy::GPUGemmCodegenConfigOptions configOptions;
    configOptions.tileConfig = tileConfig;
    configOptions.workGroup = workGroup;
    configOptions.stages = stages;
    createGemmTileConfigInsertTransform(pm, configOptions);
    pm.addPass(createTransformDialectInterpreter(true));

    mlir::buddy::GPUGemmGeneralOptions generalOptions;
    createGemmTileTransform(pm, generalOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
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

void mlir::buddy::createLinalgTensorOptPassPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options) {
  mlir::buddy::invokeOpPassPipelineBuilder(
      createLinalgTensorOptPassPipelineimpl, pm, options.target, options.arch);
}

void mlir::buddy::registerLinalgTensorOptPassPipeline() {
  PassPipelineRegistration<LinalgTensorOptPipelineOptions>(
      "linalg-tensor-opt", "Linalg with Tensor Opt Pass Pipeline",
      mlir::buddy::createLinalgTensorOptPassPipeline);
}

