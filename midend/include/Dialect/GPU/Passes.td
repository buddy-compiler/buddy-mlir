//===- Passes.td ---------------------------------------------------------===//
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

include "mlir/Pass/PassBase.td"

def GPUDistributeToWarp : Pass<"gpu-distribute-to-warp", "func::FuncOp"> {
    let summary = "GPU distribute from Block level to Warp";
    let constructor = "mlir::createGPUDistributeToWarpPass()";
}

def RemoveReduntantLoops : Pass<"remove-reduntant-loops", "func::FuncOp"> {
    let summary = "Remove the loops that only run once in gpu kernel";
    let constructor = "mlir::createRemoveReduntantLoops()";
}

def TensorCoreVectorization : Pass<"tensorcore-vectorization", "func::FuncOp"> {
    let summary = "GPU TensorCore Vectorization From Warp Level";
    let constructor = "mlir::createTensorCoreVectorizationPass()";
}

def GPUDistributeSharedMemCopy : Pass<"gpu-distribute-shared-mem-copy", "func::FuncOp"> {
    let summary = "GPU Distribute Shared Memory Copy to each thread";
    let constructor = "mlir::createGPUDistributeSharedMemCopyPass()";
}
