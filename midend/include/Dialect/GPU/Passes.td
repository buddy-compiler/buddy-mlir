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