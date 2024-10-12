include "mlir/Pass/PassBase.td"

def GPUDistributeToWarp : Pass<"gpu-distribute-to-warp", "func::FuncOp"> {
    let summary = "GPU distribute from Block level to Warp";
    let constructor = "mlir::createGPUDistributeToWarpPass()";
}

def RemoveReduntantLoops : Pass<"remove-reduntant-loops", "func::FuncOp"> {
    let summary = "Remove the loops that only run once in gpu kernel";
    let constructor = "mlir::createRemoveReduntantLoops()";
}