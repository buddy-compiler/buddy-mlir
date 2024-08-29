#ifndef PIPELINES_GPU_GEMM_CODEGEN_TRANSOFRM_H
#define PIPELINES_GPU_GEMM_CODEGEN_TRANSOFRM_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace buddy {
struct GPUGemmCodegenConfigOptions : public PassPipelineOptions<GPUGemmCodegenConfigOptions> {
    Option<std::string> funcAnchor {
        *this, "func-anchor", 
        llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
    Option<std::string> annotatePrefix {
        *this, "annotate-prefix",
        llvm::cl::desc("An optional annotate prefix attribute on target ops."),
        llvm::cl::init("__buddy_gpu_gemm__")};
    ListOption<int64_t> tileConfig {
        *this, "tile-config",
        llvm::cl::desc("An optional tile config for matmul op")};
    ListOption<int64_t> workGroup {
        *this, "work-group",
        llvm::cl::desc("An optional workgroup size config for matmul op")};
    Option<int64_t> stages {
        *this, "stages",
        llvm::cl::desc("An optional stages config for matmul op")};
};

void createGPUGemmTileConfigInsertTransform(OpPassManager &pm, const GPUGemmCodegenConfigOptions &options);

} // namespace mlir::buddy
} // namespace mlir

#endif