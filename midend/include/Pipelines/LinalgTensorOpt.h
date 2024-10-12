#ifndef PIPELINES_GPU_LINALGTENSOROPT_H
#define PIPELINES_GPU_LINALGTENSOROPT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <string>
#include <memory>

namespace mlir {
namespace buddy {

struct LinalgTensorOptPipelineOptions
    : public PassPipelineOptions<LinalgTensorOptPipelineOptions> {
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("gpu")};
  Option<std::string> arch{
      *this, "arch", llvm::cl::desc("An optional attribute to speicify arch."),
      llvm::cl::init("nv_sm_80")};
};

void createLinalgTensorOptPassPipeline(OpPassManager &pm, const LinalgTensorOptPipelineOptions &options);

void registerLinalgTensorOptPassPipeline();

    
} // namespace mlir::buddy
} // namespace mlir

#endif