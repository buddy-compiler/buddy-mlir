#ifndef PIPELINES_MEMREFOPT_H
#define PIPELINES_MEMREFOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace mlir {
namespace buddy {

struct LinalgMemrefOptPipelineOptions :
    public PassPipelineOptions<LinalgMemrefOptPipelineOptions> {
    Option<std::string> target {
        *this, "target",
        llvm::cl::desc("An optional attribute to speicify target."),
    };
};

void createLinalgMemrefOptPipeline(OpPassManager &pm, 
                                   const LinalgMemrefOptPipelineOptions &options);

void registerLinalgMemrefOptPipeline();

} // mlir::buddy
} // mlir

#endif