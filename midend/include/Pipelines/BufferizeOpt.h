#ifndef PIPELINES_BUFFERIZEOPT_H
#define PIPELINES_BUFFERIZEOPT_H

#include "Pipelines/LinalgTensorOpt.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include <string>

namespace mlir {
namespace buddy {

struct BuddyBufferizeOptOptions
    : public PassPipelineOptions<BuddyBufferizeOptOptions> {
  Option<std::string> target{
      *this,
      "target",
      llvm::cl::desc("An option to specify target"),
  };
};

void createBufferizeOptPipeline(OpPassManager &pm,
                                const BuddyBufferizeOptOptions &options);

void registerBufferizeOptPassPipeline();

} // namespace buddy
} // namespace mlir

#endif
