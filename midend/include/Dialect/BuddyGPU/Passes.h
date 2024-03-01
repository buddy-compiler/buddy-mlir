
#ifndef BUDDYGPU_PASSES_H
#define BUDDYGPU_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace buddy {
enum class PipeliningSchedulingStrategy {
  // Schedule the load from global memory into stage 0 and the associated store
  // will be in stage depth - 1.
  loadGlobalStage0 = 0,
  // Schedule both the load from global and the store to shared memory in stage
  // 0. The compute operations will be in stage depth-1. This means there won't
  // be vector registers carried between stages.
  loadStoreStage0 = 1,
  // Schedule optimized when using nvidia tensorcore with async copies. It will
  // set all the copies in stage 0 then it will prefecth part of loads in `depth
  // - 2` stage and keep the rest of the load and compute into `depth - 1`.
  nvidiaTensorCore = 2,
};

FailureOr<scf::ForOp>
pipelineSharedMemoryCopy(RewriterBase &rewriter, scf::ForOp forOp,
                         PipeliningSchedulingStrategy startegy,
                         bool peelEpilogue, int64_t depth);

} // namespace buddy
} // namespace mlir

#endif // BUDDYGPU_PASSES_H
