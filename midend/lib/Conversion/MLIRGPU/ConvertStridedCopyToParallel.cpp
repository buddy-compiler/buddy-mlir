//===- ConvertStridedCopyToParallel.cpp
//------------------------------------===//
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
//
// This file implements a pass that converts strided memref.copy operations into
// scf.parallel loops with element-wise memref.load/store. This allows strided
// copies to be lowered as GPU kernels through the standard parallel-loops →
// NVVM pipeline, instead of falling back to @memrefCopy on the CPU side.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ConvertStridedCopyToParallelPass
//===----------------------------------------------------------------------===//

namespace {

class ConvertStridedCopyToParallelPass
    : public PassWrapper<ConvertStridedCopyToParallelPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStridedCopyToParallelPass)
  StringRef getArgument() const final {
    return "convert-strided-copy-to-parallel";
  }
  StringRef getDescription() const final {
    return "Convert strided memref.copy to scf.parallel loops for GPU "
           "lowering.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, arith::ArithDialect>();
  }
};

static bool isStridedMemRef(MemRefType type) {
  auto layout = type.getLayout();
  if (layout.isIdentity())
    return false;
  // A memref with a single affine map that is not identity is strided.
  return true;
}

void ConvertStridedCopyToParallelPass::runOnOperation() {
  auto funcOp = getOperation();

  // Collect strided memref.copy ops first to avoid mutation during walk.
  SmallVector<memref::CopyOp> stridedCopies;
  funcOp.walk([&](memref::CopyOp copyOp) {
    auto srcType = dyn_cast<MemRefType>(copyOp.getSource().getType());
    auto dstType = dyn_cast<MemRefType>(copyOp.getTarget().getType());
    if (!srcType || !dstType)
      return;
    if (isStridedMemRef(srcType) || isStridedMemRef(dstType))
      stridedCopies.push_back(copyOp);
  });

  if (stridedCopies.empty())
    return;

  for (auto copyOp : stridedCopies) {
    auto src = copyOp.getSource();
    auto dst = copyOp.getTarget();
    auto dstType = cast<MemRefType>(dst.getType());
    auto loc = copyOp.getLoc();
    unsigned rank = dstType.getRank();

    OpBuilder builder(copyOp);

    // Build lower bounds (all zeros), upper bounds (shape of dst), steps (all
    // 1).
    SmallVector<Value> lowerBounds, upperBounds, steps;
    for (unsigned i = 0; i < rank; ++i) {
      lowerBounds.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
      if (dstType.isDynamicDim(i)) {
        upperBounds.push_back(memref::DimOp::create(builder, loc, dst, i));
      } else {
        upperBounds.push_back(arith::ConstantIndexOp::create(
            builder, loc, dstType.getDimSize(i)));
      }
      steps.push_back(arith::ConstantIndexOp::create(builder, loc, 1));
    }

    // Create scf.parallel with element-wise load + store.
    scf::ParallelOp::create(
        builder, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange ivs) {
          auto val = memref::LoadOp::create(bodyBuilder, bodyLoc, src, ivs);
          memref::StoreOp::create(bodyBuilder, bodyLoc, val, dst, ivs);
        });

    copyOp.erase();
  }
}

} // end anonymous namespace.

namespace mlir {
namespace buddy {
void registerConvertStridedCopyToParallelPass() {
  PassRegistration<ConvertStridedCopyToParallelPass>();
}
} // namespace buddy
} // namespace mlir
