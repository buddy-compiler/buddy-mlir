//===- EliminateMemRefCopy.cpp - Eliminate redundant memref.copy ----------===//
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
// This pass eliminates redundant memref.copy operations that copy from function
// arguments to newly allocated memrefs. Instead, it replaces all uses of the
// allocated memref with the original argument, enabling in-place modifications.
//
// Pattern matched:
//   %alloc = memref.alloc() : memref<...>
//   memref.copy %arg, %alloc : memref<...> to memref<...>
//   // ... uses of %alloc ...
//
// Transformed to:
//   // ... uses of %arg (replacing %alloc) ...
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

// Pattern to eliminate memref.copy from function arguments to allocations
struct EliminateMemRefCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Check if the source is a function argument
    Value source = copyOp.getSource();
    Value dest = copyOp.getTarget();

    // Check if source is a block argument (function argument)
    BlockArgument sourceArg = dyn_cast<BlockArgument>(source);
    if (!sourceArg)
      return failure();

    // Check if the source is a function argument (not a block argument from a
    // loop)
    Block *sourceBlock = sourceArg.getOwner();
    if (!isa<func::FuncOp>(sourceBlock->getParentOp()))
      return failure();

    // Check if destination is an allocation
    memref::AllocOp allocOp = dest.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    // Verify that the types are compatible (same shape and element type)
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto destType = dyn_cast<MemRefType>(dest.getType());
    if (!sourceType || !destType)
      return failure();

    // Check element types match
    if (sourceType.getElementType() != destType.getElementType())
      return failure();

    // Check shapes match (allowing for different layouts)
    if (sourceType.getShape() != destType.getShape())
      return failure();

    // Collect all uses of the allocation
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : allocOp->getUses()) {
      // Skip the copy operation itself
      if (use.getOwner() == copyOp.getOperation())
        continue;
      uses.push_back(&use);
    }

    // Check if source has strided layout and needs conversion
    // We need to ensure the most minor dimension has unit stride for
    // vector.load
    Value replacementValue = source;

    // Check if source has a strided layout
    if (auto stridedLayout =
            dyn_cast<StridedLayoutAttr>(sourceType.getLayout())) {
      ArrayRef<int64_t> strides = stridedLayout.getStrides();
      if (!strides.empty()) {
        // Check if the most minor dimension (last stride) is not 1
        int64_t lastStride = strides.back();
        if (ShapedType::isDynamic(lastStride) || lastStride != 1) {
          // Use ExtractStridedMetadataOp to get actual stride values
          // This ensures we preserve the correct strides from the source
          Location loc = copyOp.getLoc();
          OpBuilder builder(rewriter);
          builder.setInsertionPoint(copyOp);

          // Extract metadata to get actual stride values
          auto metadata =
              builder.create<memref::ExtractStridedMetadataOp>(loc, source);

          // Get the rank
          int64_t rank = sourceType.getRank();

          // Build sizes from shape
          SmallVector<OpFoldResult> sizes;
          sizes.reserve(rank);
          auto metadataSizes = metadata.getSizes();
          for (auto [idx, dim] : llvm::enumerate(sourceType.getShape())) {
            if (ShapedType::isDynamic(dim))
              sizes.push_back(metadataSizes[idx]);
            else
              sizes.push_back(builder.getIndexAttr(dim));
          }

          // Build strides: preserve all strides except the last one (set to 1)
          SmallVector<OpFoldResult> stridesOfr;
          stridesOfr.reserve(rank);
          auto metadataStrides = metadata.getStrides();
          for (auto [idx, stride] : llvm::enumerate(strides)) {
            if (static_cast<int64_t>(idx) == rank - 1) {
              // Last stride: set to 1
              stridesOfr.push_back(builder.getIndexAttr(1));
              continue;
            }
            // Preserve original stride
            if (ShapedType::isDynamic(stride))
              stridesOfr.push_back(metadataStrides[idx]);
            else
              stridesOfr.push_back(builder.getIndexAttr(stride));
          }

          // Get offset
          OpFoldResult offset;
          if (ShapedType::isDynamic(stridedLayout.getOffset()))
            offset = metadata.getOffset();
          else
            offset = builder.getIndexAttr(stridedLayout.getOffset());

          // Create tightened layout with last stride = 1
          SmallVector<int64_t> tightenedStrides(strides.begin(), strides.end());
          tightenedStrides.back() = 1;
          auto tightenedLayout = StridedLayoutAttr::get(
              sourceType.getContext(), stridedLayout.getOffset(),
              tightenedStrides);
          auto tightenedType = MemRefType::get(
              sourceType.getShape(), sourceType.getElementType(),
              tightenedLayout, sourceType.getMemorySpace());

          // Create reinterpret_cast with tightened layout
          Value tightened = builder.create<memref::ReinterpretCastOp>(
              loc, tightenedType, metadata.getBaseBuffer(), offset, sizes,
              stridesOfr);

          // Then cast to static layout type (no layout information)
          // This ensures compatibility with function signatures
          MemRefType finalType = MemRefType::get(
              sourceType.getShape(), sourceType.getElementType(),
              MemRefLayoutAttrInterface(), // No layout = static layout
              sourceType.getMemorySpace());

          replacementValue =
              builder.create<memref::CastOp>(loc, finalType, tightened);
        }
      }
    }

    // Replace all uses of the allocation with the replacement value
    for (OpOperand *use : uses) {
      use->set(replacementValue);
    }

    // Erase the copy operation
    rewriter.eraseOp(copyOp);

    // Erase the allocation if it has no more uses
    if (allocOp->use_empty()) {
      rewriter.eraseOp(allocOp);
    }

    return success();
  }
};

class EliminateMemRefCopyPass
    : public PassWrapper<EliminateMemRefCopyPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateMemRefCopyPass)
  StringRef getArgument() const final { return "eliminate-memref-copy"; }
  StringRef getDescription() const final {
    return "Eliminate redundant memref.copy operations from function arguments "
           "to allocations.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<EliminateMemRefCopyPattern>(ctx);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::AnyOp);
    (void)applyPatternsGreedily(func, std::move(patterns), config);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
};
} // namespace

namespace mlir {
namespace buddy {
void registerEliminateMemRefCopyPass() {
  PassRegistration<EliminateMemRefCopyPass>();
}
} // namespace buddy
} // namespace mlir
