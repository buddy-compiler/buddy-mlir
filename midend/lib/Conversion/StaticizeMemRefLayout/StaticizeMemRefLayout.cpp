//===- StaticizeMemRefLayout.cpp ------------------------------------------===//
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
// This file implements a pass that finds memref.copy operations with dynamic
// layouts, traces back to their source reinterpret_cast operations, and
// converts them to use static layouts when the shape is fully static.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

/// Returns true if the memref type has dynamic strides or offset
static bool hasDynamicLayout(MemRefType type) {
  auto layout = dyn_cast<StridedLayoutAttr>(type.getLayout());
  if (!layout)
    return false;

  // Check offset
  if (ShapedType::isDynamic(layout.getOffset()))
    return true;

  // Check strides
  ArrayRef<int64_t> strides = layout.getStrides();
  for (int64_t stride : strides) {
    if (ShapedType::isDynamic(stride))
      return true;
  }

  return false;
}

/// Computes row-major strides from shape
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  for (unsigned i = 0; i < shape.size(); ++i) {
    int64_t stride = 1;
    for (unsigned j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

class StaticizeMemRefLayoutPass
    : public PassWrapper<StaticizeMemRefLayoutPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticizeMemRefLayoutPass)

  StringRef getArgument() const final { return "staticize-memref-layout"; }

  StringRef getDescription() const final {
    return "Convert dynamic layouts in memref.reinterpret_cast operations used"
           "by memref.copy to static layouts when shapes are fully static.";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());
    bool changed = false;

    // Collect all memref.copy operations
    func.walk([&](memref::CopyOp copyOp) {
      Value source = copyOp.getSource();
      auto sourceType = dyn_cast<MemRefType>(source.getType());
      if (!sourceType || !hasDynamicLayout(sourceType))
        return;

      // Check if source comes from a reinterpret_cast
      Operation *defOp = source.getDefiningOp();
      auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(defOp);
      if (!reinterpretOp)
        return;

      // Check if shape is fully static
      ArrayRef<int64_t> shape = sourceType.getShape();
      bool allDimensionsStatic = true;
      for (int64_t dim : shape) {
        if (ShapedType::isDynamic(dim)) {
          allDimensionsStatic = false;
          break;
        }
      }

      if (!allDimensionsStatic)
        return;

      // Compute static strides
      SmallVector<int64_t> staticStrides = computeRowMajorStrides(shape);
      // Ensure last stride is 1
      staticStrides.back() = 1;

      // Create static offset (assume 0)
      int64_t staticOffset = 0;

      // Get the base buffer from the original reinterpret_cast (first operand)
      Value baseBuffer = reinterpretOp.getOperand(0);

      // Create static sizes from shape
      SmallVector<OpFoldResult> staticSizes;
      staticSizes.reserve(shape.size());
      for (int64_t dim : shape) {
        staticSizes.push_back(builder.getIndexAttr(dim));
      }

      // Create static strides as OpFoldResult
      SmallVector<OpFoldResult> staticStridesOfr;
      staticStridesOfr.reserve(staticStrides.size());
      for (int64_t stride : staticStrides) {
        staticStridesOfr.push_back(builder.getIndexAttr(stride));
      }

      // Create static offset as OpFoldResult
      OpFoldResult staticOffsetOfr = builder.getIndexAttr(staticOffset);

      // Create new static layout
      auto staticLayout = StridedLayoutAttr::get(func.getContext(),
                                                 staticOffset, staticStrides);
      auto staticType =
          MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
                          staticLayout, sourceType.getMemorySpace());

      // Create new reinterpret_cast with static layout
      builder.setInsertionPointAfter(reinterpretOp);
      auto newReinterpretOp = builder.create<memref::ReinterpretCastOp>(
          reinterpretOp.getLoc(), staticType, baseBuffer, staticOffsetOfr,
          staticSizes, staticStridesOfr);

      // Replace all uses of the old reinterpret_cast result with the new one
      Value oldResult = reinterpretOp.getResult();
      Value newResult = newReinterpretOp.getResult();
      oldResult.replaceAllUsesWith(newResult);

      // Erase the old operation
      reinterpretOp.erase();
      changed = true;
    });

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

namespace mlir {
namespace buddy {

void registerStaticizeMemRefLayoutPass() {
  PassRegistration<StaticizeMemRefLayoutPass>();
}

} // namespace buddy
} // namespace mlir
