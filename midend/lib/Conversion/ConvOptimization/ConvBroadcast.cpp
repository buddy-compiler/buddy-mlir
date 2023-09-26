//====- ConvBroadcast.cpp --------------------------------------------------===//
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
// This file implements the Conv Broadcast Optmize for linalg.conv_2d_nhwc_hwcf
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===--------------------------------------------
// Rewrite Pattern
//===--------------------------------------------

namespace {
class ConvBroadcastOptimizePattern : public ConversionPattern {
public:
  explicit ConvBroadcastOptimizePattern(MLIRContext *context, int64_t strideParam,
                                        ArrayRef<int64_t> tileParam)
      : ConversionPattern(linalg::Conv2DNhwcHwcfOp::getOperationName(), 1, context) {
    
    stride = strideParam;
    tileSizes = tileParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return failure();
  }

private:
  int64_t stride;
  ArrayRef<int64_t> tileSizes;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvBroadcastNhwcHwcf
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg conv2d_nhwc_hwcf to mixture of
/// Affine + Vector + Std operations.
namespace
{
class ConvBroadcastNhwcHwcfPass
    : public PassWrapper<ConvBroadcastNhwcHwcfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvBroadcastNhwcHwcfPass)
  StringRef getArgument() const final { return "conv-broadcast"; }
  StringRef getDescription() const final {
    return "Convolution Broadcast optimize for conv2d_nhwc_hwcf";
  }
  ConvBroadcastNhwcHwcfPass() = default;
  ConvBroadcastNhwcHwcfPass(const ConvBroadcastNhwcHwcfPass &) {}
  explicit ConvBroadcastNhwcHwcfPass(int64_t strideParam,
                                     ArrayRef<int64_t> tileParam) {
    stride = strideParam;
    tile = tileParam;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &regiistery) const override {
    regiistery.insert<linalg::LinalgDialect, scf::SCFDialect,
                      affine::AffineDialect, VectorDialect, func::FuncDialect>();
  }

  Option<int64_t> stride{*this, "strip-mining",
                         llvm::cl::desc("Strip mining size."),
                         llvm::cl::init(32)};
  ListOption<int64_t> tile{*this, "tile-sizes", llvm::cl::desc("Tile sizes"),
                           llvm::cl::ZeroOrMore};
};
} // end anonymous namespace

void ConvBroadcastNhwcHwcfPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                          scf::SCFDialect, func::FuncDialect,
                          memref::MemRefDialect, VectorDialect,
                          math::MathDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvBroadcastOptimizePattern>(context, stride, tile);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
  {
      signalPassFailure();
  }
    
}

namespace mlir {
namespace buddy {
void registerConvBroadcastNhwcHwcfPass() {
    PassRegistration<ConvBroadcastNhwcHwcfPass>();
}
}
}
