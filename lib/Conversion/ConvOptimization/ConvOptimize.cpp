//====- ConvOptimize.cpp ----------------------------------------===//
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
// This file implements the conv optimize.
//
//===----------------------------------------------------------------------===//
#include <iostream>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace vector;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// PoolingNhwcSum vectorization pattern
class ConvOptimizePattern : public ConversionPattern {
public:
  explicit ConvOptimizePattern(MLIRContext *context,
                                                int64_t stripParam, int64_t kernelMParam, int64_t kernelNParam)
      : ConversionPattern(linalg::Conv2DNchwFchwOp::getOperationName(), 1,
                          context) {
    strip = stripParam;
    kernelM = kernelMParam;
    kernelN = kernelNParam;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // here we need coding.
    
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t strip;
  int64_t kernelM;
  int64_t kernelN;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConvOptimizePass
//===----------------------------------------------------------------------===//

/// This is a partial lowering linalg pooling operations to mixture of
/// Affine + Vector operations.
namespace {
class ConvOptimizePass
    : public PassWrapper<ConvOptimizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvOptimizePass)
  StringRef getArgument() const final { return "conv-optimize"; }
  StringRef getDescription() const final { return "Conv vectorization."; }
  ConvOptimizePass() = default;
  ConvOptimizePass(const ConvOptimizePass &) {}
  explicit ConvOptimizePass(int64_t stripParam, int64_t kernelMParam, int64_t kernelNParam) { strip = stripParam; kernelM = kernelMParam; kernelN = kernelNParam; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AffineDialect,
                    VectorDialect>();
  }

  Option<int64_t> strip{*this, "strip-mining",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(16)};
  
  Option<int64_t> kernelM{*this, "kernel-m",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(4)};

  Option<int64_t> kernelN{*this, "kernel-n",
                        llvm::cl::desc("Strip mining size."),
                        llvm::cl::init(2)};
};
} // end anonymous namespace.

void ConvOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target
      .addLegalDialect<arith::ArithmeticDialect, AffineDialect, scf::SCFDialect,
                       memref::MemRefDialect, VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<linalg::FillOp>();

  RewritePatternSet patterns(context);
  patterns.add<ConvOptimizePattern>(context, strip, kernelM, kernelN);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerConvOptimizePass() {
  PassRegistration<ConvOptimizePass>();
}
} // namespace buddy
} // namespace mlir
