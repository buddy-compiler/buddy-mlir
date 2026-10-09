//===- BgeM3MatMulA100.cpp ------------------------------------------------===//
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
// BGE-M3 K3 pass: offload f32 linalg.matmul to the SpacemiT A100 compute
// cores through the spine-runtime. Each memref linalg.matmul is replaced
// with a call to `bge_m3_a100_gemm` (implemented in
// examples/BuddyBgeM3/k3/a100_kernel.cpp and linked into the model .so).
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

constexpr const char *kOffloadFn = "bge_m3_a100_gemm";

// Debug knobs: offload only matmuls in [offload-start, start+offload-limit).
// Used to bisect which matmul group breaks numerical correctness.
static llvm::cl::opt<int> clOffloadStart(
    "bge-m3-a100-start", llvm::cl::init(0),
    llvm::cl::desc("First matmul index to offload (0-based)"));
static llvm::cl::opt<int> clOffloadLimit(
    "bge-m3-a100-limit", llvm::cl::init(-1),
    llvm::cl::desc("Max number of matmuls to offload (-1 = all)"));

class BgeM3MatMulA100Pattern : public ConversionPattern {
public:
  explicit BgeM3MatMulA100Pattern(MLIRContext *context)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType)
      return failure();
    if (!aType.getElementType().isF32() || !bType.getElementType().isF32() ||
        !cType.getElementType().isF32())
      return failure();

    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    // One callee with dynamic-shape memrefs; cast each operand to it so all
    // matmul shapes share a single function signature.
    auto dynF32 = MemRefType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic},
        aType.getElementType());

    func::FuncOp fn = module.lookupSymbol<func::FuncOp>(kOffloadFn);
    if (!fn) {
      // Declare the external kernel once per module.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      fn = rewriter.create<func::FuncOp>(
          loc, kOffloadFn,
          rewriter.getFunctionType(
              TypeRange{dynF32, dynF32, dynF32}, TypeRange{}));
      fn.setPrivate();
    }

    Value dynA = rewriter.create<memref::CastOp>(loc, dynF32, A);
    Value dynB = rewriter.create<memref::CastOp>(loc, dynF32, B);
    Value dynC = rewriter.create<memref::CastOp>(loc, dynF32, C);

    // The offload kernel fully overwrites C, so a zero-fill producer is dead.
    // Remove it: if the fill loop survives as a standalone affine loop,
    // affine-loop-fusion can hoist it and zero buffers that alias C across
    // layers (bufferization reuses one buffer for layer inputs).
    Value cBase = C;
    while (auto cast = cBase.getDefiningOp<memref::CastOp>())
      cBase = cast.getSource();
    if (auto fill = cBase.getDefiningOp<linalg::FillOp>()) {
      bool isZero = false;
      if (auto cst = fill.value().getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = dyn_cast<FloatAttr>(cst.getValue()))
          isZero = attr.getValueAsDouble() == 0.0;
      }
      if (isZero)
        rewriter.replaceOp(fill, fill.getInputs().front());
    }

    rewriter.create<func::CallOp>(loc, fn, ValueRange{dynA, dynB, dynC});
    rewriter.eraseOp(op);
    return success();
  }
};

class BgeM3MatMulA100Pass
    : public PassWrapper<BgeM3MatMulA100Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BgeM3MatMulA100Pass)
  StringRef getArgument() const final { return "bge-m3-matmul-a100"; }
  StringRef getDescription() const final {
    return "Offload BGE-M3 f32 matmuls to the A100 cores (spine-runtime).";
  }
  BgeM3MatMulA100Pass() = default;
  BgeM3MatMulA100Pass(const BgeM3MatMulA100Pass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
};

} // namespace

void BgeM3MatMulA100Pass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Pre-number every matmul in module order so the dynamic legality check is
  // idempotent (the framework may query legality multiple times per op).
  llvm::DenseMap<Operation *, unsigned> matmulIndex;
  SmallVector<linalg::MatmulOp> matmuls;
  module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  for (const auto &en : llvm::enumerate(matmuls))
    matmulIndex[en.value().getOperation()] = en.index();

  ConversionTarget target(*context);
  target.addLegalDialect<func::FuncDialect, linalg::LinalgDialect,
                         memref::MemRefDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>();
  target.addDynamicallyLegalOp<linalg::MatmulOp>(
      [&matmulIndex](linalg::MatmulOp op) {
        // Legal (untouched) when not a memref f32 matmul.
        auto aType = dyn_cast<MemRefType>(op.getOperand(0).getType());
        auto bType = dyn_cast<MemRefType>(op.getOperand(1).getType());
        auto cType = dyn_cast<MemRefType>(op.getOperand(2).getType());
        if (!aType || !bType || !cType)
          return true;
        if (!aType.getElementType().isF32())
          return true;
        // Debug bisection knobs: offload only [start, start+limit).
        auto it = matmulIndex.find(op.getOperation());
        if (it == matmulIndex.end())
          return true;
        const int idx = (int)it->second;
        if (idx < clOffloadStart)
          return true;
        if (clOffloadLimit >= 0 && idx >= clOffloadStart + clOffloadLimit)
          return true;
        return false;
      });

  RewritePatternSet patterns(context);
  patterns.add<BgeM3MatMulA100Pattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerBgeM3MatMulA100Pass() {
  PassRegistration<BgeM3MatMulA100Pass>();
}
} // namespace buddy
} // namespace mlir
