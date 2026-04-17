//===- MatMulVectorizationDecode.cpp --------------------------------------===//
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
// This file implements a specialized matmul vectorization for decode
// workloads, where the `m` dimension is statically known to be 1.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
static bool isZeroAttribute(Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

static bool isZeroGlobal(Value value, ModuleOp module) {
  auto getGlobal = value.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return false;

  auto global = module.lookupSymbol<memref::GlobalOp>(getGlobal.getNameAttr());
  if (!global || !global.getInitialValue().has_value())
    return false;

  auto dense = dyn_cast<DenseElementsAttr>(*global.getInitialValue());
  if (!dense || !dense.isSplat())
    return false;

  return isZeroAttribute(dense.getSplatValue<Attribute>());
}

static memref::CopyOp findZeroInitCopy(Value output, linalg::MatmulOp matmulOp,
                                       ModuleOp module) {
  for (Operation *user : output.getUsers()) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != output)
      continue;
    if (copyOp->getBlock() != matmulOp->getBlock() ||
        !copyOp->isBeforeInBlock(matmulOp))
      continue;
    if (isZeroGlobal(copyOp.getSource(), module))
      return copyOp;
  }
  return nullptr;
}

class MatMulVectorizationDecodePattern : public ConversionPattern {
public:
  MatMulVectorizationDecodePattern(MLIRContext *ctx, ModuleOp module,
                                   int64_t vecSize, bool scalableParam)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx),
        module(module), vecSize(vecSize), scalable(scalableParam) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    auto loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType)
      return failure();
    if (!aType.hasStaticShape() || aType.getRank() != 2)
      return failure();
    if (aType.getDimSize(0) != 1)
      return failure();
    if (cType.getRank() != 2 || cType.getDimSize(0) != 1)
      return failure();

    Type elementType = cType.getElementType();
    auto vectorType = VectorType::get({vecSize}, elementType, {scalable});

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, vecSize);
    if (scalable) {
      Value vscale = vector::VectorScaleOp::create(rewriter, loc);
      step = arith::MulIOp::create(rewriter, loc, step, vscale);
    }

    Value n = memref::DimOp::create(rewriter, loc, C, c1);
    Value k = memref::DimOp::create(rewriter, loc, A, c1);
    memref::CopyOp zeroInitCopy = findZeroInitCopy(C, matmulOp, module);
    bool zeroInitialized = zeroInitCopy != nullptr;

    scf::ParallelOp::create(rewriter, 
        loc,
        /*lowerBounds=*/ValueRange{c0},
        /*upperBounds=*/ValueRange{n},
        /*steps=*/ValueRange{step},
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value nIdx = ivs.front();
          Value cVec;
          if (zeroInitialized) {
            Value zero = arith::ConstantOp::create(builder, 
                loc, elementType, builder.getZeroAttr(elementType));
            cVec = vector::BroadcastOp::create(builder, loc, vectorType, zero);
          } else {
            cVec = vector::LoadOp::create(builder, loc, vectorType, C,
                                                  ValueRange{c0, nIdx});
          }
          auto sumIter = scf::ForOp::create(builder, 
              loc, c0, k, c1, ValueRange{cVec},
              [&](OpBuilder &builder, Location loc, Value kIdx,
                  ValueRange iterArgs) {
                Value aElem = memref::LoadOp::create(builder, 
                    loc, A, ValueRange{c0, kIdx});
                Value aVec =
                    vector::BroadcastOp::create(builder, loc, vectorType, aElem);
                Value bVec = vector::LoadOp::create(builder, 
                    loc, vectorType, B, ValueRange{kIdx, nIdx});
                Value res = vector::FMAOp::create(builder, loc, aVec, bVec,
                                                          iterArgs.front());
                scf::YieldOp::create(builder, loc, res);
              });

          vector::StoreOp::create(builder, loc, sumIter.getResult(0), C,
                                          ValueRange{c0, nIdx});
        });

    rewriter.eraseOp(op);
    if (zeroInitCopy) {
      auto getGlobal =
          zeroInitCopy.getSource().getDefiningOp<memref::GetGlobalOp>();
      rewriter.eraseOp(zeroInitCopy);
      if (getGlobal && getGlobal->use_empty())
        rewriter.eraseOp(getGlobal);
    }

    return success();
  }

private:
  ModuleOp module;
  int64_t vecSize;
  bool scalable;
};

class MatMulVectorizationDecodePass
    : public PassWrapper<MatMulVectorizationDecodePass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulVectorizationDecodePass)

  StringRef getArgument() const final { return "matmul-vectorization-decode"; }
  StringRef getDescription() const final {
    return "Vectorize linalg.matmul with m==1 for decode workloads.";
  }

  MatMulVectorizationDecodePass() = default;
  MatMulVectorizationDecodePass(const MatMulVectorizationDecodePass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect,
                           func::FuncDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) -> bool {
          auto aType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
          auto cType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
          if (!aType || !cType)
            return true;
          if (!aType.hasStaticShape() || aType.getRank() != 2)
            return true;
          if (aType.getDimSize(0) != 1)
            return true;
          if (cType.getRank() != 2 || cType.getDimSize(0) != 1)
            return true;
          return false;
        });

    RewritePatternSet patterns(context);
    bool isScalable = (vectorType == "scalable");
    patterns.add<MatMulVectorizationDecodePattern>(context, module, vectorSize,
                                                   isScalable);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<int64_t> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Specify the vector width for n-dimension iteration."),
      llvm::cl::init(32)};
  Option<std::string> vectorType{
      *this, "vector-type",
      llvm::cl::desc("Specify vector type: fixed or scalable."),
      llvm::cl::init("fixed")};
};
} // namespace

namespace mlir {
namespace buddy {
void registerMatMulVectorizationDecodePass() {
  PassRegistration<MatMulVectorizationDecodePass>();
}
} // namespace buddy
} // namespace mlir
