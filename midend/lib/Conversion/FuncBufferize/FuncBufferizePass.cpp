#include "mlir-c/BuiltinTypes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <mlir/IR/OperationSupport.h>
using namespace mlir;
using namespace mlir::func;

namespace {

class FuncBufferizeDynamicOffsetPattern : public ConversionPattern {
public:
  explicit FuncBufferizeDynamicOffsetPattern(MLIRContext *context)
      : ConversionPattern(FuncOp::getOperationName(), 1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto func_op = llvm::cast<func::FuncOp>(*op);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {
class FuncBufferizeDynamicOffsetPass
    : public PassWrapper<FuncBufferizeDynamicOffsetPass,
                         OperationPass<ModuleOp>> {
public:
  FuncBufferizeDynamicOffsetPass()=default;
  llvm::StringRef getArgument() const final {
    return "func-bufferize-dynamic-offset";
  }
  llvm::StringRef getName() const final {
    return "func-bufferize-dynamic-offset";
  }
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<FuncBufferizeDynamicOffsetPass>();
  }
  
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    
  registry.insert<bufferization::BufferizationDialect>();

  registry.insert<memref::MemRefDialect>();

  }

  void runOnOperation() override;
};
} // namespace

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<BaseMemRefType>(inputs[0].getType()));
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

void FuncBufferizeDynamicOffsetPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](RankedTensorType type){
    auto shape = type.getShape();
    llvm::SmallVector<int64_t, 8> stride;
    stride.reserve(shape.size());
    int64_t initial_value = 1;
    stride.insert(stride.begin(), initial_value);
    for(auto i=shape.size()-1;i>0;i--){
      initial_value*=shape[i];
      stride.insert(stride.begin(), initial_value);
    }
    return MemRefType::get(type.getShape(), type.getElementType(), StridedLayoutAttr::get(context, mlirShapedTypeGetDynamicSize(), ArrayRef<int64_t>(stride)));
  });
  typeConverter.addArgumentMaterialization(materializeToTensor);
  typeConverter.addSourceMaterialization(materializeToTensor);
  typeConverter.addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "expected exactly one input");

    if (auto inputType = dyn_cast<MemRefType>(inputs[0].getType())) {
      // MemRef to MemRef cast.
      assert(inputType != type && "expected different types");
      // Unranked to ranked and ranked to unranked casts must be explicit.
      auto rankedDestType = dyn_cast<MemRefType>(type);
      if (!rankedDestType)
        return nullptr;
      FailureOr<Value> replacement =
          bufferization::castOrReallocMemRefValue(builder, inputs[0], rankedDestType);
      if (failed(replacement))
        return nullptr;
      return *replacement;
    }

    if (isa<TensorType>(inputs[0].getType())) {
      // Tensor to MemRef cast.
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    }

    llvm_unreachable("only tensor/memref input types supported");
  });
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    // #define DEBUG_TYPE "wlq"
    // for(auto ty:op.getFunctionType().getInputs()){
    //   LLVM_DEBUG(llvm::dbgs()<< typeConverter.convertType(ty)<<" "<<ty<<"\n");
    // }
    // LLVM_DEBUG(llvm::dbgs()<< "--------------------------------\n");
    // for(auto ty:op.getFunctionType().getResults()){
    //   LLVM_DEBUG(llvm::dbgs()<< typeConverter.convertType(ty)<<" "<<ty<<"\n");
    // }
    // #undef DEBUG_TYPE
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  RewritePatternSet patterns(context);
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addLegalOp<ModuleOp, bufferization::ToTensorOp,
                      bufferization::ToMemrefOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation* op){
    return isLegalForReturnOpTypeConversionPattern(op, typeConverter) || isNotBranchOpInterfaceOrReturnLikeOp(op);
  });
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {
namespace buddy {
void registerFuncBufferizeDynamicOffsetPass() {
  PassRegistration<FuncBufferizeDynamicOffsetPass>();
}
} // namespace buddy
} // namespace mlir