//===- FuncBufferizePass.cpp ----------------------------------------------===//
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
// This file implements the func-bufferize with dynamic offset.
//
//===----------------------------------------------------------------------===//
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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <memory>
#include <utility>
using namespace mlir;
using namespace mlir::func;

namespace {
class FuncBufferizeDynamicOffsetPass
    : public PassWrapper<FuncBufferizeDynamicOffsetPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuncBufferizeDynamicOffsetPass)
  FuncBufferizeDynamicOffsetPass() = default;
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

  // Convert RankedTensorType to MemrefType with dynamic offset.
  typeConverter.addConversion([&](RankedTensorType type) {
    auto shape = type.getShape();
    llvm::SmallVector<int64_t, 8> stride;
    stride.reserve(shape.size());
    int64_t initial_value = 1;

    // Compute MemrefType's stride.
    stride.insert(stride.begin(), initial_value);
    for (auto i = shape.size() - 1; i > 0; i--) {
      initial_value *= shape[i];
      stride.insert(stride.begin(), initial_value);
    }
    return MemRefType::get(
        type.getShape(), type.getElementType(),
        StridedLayoutAttr::get(context, mlirShapedTypeGetDynamicSize(),
                               ArrayRef<int64_t>(stride)));
  });
  typeConverter.addArgumentMaterialization(materializeToTensor);
  typeConverter.addSourceMaterialization(materializeToTensor);
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            BaseMemRefType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1 && "expected exactly one input");

    if (auto inputType = dyn_cast<MemRefType>(inputs[0].getType())) {
      // MemRef to MemRef cast.
      assert(inputType != type && "expected different types");
      // Unranked to ranked and ranked to unranked casts must be explicit.
      auto rankedDestType = dyn_cast<MemRefType>(type);
      if (!rankedDestType)
        return nullptr;
      mlir::bufferization::BufferizationOptions bufferizationOptions;
      FailureOr<Value> replacement = bufferization::castOrReallocMemRefValue(
          builder, inputs[0], rankedDestType, bufferizationOptions);
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
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  RewritePatternSet patterns(context);

  // Bufferize func's input args
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);

  // Bufferize func's return op.
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addLegalOp<ModuleOp, bufferization::ToTensorOp,
                    bufferization::ToMemrefOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isLegalForReturnOpTypeConversionPattern(op, typeConverter) ||
           isNotBranchOpInterfaceOrReturnLikeOp(op);
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
