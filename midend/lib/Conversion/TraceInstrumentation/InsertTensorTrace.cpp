//===- InsertTensorTrace.cpp - Insert tensor trace calls ------------------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

constexpr StringLiteral kTraceIdAttr = "buddy.trace_id";
constexpr StringLiteral kTraceFuncName = "buddyTraceTensorF32";

class InsertTensorTracePass
    : public PassWrapper<InsertTensorTracePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertTensorTracePass)

  StringRef getArgument() const final { return "insert-tensor-trace"; }

  StringRef getDescription() const final {
    return "Insert trace calls for tensor ops marked with buddy.trace_id";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ensureTraceDecl(module);

    SmallVector<Operation *> targets;
    module.walk([&](Operation *op) {
      if (op->hasAttr(kTraceIdAttr))
        targets.push_back(op);
    });

    for (Operation *op : targets) {
      if (failed(insertTraceCall(op))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  void ensureTraceDecl(ModuleOp module) {
    if (module.lookupSymbol<func::FuncOp>(kTraceFuncName))
      return;

    OpBuilder builder(module.getBodyRegion());
    Location loc = module.getLoc();
    Type i64 = builder.getI64Type();
    Type f32 = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32);
    auto funcType = builder.getFunctionType({i64, memrefType}, {});
    auto func = builder.create<func::FuncOp>(loc, kTraceFuncName, funcType);
    func.setPrivate();
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  }

  LogicalResult insertTraceCall(Operation *op) {
    auto traceId = dyn_cast<IntegerAttr>(op->getAttr(kTraceIdAttr));
    if (!traceId)
      return op->emitError("buddy.trace_id must be an integer attribute");
    if (op->getNumResults() != 1)
      return op->emitError("buddy.trace_id only supports single-result ops");

    Value value = op->getResult(0);
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType)
      return op->emitError("buddy.trace_id result must be a ranked tensor");
    if (!tensorType.hasStaticShape())
      return op->emitError("buddy.trace_id requires static tensor shape");
    if (!tensorType.getElementType().isF32())
      return op->emitError(
          "buddy.trace_id currently only supports f32 tensors");

    int64_t flatSize = tensorType.getNumElements();
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    Location loc = op->getLoc();

    Value flat = value;
    auto flatType =
        RankedTensorType::get({flatSize}, tensorType.getElementType());
    if (tensorType.getRank() != 1 || tensorType.getShape()[0] != flatSize) {
      SmallVector<ReassociationIndices> reassociation(1);
      for (int64_t i = 0, e = tensorType.getRank(); i < e; ++i)
        reassociation[0].push_back(i);
      flat = builder.create<tensor::CollapseShapeOp>(loc, flatType, value,
                                                     reassociation);
    }

    auto bufType = MemRefType::get({flatSize}, tensorType.getElementType());
    Value buf = bufferization::ToBufferOp::create(builder, loc, bufType, flat);
    auto castType =
        MemRefType::get({ShapedType::kDynamic}, tensorType.getElementType());
    Value cast = builder.create<memref::CastOp>(loc, castType, buf);
    Value id = builder.create<arith::ConstantIntOp>(loc, traceId.getInt(), 64);
    builder.create<func::CallOp>(loc, kTraceFuncName, TypeRange{},
                                 ValueRange{id, cast});
    return success();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerInsertTensorTracePass() {
  PassRegistration<InsertTensorTracePass>();
}
} // namespace buddy
} // namespace mlir
