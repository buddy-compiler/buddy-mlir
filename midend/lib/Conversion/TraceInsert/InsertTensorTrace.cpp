//===- InsertTensorTrace.cpp - Convert trace ops to runtime calls ---------===//
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
// This pass lowers buddy_trace ops to selected runtime trace calls.
//
//===----------------------------------------------------------------------===//

#include "Trace/TraceOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

constexpr StringLiteral kTensorTraceF32PathFuncName = "buddyTraceTensorF32Path";
constexpr StringLiteral kTensorTraceBF16PathFuncName =
    "buddyTraceTensorBF16Path";
constexpr StringLiteral kTensorTraceI8PathFuncName = "buddyTraceTensorI8Path";
constexpr StringLiteral kTensorTraceI32PathFuncName = "buddyTraceTensorI32Path";
constexpr StringLiteral kBuckyballStageTraceI8PathFuncName =
    "buckyballTraceStageI8Path";
constexpr StringLiteral kCycleTraceStartPathFuncName =
    "buddyTraceCycleStartPath";
constexpr StringLiteral kCycleTraceEndPathFuncName = "buddyTraceCycleEndPath";
constexpr StringLiteral kIdPathAttrName = "id_path";
constexpr int64_t kRuntimeTracePathMaxDepth = 4;

SmallVector<int64_t> getIdPath(Operation *op, int64_t id) {
  if (Attribute attr = op->getAttr(kIdPathAttrName)) {
    SmallVector<int64_t> path;
    for (Attribute value : cast<ArrayAttr>(attr))
      path.push_back(cast<IntegerAttr>(value).getInt());
    return path;
  }
  return {id};
}

class ConvertTraceToLLVMPass
    : public PassWrapper<ConvertTraceToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTraceToLLVMPass)

  StringRef getArgument() const final { return "convert-trace-to-llvm"; }

  StringRef getDescription() const final {
    return "Lower buddy_trace ops to runtime trace calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, ::buddy::trace::BuddyTraceDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<::buddy::trace::StartOp> starts;
    SmallVector<::buddy::trace::EndOp> ends;
    module.walk([&](Operation *op) {
      if (auto start = dyn_cast<::buddy::trace::StartOp>(op))
        starts.push_back(start);
      else if (auto end = dyn_cast<::buddy::trace::EndOp>(op))
        ends.push_back(end);
    });

    for (::buddy::trace::StartOp start : starts) {
      if (start.getCycle() && failed(insertCycleStartCall(start))) {
        signalPassFailure();
        return;
      }
      start.erase();
    }

    for (::buddy::trace::EndOp end : ends) {
      if (failed(lowerEndOp(end))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  func::FuncOp ensureRuntimeDecl(ModuleOp module, StringRef funcName,
                                 TypeRange argTypes) {
    if (auto func = module.lookupSymbol<func::FuncOp>(funcName))
      return func;

    OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType(argTypes, {});
    auto func =
        func::FuncOp::create(builder, module.getLoc(), funcName, funcType);
    func.setPrivate();
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    return func;
  }

  func::FuncOp ensureTensorTracePathDecl(ModuleOp module, StringRef funcName,
                                         Type elemType) {
    OpBuilder builder(module.getBodyRegion());
    Type i64 = builder.getI64Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    return ensureRuntimeDecl(module, funcName,
                             {i64, i64, i64, i64, i64, i64, memrefType});
  }

  void ensureCycleTraceDecls(ModuleOp module) {
    OpBuilder builder(module.getBodyRegion());
    Type i64 = builder.getI64Type();
    ensureRuntimeDecl(module, kCycleTraceStartPathFuncName,
                      {i64, i64, i64, i64, i64, i64});
    ensureRuntimeDecl(module, kCycleTraceEndPathFuncName,
                      {i64, i64, i64, i64, i64, i64});
  }

  LogicalResult buildTracePathValues(Operation *op, OpBuilder &builder,
                                     SmallVectorImpl<Value> &values) {
    int64_t id = 0;
    if (auto start = dyn_cast<::buddy::trace::StartOp>(op))
      id = start.getId();
    else if (auto end = dyn_cast<::buddy::trace::EndOp>(op))
      id = end.getId();
    else
      return failure();

    SmallVector<int64_t> path = getIdPath(op, id);
    if (path.size() > kRuntimeTracePathMaxDepth)
      return op->emitError("trace id_path exceeds runtime depth limit");

    Location loc = op->getLoc();
    values.push_back(arith::ConstantIntOp::create(builder, loc, id, 64));
    values.push_back(arith::ConstantIntOp::create(
        builder, loc, static_cast<int64_t>(path.size()), 64));
    for (int64_t i = 0; i < kRuntimeTracePathMaxDepth; ++i) {
      int64_t component = i < static_cast<int64_t>(path.size()) ? path[i] : -1;
      values.push_back(
          arith::ConstantIntOp::create(builder, loc, component, 64));
    }
    return success();
  }

  LogicalResult insertCycleStartCall(::buddy::trace::StartOp op) {
    ensureCycleTraceDecls(op->getParentOfType<ModuleOp>());
    OpBuilder builder(op);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    func::CallOp::create(builder, op.getLoc(), kCycleTraceStartPathFuncName,
                         TypeRange{}, args);
    return success();
  }

  LogicalResult insertCycleEndCall(::buddy::trace::EndOp op) {
    ensureCycleTraceDecls(op->getParentOfType<ModuleOp>());
    OpBuilder builder(op);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    func::CallOp::create(builder, op.getLoc(), kCycleTraceEndPathFuncName,
                         TypeRange{}, args);
    return success();
  }

  LogicalResult lowerEndOp(::buddy::trace::EndOp op) {
    if (op.getCycle() && failed(insertCycleEndCall(op)))
      return failure();
    if (op.getTensor() && failed(insertTensorTraceCall(op)))
      return failure();

    op.getOutput().replaceAllUsesWith(op.getInput());
    op.erase();
    return success();
  }

  LogicalResult insertTensorTraceCall(::buddy::trace::EndOp op) {
    Value value = op.getInput();
    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType)
      return op.emitError("trace end input must be a ranked memref");
    if (!memrefType.hasStaticShape())
      return op.emitError("trace end requires static memref shape");
    Type elemType = memrefType.getElementType();
    StringRef funcName;
    bool buckyballStageTrace = op->hasAttr("buckyball.stage_trace");
    if (buckyballStageTrace && elemType.isInteger(8))
      funcName = kBuckyballStageTraceI8PathFuncName;
    else if (elemType.isF32())
      funcName = kTensorTraceF32PathFuncName;
    else if (elemType.isBF16())
      funcName = kTensorTraceBF16PathFuncName;
    else if (elemType.isInteger(8))
      funcName = kTensorTraceI8PathFuncName;
    else if (elemType.isInteger(32))
      funcName = kTensorTraceI32PathFuncName;
    else
      return op.emitError(
          "trace end only supports f32, bf16, i8, and i32 memrefs");

    ensureTensorTracePathDecl(op->getParentOfType<ModuleOp>(), funcName,
                              elemType);

    int64_t flatSize = memrefType.getNumElements();
    OpBuilder builder(op);
    Location loc = op.getLoc();

    Value flat = value;
    if (memrefType.getRank() != 1 || memrefType.getShape()[0] != flatSize) {
      SmallVector<ReassociationIndices> reassociation(1);
      for (int64_t i = 0, e = memrefType.getRank(); i < e; ++i)
        reassociation[0].push_back(i);
      auto flatType = MemRefType::get({flatSize}, memrefType.getElementType());
      flat = memref::CollapseShapeOp::create(builder, loc, flatType, value,
                                             reassociation);
    }

    auto castType =
        MemRefType::get({ShapedType::kDynamic}, memrefType.getElementType());
    Value cast = memref::CastOp::create(builder, loc, castType, flat);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    args.push_back(cast);
    func::CallOp::create(builder, loc, funcName, TypeRange{}, args);
    return success();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerConvertTraceToLLVMPass() {
  PassRegistration<ConvertTraceToLLVMPass>();
}
} // namespace buddy
} // namespace mlir
