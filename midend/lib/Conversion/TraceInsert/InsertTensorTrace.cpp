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

constexpr StringLiteral kTensorTraceFuncName = "buddyTraceTensorF32";
constexpr StringLiteral kCycleTraceStartFuncName = "buddyTraceCycleStart";
constexpr StringLiteral kCycleTraceEndFuncName = "buddyTraceCycleEnd";

struct TraceModes {
  bool tensorTrace = false;
  bool cycleTrace = false;
};

FailureOr<TraceModes> parseTraceModes(StringRef modes) {
  TraceModes parsed;
  StringRef trimmed = modes.trim();
  if ((trimmed.starts_with("\"") && trimmed.ends_with("\"")) ||
      (trimmed.starts_with("'") && trimmed.ends_with("'")))
    trimmed = trimmed.drop_front().drop_back().trim();
  else if (trimmed.starts_with("\"") || trimmed.ends_with("\"") ||
           trimmed.starts_with("'") || trimmed.ends_with("'"))
    return failure();
  if (trimmed.starts_with("\"") || trimmed.ends_with("\"") ||
      trimmed.starts_with("'") || trimmed.ends_with("'"))
    return failure();
  if (trimmed.empty())
    return failure();

  SmallVector<StringRef> parts;
  trimmed.split(parts, ",");
  for (StringRef part : parts) {
    StringRef mode = part.trim();
    if (mode.empty())
      return failure();
    if (mode == "tensor-trace") {
      if (parsed.tensorTrace)
        return failure();
      parsed.tensorTrace = true;
      continue;
    }
    if (mode == "cycle-trace") {
      if (parsed.cycleTrace)
        return failure();
      parsed.cycleTrace = true;
      continue;
    }
    return failure();
  }
  return parsed;
}

class ConvertTraceToLLVMPass
    : public PassWrapper<ConvertTraceToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTraceToLLVMPass)

  ConvertTraceToLLVMPass(TraceModes modes) : modes(modes) {}

  StringRef getDescription() const final {
    return "Lower buddy_trace ops to runtime trace calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, ::buddy::trace::BuddyTraceDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (modes.tensorTrace)
      ensureTensorTraceDecl(module);
    if (modes.cycleTrace)
      ensureCycleTraceDecls(module);

    SmallVector<::buddy::trace::StartOp> starts;
    SmallVector<::buddy::trace::EndOp> ends;
    module.walk([&](Operation *op) {
      if (auto start = dyn_cast<::buddy::trace::StartOp>(op))
        starts.push_back(start);
      else if (auto end = dyn_cast<::buddy::trace::EndOp>(op))
        ends.push_back(end);
    });

    for (::buddy::trace::StartOp start : starts) {
      if (modes.cycleTrace)
        insertCycleStartCall(start);
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
  TraceModes modes;

  func::FuncOp ensureRuntimeDecl(ModuleOp module, StringRef funcName,
                                 TypeRange argTypes) {
    if (auto func = module.lookupSymbol<func::FuncOp>(funcName))
      return func;

    OpBuilder builder(module.getBodyRegion());
    auto funcType = builder.getFunctionType(argTypes, {});
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();
    func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    return func;
  }

  void ensureTensorTraceDecl(ModuleOp module) {
    if (module.lookupSymbol<func::FuncOp>(kTensorTraceFuncName))
      return;

    OpBuilder builder(module.getBodyRegion());
    Type i64 = builder.getI64Type();
    Type f32 = builder.getF32Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, f32);
    ensureRuntimeDecl(module, kTensorTraceFuncName, {i64, memrefType});
  }

  void ensureCycleTraceDecls(ModuleOp module) {
    OpBuilder builder(module.getBodyRegion());
    Type i64 = builder.getI64Type();
    ensureRuntimeDecl(module, kCycleTraceStartFuncName, {i64});
    ensureRuntimeDecl(module, kCycleTraceEndFuncName, {i64});
  }

  void insertCycleStartCall(::buddy::trace::StartOp op) {
    OpBuilder builder(op);
    Value id =
        builder.create<arith::ConstantIntOp>(op.getLoc(), op.getId(), 64);
    builder.create<func::CallOp>(op.getLoc(), kCycleTraceStartFuncName,
                                 TypeRange{}, ValueRange{id});
  }

  void insertCycleEndCall(::buddy::trace::EndOp op) {
    OpBuilder builder(op);
    Value id =
        builder.create<arith::ConstantIntOp>(op.getLoc(), op.getId(), 64);
    builder.create<func::CallOp>(op.getLoc(), kCycleTraceEndFuncName,
                                 TypeRange{}, ValueRange{id});
  }

  LogicalResult lowerEndOp(::buddy::trace::EndOp op) {
    if (modes.tensorTrace && failed(insertTensorTraceCall(op)))
      return failure();
    if (modes.cycleTrace)
      insertCycleEndCall(op);

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
    if (!memrefType.getElementType().isF32())
      return op.emitError("trace end currently only supports f32 memrefs");

    int64_t flatSize = memrefType.getNumElements();
    OpBuilder builder(op);
    Location loc = op.getLoc();

    Value flat = value;
    if (memrefType.getRank() != 1 || memrefType.getShape()[0] != flatSize) {
      SmallVector<ReassociationIndices> reassociation(1);
      for (int64_t i = 0, e = memrefType.getRank(); i < e; ++i)
        reassociation[0].push_back(i);
      auto flatType = MemRefType::get({flatSize}, memrefType.getElementType());
      flat = builder.create<memref::CollapseShapeOp>(loc, flatType, value,
                                                     reassociation);
    }

    auto castType =
        MemRefType::get({ShapedType::kDynamic}, memrefType.getElementType());
    Value cast = builder.create<memref::CastOp>(loc, castType, flat);
    Value id = builder.create<arith::ConstantIntOp>(loc, op.getId(), 64);
    builder.create<func::CallOp>(loc, kTensorTraceFuncName, TypeRange{},
                                 ValueRange{id, cast});
    return success();
  }
};

LogicalResult buildConvertTraceToLLVMPipeline(
    OpPassManager &pm, StringRef options,
    function_ref<LogicalResult(const Twine &)> errorHandler) {
  FailureOr<TraceModes> modes = parseTraceModes(options);
  if (failed(modes)) {
    return errorHandler(
        "expected comma-separated trace modes: tensor-trace,cycle-trace");
  }
  pm.addPass(std::make_unique<ConvertTraceToLLVMPass>(*modes));
  return success();
}

} // namespace

namespace mlir {
namespace buddy {
void registerConvertTraceToLLVMPass() {
  registerPassPipeline("convert-trace-to-llvm",
                       "Lower buddy_trace ops to selected runtime trace calls",
                       buildConvertTraceToLLVMPipeline,
                       [](function_ref<void(const detail::PassOptions &)>) {});
}
} // namespace buddy
} // namespace mlir
