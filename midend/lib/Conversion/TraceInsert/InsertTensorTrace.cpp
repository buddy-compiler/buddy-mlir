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

constexpr StringLiteral kTensorTraceF32FuncName = "buddyTraceTensorF32";
constexpr StringLiteral kTensorTraceBF16FuncName = "buddyTraceTensorBF16";
constexpr StringLiteral kTensorTraceF32PathFuncName = "buddyTraceTensorF32Path";
constexpr StringLiteral kTensorTraceBF16PathFuncName =
    "buddyTraceTensorBF16Path";
constexpr StringLiteral kCycleTraceStartFuncName = "buddyTraceCycleStart";
constexpr StringLiteral kCycleTraceEndFuncName = "buddyTraceCycleEnd";
constexpr StringLiteral kCycleTraceStartPathFuncName =
    "buddyTraceCycleStartPath";
constexpr StringLiteral kCycleTraceEndPathFuncName = "buddyTraceCycleEndPath";
constexpr StringLiteral kIdPathAttrName = "id_path";
constexpr StringLiteral kLevelAttrName = "level";
constexpr StringLiteral kParentAttrName = "parent";
constexpr StringLiteral kGeneratedAttrName = "generated";
constexpr StringLiteral kTraceTypeAttrName = "trace_type";
constexpr int64_t kRuntimeTracePathMaxDepth = 4;

struct TraceModes {
  bool tensorTrace = false;
  bool cycleTrace = false;
};

std::string extendAttrName(StringRef dialect) {
  return ("trace_extend_" + dialect).str();
}

std::string includeAttrName(StringRef dialect) {
  return ("trace_" + dialect + "_include").str();
}

bool hasUnitOrTrueAttr(Operation *op, StringRef name) {
  Attribute attr = op->getAttr(name);
  if (!attr)
    return false;
  if (isa<UnitAttr>(attr))
    return true;
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  return false;
}

SmallVector<int64_t> getIdPath(Operation *op, int64_t fallback) {
  SmallVector<int64_t> result;
  if (auto arrayAttr =
          dyn_cast_or_null<ArrayAttr>(op->getAttr(kIdPathAttrName))) {
    for (Attribute attr : arrayAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        return {fallback};
      result.push_back(intAttr.getInt());
    }
    if (!result.empty())
      return result;
  }
  return {fallback};
}

ArrayAttr getIdPathAttr(Builder &builder, ArrayRef<int64_t> path) {
  SmallVector<Attribute> attrs;
  for (int64_t value : path)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

SmallVector<std::string> getIncludeList(Operation *op, StringRef dialect) {
  SmallVector<std::string> result;
  auto arrayAttr =
      dyn_cast_or_null<ArrayAttr>(op->getAttr(includeAttrName(dialect)));
  if (!arrayAttr)
    return result;
  for (Attribute attr : arrayAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (strAttr)
      result.push_back(strAttr.getValue().str());
  }
  return result;
}

ArrayAttr getStringArrayAttr(OpBuilder &builder, ArrayRef<std::string> values) {
  SmallVector<Attribute> attrs;
  for (const std::string &value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

StringRef getShortOpName(Operation *op) {
  StringRef fullName = op->getName().getStringRef();
  auto split = fullName.split('.');
  return split.second.empty() ? fullName : split.second;
}

bool matchesDialect(Operation *op, StringRef dialect) {
  return op->getName().getDialectNamespace() == dialect;
}

bool matchesInclude(Operation *op, ArrayRef<std::string> include) {
  if (include.empty())
    return true;
  StringRef fullName = op->getName().getStringRef();
  StringRef shortName = getShortOpName(op);
  for (const std::string &name : include) {
    if (fullName == name || shortName == name)
      return true;
  }
  return false;
}

int64_t getMaxTraceId(ModuleOp module) {
  int64_t maxId = -1;
  module.walk([&](Operation *op) {
    if (auto start = dyn_cast<::buddy::trace::StartOp>(op))
      maxId = std::max(maxId, static_cast<int64_t>(start.getId()));
    else if (auto end = dyn_cast<::buddy::trace::EndOp>(op))
      maxId = std::max(maxId, static_cast<int64_t>(end.getId()));
  });
  return maxId;
}

::buddy::trace::EndOp findMatchingEnd(::buddy::trace::StartOp start) {
  Block *block = start->getBlock();
  if (!block)
    return {};

  int64_t id = static_cast<int64_t>(start.getId());
  int64_t depth = 0;
  for (auto it = std::next(start->getIterator()), e = block->end(); it != e;
       ++it) {
    Operation *op = &*it;
    if (isa<::buddy::trace::StartOp>(op)) {
      ++depth;
      continue;
    }
    if (auto end = dyn_cast<::buddy::trace::EndOp>(op)) {
      if (depth == 0 && static_cast<int64_t>(end.getId()) == id)
        return end;
      if (depth > 0)
        --depth;
    }
  }
  return {};
}

SmallVector<Operation *> collectOpsInScope(::buddy::trace::StartOp start,
                                           ::buddy::trace::EndOp end,
                                           StringRef dialect,
                                           ArrayRef<std::string> include) {
  SmallVector<Operation *> result;
  auto collect = [&](Operation *op) {
    if (op->getName().getDialectNamespace() == "buddy_trace")
      return;
    if (matchesDialect(op, dialect) && matchesInclude(op, include))
      result.push_back(op);
  };
  for (auto it = std::next(start->getIterator()); &*it != end.getOperation();
       ++it) {
    Operation *op = &*it;
    collect(op);
    op->walk([&](Operation *nested) {
      if (nested != op)
        collect(nested);
    });
  }
  return result;
}

void setGeneratedTraceAttrs(Operation *op, OpBuilder &builder, int64_t parent,
                            ArrayRef<int64_t> idPath, int64_t level,
                            StringRef traceType,
                            ArrayRef<std::string> buckyballInclude,
                            bool propagateBuckyball) {
  op->setAttr(kIdPathAttrName, getIdPathAttr(builder, idPath));
  op->setAttr(kParentAttrName, builder.getI64IntegerAttr(parent));
  op->setAttr(kLevelAttrName, builder.getI64IntegerAttr(level));
  op->setAttr(kGeneratedAttrName, builder.getBoolAttr(true));
  op->setAttr(kTraceTypeAttrName, builder.getStringAttr(traceType));
  if (propagateBuckyball) {
    op->setAttr(extendAttrName("buckyball"), builder.getBoolAttr(true));
    if (!buckyballInclude.empty())
      op->setAttr(includeAttrName("buckyball"),
                  getStringArrayAttr(builder, buckyballInclude));
  }
}

void insertTraceAround(Operation *target, ::buddy::trace::StartOp parent,
                       int64_t id, ArrayRef<int64_t> idPath, StringRef tag,
                       StringRef traceType, bool propagateBuckyball,
                       ArrayRef<std::string> buckyballInclude) {
  OpBuilder startBuilder(target);
  auto idAttr = startBuilder.getI64IntegerAttr(id);
  auto tagAttr = startBuilder.getStringAttr(tag);
  auto start = startBuilder.create<::buddy::trace::StartOp>(target->getLoc(),
                                                            idAttr, tagAttr);
  int64_t level = static_cast<int64_t>(idPath.size()) - 1;
  setGeneratedTraceAttrs(start.getOperation(), startBuilder, parent.getId(),
                         idPath, level, traceType, buckyballInclude,
                         propagateBuckyball);

  OpBuilder endBuilder(target->getBlock(), std::next(Block::iterator(target)));
  Value input;
  if (target->getNumResults() == 1)
    input = target->getResult(0);
  else
    input = endBuilder.create<arith::ConstantIndexOp>(target->getLoc(), 0);

  auto end = endBuilder.create<::buddy::trace::EndOp>(
      target->getLoc(), input.getType(), input, idAttr, tagAttr);
  setGeneratedTraceAttrs(end.getOperation(), endBuilder, parent.getId(), idPath,
                         level, traceType, buckyballInclude,
                         propagateBuckyball);
}

class ExtendTracePass
    : public PassWrapper<ExtendTracePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtendTracePass)

  ExtendTracePass(StringRef dialect) : dialect(dialect.str()) {}

  StringRef getDescription() const override {
    return "Extend trace scopes into dialect-specific child trace scopes";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, ::buddy::trace::BuddyTraceDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<::buddy::trace::StartOp> starts;
    module.walk([&](::buddy::trace::StartOp start) {
      if (hasUnitOrTrueAttr(start.getOperation(), extendAttrName(dialect)))
        starts.push_back(start);
    });

    int64_t nextId = getMaxTraceId(module) + 1;
    for (::buddy::trace::StartOp start : starts) {
      std::string extendedAttr = "trace_extended_" + dialect;
      if (start->getAttr(extendedAttr)) {
        start.emitError("trace scope already extended to ") << dialect;
        signalPassFailure();
        return;
      }

      auto end = findMatchingEnd(start);
      if (!end) {
        start.emitError("failed to find matching trace end");
        signalPassFailure();
        return;
      }

      SmallVector<std::string> include =
          getIncludeList(start.getOperation(), dialect);
      SmallVector<Operation *> targets =
          collectOpsInScope(start, end, dialect, include);
      if (targets.empty()) {
        start.emitError("trace extension found no matching ")
            << dialect << " ops";
        signalPassFailure();
        return;
      }

      SmallVector<int64_t> parentPath =
          getIdPath(start.getOperation(), start.getId());
      StringRef parentTag =
          start.getTagAttr() ? start.getTagAttr().getValue() : "trace";
      bool propagateBuckyball =
          dialect == "linalg" &&
          hasUnitOrTrueAttr(start.getOperation(), extendAttrName("buckyball"));
      SmallVector<std::string> buckyballInclude =
          getIncludeList(start.getOperation(), "buckyball");

      int64_t ordinal = 0;
      for (Operation *target : targets) {
        SmallVector<int64_t> childPath(parentPath.begin(), parentPath.end());
        childPath.push_back(ordinal);
        std::string tag = parentTag.str() + "." + dialect + "." +
                          getShortOpName(target).str() + "." +
                          std::to_string(ordinal);
        insertTraceAround(target, start, nextId++, childPath, tag, dialect,
                          propagateBuckyball, buckyballInclude);
        ++ordinal;
      }

      OpBuilder builder(start);
      start->setAttr(extendedAttr, builder.getBoolAttr(true));
      end->setAttr(extendedAttr, builder.getBoolAttr(true));
      if (dialect == "linalg" && propagateBuckyball) {
        start->removeAttr(extendAttrName("buckyball"));
        start->removeAttr(includeAttrName("buckyball"));
        end->removeAttr(extendAttrName("buckyball"));
        end->removeAttr(includeAttrName("buckyball"));
      }
    }
  }

private:
  std::string dialect;
};

class ExtendTraceToLinalgPass : public ExtendTracePass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtendTraceToLinalgPass)

  ExtendTraceToLinalgPass() : ExtendTracePass("linalg") {}

  StringRef getArgument() const final { return "extend-trace-to-linalg"; }

  StringRef getDescription() const final {
    return "Extend marked trace scopes to matching linalg ops";
  }
};

class ExtendTraceToBuckyballPass : public ExtendTracePass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtendTraceToBuckyballPass)

  ExtendTraceToBuckyballPass() : ExtendTracePass("buckyball") {}

  StringRef getArgument() const final { return "extend-trace-to-buckyball"; }

  StringRef getDescription() const final {
    return "Extend marked trace scopes to matching buckyball ops";
  }
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
      if (modes.cycleTrace && failed(insertCycleStartCall(start))) {
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

  func::FuncOp ensureTensorTraceDecl(ModuleOp module, StringRef funcName,
                                     Type elemType) {
    OpBuilder builder(module.getBodyRegion());
    Type i64 = builder.getI64Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, elemType);
    return ensureRuntimeDecl(module, funcName, {i64, memrefType});
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
    values.push_back(builder.create<arith::ConstantIntOp>(loc, id, 64));
    values.push_back(builder.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(path.size()), 64));
    for (int64_t i = 0; i < kRuntimeTracePathMaxDepth; ++i) {
      int64_t component = i < static_cast<int64_t>(path.size()) ? path[i] : -1;
      values.push_back(
          builder.create<arith::ConstantIntOp>(loc, component, 64));
    }
    return success();
  }

  LogicalResult insertCycleStartCall(::buddy::trace::StartOp op) {
    OpBuilder builder(op);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    builder.create<func::CallOp>(op.getLoc(), kCycleTraceStartPathFuncName,
                                 TypeRange{}, args);
    return success();
  }

  LogicalResult insertCycleEndCall(::buddy::trace::EndOp op) {
    OpBuilder builder(op);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    builder.create<func::CallOp>(op.getLoc(), kCycleTraceEndPathFuncName,
                                 TypeRange{}, args);
    return success();
  }

  LogicalResult lowerEndOp(::buddy::trace::EndOp op) {
    if (modes.tensorTrace && failed(insertTensorTraceCall(op)))
      return failure();
    if (modes.cycleTrace && failed(insertCycleEndCall(op)))
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
    if (elemType.isF32())
      funcName = kTensorTraceF32PathFuncName;
    else if (elemType.isBF16())
      funcName = kTensorTraceBF16PathFuncName;
    else
      return op.emitError("trace end only supports f32 and bf16 memrefs");

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
      flat = builder.create<memref::CollapseShapeOp>(loc, flatType, value,
                                                     reassociation);
    }

    auto castType =
        MemRefType::get({ShapedType::kDynamic}, memrefType.getElementType());
    Value cast = builder.create<memref::CastOp>(loc, castType, flat);
    SmallVector<Value> args;
    if (failed(buildTracePathValues(op.getOperation(), builder, args)))
      return failure();
    args.push_back(cast);
    builder.create<func::CallOp>(loc, funcName, TypeRange{}, args);
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
  PassRegistration<ExtendTraceToLinalgPass>();
  PassRegistration<ExtendTraceToBuckyballPass>();
  registerPassPipeline("convert-trace-to-llvm",
                       "Lower buddy_trace ops to selected runtime trace calls",
                       buildConvertTraceToLLVMPipeline,
                       [](function_ref<void(const detail::PassOptions &)>) {});
}
} // namespace buddy
} // namespace mlir
