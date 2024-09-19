//====- LowerTracePass.cpp - Trace Dialect Lowering Pass
//---------------------===//
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
// This file defines trace dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

#include "Trace/TraceDialect.h"
#include "Trace/TraceOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

class TraceStartLowering : public OpRewritePattern<trace::StartOp> {
public:
  using OpRewritePattern<trace::StartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(trace::StartOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 确保rtclock函数的符号存在
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FlatSymbolRefAttr funcSymbol;
    auto rtclockFunc = module.lookupSymbol<mlir::func::FuncOp>("rtclock");
    if (!rtclockFunc) {
      // 如果函数不存在，则声明它
      auto funcType = rewriter.getFunctionType({}, rewriter.getF64Type());
      rtclockFunc = rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                                        "rtclock", funcType);
      rtclockFunc.setPrivate();
      funcSymbol = mlir::SymbolRefAttr::get(op.getContext(), "rtclock");
    } else {
      funcSymbol =
          mlir::SymbolRefAttr::get(op.getContext(), rtclockFunc.getName());
    }

    // 创建一个调用rtclock的操作
    auto callOp = rewriter.create<mlir::func::CallOp>(
        loc, rewriter.getF64Type(), funcSymbol, ValueRange());

    // 用新的call操作替换原有的StartOp
    rewriter.replaceOp(op, callOp.getResults());

    return success();
  }
};

class TraceEndLowering : public OpRewritePattern<trace::EndOp> {
public:
  using OpRewritePattern<trace::EndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(trace::EndOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 确保rtclock函数的符号存在
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FlatSymbolRefAttr funcSymbol;
    auto rtclockFunc = module.lookupSymbol<mlir::func::FuncOp>("rtclock");
    if (!rtclockFunc) {
      // 如果函数不存在，则声明它
      auto funcType = rewriter.getFunctionType({}, rewriter.getF64Type());
      rtclockFunc = rewriter.create<mlir::func::FuncOp>(module.getLoc(),
                                                        "rtclock", funcType);
      rtclockFunc.setPrivate();
      funcSymbol = mlir::SymbolRefAttr::get(op.getContext(), "rtclock");
    } else {
      funcSymbol =
          mlir::SymbolRefAttr::get(op.getContext(), rtclockFunc.getName());
    }

    // 创建一个调用rtclock的操作
    auto callOp = rewriter.create<mlir::func::CallOp>(
        loc, rewriter.getF64Type(), funcSymbol, ValueRange());

    // 用新的call操作替换原有的StartOp
    rewriter.replaceOp(op, callOp.getResults());

    return success();
  }
};

} // end anonymous namespace

void populateLowerTraceConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      TraceStartLowering,
      TraceEndLowering>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerTracePass
//===----------------------------------------------------------------------===//

namespace {
class LowerTracePass
    : public PassWrapper<LowerTracePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTracePass)
  LowerTracePass() = default;
  LowerTracePass(const LowerTracePass &) {}

  StringRef getArgument() const final { return "lower-trace"; }
  StringRef getDescription() const final { return "Lower Trace Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::trace::TraceDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};

class TracePass : public PassWrapper<TracePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TracePass)
  TracePass() = default;
  TracePass(const TracePass &) {}

  Option<std::string> OpNameOption{
      *this, "trace", llvm::cl::desc("Trace an op name"),
      llvm::cl::init(
          "linalg.conv_2d")}; // linalg.conv_2d  linalg.generic
                              // tosa.mul linalg.matmul tosa.transpose

  StringRef getArgument() const final { return "trace"; }
  StringRef getDescription() const final { return "Trace Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        trace::TraceDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};

} // end anonymous namespace.

void LowerTracePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      vector::VectorDialect,
      memref::MemRefDialect,
      LLVM::LLVMDialect>();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerTraceConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void TracePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  OpBuilder builder(context);

  // 在module的开头插入rtclock函数声明，memref 计时器的内存。
  builder.setInsertionPointToStart(module.getBody());

  // func.func @rtclock
  auto funcType = builder.getFunctionType({}, builder.getF64Type());
  auto rtclockFunc =
      builder.create<mlir::func::FuncOp>(module.getLoc(), "rtclock", funcType);
  rtclockFunc.setPrivate();

  // func.func private @printMemrefF64(memref<*xf64>)
  mlir::Type memrefType1 =
      mlir::UnrankedMemRefType::get(builder.getF64Type(), /*memorySpace=*/0);
  auto printmemrefFuncType = builder.getFunctionType({memrefType1}, {});
  auto printMemrefFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "printMemrefF64", printmemrefFuncType);
  printMemrefFunc.setPrivate();

  // func.func private @printF64(f64)
  auto printF64FuncType = builder.getFunctionType({builder.getF64Type()}, {});
  auto printF64Func = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "printF64", printF64FuncType);
  printF64Func.setPrivate();

  // printNewline ()
  auto printNewlineType = builder.getFunctionType({}, {});
  auto printNewlineFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "printNewline", printNewlineType);
  printNewlineFunc.setPrivate();

  // 遍历
  module.walk([&](mlir::func::FuncOp funcOp) {
    // llvm::outs() << "Function name: " << funcOp.getName() << "\n";
    if (funcOp == rtclockFunc || funcOp == printMemrefFunc ||
        funcOp == printF64Func || funcOp == printNewlineFunc) {
      // llvm::outs() << "return\n";
      return;
    }

    // 先遍历一遍，记录op的数量
    int opNum = 0;
    std::vector<Operation *> ops;
    Operation *returnOp;
    funcOp.walk([&](Operation *op) {
      // llvm::outs << op->getName().getStringRef() << "\n";
      // 检查操作是否是我们感兴趣的特定类型
      if (op->getName().getStringRef() == OpNameOption) {
        opNum++;
        ops.push_back(op);
      } else if (op->getName().getStringRef() == "return") {
        returnOp = op;
      }
    });

    if (ops.empty()) {
      // llvm::outs() << "No " << OpNameOption << '\n';
      return;
    }

    // 将插入点插入到函数的开头
    builder.setInsertionPointToStart(&funcOp.getBody().front());

    // 插入memref.alloc
    auto memrefType = MemRefType::get({opNum}, builder.getF64Type());
    Value memrefAlloc =
        builder.create<mlir::memref::AllocOp>(module.getLoc(), memrefType);

    // 常数
    Value idx =
        builder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 0);
    Value c1 = builder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 1);
    Value all_duration = builder.create<arith::ConstantOp>(
        module.getLoc(), builder.getF64Type(), builder.getF64FloatAttr(0.0));

    // 遍历所有对应的Op
    for (auto op : ops) {
      // 在op的前面添加startOp
      builder.setInsertionPoint(op);
      Value startOp = builder.create<trace::StartOp>(op->getLoc());

      // 在op的后面添加endOp，subop，storeOp
      builder.setInsertionPointAfter(op);
      Value endOp = builder.create<trace::EndOp>(op->getLoc());
      Value SubOp =
          builder.create<mlir::arith::SubFOp>(op->getLoc(), endOp, startOp);
      builder.create<mlir::memref::StoreOp>(op->getLoc(), SubOp, memrefAlloc,
                                            ValueRange(idx));
      all_duration = builder.create<mlir::arith::AddFOp>(op->getLoc(),
                                                         all_duration, SubOp);
      idx = builder.create<mlir::arith::AddIOp>(op->getLoc(), idx, c1);
    }

    // builder.setInsertionPoint(returnOp);
    // cast memref<opNumxf64> -> memref<*xf64>
    Value result = builder.create<memref::CastOp>(ops.back()->getLoc(),
                                                  memrefType1, memrefAlloc);

    // 在代码的最后添加打印时间的op
    builder.create<mlir::func::CallOp>(ops.back()->getLoc(), printMemrefFunc,
                                       ValueRange{result});
    builder.create<mlir::func::CallOp>(ops.back()->getLoc(), printF64Func,
                                       ValueRange{all_duration});
    builder.create<mlir::func::CallOp>(ops.back()->getLoc(), printNewlineFunc,
                                       ValueRange{});
  });
}

namespace mlir {
namespace buddy {
void registerLowerTracePass() { PassRegistration<LowerTracePass>(); }

void registerTracePass() { PassRegistration<TracePass>(); }
} // namespace buddy
} // namespace mlir
