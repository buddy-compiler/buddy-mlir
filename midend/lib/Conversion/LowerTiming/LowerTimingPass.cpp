//====- LowerTimingPass.cpp - Timing Dialect Lowering Pass
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
// This file defines timing dialect lowering pass.
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

#include "Timing/TimingDialect.h"
#include "Timing/TimingOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {
class TimingTestConstantLowering
    : public OpRewritePattern<timing::TestConstantOp> {
public:
  using OpRewritePattern<timing::TestConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::TestConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Create constant operation.
    Value c0 = rewriter.create<mlir::arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));

    rewriter.replaceOp(op, c0);
    return success();
  }
};

class TimingTestPrintLowering : public OpRewritePattern<timing::TestPrintOp> {
public:
  using OpRewritePattern<timing::TestPrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::TestPrintOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Create constant operation.
    Value c0 = rewriter.create<mlir::arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));
    // Create print operation for the scalar value.
    rewriter.create<vector::PrintOp>(loc, c0);
    VectorType vectorTy4 =
        VectorType::get({4 /*number of elements in the vector*/}, resultType);
    // Broadcast element of the kernel.
    Value broadcastVector =
        rewriter.create<vector::BroadcastOp>(loc, vectorTy4, c0);
    // Create print operation for the vector value.
    rewriter.create<vector::PrintOp>(loc, broadcastVector);

    rewriter.eraseOp(op);
    return success();
  }
};

class TimingTestEnumAttrLowering
    : public OpRewritePattern<timing::TestEnumAttrOp> {
public:
  using OpRewritePattern<timing::TestEnumAttrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::TestEnumAttrOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get type from the origin operation.
    Type resultType = op.getResult().getType();
    // Get the attribute.
    auto arithAttr = op.getArith();
    // Get the lhs and rhs.
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result;
    // Lowering to different ops according to the attribute.
    if (arithAttr == buddy::timing::TestEnumAttrOperation::ADD)
      // Create addi operation.
      result = rewriter.create<arith::AddIOp>(loc, resultType, lhs, rhs);
    if (arithAttr == buddy::timing::TestEnumAttrOperation::SUB)
      // Create subi operation.
      result = rewriter.create<arith::SubIOp>(loc, resultType, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TimingTestArrayAttrLowering
    : public OpRewritePattern<timing::TestArrayAttrOp> {
public:
  using OpRewritePattern<timing::TestArrayAttrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::TestArrayAttrOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Get the attribute and the value.
    ArrayAttr coordinateAttr = op.getCoordinate();
    int64_t valX = coordinateAttr[0].cast<IntegerAttr>().getInt();
    int64_t valY = coordinateAttr[1].cast<IntegerAttr>().getInt();
    // Get the index attribute and constant value.
    IntegerAttr attrX = rewriter.getIntegerAttr(rewriter.getIndexType(), valX);
    IntegerAttr attrY = rewriter.getIntegerAttr(rewriter.getIndexType(), valY);
    Value idxX = rewriter.create<arith::ConstantOp>(loc, attrX);
    Value idxY = rewriter.create<arith::ConstantOp>(loc, attrY);
    SmallVector<Value, 2> memrefIdx = {idxX, idxY};
    // Get base memref.
    Value memref = op.getBase();
    // Create memref load operation.
    Value result = rewriter.create<memref::LoadOp>(loc, memref, memrefIdx);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TimingStartLowering : public OpRewritePattern<timing::StartOp> {
public:
  using OpRewritePattern<timing::StartOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::StartOp op,
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

class TimingEndLowering : public OpRewritePattern<timing::EndOp> {
public:
  using OpRewritePattern<timing::EndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(timing::EndOp op,
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

void populateLowerTimingConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      TimingTestConstantLowering,
      TimingTestPrintLowering,
      TimingTestEnumAttrLowering,
      TimingTestArrayAttrLowering,
      TimingStartLowering,
      TimingEndLowering>(patterns.getContext());
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerTimingPass
//===----------------------------------------------------------------------===//

namespace {
class LowerTimingPass
    : public PassWrapper<LowerTimingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTimingPass)
  LowerTimingPass() = default;
  LowerTimingPass(const LowerTimingPass &) {}

  StringRef getArgument() const final { return "lower-timing"; }
  StringRef getDescription() const final { return "Lower Timing Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        buddy::timing::TimingDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};

class TimingPass : public PassWrapper<TimingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TimingPass)
  TimingPass() = default;
  TimingPass(const TimingPass &) {}

  Option<std::string> OpNameOption{*this, "timing",
                                   llvm::cl::desc("Timing an op name"),
                                   llvm::cl::init("linalg.generic")};

  StringRef getArgument() const final { return "timing"; }
  StringRef getDescription() const final { return "Timing Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        timing::TimingDialect,
        func::FuncDialect,
        vector::VectorDialect,
        memref::MemRefDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};

} // end anonymous namespace.

void LowerTimingPass::runOnOperation() {
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
  populateLowerTimingConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void TimingPass::runOnOperation() {
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

  // func.func private @printMemrefF32(memref<*xf32>)
  mlir::Type memrefType1 =
      mlir::UnrankedMemRefType::get(builder.getF64Type(), /*memorySpace=*/0);
  auto printFuncType = builder.getFunctionType({memrefType1}, {});
  auto printMemrefFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "printMemrefF64", printFuncType);
  printMemrefFunc.setPrivate();

  // 遍历
  module.walk([&](mlir::func::FuncOp funcOp) {
    // llvm::outs() << "Function name: " << funcOp.getName() << "\n";
    if (funcOp == rtclockFunc || funcOp == printMemrefFunc) {
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

    builder.setInsertionPointToStart(&funcOp.getBody().front());

    // memref.alloc
    auto memrefType = MemRefType::get({opNum}, builder.getF64Type());
    Value memrefAlloc =
        builder.create<mlir::memref::AllocOp>(module.getLoc(), memrefType);

    // 常数
    Value idx =
        builder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 0);
    Value c1 = builder.create<mlir::arith::ConstantIndexOp>(module.getLoc(), 1);

    // 遍历所有对应的Op
    for (auto op : ops) {
      // 在op的前面添加startOp
      builder.setInsertionPoint(op);
      auto startOp = builder.create<timing::StartOp>(op->getLoc());

      // 在op的后面添加endOp，subop，storeOp
      builder.setInsertionPointAfter(op);
      auto endOp = builder.create<timing::EndOp>(op->getLoc());
      auto SubOp =
          builder.create<mlir::arith::SubFOp>(op->getLoc(), endOp, startOp);
      builder.create<mlir::memref::StoreOp>(op->getLoc(), SubOp.getResult(),
                                            memrefAlloc, ValueRange(idx));

      idx = builder.create<mlir::arith::AddIOp>(op->getLoc(), idx, c1);
    }

    // builder.setInsertionPoint(returnOp);
    // cast memref<opNumxf64> -> memref<*xf64>
    Value result = builder.create<memref::CastOp>(ops.back()->getLoc(),
                                                  memrefType1, memrefAlloc);

    // 在代码的最后添加打印时间的op
    builder.create<mlir::func::CallOp>(ops.back()->getLoc(), printMemrefFunc,
                                       ValueRange{result});
  });
}

namespace mlir {
namespace buddy {
void registerLowerTimingPass() { PassRegistration<LowerTimingPass>(); }

void registerTimingPass() { PassRegistration<TimingPass>(); }
} // namespace buddy
} // namespace mlir
