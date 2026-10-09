//====- LowerRVProfPass.cpp - RVProf Dialect Lowering Pass ----------------===//
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
// This file defines RVProf dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "RVProf/RVProfDialect.h"
#include "RVProf/RVProfOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

// Get or create a global string constant
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

// Get LLVM function type for RVProf runtime functions
static LLVM::LLVMFunctionType getRVProfFunctionType(MLIRContext *context) {
  auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
  auto llvmPtr = LLVM::LLVMPointerType::get(context);
  return LLVM::LLVMFunctionType::get(llvmVoidTy, llvmPtr,
                                     /*isVarArg=*/false);
}

// Get or insert __rvprof_region_begin function declaration
static FlatSymbolRefAttr getOrInsertRVProfRegionBegin(PatternRewriter &rewriter,
                                                      ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("__rvprof_region_begin"))
    return SymbolRefAttr::get(context, "__rvprof_region_begin");

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "__rvprof_region_begin",
                                    getRVProfFunctionType(context));
  return SymbolRefAttr::get(context, "__rvprof_region_begin");
}

// Get or insert __rvprof_region_end function declaration
static FlatSymbolRefAttr getOrInsertRVProfRegionEnd(PatternRewriter &rewriter,
                                                    ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("__rvprof_region_end"))
    return SymbolRefAttr::get(context, "__rvprof_region_end");

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "__rvprof_region_end",
                                    getRVProfFunctionType(context));
  return SymbolRefAttr::get(context, "__rvprof_region_end");
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class RVProfRegionBeginLowering
    : public OpRewritePattern<rvprof::RegionBeginOp> {
public:
  using OpRewritePattern<rvprof::RegionBeginOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rvprof::RegionBeginOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *context = rewriter.getContext();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get or insert the runtime function declaration
    auto funcRef = getOrInsertRVProfRegionBegin(rewriter, parentModule);

    // Create global string for the region name
    std::string globalName = "rvprof_region_" + op.getName().str();
    std::string regionName = op.getName().str() + "\0";
    Value namePtr = getOrCreateGlobalString(loc, rewriter, globalName,
                                            regionName, parentModule);

    // Create call to __rvprof_region_begin
    rewriter.create<LLVM::CallOp>(loc, getRVProfFunctionType(context), funcRef,
                                  ValueRange{namePtr});

    rewriter.eraseOp(op);
    return success();
  }
};

class RVProfRegionEndLowering : public OpRewritePattern<rvprof::RegionEndOp> {
public:
  using OpRewritePattern<rvprof::RegionEndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rvprof::RegionEndOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *context = rewriter.getContext();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get or insert the runtime function declaration
    auto funcRef = getOrInsertRVProfRegionEnd(rewriter, parentModule);

    // Create global string for the region name
    std::string globalName = "rvprof_region_" + op.getName().str();
    std::string regionName = op.getName().str() + "\0";
    Value namePtr = getOrCreateGlobalString(loc, rewriter, globalName,
                                            regionName, parentModule);

    // Create call to __rvprof_region_end
    rewriter.create<LLVM::CallOp>(loc, getRVProfFunctionType(context), funcRef,
                                  ValueRange{namePtr});

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateLowerRVProfConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<RVProfRegionBeginLowering, RVProfRegionEndLowering>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// LowerRVProfPass
//===----------------------------------------------------------------------===//

namespace {
class LowerRVProfPass
    : public PassWrapper<LowerRVProfPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRVProfPass)
  LowerRVProfPass() = default;
  LowerRVProfPass(const LowerRVProfPass &) {}

  StringRef getArgument() const final { return "lower-rvprof"; }
  StringRef getDescription() const final { return "Lower RVProf Dialect."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<rvprof::RVProfDialect, func::FuncDialect, LLVM::LLVMDialect>();
  }
};
} // end anonymous namespace.

void LowerRVProfPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  populateLowerRVProfConversionPatterns(patterns);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerRVProfPass() { PassRegistration<LowerRVProfPass>(); }
} // namespace buddy
} // namespace mlir
