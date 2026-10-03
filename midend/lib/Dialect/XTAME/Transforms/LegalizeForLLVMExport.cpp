//====- LegalizeForLLVMExport.cpp - Prepare XTAME for LLVM translation ----===//
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

#include "Dialect/XTAME/Transform.h"
#include "Dialect/XTAME/XTAMEDialect.h"
#include "Dialect/XTAME/XTAMEOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace buddy::xtame;

namespace {

// Preserve the fixed-register contract of the high-level XTAME operations.
// LLVM's native XTAME intrinsics now take SSA matrix values; passing register
// indices to those intrinsics would produce invalid IR. Use side-effecting asm
// for the legacy interface and declare the matrix state and memory clobbers.
static FailureOr<LLVM::InlineAsmOp>
createXTAMEAsm(ConversionPatternRewriter &rewriter, Operation *op,
               StringRef mnemonic, const Twine &arguments,
               ValueRange operands = {}, TypeRange results = {}) {
  for (StringRef name : {"md", "ms1", "ms2", "ms3"}) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(name)) {
      if (attr.getInt() < 0 || attr.getInt() > 7) {
        op->emitOpError("matrix register index must be in [0, 7]");
        return failure();
      }
    }
  }

  std::string constraints;
  if (!results.empty())
    constraints = "=r,";
  for (size_t i = 0; i < operands.size(); ++i)
    constraints += "r,";
  constraints += "~{memory},~{m0},~{m1},~{m2},~{m3},~{m4},~{m5},~{m6},~{m7}";
  return LLVM::InlineAsmOp::create(
      rewriter, op->getLoc(), results, operands,
      (mnemonic + " " + arguments).str(), constraints,
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::tailcallkind::TailCallKind::None, /*convergent=*/false,
      LLVM::AsmDialectAttr{}, ArrayAttr{});
}

static std::string matrixReg(uint64_t index) {
  return (Twine("m") + Twine(index)).str();
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64Type = IntegerType::get(ctx, 64);
  Value idx =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value i64Val = arith::IndexCastOp::create(rewriter, loc, i64Type, idx);
  return LLVM::IntToPtrOp::create(rewriter, loc, ptrType, i64Val);
}

template <typename OpTy>
struct XTAMEAsmLowering : public ConvertOpToLLVMPattern<OpTy> {
  StringRef mnemonic;
  XTAMEAsmLowering(LLVMTypeConverter &converter, StringRef mnemonic)
      : ConvertOpToLLVMPattern<OpTy>(converter), mnemonic(mnemonic) {}

  LogicalResult emit(OpTy op, ConversionPatternRewriter &rewriter,
                     const Twine &arguments, ValueRange operands = {}) const {
    auto result = createXTAMEAsm(rewriter, op, mnemonic, arguments, operands,
                                 op->getResultTypes());
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

template <typename OpTy>
struct XTAMEConfigLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, "$0", adaptor.getOperands());
  }
};

template <typename OpTy, uint64_t (OpTy::*AttrGetter)()>
struct XTAMEConfigImmLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, Twine((op.*AttrGetter)()));
  }
};

template <typename OpTy>
struct XTAMEZeroLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, matrixReg(op.getMd()));
  }
};

template <typename OpTy>
struct XTAMEDualAttrLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter,
                      matrixReg(op.getMd()) + ", " + matrixReg(op.getMs1()));
  }
};

template <typename OpTy>
struct XTAMEDupLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, matrixReg(op.getMd()) + ", $0",
                      adaptor.getOperands());
  }
};

template <typename OpTy>
struct XTAMEMmovMXLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, matrixReg(op.getMd()) + ", $0, $1",
                      adaptor.getOperands());
  }
};

template <typename OpTy>
struct XTAMEMmovXMLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter, "$0, " + matrixReg(op.getMs2()) + ", $1",
                      adaptor.getOperands());
  }
};

template <typename OpTy>
struct XTAMECmovMvILowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getUimm3() > 7)
      return op.emitOpError("broadcast index must be in [0, 7]");

    // The current LLVM XTAME instruction printer emits ms1[index], but its
    // assembler cannot parse that operand. All broadcast operands are constant,
    // so emit the encoding from RVInstXTAMEDB via .insn until the parser supports
    // this syntax. Matrix and memory clobbers still model the legacy state.
    uint32_t encoding = 0x0400002b; // th.mmov.mv.i: func4=0, uop=2.
    if (this->mnemonic.starts_with("th.mcmov")) {
      unsigned size = llvm::StringSwitch<unsigned>(this->mnemonic)
                          .Case("th.mcmovb.mv.i", 0)
                          .Case("th.mcmovh.mv.i", 1)
                          .Case("th.mcmovw.mv.i", 2)
                          .Case("th.mcmovd.mv.i", 3);
      encoding = 0x5c00002b | (size << 10); // func4=5, uop=6.
    }
    encoding |= op.getMs1() << 18 | op.getMd() << 15 | op.getUimm3() << 7;
    auto result = createXTAMEAsm(rewriter, op, ".insn", "4, " + Twine(encoding));
    if (failed(result))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
struct XTAMELoadLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = extractPointerFromMemref(rewriter, op.getLoc(), op.getBase());
    return this->emit(op, rewriter, matrixReg(op.getMd()) + ", $0, $1",
                      {adaptor.getStride(), base});
  }
};

template <typename OpTy>
struct XTAMEPrefetchLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = extractPointerFromMemref(rewriter, op.getLoc(), op.getBase());
    return this->emit(op, rewriter, "$0, $1", {adaptor.getStride(), base});
  }
};

template <typename OpTy>
struct XTAMEStoreLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = extractPointerFromMemref(rewriter, op.getLoc(), op.getBase());
    return this->emit(op, rewriter, matrixReg(op.getMs3()) + ", $0, $1",
                      {adaptor.getStride(), base});
  }
};

template <typename OpTy>
struct XTAMETernaryOpLowering : public XTAMEAsmLowering<OpTy> {
  using XTAMEAsmLowering<OpTy>::XTAMEAsmLowering;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return this->emit(op, rewriter,
                      matrixReg(op.getMd()) + ", " + matrixReg(op.getMs2()) +
                          ", " + matrixReg(op.getMs1()));
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LegalizeXTAMEForLLVMExport
    : public PassWrapper<LegalizeXTAMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeXTAMEForLLVMExport)

  StringRef getArgument() const final { return "lower-xt-ame"; }
  StringRef getDescription() const final {
    return "XTAME dialect lowering pass.";
  }

  LegalizeXTAMEForLLVMExport() = default;
  LegalizeXTAMEForLLVMExport(const LegalizeXTAMEForLLVMExport &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<XTAMEDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    LLVMConversionTarget target(context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    target.addIllegalDialect<buddy::xtame::XTAMEDialect>();

    LLVMTypeConverter typeConverter(&context);
    RewritePatternSet patterns(&context);

    // Configuration patterns
    patterns.add<XTAMEConfigLowering<ThMcfgOp>>(typeConverter,
                                                "th.mcfg");
    patterns.add<XTAMEConfigLowering<ThMcfgmOp>>(typeConverter,
                                                 "th.mcfgm");
    patterns.add<XTAMEConfigLowering<ThMcfgnOp>>(typeConverter,
                                                 "th.mcfgn");
    patterns.add<XTAMEConfigLowering<ThMcfgkOp>>(typeConverter,
                                                 "th.mcfgk");
    patterns.add<XTAMEConfigImmLowering<ThMcfgmiOp, &ThMcfgmiOp::getTilem>>(
        typeConverter, "th.mcfgmi");
    patterns.add<XTAMEConfigImmLowering<ThMcfgniOp, &ThMcfgniOp::getTilen>>(
        typeConverter, "th.mcfgni");
    patterns.add<XTAMEConfigImmLowering<ThMcfgkiOp, &ThMcfgkiOp::getTilek>>(
        typeConverter, "th.mcfgki");

    // MISC patterns
    patterns.add<XTAMEZeroLowering<ThMzeroOp>>(typeConverter,
                                               "th.mzero");
    patterns.add<XTAMEZeroLowering<ThMzero2rOp>>(typeConverter,
                                                 "th.mzero2r");
    patterns.add<XTAMEZeroLowering<ThMzero4rOp>>(typeConverter,
                                                 "th.mzero4r");
    patterns.add<XTAMEZeroLowering<ThMzero8rOp>>(typeConverter,
                                                 "th.mzero8r");
    patterns.add<XTAMEDualAttrLowering<ThMmovMmOp>>(typeConverter,
                                                    "th.mmov.mm");
    patterns.add<XTAMEDupLowering<ThMdupbMXOp>>(typeConverter,
                                                "th.mdupb.m.x");
    patterns.add<XTAMEDupLowering<ThMduphMXOp>>(typeConverter,
                                                "th.mduph.m.x");
    patterns.add<XTAMEDupLowering<ThMdupwMXOp>>(typeConverter,
                                                "th.mdupw.m.x");
    patterns.add<XTAMEDupLowering<ThMdupdMXOp>>(typeConverter,
                                                "th.mdupd.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovbMXOp>>(typeConverter,
                                                   "th.mmovb.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovhMXOp>>(typeConverter,
                                                   "th.mmovh.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovwMXOp>>(typeConverter,
                                                   "th.mmovw.m.x");
    patterns.add<XTAMEMmovMXLowering<ThMmovdMXOp>>(typeConverter,
                                                   "th.mmovd.m.x");
    patterns.add<XTAMEMmovXMLowering<ThMmovbXMOp>>(typeConverter,
                                                   "th.mmovb.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovhXMOp>>(typeConverter,
                                                   "th.mmovh.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovwXMOp>>(typeConverter,
                                                   "th.mmovw.x.m");
    patterns.add<XTAMEMmovXMLowering<ThMmovdXMOp>>(typeConverter,
                                                   "th.mmovd.x.m");
    patterns.add<XTAMECmovMvILowering<ThMmovMvIOp>>(typeConverter,
                                                    "th.mmov.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovbMvIOp>>(
        typeConverter, "th.mcmovb.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovhMvIOp>>(
        typeConverter, "th.mcmovh.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovwMvIOp>>(
        typeConverter, "th.mcmovw.mv.i");
    patterns.add<XTAMECmovMvILowering<ThMcmovdMvIOp>>(
        typeConverter, "th.mcmovd.mv.i");
    patterns.add<XTAMETernaryOpLowering<ThMpackMmOp>>(typeConverter,
                                                      "th.mpack.mm");
    patterns.add<XTAMETernaryOpLowering<ThMpackhlMmOp>>(
        typeConverter, "th.mpackhl.mm");
    patterns.add<XTAMETernaryOpLowering<ThMpackhhMmOp>>(
        typeConverter, "th.mpackhh.mm");

    // Load/Store patterns
    patterns.add<XTAMELoadLowering<ThMlde8Op>>(typeConverter,
                                               "th.mlde8");
    patterns.add<XTAMELoadLowering<ThMlde16Op>>(typeConverter,
                                                "th.mlde16");
    patterns.add<XTAMELoadLowering<ThMlde32Op>>(typeConverter,
                                                "th.mlde32");
    patterns.add<XTAMELoadLowering<ThMlde64Op>>(typeConverter,
                                                "th.mlde64");
    patterns.add<XTAMELoadLowering<ThMldte8Op>>(typeConverter,
                                                "th.mldte8");
    patterns.add<XTAMELoadLowering<ThMldte16Op>>(typeConverter,
                                                 "th.mldte16");
    patterns.add<XTAMELoadLowering<ThMldte32Op>>(typeConverter,
                                                 "th.mldte32");
    patterns.add<XTAMELoadLowering<ThMldte64Op>>(typeConverter,
                                                 "th.mldte64");
    patterns.add<XTAMELoadLowering<ThMslde8Op>>(typeConverter,
                                                "th.mslde8");
    patterns.add<XTAMELoadLowering<ThMslde16Op>>(typeConverter,
                                                 "th.mslde16");
    patterns.add<XTAMELoadLowering<ThMslde32Op>>(typeConverter,
                                                 "th.mslde32");
    patterns.add<XTAMELoadLowering<ThMslde64Op>>(typeConverter,
                                                 "th.mslde64");
    patterns.add<XTAMELoadLowering<ThMsldte8Op>>(typeConverter,
                                                 "th.msldte8");
    patterns.add<XTAMELoadLowering<ThMsldte16Op>>(typeConverter,
                                                  "th.msldte16");
    patterns.add<XTAMELoadLowering<ThMsldte32Op>>(typeConverter,
                                                  "th.msldte32");
    patterns.add<XTAMELoadLowering<ThMsldte64Op>>(typeConverter,
                                                  "th.msldte64");

    patterns.add<XTAMEPrefetchLowering<ThMplde8Op>>(typeConverter,
                                                    "th.mplde8");
    patterns.add<XTAMEPrefetchLowering<ThMplde16Op>>(typeConverter,
                                                     "th.mplde16");
    patterns.add<XTAMEPrefetchLowering<ThMplde32Op>>(typeConverter,
                                                     "th.mplde32");
    patterns.add<XTAMEPrefetchLowering<ThMplde64Op>>(typeConverter,
                                                     "th.mplde64");
    patterns.add<XTAMEPrefetchLowering<ThMpldte8Op>>(typeConverter,
                                                     "th.mpldte8");
    patterns.add<XTAMEPrefetchLowering<ThMpldte16Op>>(typeConverter,
                                                      "th.mpldte16");
    patterns.add<XTAMEPrefetchLowering<ThMpldte32Op>>(typeConverter,
                                                      "th.mpldte32");
    patterns.add<XTAMEPrefetchLowering<ThMpldte64Op>>(typeConverter,
                                                      "th.mpldte64");

    patterns.add<XTAMEStoreLowering<ThMste8Op>>(typeConverter,
                                                "th.mste8");
    patterns.add<XTAMEStoreLowering<ThMste16Op>>(typeConverter,
                                                 "th.mste16");
    patterns.add<XTAMEStoreLowering<ThMste32Op>>(typeConverter,
                                                 "th.mste32");
    patterns.add<XTAMEStoreLowering<ThMste64Op>>(typeConverter,
                                                 "th.mste64");
    patterns.add<XTAMEStoreLowering<ThMstte8Op>>(typeConverter,
                                                 "th.mstte8");
    patterns.add<XTAMEStoreLowering<ThMstte16Op>>(typeConverter,
                                                  "th.mstte16");
    patterns.add<XTAMEStoreLowering<ThMstte32Op>>(typeConverter,
                                                  "th.mstte32");
    patterns.add<XTAMEStoreLowering<ThMstte64Op>>(typeConverter,
                                                  "th.mstte64");
    patterns.add<XTAMEStoreLowering<ThMsste8Op>>(typeConverter,
                                                 "th.msste8");
    patterns.add<XTAMEStoreLowering<ThMsste16Op>>(typeConverter,
                                                  "th.msste16");
    patterns.add<XTAMEStoreLowering<ThMsste32Op>>(typeConverter,
                                                  "th.msste32");
    patterns.add<XTAMEStoreLowering<ThMsste64Op>>(typeConverter,
                                                  "th.msste64");
    patterns.add<XTAMEStoreLowering<ThMsstte8Op>>(typeConverter,
                                                  "th.msstte8");
    patterns.add<XTAMEStoreLowering<ThMsstte16Op>>(typeConverter,
                                                   "th.msstte16");
    patterns.add<XTAMEStoreLowering<ThMsstte32Op>>(typeConverter,
                                                   "th.msstte32");
    patterns.add<XTAMEStoreLowering<ThMsstte64Op>>(typeConverter,
                                                   "th.msstte64");

    // Tile register matrix multiply patterns
    patterns.add<XTAMETernaryOpLowering<ThMmaccWBOp>>(
        typeConverter, "th.mmacc.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccuWBOp>>(
        typeConverter, "th.mmaccu.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccusWBOp>>(
        typeConverter, "th.mmaccus.w.b");
    patterns.add<XTAMETernaryOpLowering<ThMmaccsuWBOp>>(
        typeConverter, "th.mmaccsu.w.b");

    patterns.add<XTAMETernaryOpLowering<ThMfmaccHOp>>(typeConverter,
                                                      "th.mfmacc.h");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16Op>>(
        typeConverter, "th.mfmacc.bf16");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSOp>>(typeConverter,
                                                      "th.mfmacc.s");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccDOp>>(typeConverter,
                                                      "th.mfmacc.d");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccHE4m3Op>>(
        typeConverter, "th.mfmacc.h.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccHE5m2Op>>(
        typeConverter, "th.mfmacc.h.e5m2");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E4m3Op>>(
        typeConverter, "th.mfmacc.bf16.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E5m2Op>>(
        typeConverter, "th.mfmacc.bf16.e5m2");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSHOp>>(
        typeConverter, "th.mfmacc.s.h");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSBf16Op>>(
        typeConverter, "th.mfmacc.s.bf16");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccDSOp>>(
        typeConverter, "th.mfmacc.d.s");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSE4m3Op>>(
        typeConverter, "th.mfmacc.s.e4m3");
    patterns.add<XTAMETernaryOpLowering<ThMfmaccSE5m2Op>>(
        typeConverter, "th.mfmacc.s.e5m2");

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::populateXTAMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Configuration patterns
  patterns.add<XTAMEConfigLowering<ThMcfgOp>>(converter, "th.mcfg");
  patterns.add<XTAMEConfigLowering<ThMcfgmOp>>(converter,
                                               "th.mcfgm");
  patterns.add<XTAMEConfigLowering<ThMcfgnOp>>(converter,
                                               "th.mcfgn");
  patterns.add<XTAMEConfigLowering<ThMcfgkOp>>(converter,
                                               "th.mcfgk");
  patterns.add<XTAMEConfigImmLowering<ThMcfgmiOp, &ThMcfgmiOp::getTilem>>(
      converter, "th.mcfgmi");
  patterns.add<XTAMEConfigImmLowering<ThMcfgniOp, &ThMcfgniOp::getTilen>>(
      converter, "th.mcfgni");
  patterns.add<XTAMEConfigImmLowering<ThMcfgkiOp, &ThMcfgkiOp::getTilek>>(
      converter, "th.mcfgki");

  // MISC patterns
  patterns.add<XTAMEZeroLowering<ThMzeroOp>>(converter, "th.mzero");
  patterns.add<XTAMEZeroLowering<ThMzero2rOp>>(converter,
                                               "th.mzero2r");
  patterns.add<XTAMEZeroLowering<ThMzero4rOp>>(converter,
                                               "th.mzero4r");
  patterns.add<XTAMEZeroLowering<ThMzero8rOp>>(converter,
                                               "th.mzero8r");
  patterns.add<XTAMEDualAttrLowering<ThMmovMmOp>>(converter,
                                                  "th.mmov.mm");
  patterns.add<XTAMEDupLowering<ThMdupbMXOp>>(converter,
                                              "th.mdupb.m.x");
  patterns.add<XTAMEDupLowering<ThMduphMXOp>>(converter,
                                              "th.mduph.m.x");
  patterns.add<XTAMEDupLowering<ThMdupwMXOp>>(converter,
                                              "th.mdupw.m.x");
  patterns.add<XTAMEDupLowering<ThMdupdMXOp>>(converter,
                                              "th.mdupd.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovbMXOp>>(converter,
                                                 "th.mmovb.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovhMXOp>>(converter,
                                                 "th.mmovh.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovwMXOp>>(converter,
                                                 "th.mmovw.m.x");
  patterns.add<XTAMEMmovMXLowering<ThMmovdMXOp>>(converter,
                                                 "th.mmovd.m.x");
  patterns.add<XTAMEMmovXMLowering<ThMmovbXMOp>>(converter,
                                                 "th.mmovb.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovhXMOp>>(converter,
                                                 "th.mmovh.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovwXMOp>>(converter,
                                                 "th.mmovw.x.m");
  patterns.add<XTAMEMmovXMLowering<ThMmovdXMOp>>(converter,
                                                 "th.mmovd.x.m");
  patterns.add<XTAMECmovMvILowering<ThMmovMvIOp>>(converter,
                                                  "th.mmov.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovbMvIOp>>(
      converter, "th.mcmovb.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovhMvIOp>>(
      converter, "th.mcmovh.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovwMvIOp>>(
      converter, "th.mcmovw.mv.i");
  patterns.add<XTAMECmovMvILowering<ThMcmovdMvIOp>>(
      converter, "th.mcmovd.mv.i");
  patterns.add<XTAMETernaryOpLowering<ThMpackMmOp>>(converter,
                                                    "th.mpack.mm");
  patterns.add<XTAMETernaryOpLowering<ThMpackhlMmOp>>(
      converter, "th.mpackhl.mm");
  patterns.add<XTAMETernaryOpLowering<ThMpackhhMmOp>>(
      converter, "th.mpackhh.mm");

  // Load/Store patterns
  patterns.add<XTAMELoadLowering<ThMlde8Op>>(converter, "th.mlde8");
  patterns.add<XTAMELoadLowering<ThMlde16Op>>(converter,
                                              "th.mlde16");
  patterns.add<XTAMELoadLowering<ThMlde32Op>>(converter,
                                              "th.mlde32");
  patterns.add<XTAMELoadLowering<ThMlde64Op>>(converter,
                                              "th.mlde64");
  patterns.add<XTAMELoadLowering<ThMldte8Op>>(converter,
                                              "th.mldte8");
  patterns.add<XTAMELoadLowering<ThMldte16Op>>(converter,
                                               "th.mldte16");
  patterns.add<XTAMELoadLowering<ThMldte32Op>>(converter,
                                               "th.mldte32");
  patterns.add<XTAMELoadLowering<ThMldte64Op>>(converter,
                                               "th.mldte64");
  patterns.add<XTAMELoadLowering<ThMslde8Op>>(converter,
                                              "th.mslde8");
  patterns.add<XTAMELoadLowering<ThMslde16Op>>(converter,
                                               "th.mslde16");
  patterns.add<XTAMELoadLowering<ThMslde32Op>>(converter,
                                               "th.mslde32");
  patterns.add<XTAMELoadLowering<ThMslde64Op>>(converter,
                                               "th.mslde64");
  patterns.add<XTAMELoadLowering<ThMsldte8Op>>(converter,
                                               "th.msldte8");
  patterns.add<XTAMELoadLowering<ThMsldte16Op>>(converter,
                                                "th.msldte16");
  patterns.add<XTAMELoadLowering<ThMsldte32Op>>(converter,
                                                "th.msldte32");
  patterns.add<XTAMELoadLowering<ThMsldte64Op>>(converter,
                                                "th.msldte64");

  patterns.add<XTAMEPrefetchLowering<ThMplde8Op>>(converter,
                                                  "th.mplde8");
  patterns.add<XTAMEPrefetchLowering<ThMplde16Op>>(converter,
                                                   "th.mplde16");
  patterns.add<XTAMEPrefetchLowering<ThMplde32Op>>(converter,
                                                   "th.mplde32");
  patterns.add<XTAMEPrefetchLowering<ThMplde64Op>>(converter,
                                                   "th.mplde64");
  patterns.add<XTAMEPrefetchLowering<ThMpldte8Op>>(converter,
                                                   "th.mpldte8");
  patterns.add<XTAMEPrefetchLowering<ThMpldte16Op>>(converter,
                                                    "th.mpldte16");
  patterns.add<XTAMEPrefetchLowering<ThMpldte32Op>>(converter,
                                                    "th.mpldte32");
  patterns.add<XTAMEPrefetchLowering<ThMpldte64Op>>(converter,
                                                    "th.mpldte64");

  patterns.add<XTAMEStoreLowering<ThMste8Op>>(converter, "th.mste8");
  patterns.add<XTAMEStoreLowering<ThMste16Op>>(converter,
                                               "th.mste16");
  patterns.add<XTAMEStoreLowering<ThMste32Op>>(converter,
                                               "th.mste32");
  patterns.add<XTAMEStoreLowering<ThMste64Op>>(converter,
                                               "th.mste64");
  patterns.add<XTAMEStoreLowering<ThMstte8Op>>(converter,
                                               "th.mstte8");
  patterns.add<XTAMEStoreLowering<ThMstte16Op>>(converter,
                                                "th.mstte16");
  patterns.add<XTAMEStoreLowering<ThMstte32Op>>(converter,
                                                "th.mstte32");
  patterns.add<XTAMEStoreLowering<ThMstte64Op>>(converter,
                                                "th.mstte64");
  patterns.add<XTAMEStoreLowering<ThMsste8Op>>(converter,
                                               "th.msste8");
  patterns.add<XTAMEStoreLowering<ThMsste16Op>>(converter,
                                                "th.msste16");
  patterns.add<XTAMEStoreLowering<ThMsste32Op>>(converter,
                                                "th.msste32");
  patterns.add<XTAMEStoreLowering<ThMsste64Op>>(converter,
                                                "th.msste64");
  patterns.add<XTAMEStoreLowering<ThMsstte8Op>>(converter,
                                                "th.msstte8");
  patterns.add<XTAMEStoreLowering<ThMsstte16Op>>(converter,
                                                 "th.msstte16");
  patterns.add<XTAMEStoreLowering<ThMsstte32Op>>(converter,
                                                 "th.msstte32");
  patterns.add<XTAMEStoreLowering<ThMsstte64Op>>(converter,
                                                 "th.msstte64");

  patterns.add<XTAMETernaryOpLowering<ThMmaccWBOp>>(converter,
                                                    "th.mmacc.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccuWBOp>>(
      converter, "th.mmaccu.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccusWBOp>>(
      converter, "th.mmaccus.w.b");
  patterns.add<XTAMETernaryOpLowering<ThMmaccsuWBOp>>(
      converter, "th.mmaccsu.w.b");

  // Tile register matrix multiply patterns (float-point types)
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHOp>>(converter,
                                                    "th.mfmacc.h");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16Op>>(
      converter, "th.mfmacc.bf16");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSOp>>(converter,
                                                    "th.mfmacc.s");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccDOp>>(converter,
                                                    "th.mfmacc.d");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHE4m3Op>>(
      converter, "th.mfmacc.h.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccHE5m2Op>>(
      converter, "th.mfmacc.h.e5m2");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E4m3Op>>(
      converter, "th.mfmacc.bf16.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccBf16E5m2Op>>(
      converter, "th.mfmacc.bf16.e5m2");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSHOp>>(
      converter, "th.mfmacc.s.h");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSBf16Op>>(
      converter, "th.mfmacc.s.bf16");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccDSOp>>(
      converter, "th.mfmacc.d.s");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSE4m3Op>>(
      converter, "th.mfmacc.s.e4m3");
  patterns.add<XTAMETernaryOpLowering<ThMfmaccSE5m2Op>>(
      converter, "th.mfmacc.s.e5m2");
}

void mlir::configureXTAMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  target.addIllegalDialect<buddy::xtame::XTAMEDialect>();
}

std::unique_ptr<Pass> buddy::xtame::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeXTAMEForLLVMExport>();
}

namespace mlir {
namespace buddy {
void registerLowerXTAMEPass() {
  PassRegistration<LegalizeXTAMEForLLVMExport>();
}
} // namespace buddy
} // namespace mlir
