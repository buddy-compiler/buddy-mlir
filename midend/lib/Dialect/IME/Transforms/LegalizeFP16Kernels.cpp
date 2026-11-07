//====- LegalizeFP16Kernels.cpp - Lower A100 FP16 microkernels ------===//
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

#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace buddy::ime;
namespace {
static bool contiguousBuffer(Value value, int64_t elements) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.hasStaticShape() && type.getRank() == 1 &&
         type.getDimSize(0) == elements &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides[0] == 1;
}
static Value tilePointer(ConversionPatternRewriter &rewriter, Location loc,
                         Value tile) {
  auto metadata = memref::ExtractStridedMetadataOp::create(rewriter, loc, tile);
  Value aligned =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, tile);
  Value address =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), aligned);
  auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value base = LLVM::IntToPtrOp::create(rewriter, loc, ptr, address);
  Value offset = arith::IndexCastOp::create(
      rewriter, loc, rewriter.getI64Type(), metadata.getOffset());
  return LLVM::GEPOp::create(rewriter, loc, ptr,
                             cast<MemRefType>(tile.getType()).getElementType(),
                             base, ValueRange{offset});
}
struct IMEVfwmadotLowering : public ConvertOpToLLVMPattern<VfwmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VfwmadotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> tiles{op.getVd(), op.getVs1(), op.getVs2()};
    for (Value tile : tiles) {
      auto type = cast<MemRefType>(tile.getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      if (type.getShape() != ArrayRef<int64_t>({8, 8}) ||
          failed(type.getStridesAndOffset(strides, offset)) ||
          strides != SmallVector<int64_t>{8, 1})
        return rewriter.notifyMatchFailure(
            op, "A100 requires contiguous 8x8 tiles");
    }
    Value c = tilePointer(rewriter, op.getLoc(), op.getVd());
    Value a = tilePointer(rewriter, op.getLoc(), op.getVs1());
    Value b = tilePointer(rewriter, op.getLoc(), op.getVs2());
    // Official A100 encoding: funct7=1001111, funct3=100, opcode=0101011.
    // A literal encoding works with the current LLVM assembler, which only
    // knows the incompatible legacy vfmadot mnemonic. All vector-state
    // changes stay in one block, and both accumulator registers are clobbered.
    LLVM::InlineAsmOp::create(
        rewriter, op.getLoc(), Type{}, ValueRange{c, a, b},
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vle32.v v16, ($0)\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"
        "vle16.v v8, ($1)\n\t"
        "vle16.v v12, ($2)\n\t"
        ".word 0x9ec4482b # smt.vfwmadot v16, v8, v12\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v16, ($0)",
        "r,r,r,~{t0},~{v8},~{v12},~{v16},~{v17},~{vl},~{vtype},~{memory}", true,
        false, LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr{},
        ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Include view offsets: model weights are subviews of a single large blob.
static Value getF16Pointer(ConversionPatternRewriter &rewriter, Location loc,
                           Value memref, Value extra) {
  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
  Value aligned =
      memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
  Value offset =
      arith::AddIOp::create(rewriter, loc, metadata.getOffset(), extra);
  Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
  Value bytes = arith::MulIOp::create(rewriter, loc, offset, two);
  Value address = arith::AddIOp::create(rewriter, loc, aligned, bytes);
  Value integer =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), address);
  return LLVM::IntToPtrOp::create(
      rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
      integer);
}

struct IMEPackBF16Lowering : public ConvertOpToLLVMPattern<PackBF16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PackBF16Op op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t k = op.getKAttr().getInt();
    if (k <= 0 || k % 8)
      return op.emitError(
          "A100 panel packing requires positive K divisible by 8");
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    if (!sourceType.hasStaticShape() || sourceType.getDimSize(0) != k ||
        sourceType.getDimSize(1) <= 0 || sourceType.getDimSize(1) % 32 ||
        !contiguousBuffer(op.getPacked(), k * 32))
      return op.emitError(
          "A100 packing requires KxN source and contiguous Kx32 scratch");
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
    auto metadata =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, op.getSource());
    Value offset = arith::MulIOp::create(rewriter, loc, op.getColumn(),
                                         metadata.getStrides()[1]);
    Value rowBytes =
        arith::MulIOp::create(rewriter, loc, metadata.getStrides()[0], two);
    Value colBytes =
        arith::MulIOp::create(rewriter, loc, metadata.getStrides()[1], two);
    rowBytes = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                          rowBytes);
    colBytes = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                          colBytes);
    Value blocks = arith::ConstantIntOp::create(rewriter, loc, k / 8, 64);
    SmallVector<Value> operands{
        getF16Pointer(rewriter, loc, op.getPacked(), zero),
        getF16Pointer(rewriter, loc, op.getSource(), offset), rowBytes,
        colBytes, blocks};
    std::string assembly = "mv t1, $1\n\tmv t2, $0\n\tmv t3, $4\n\t"
                           "slli t0, $2, 3\n\t"
                           "li t4, 2\n\tbne $3, t4, 3f\n\t"
                           "li t4, 32\n\tvsetvli zero, t4, e16, m1, ta, ma\n\t"
                           "1:\n\tmv t5, t1\n\t";
    for (int reg = 8; reg < 16; ++reg)
      assembly +=
          "vle16.v v" + std::to_string(reg) + ", (t5)\n\tadd t5, t5, $2\n\t";
    assembly += "vsseg8e16.v v8, (t2)\n\t"
                "add t1, t1, t0\n\taddi t2, t2, 512\n\t"
                "addi t3, t3, -1\n\tbnez t3, 1b\n\tj 5f\n\t"
                // Non-unit column strides retain the same packed layout.
                "3:\n\tvsetivli zero, 8, e16, m1, ta, ma\n\t"
                "mv t5, t1\n\tmv t6, t2\n\tli t4, 32\n\t"
                "4:\n\tvlse16.v v8, (t5), $2\n\tvse16.v v8, (t6)\n\t"
                "add t5, t5, $3\n\taddi t6, t6, 16\n\t"
                "addi t4, t4, -1\n\tbnez t4, 4b\n\t"
                "add t1, t1, t0\n\taddi t2, t2, 512\n\t"
                "addi t3, t3, -1\n\tbnez t3, 3b\n\t5:";
    LLVM::InlineAsmOp::create(
        rewriter, loc, Type{}, operands, assembly,
        "r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
        "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},"
        "~{vl},~{vtype},~{memory}",
        true, false, LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr{},
        ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEGemmF16Lowering : public ConvertOpToLLVMPattern<GemmF16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(GemmF16Op op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t k = op.getKAttr().getInt(), n = op.getNAttr().getInt();
    if (k <= 0 || k % 8 || n <= 0 || n % 32)
      return op.emitError("A100 GEMM requires K%8=0 and N%32=0");
    auto outputType = cast<MemRefType>(op.getVd().getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (!outputType.hasStaticShape() || outputType.getDimSize(0) <= 0 ||
        outputType.getDimSize(0) % 8 || outputType.getDimSize(1) != n ||
        failed(outputType.getStridesAndOffset(strides, offset)) ||
        strides != SmallVector<int64_t>{n, 1} ||
        !contiguousBuffer(op.getVs1(), outputType.getDimSize(0) * k) ||
        !contiguousBuffer(op.getVs2(), k * 32))
      return op.emitError(
          "A100 GEMM requires row-major C and matching packed A/B buffers");
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    Value aOffset = arith::MulIOp::create(rewriter, loc, op.getRow(), index(k));
    Value cOffset = arith::MulIOp::create(rewriter, loc, op.getRow(), index(n));
    cOffset = arith::AddIOp::create(rewriter, loc, cOffset, op.getColumn());
    SmallVector<Value> operands{
        getF16Pointer(rewriter, loc, op.getVs1(), aOffset),
        getF16Pointer(rewriter, loc, op.getVs2(), index(0)),
        getF16Pointer(rewriter, loc, op.getVd(), cOffset),
        arith::ConstantIntOp::create(rewriter, loc, k / 8, 64),
        arith::ConstantIntOp::create(rewriter, loc, n * 2, 64)};
    std::string assembly =
        "mv t1, $0\n\tmv t2, $1\n\tmv t3, $2\n\tmv t4, $3\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vid.v v4\n\tvsrl.vi v4, v4, 3\n\tvmul.vx v4, v4, $4\n\t"
        "vid.v v6\n\tvand.vi v6, v6, 7\n\tvsll.vi v6, v6, 1\n\t"
        "vadd.vv v4, v4, v6\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t";
    for (int panel = 0; panel < 4; ++panel) {
      assembly += "addi t0, t3, " + std::to_string(panel * 16) +
                  "\n\tvluxei32.v v6, (t0), v4\n\tvfwcvt.f.f.v v" +
                  std::to_string(16 + panel * 2) + ", v6\n\t";
    }
    assembly += "1:\n\tvle16.v v2, (t1)\n\tmv t6, t2\n\t";
    for (int panel = 0; panel < 4; ++panel) {
      assembly += "vle16.v v" + std::to_string(8 + panel * 2) +
                  ", (t6)\n\taddi t6, t6, 128\n\t";
    }
    assembly += ".word 0x9e81482b # smt.vfwmadot v16, v2, v8\n\t"
                ".word 0x9ea1492b # smt.vfwmadot v18, v2, v10\n\t"
                ".word 0x9ec14a2b # smt.vfwmadot v20, v2, v12\n\t"
                ".word 0x9ee14b2b # smt.vfwmadot v22, v2, v14\n\t"
                "addi t1, t1, 128\n\taddi t2, t2, 512\n\t"
                "addi t4, t4, -1\n\tbnez t4, 1b\n\t";
    for (int panel = 0; panel < 4; ++panel) {
      assembly += "addi t0, t3, " + std::to_string(panel * 16) +
                  "\n\tvfncvt.f.f.w v6, v" + std::to_string(16 + panel * 2) +
                  "\n\tvsuxei32.v v6, (t0), v4\n\t";
    }
    LLVM::InlineAsmOp::create(
        rewriter, loc, Type{}, operands, assembly,
        "r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t6},"
        "~{v2},~{v4},~{v5},~{v6},~{v7},~{v8},~{v10},~{v12},~{v14},"
        "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
        "~{vl},~{vtype},~{memory}",
        true, false, LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr{},
        ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEGemvF16Lowering : public ConvertOpToLLVMPattern<GemvF16Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemvF16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t k = op.getKAttr().getInt();
    int64_t distance = op.getPrefetchAttr().getInt();
    bool unroll = op.getUnroll2();
    if (k <= 0 || k % (unroll ? 16 : 8) || distance < 0 || distance > 2016 ||
        distance % 32)
      return op.emitError(
          "invalid K/unroll or prefetch distance for A100 GEMV");
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    // Constant matrices are views into a large weights blob: preserve the
    // descriptor offset rather than using just its aligned base pointer.
    auto pointer = [&](Value memref, Value extra) -> Value {
      auto metadata =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, memref);
      Value aligned =
          memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, memref);
      Value offset =
          arith::AddIOp::create(rewriter, loc, metadata.getOffset(), extra);
      Value bytes = arith::MulIOp::create(rewriter, loc, offset, index(2));
      Value address = arith::AddIOp::create(rewriter, loc, aligned, bytes);
      Value integer = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), address);
      return LLVM::IntToPtrOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          integer);
    };
    Value a = pointer(op.getVs1(), index(0));
    Value base = arith::MulIOp::create(rewriter, loc, op.getColumn(), index(k));
    SmallVector<Value> operands{a};
    for (int64_t panel = 0; panel < 4; ++panel) {
      Value offset =
          arith::AddIOp::create(rewriter, loc, base, index(panel * 64));
      operands.push_back(pointer(op.getVs2(), offset));
    }
    operands.push_back(pointer(op.getVd(), op.getColumn()));
    operands.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(k / (unroll ? 16 : 8))));
    std::string assembly =
        "mv t1, $0\n\tmv t2, $1\n\tmv t3, $2\n\tmv t4, $3\n\tmv t5, $4\n\tmv "
        "t6, $6\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\tvmv.v.i v18, 0\n\tvmv.v.i v20, 0\n\tvmv.v.i v22, "
        "0\n\t"
        "vsetivli zero, 8, e16, m1, tu, mu\n\t"
        "vle16.v v6, ($5)\n\tvfwcvt.f.f.v v16, v6\n\t"
        "addi t0, $5, 16\n\tvle16.v v6, (t0)\n\tvfwcvt.f.f.v v18, v6\n\t"
        "addi t0, $5, 32\n\tvle16.v v6, (t0)\n\tvfwcvt.f.f.v v20, v6\n\t"
        "addi t0, $5, 48\n\tvle16.v v6, (t0)\n\tvfwcvt.f.f.v v22, v6\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vmv.v.i v2, 0\n\tvid.v v4\n\tvmsltu.vi v0, v4, 8\n\t1:\n\t";
    if (distance)
      assembly += "prefetch.r " + std::to_string(distance) +
                  "(t2)\n\t"
                  "prefetch.r " +
                  std::to_string(distance) + "(t4)\n\t";
    std::string body =
        "vle16.v v2, (t1), v0.t\n\t"
        "vle16.v v8, (t2)\n\tvle16.v v10, (t3)\n\t"
        "vle16.v v12, (t4)\n\tvle16.v v14, (t5)\n\t"
        ".word 0x9e81482b # smt.vfwmadot v16, v2, v8\n\t"
        ".word 0x9ea1492b # smt.vfwmadot v18, v2, v10\n\t"
        ".word 0x9ec14a2b # smt.vfwmadot v20, v2, v12\n\t"
        ".word 0x9ee14b2b # smt.vfwmadot v22, v2, v14\n\t"
        "addi t1, t1, 16\n\taddi t2, t2, 512\n\taddi t3, t3, 512\n\t"
        "addi t4, t4, 512\n\taddi t5, t5, 512\n\t";
    assembly += body;
    if (unroll)
      assembly += body;
    assembly +=
        "addi t6, t6, -1\n\tbnez t6, 1b\n\t"
        "vsetivli zero, 8, e16, m1, tu, mu\n\t"
        "vfncvt.f.f.w v6, v16\n\tvse16.v v6, ($5)\n\t"
        "addi t0, $5, 16\n\tvfncvt.f.f.w v6, v18\n\tvse16.v v6, (t0)\n\t"
        "addi t0, $5, 32\n\tvfncvt.f.f.w v6, v20\n\tvse16.v v6, (t0)\n\t"
        "addi t0, $5, 48\n\tvfncvt.f.f.w v6, v22\n\tvse16.v v6, (t0)";
    LLVM::InlineAsmOp::create(
        rewriter, loc, Type{}, operands, assembly,
        "r,r,r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
        "~{v0},~{v2},~{v4},~{v6},~{v8},~{v10},~{v12},~{v14},"
        "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
        "~{vl},~{vtype},~{memory}",
        true, false, LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr{},
        ArrayAttr{});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace
void mlir::populateIMEFP16KernelPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<IMEVfwmadotLowering, IMEPackBF16Lowering, IMEGemmF16Lowering,
               IMEGemvF16Lowering>(converter);
}
