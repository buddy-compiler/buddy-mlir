//====- LegalizeForLLVMExport.cpp - Prepare IME for LLVM translation ------===//
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace buddy::ime;

namespace {
// Include the descriptor offset; an aligned pointer alone is not the start of
// a subview. Contiguous native and scratch tiles use whole-register
// loads/stores.
Value tilePointer(ConversionPatternRewriter &rewriter, Location loc,
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
  Type element = cast<MemRefType>(tile.getType()).getElementType();
  return LLVM::GEPOp::create(rewriter, loc, ptr, element, base,
                             ValueRange{offset});
}

template <typename OpTy, typename IntrOpTy>
struct IMEK3DotLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto aType = cast<MemRefType>(op.getVs1().getType());
    auto bType = cast<MemRefType>(op.getVs2().getType());
    auto cType = cast<MemRefType>(op.getVd().getType());
    Type inputElement = aType.getElementType();
    bool fp = inputElement.isF16();
    int64_t tileK = fp ? 8 : 16;
    if ((!fp && !inputElement.isInteger(8)) ||
        (fp ? !bType.getElementType().isF16()
            : !bType.getElementType().isInteger(8)) ||
        !aType.hasStaticShape() || !bType.hasStaticShape() ||
        !cType.hasStaticShape() ||
        (fp ? !cType.getElementType().isF16()
            : !cType.getElementType().isInteger(32)))
      return op.emitError(
          "K3 IME requires static packed int8/int32 or fp16 tiles");

    int64_t m = cType.getDimSize(0), n = cType.getDimSize(1);
    int64_t k = aType.getDimSize(1);
    if (m <= 0 || n <= 0 || k <= 0 || m > 8 || n > 8 || k > tileK ||
        aType.getDimSize(0) != m || bType.getDimSize(0) != n ||
        bType.getDimSize(1) != k)
      return op.emitError("K3 IME expects C[M,N], A[M,K], packed B[N,K], "
                          "with M,N <= 8 and K <= ")
             << tileK;

    Type accumulatorElement =
        fp ? Type(rewriter.getF32Type()) : Type(rewriter.getI32Type());
    auto inputVector = VectorType::get(
        {fp ? 4 : 8},
        fp ? Type(rewriter.getF16Type()) : Type(rewriter.getI8Type()), true);
    auto outputVector = VectorType::get({4}, accumulatorElement, true);
    auto contiguous = [](MemRefType type) {
      SmallVector<int64_t> strides;
      int64_t offset;
      return succeeded(type.getStridesAndOffset(strides, offset)) &&
             strides[1] == 1 && strides[0] == type.getDimSize(1);
    };
    // Native complete integer tiles need no packing or scratch copies. The
    // descriptor offset is still honored, including for contiguous subviews.
    if (!fp && m == 8 && n == 8 && k == 16 &&
        inputElement.isSignlessInteger() &&
        bType.getElementType().isSignlessInteger() && contiguous(aType) &&
        contiguous(bType) && contiguous(cType)) {
      Value a =
          LLVM::LoadOp::create(rewriter, loc, inputVector,
                               tilePointer(rewriter, loc, op.getVs1()), 1);
      Value b =
          LLVM::LoadOp::create(rewriter, loc, inputVector,
                               tilePointer(rewriter, loc, op.getVs2()), 1);
      Value cPtr = tilePointer(rewriter, loc, op.getVd());
      Value c = LLVM::LoadOp::create(rewriter, loc, outputVector, cPtr, 1);
      Value result = IntrOpTy::create(rewriter, loc, outputVector, c, a, b);
      LLVM::StoreOp::create(rewriter, loc, result, cPtr, 1);
      rewriter.eraseOp(op);
      return success();
    }

    // Bound scratch storage to this invocation, including when the dot is
    // nested in a loop or a parallel region.
    OpBuilder::InsertionGuard guard(rewriter);
    auto scope = memref::AllocaScopeOp::create(rewriter, loc, TypeRange{});
    rewriter.createBlock(&scope.getBodyRegion());

    auto index = [&](int64_t i) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, i);
    };
    auto makeTile = [&](ArrayRef<int64_t> shape, Type element) -> Value {
      auto type = MemRefType::get(shape, element);
      return memref::AllocaOp::create(rewriter, loc, type);
    };
    Type packedInputElement =
        fp ? Type(rewriter.getF16Type()) : Type(rewriter.getI8Type());
    Value aTile = makeTile({8, tileK}, packedInputElement);
    Value bTile = makeTile({8, tileK}, packedInputElement);
    Value cTile = makeTile({8, 8}, accumulatorElement);

    auto pack = [&](Value src, Value dst, int64_t rows, int64_t cols,
                    int64_t capacityCols, bool widen) {
      Type element = cast<MemRefType>(dst.getType()).getElementType();
      Value zero = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getZeroAttr(element));
      for (int64_t i = 0; i < 8; ++i)
        for (int64_t j = 0; j < capacityCols; ++j) {
          SmallVector<Value> indices{index(i), index(j)};
          Value value = zero;
          if (i < rows && j < cols) {
            value = memref::LoadOp::create(rewriter, loc, src, indices);
            if (widen)
              value = arith::ExtFOp::create(rewriter, loc, element, value);
            else if (value.getType() != element)
              value = UnrealizedConversionCastOp::create(
                          rewriter, loc, TypeRange{element}, ValueRange{value})
                          .getResult(0);
          }
          memref::StoreOp::create(rewriter, loc, value, dst, indices);
        }
    };
    pack(op.getVs1(), aTile, m, k, tileK, false);
    pack(op.getVs2(), bTile, n, k, tileK, false);
    pack(op.getVd(), cTile, m, n, 8, fp);

    auto load = [&](Value tile, Type vector) -> Value {
      return LLVM::LoadOp::create(rewriter, loc, vector,
                                  tilePointer(rewriter, loc, tile), 1);
    };
    Value a = load(aTile, inputVector);
    Value b = load(bTile, inputVector);
    Value c = load(cTile, outputVector);
    Value result = IntrOpTy::create(rewriter, loc, outputVector, c, a, b);
    LLVM::StoreOp::create(rewriter, loc, result,
                          tilePointer(rewriter, loc, cTile), 1);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) {
        SmallVector<Value> indices{index(i), index(j)};
        Value value = memref::LoadOp::create(rewriter, loc, cTile, indices);
        if (fp)
          value = arith::TruncFOp::create(rewriter, loc, rewriter.getF16Type(),
                                          value);
        memref::StoreOp::create(rewriter, loc, value, op.getVd(), indices);
      }
    memref::AllocaScopeReturnOp::create(rewriter, loc, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

static bool contiguousBuffer(Value value, int64_t elements) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.hasStaticShape() && type.getRank() == 1 &&
         type.getDimSize(0) == elements &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides[0] == 1;
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
    // Conversion must also reject unsupported views when pass verification is
    // off.
    if (failed(op.verify()))
      return failure();
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
    int64_t width = op.getTileN();
    if (width < 8 || width > 32 || width % 8)
      return op.emitError("expected one to four native 8-output tiles");
    int64_t groups = width / 8;
    // Pass each panel address as an operand so LLVM hoists the same address
    // calculations as in the original four-panel kernel.
    SmallVector<Value> operands{a};
    for (int g = 0; g < groups; ++g) {
      Value offset = arith::AddIOp::create(rewriter, loc, base, index(g * 64));
      operands.push_back(pointer(op.getVs2(), offset));
    }
    operands.push_back(pointer(op.getVd(), op.getColumn()));
    operands.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(k / (unroll ? 16 : 8))));

    std::string assembly;
    llvm::raw_string_ostream out(assembly);
    out << "mv t1, $0\n\t";
    for (int g = 0; g < groups; ++g)
      out << "mv t" << 2 + g << ", $" << 1 + g << "\n\t";
    out << "mv t6, $" << groups + 2 << "\n\t";
    out << "vsetvli t0, zero, e32, m2, tu, mu\n\t";
    for (int g = 0; g < groups; ++g)
      out << "vmv.v.i v" << 16 + 2 * g << ", 0\n\t";
    out << "vsetivli zero, 8, e16, m1, tu, mu\n\t";
    std::string outputOperand = "$" + std::to_string(groups + 1);
    auto outputAddress = [&](int group) -> StringRef {
      if (group == 0)
        return outputOperand;
      out << "addi t0, " << outputOperand << ", " << 16 * group << "\n\t";
      return "t0";
    };
    for (int g = 0; g < groups; ++g) {
      auto addr = outputAddress(g);
      out << "vle16.v v6, (" << addr << ")\n\tvfwcvt.f.f.v v" << 16 + 2 * g
          << ", v6\n\t";
    }
    out << "vsetvli t0, zero, e16, m1, tu, mu\n\t"
           "vmv.v.i v2, 0\n\tvid.v v4\n\tvmsltu.vi v0, v4, 8\n\t1:\n\t";
    if (distance)
      for (int g = 0; g < groups; g += 2)
        out << "prefetch.r " << distance << "(t" << 2 + g << ")\n\t";
    for (int i = 0; i < (unroll ? 2 : 1); ++i) {
      out << "vle16.v v2, (t1), v0.t\n\t";
      for (int g = 0; g < groups; ++g)
        out << "vle16.v v" << 8 + 2 * g << ", (t" << 2 + g << ")\n\t";
      for (int g = 0; g < groups; ++g) {
        // smt.vfwmadot vd, v2, vs2: vd and vs2 advance by two registers.
        uint32_t opcode = 0x9e81482b + (g << 21) + (g << 8);
        out << ".word 0x" << llvm::utohexstr(opcode, true)
            << " # smt.vfwmadot v" << 16 + 2 * g << ", v2, v" << 8 + 2 * g
            << "\n\t";
      }
      out << "addi t1, t1, 16\n\t";
      for (int g = 0; g < groups; ++g)
        out << "addi t" << 2 + g << ", t" << 2 + g << ", " << groups * 128
            << "\n\t";
    }
    out << "addi t6, t6, -1\n\tbnez t6, 1b\n\t"
           "vsetivli zero, 8, e16, m1, tu, mu\n\t";
    for (int g = 0; g < groups; ++g) {
      auto addr = outputAddress(g);
      out << "vfncvt.f.f.w v6, v" << 16 + 2 * g << "\n\tvse16.v v6, (" << addr
          << ")";
      if (g + 1 < groups)
        out << "\n\t";
    }
    std::string constraints;
    for (size_t i = 0; i < operands.size(); ++i)
      constraints += "r,";
    constraints += "~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
                   "~{v0},~{v2},~{v4},~{v6},~{v8},~{v10},~{v12},~{v14},"
                   "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
                   "~{vl},~{vtype},~{memory}";
    LLVM::InlineAsmOp::create(
        rewriter, loc, Type{}, operands, assembly, constraints, true, false,
        LLVM::TailCallKind::None, false, LLVM::AsmDialectAttr{}, ArrayAttr{});
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

void mlir::populateIMEK3LegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  populateIMEFP16KernelPatterns(converter, patterns);
  patterns.add<IMEK3DotLowering<VmadotOp, Vmadot_IntrOp>,
               IMEK3DotLowering<VmadotuOp, Vmadotu_IntrOp>,
               IMEK3DotLowering<VmadotsuOp, Vmadotsu_IntrOp>,
               IMEK3DotLowering<VmadotusOp, Vmadotus_IntrOp>,
               IMEK3DotLowering<VfmadotOp, Vfmadot_IntrOp>>(converter);
}
