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

} // namespace

void mlir::populateIMEK3LegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<IMEK3DotLowering<VmadotOp, Vmadot_IntrOp>,
               IMEK3DotLowering<VmadotuOp, Vmadotu_IntrOp>,
               IMEK3DotLowering<VmadotsuOp, Vmadotsu_IntrOp>,
               IMEK3DotLowering<VmadotusOp, Vmadotus_IntrOp>,
               IMEK3DotLowering<VfmadotOp, Vfmadot_IntrOp>>(converter);
}
