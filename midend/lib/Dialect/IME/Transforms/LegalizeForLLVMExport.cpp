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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/IME/IMEDialect.h"
#include "Dialect/IME/IMEOps.h"
#include "Dialect/IME/Transform.h"

using namespace mlir;
using namespace buddy::ime;

namespace {

static FlatSymbolRefAttr
getOrInsertIntrinsic(ConversionPatternRewriter &rewriter, ModuleOp module,
                     StringRef intrinsicName, LLVM::LLVMFunctionType funcType) {
  auto *ctx = rewriter.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(intrinsicName))
    return FlatSymbolRefAttr::get(ctx, intrinsicName);

  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), intrinsicName, funcType,
                                    LLVM::Linkage::External, false,
                                    LLVM::CConv::C);
  rewriter.restoreInsertionPoint(savedInsertionPoint);
  return FlatSymbolRefAttr::get(ctx, intrinsicName);
}

static Value createRVVVectorLoad(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value pointer,
                                 Type vectorType, StringRef baseIntrinsicName) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  std::string mangledName = (baseIntrinsicName + ".p0.i64").str();
  auto funcType = LLVM::LLVMFunctionType::get(
      vectorType, {vectorType, ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
  auto undefPassthru = rewriter.create<LLVM::UndefOp>(loc, vectorType);
  auto vl = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                              rewriter.getI64IntegerAttr(-1));
  auto call =
      rewriter.create<LLVM::CallOp>(loc, TypeRange{vectorType}, funcRef,
                                    ValueRange{undefPassthru, pointer, vl});
  return call.getResult();
}

static void createRVVVectorStore(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value vector,
                                 Value pointer, StringRef baseIntrinsicName) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto vectorType = vector.getType();
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  std::string mangledName = (baseIntrinsicName + ".p0.i64").str();
  auto funcType = LLVM::LLVMFunctionType::get(
      voidType, {vectorType, ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
  auto vl = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                              rewriter.getI64IntegerAttr(-1));
  rewriter.create<LLVM::CallOp>(loc, TypeRange{}, funcRef,
                                ValueRange{vector, pointer, vl});
}

static Value createIMEVmadotIntrinsic(ConversionPatternRewriter &rewriter,
                                      Location loc, ModuleOp module,
                                      Value vdVector, Value vs1Vector,
                                      Value vs2Vector,
                                      StringRef baseIntrinsicName,
                                      StringRef typeSuffix) {
  auto vdType = vdVector.getType();
  auto vs1Type = vs1Vector.getType();
  auto vs2Type = vs2Vector.getType();
  std::string mangledName = (baseIntrinsicName + typeSuffix).str();
  auto funcType =
      LLVM::LLVMFunctionType::get(vdType, {vdType, vs1Type, vs2Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
  auto call =
      rewriter.create<LLVM::CallOp>(loc, TypeRange{vdType}, funcRef,
                                    ValueRange{vdVector, vs1Vector, vs2Vector});
  return call.getResult();
}

static Value createIMEVmadotnIntrinsic(ConversionPatternRewriter &rewriter,
                                       Location loc, ModuleOp module,
                                       Value vdVector, Value vs1Vector,
                                       Value vs2Vector, Value slideVal,
                                       StringRef baseIntrinsicName,
                                       StringRef typeSuffix) {
  auto *ctx = rewriter.getContext();
  auto vdType = vdVector.getType();
  auto vs1Type = vs1Vector.getType();
  auto vs2Type = vs2Vector.getType();
  auto i64Type = IntegerType::get(ctx, 64);
  std::string mangledName = (baseIntrinsicName + typeSuffix).str();
  auto funcType = LLVM::LLVMFunctionType::get(
      vdType, {vdType, vs1Type, vs2Type, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
  auto call = rewriter.create<LLVM::CallOp>(
      loc, TypeRange{vdType}, funcRef,
      ValueRange{vdVector, vs1Vector, vs2Vector, slideVal});
  return call.getResult();
}

static Value extractPointerFromMemref(ConversionPatternRewriter &rewriter,
                                      Location loc, Value memref) {
  auto *ctx = rewriter.getContext();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto i64Type = IntegerType::get(ctx, 64);
  Value idx =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value i64Val = rewriter.create<arith::IndexCastOp>(loc, i64Type, idx);
  Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, i64Val);
  return ptr;
}

struct IMEVmadotLowering : public ConvertOpToLLVMPattern<VmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8VecType = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot",
        ".nxv8i32.nxv32i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotuLowering : public ConvertOpToLLVMPattern<VmadotuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotuOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8VecType = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotu",
        ".nxv8i32.nxv32i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotsuLowering : public ConvertOpToLLVMPattern<VmadotsuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotsuOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8VecType = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotsu",
        ".nxv8i32.nxv32i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotusLowering : public ConvertOpToLLVMPattern<VmadotusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotusOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8VecType = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, i8VecType,
                                       "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotus",
        ".nxv8i32.nxv32i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadotLowering : public ConvertOpToLLVMPattern<VfmadotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VfmadotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto f16Type = Float16Type::get(ctx);
    auto f16VecType = VectorType::get({32}, f16Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       f16VecType, "llvm.riscv.vle.nxv32f16");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       f16VecType, "llvm.riscv.vle.nxv32f16");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, f16VecType,
                                      "llvm.riscv.vle.nxv32f16");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot",
        ".nxv32f16.nxv32f16.nxv32f16");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv32f16");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot1Lowering : public ConvertOpToLLVMPattern<Vmadot1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);

    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    // Load 64 i8 elements for VS1 (two consecutive vector registers)
    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot1",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot1uLowering : public ConvertOpToLLVMPattern<Vmadot1uOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot1uOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot1u",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot1suLowering : public ConvertOpToLLVMPattern<Vmadot1suOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot1suOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot1su", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot1usLowering : public ConvertOpToLLVMPattern<Vmadot1usOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot1usOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot1us", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot2Lowering : public ConvertOpToLLVMPattern<Vmadot2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot2",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot2uLowering : public ConvertOpToLLVMPattern<Vmadot2uOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot2uOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot2u",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot2suLowering : public ConvertOpToLLVMPattern<Vmadot2suOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot2suOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot2su", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot2usLowering : public ConvertOpToLLVMPattern<Vmadot2usOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot2usOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot2us", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot3Lowering : public ConvertOpToLLVMPattern<Vmadot3Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot3Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot3",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot3uLowering : public ConvertOpToLLVMPattern<Vmadot3uOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot3uOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot3u",
        ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot3suLowering : public ConvertOpToLLVMPattern<Vmadot3suOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot3suOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot3su", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadot3usLowering : public ConvertOpToLLVMPattern<Vmadot3usOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vmadot3usOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot3us", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadot1Lowering : public ConvertOpToLLVMPattern<Vfmadot1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vfmadot1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto f16Type = Float16Type::get(ctx);

    auto f16Vec32Type = VectorType::get({32}, f16Type, /*scalableDims=*/true);

    auto f16Vec16Type = VectorType::get({16}, f16Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       f16Vec32Type, "llvm.riscv.vle.nxv32f16");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot1",
        ".nxv16f16.nxv32f16.nxv16f16");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv16f16");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadot2Lowering : public ConvertOpToLLVMPattern<Vfmadot2Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vfmadot2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto f16Type = Float16Type::get(ctx);

    auto f16Vec32Type = VectorType::get({32}, f16Type, /*scalableDims=*/true);

    auto f16Vec16Type = VectorType::get({16}, f16Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       f16Vec32Type, "llvm.riscv.vle.nxv32f16");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot2",
        ".nxv16f16.nxv32f16.nxv16f16");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv16f16");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadot3Lowering : public ConvertOpToLLVMPattern<Vfmadot3Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Vfmadot3Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    auto f16Type = Float16Type::get(ctx);

    auto f16Vec32Type = VectorType::get({32}, f16Type, /*scalableDims=*/true);

    auto f16Vec16Type = VectorType::get({16}, f16Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       f16Vec32Type, "llvm.riscv.vle.nxv32f16");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot3",
        ".nxv16f16.nxv32f16.nxv16f16");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv16f16");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotnLowering : public ConvertOpToLLVMPattern<VmadotnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotn", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotnuLowering : public ConvertOpToLLVMPattern<VmadotnuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotnuOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnu", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotnsuLowering : public ConvertOpToLLVMPattern<VmadotnsuOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotnsuOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnsu", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVmadotnusLowering : public ConvertOpToLLVMPattern<VmadotnusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VmadotnusOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    auto i8Type = IntegerType::get(ctx, 8);
    auto i32Type = IntegerType::get(ctx, 32);
    auto i8Vec64Type = VectorType::get({64}, i8Type, /*scalableDims=*/true);
    auto i8Vec32Type = VectorType::get({32}, i8Type, /*scalableDims=*/true);
    auto i32VecType = VectorType::get({8}, i32Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       i8Vec64Type, "llvm.riscv.vle.nxv64i8");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       i8Vec32Type, "llvm.riscv.vle.nxv32i8");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, i32VecType,
                                      "llvm.riscv.vle.nxv8i32");
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnus", ".nxv8i32.nxv64i8.nxv32i8");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv8i32");
    rewriter.eraseOp(op);
    return success();
  }
};

struct IMEVfmadotnLowering : public ConvertOpToLLVMPattern<VfmadotnOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VfmadotnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    auto f16Type = Float16Type::get(ctx);

    auto f16Vec32Type = VectorType::get({32}, f16Type, /*scalableDims=*/true);

    auto f16Vec16Type = VectorType::get({16}, f16Type, /*scalableDims=*/true);

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       f16Vec32Type, "llvm.riscv.vle.nxv32f16");
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      f16Vec16Type, "llvm.riscv.vle.nxv16f16");
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vfmadotn", ".nxv16f16.nxv32f16.nxv16f16");
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         "llvm.riscv.vse.nxv16f16");
    rewriter.eraseOp(op);
    return success();
  }
};

struct LegalizeIMEForLLVMExport
    : public PassWrapper<LegalizeIMEForLLVMExport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeIMEForLLVMExport)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
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
    target.addIllegalOp<VmadotOp, VmadotuOp, VmadotsuOp, VmadotusOp, VfmadotOp,
                        Vmadot1Op, Vmadot1uOp, Vmadot1suOp, Vmadot1usOp,
                        Vmadot2Op, Vmadot2uOp, Vmadot2suOp, Vmadot2usOp,
                        Vmadot3Op, Vmadot3uOp, Vmadot3suOp, Vmadot3usOp,
                        Vfmadot1Op, Vfmadot2Op, Vfmadot3Op, VmadotnOp,
                        VmadotnuOp, VmadotnsuOp, VmadotnusOp, VfmadotnOp>();

    LLVMTypeConverter typeConverter(&context);

    RewritePatternSet patterns(&context);
    patterns
        .add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
             IMEVmadotusLowering, IMEVfmadotLowering, IMEVmadot1Lowering,
             IMEVmadot1uLowering, IMEVmadot1suLowering, IMEVmadot1usLowering,
             IMEVmadot2Lowering, IMEVmadot2uLowering, IMEVmadot2suLowering,
             IMEVmadot2usLowering, IMEVmadot3Lowering, IMEVmadot3uLowering,
             IMEVmadot3suLowering, IMEVmadot3usLowering, IMEVfmadot1Lowering,
             IMEVfmadot2Lowering, IMEVfmadot3Lowering, IMEVmadotnLowering,
             IMEVmadotnuLowering, IMEVmadotnsuLowering, IMEVmadotnusLowering,
             IMEVfmadotnLowering>(typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateIMELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<IMEVmadotLowering, IMEVmadotuLowering, IMEVmadotsuLowering,
               IMEVmadotusLowering, IMEVfmadotLowering, IMEVmadot1Lowering,
               IMEVmadot1uLowering, IMEVmadot1suLowering, IMEVmadot1usLowering,
               IMEVmadot2Lowering, IMEVmadot2uLowering, IMEVmadot2suLowering,
               IMEVmadot2usLowering, IMEVmadot3Lowering, IMEVmadot3uLowering,
               IMEVmadot3suLowering, IMEVmadot3usLowering, IMEVfmadot1Lowering,
               IMEVfmadot2Lowering, IMEVfmadot3Lowering, IMEVmadotnLowering,
               IMEVmadotnuLowering, IMEVmadotnsuLowering, IMEVmadotnusLowering,
               IMEVfmadotnLowering>(converter);
}

void mlir::configureIMELegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addIllegalOp<VmadotOp, VmadotuOp, VmadotsuOp, VmadotusOp, VfmadotOp,
                      Vmadot1Op, Vmadot1uOp, Vmadot1suOp, Vmadot1usOp,
                      Vmadot2Op, Vmadot2uOp, Vmadot2suOp, Vmadot2usOp,
                      Vmadot3Op, Vmadot3uOp, Vmadot3suOp, Vmadot3usOp,
                      Vfmadot1Op, Vfmadot2Op, Vfmadot3Op, VmadotnOp, VmadotnuOp,
                      VmadotnsuOp, VmadotnusOp, VfmadotnOp>();
}

std::unique_ptr<Pass> buddy::ime::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeIMEForLLVMExport>();
}
