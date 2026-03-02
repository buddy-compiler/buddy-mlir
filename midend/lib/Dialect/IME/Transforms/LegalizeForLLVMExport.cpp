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

//===----------------------------------------------------------------------===//
// IME Type Configuration
//===----------------------------------------------------------------------===//

/// Enumeration of supported IME data types
enum class IMEDataType {
  Int8,   // 8-bit integer (signed/unsigned)
  Int16,  // 16-bit integer (signed/unsigned)
  FP16    // 16-bit floating point
};

/// Configuration structure for IME operations based on data type
struct IMETypeConfig {
  int64_t targetVL;        // Vector length
  int64_t sew;             // SEW encoding: 0=e8, 1=e16, 2=e32, 3=e64
  int64_t lmul;            // LMUL encoding: 0=m1, 1=m2, 2=m4, 3=m8
  int64_t extendedVL;      // Extended VL for sliding operations (2x targetVL)
  Type elementType;        // Element type for input vectors
  Type outputElementType;  // Element type for output vectors
  VectorType inputVecType; // Input vector type (e.g., nxv32i8)
  VectorType outputVecType;// Output vector type (e.g., nxv8i32)
  VectorType extendedInputVecType; // Extended input vector for sliding ops
  std::string inputVecSuffix;  // e.g., "nxv32i8"
  std::string outputVecSuffix; // e.g., "nxv8i32"
  std::string extendedInputVecSuffix; // e.g., "nxv64i8"
};

/// Get IME type configuration from memref element type
static IMETypeConfig getIMETypeConfig(MLIRContext *ctx, Type elementType) {
  IMETypeConfig config;
  config.lmul = 0; // Always m1 for IME operations
  
  if (elementType.isInteger(8)) {
    // int8: SEW=e8, VL=32, MAC unit 4x4x8
    config.targetVL = 32;
    config.sew = 0;  // e8
    config.extendedVL = 64;
    config.elementType = IntegerType::get(ctx, 8);
    config.outputElementType = IntegerType::get(ctx, 32);
    config.inputVecType = VectorType::get({32}, config.elementType, /*scalableDims=*/true);
    config.outputVecType = VectorType::get({8}, config.outputElementType, /*scalableDims=*/true);
    config.extendedInputVecType = VectorType::get({64}, config.elementType, /*scalableDims=*/true);
    config.inputVecSuffix = "nxv32i8";
    config.outputVecSuffix = "nxv8i32";
    config.extendedInputVecSuffix = "nxv64i8";
  } else if (elementType.isInteger(16)) {
    // int16: SEW=e16, VL=16, MAC unit 4x4x4
    config.targetVL = 16;
    config.sew = 1;  // e16
    config.extendedVL = 32;
    config.elementType = IntegerType::get(ctx, 16);
    config.outputElementType = IntegerType::get(ctx, 32);
    config.inputVecType = VectorType::get({16}, config.elementType, /*scalableDims=*/true);
    config.outputVecType = VectorType::get({8}, config.outputElementType, /*scalableDims=*/true);
    config.extendedInputVecType = VectorType::get({32}, config.elementType, /*scalableDims=*/true);
    config.inputVecSuffix = "nxv16i16";
    config.outputVecSuffix = "nxv8i32";
    config.extendedInputVecSuffix = "nxv32i16";
  } else if (elementType.isF16()) {
    // fp16: SEW=e16, VL=16, MAC unit 4x4x4
    config.targetVL = 16;
    config.sew = 1;  // e16
    config.extendedVL = 32;
    config.elementType = Float16Type::get(ctx);
    config.outputElementType = Float16Type::get(ctx); // fp16 output is also f16
    config.inputVecType = VectorType::get({16}, config.elementType, /*scalableDims=*/true);
    config.outputVecType = VectorType::get({16}, config.outputElementType, /*scalableDims=*/true);
    config.extendedInputVecType = VectorType::get({32}, config.elementType, /*scalableDims=*/true);
    config.inputVecSuffix = "nxv16f16";
    config.outputVecSuffix = "nxv16f16";
    config.extendedInputVecSuffix = "nxv32f16";
  } else {
    // Default to int8 if unknown type
    config.targetVL = 32;
    config.sew = 0;
    config.extendedVL = 64;
    config.elementType = IntegerType::get(ctx, 8);
    config.outputElementType = IntegerType::get(ctx, 32);
    config.inputVecType = VectorType::get({32}, config.elementType, /*scalableDims=*/true);
    config.outputVecType = VectorType::get({8}, config.outputElementType, /*scalableDims=*/true);
    config.extendedInputVecType = VectorType::get({64}, config.elementType, /*scalableDims=*/true);
    config.inputVecSuffix = "nxv32i8";
    config.outputVecSuffix = "nxv8i32";
    config.extendedInputVecSuffix = "nxv64i8";
  }
  
  return config;
}

/// Get element type from a memref value
static Type getMemRefElementType(Value memref) {
  if (auto memrefType = dyn_cast<MemRefType>(memref.getType())) {
    return memrefType.getElementType();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

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

// Create vsetvli intrinsic call to configure vector type
// SEW encoding: 0=e8, 1=e16, 2=e32, 3=e64
// LMUL encoding: 0=m1, 1=m2, 2=m4, 3=m8, 5=mf8, 6=mf4, 7=mf2
static Value createVsetvli(ConversionPatternRewriter &rewriter, Location loc,
                           ModuleOp module, int64_t avl, int64_t sew,
                           int64_t lmul) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);

  // vsetvli intrinsic: i64 @llvm.riscv.vsetvli.i64(i64 avl, i64 sew, i64 lmul)
  auto funcType =
      LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type, i64Type}, false);
  auto funcRef =
      getOrInsertIntrinsic(rewriter, module, "llvm.riscv.vsetvli.i64", funcType);

  auto avlVal = rewriter.create<LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(avl));
  auto sewVal = rewriter.create<LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(sew));
  auto lmulVal = rewriter.create<LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(lmul));

  auto call = rewriter.create<LLVM::CallOp>(loc, TypeRange{i64Type}, funcRef,
                                            ValueRange{avlVal, sewVal, lmulVal});
  return call.getResult();
}

static Value createRVVVectorLoad(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value pointer,
                                 Type vectorType, StringRef baseIntrinsicName,
                                 Value vl) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  std::string mangledName = (baseIntrinsicName + ".p0.i64").str();
  auto funcType = LLVM::LLVMFunctionType::get(
      vectorType, {vectorType, ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
  auto undefPassthru = rewriter.create<LLVM::UndefOp>(loc, vectorType);
  auto call =
      rewriter.create<LLVM::CallOp>(loc, TypeRange{vectorType}, funcRef,
                                    ValueRange{undefPassthru, pointer, vl});
  return call.getResult();
}

static void createRVVVectorStore(ConversionPatternRewriter &rewriter,
                                 Location loc, ModuleOp module, Value vector,
                                 Value pointer, StringRef baseIntrinsicName,
                                 Value vl) {
  auto *ctx = rewriter.getContext();
  auto i64Type = IntegerType::get(ctx, 64);
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto vectorType = vector.getType();
  auto voidType = LLVM::LLVMVoidType::get(ctx);
  std::string mangledName = (baseIntrinsicName + ".p0.i64").str();
  auto funcType = LLVM::LLVMFunctionType::get(
      voidType, {vectorType, ptrType, i64Type}, false);
  auto funcRef = getOrInsertIntrinsic(rewriter, module, mangledName, funcType);
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

    // Get element type from vs1 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs1());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.inputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs1 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs1());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.inputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotu",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs1 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs1());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.inputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotsu",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs1 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs1());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.inputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr, 
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadotus",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs1 memref - should be f16
    Type elemType = getMemRefElementType(op.getVs1());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type (fp16: vl=16, SEW=e16, LMUL=m1)
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.inputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    // For extended loads (2x elements for vs1)
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    // Load extended elements for VS1 (two consecutive vector registers)
    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot1",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot1u",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot1su", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot1us", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot2",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot2u",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot2su", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot2us", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot3",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vmadot3u",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot3su", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec,
        "llvm.riscv.ime.vmadot3us", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref - should be f16
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type (fp16: vl=16, SEW=e16, LMUL=m1)
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot1",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref - should be f16
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type (fp16: vl=16, SEW=e16, LMUL=m1)
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot2",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref - should be f16
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type (fp16: vl=16, SEW=e16, LMUL=m1)
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, "llvm.riscv.ime.vfmadot3",
        typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotn", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnu", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnsu", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string outputLoadIntrinsic = "llvm.riscv.vle." + config.outputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr, 
                                      config.outputVecType, outputLoadIntrinsic, vlValue);
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vmadotnus", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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

    // Get element type from vs2 memref and configure accordingly
    Type elemType = getMemRefElementType(op.getVs2());
    if (!elemType)
      return failure();
    
    IMETypeConfig config = getIMETypeConfig(ctx, elemType);
    auto i64Type = IntegerType::get(ctx, 64);

    // Configure vtype based on element type
    createVsetvli(rewriter, loc, module, config.targetVL, config.sew, config.lmul);
    Value vlValue = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.targetVL));
    Value vlValueExtended = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(config.extendedVL));

    Value vdPtr = extractPointerFromMemref(rewriter, loc, op.getVd());
    Value vs1Ptr = extractPointerFromMemref(rewriter, loc, op.getVs1());
    Value vs2Ptr = extractPointerFromMemref(rewriter, loc, op.getVs2());
    Value slideVal = op.getSlide();

    std::string extLoadIntrinsic = "llvm.riscv.vle." + config.extendedInputVecSuffix;
    std::string loadIntrinsic = "llvm.riscv.vle." + config.inputVecSuffix;
    std::string storeIntrinsic = "llvm.riscv.vse." + config.outputVecSuffix;
    std::string typeSuffix = "." + config.outputVecSuffix + "." + 
                             config.extendedInputVecSuffix + "." + config.inputVecSuffix;

    Value vs1Vec = createRVVVectorLoad(rewriter, loc, module, vs1Ptr,
                                       config.extendedInputVecType, extLoadIntrinsic, vlValueExtended);
    Value vs2Vec = createRVVVectorLoad(rewriter, loc, module, vs2Ptr,
                                       config.inputVecType, loadIntrinsic, vlValue);
    Value vdVec = createRVVVectorLoad(rewriter, loc, module, vdPtr,
                                      config.outputVecType, loadIntrinsic, vlValue);
    Value result = createIMEVmadotnIntrinsic(
        rewriter, loc, module, vdVec, vs1Vec, vs2Vec, slideVal,
        "llvm.riscv.ime.vfmadotn", typeSuffix);
    createRVVVectorStore(rewriter, loc, module, result, vdPtr,
                         storeIntrinsic, vlValue);
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
