//====- LowerVectorExpPass.cpp - Vector Experiment Dialect Lowering Pass  -===//
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
// This file defines vector experiment dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "RVV/RVVDialect.h"
#include "RVV/Transforms.h"
#include "VectorExp/VectorExpDialect.h"
#include "VectorExp/VectorExpOps.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

/// Predication Operation Lowering Pass
class VectorExpPredicationLowering
    : public ConvertOpToLLVMPattern<vector_exp::PredicationOp> {
public:
  using ConvertOpToLLVMPattern<
      vector_exp::PredicationOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector_exp::PredicationOp op,
                  vector_exp::PredicationOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the region and block from the predication operation.
    Location loc = op.getLoc();
    Region &configRegion = op.getRegion();
    mlir::Block &configBlock = configRegion.front();
    // Iterate region and get the operations inside.
    for (mlir::Operation &innerOp : configBlock.getOperations()) {
      //
      if (isa<arith::AddFOp>(innerOp)) {
        Type resultType = cast<arith::AddFOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPFAddOp>(
            loc, resultType, cast<arith::AddFOp>(innerOp).getLhs(),
            cast<arith::AddFOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<arith::MulFOp>(innerOp)) {
        Type resultType = cast<arith::MulFOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPFMulOp>(
            loc, resultType, cast<arith::MulFOp>(innerOp).getLhs(),
            cast<arith::MulFOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<arith::AddIOp>(innerOp)) {
        Type resultType = cast<arith::AddIOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPAddOp>(
            loc, resultType, cast<arith::AddIOp>(innerOp).getLhs(),
            cast<arith::AddIOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<arith::MulIOp>(innerOp)) {
        Type resultType = cast<arith::MulIOp>(innerOp).getResult().getType();
        Value result = rewriter.create<LLVM::VPMulOp>(
            loc, resultType, cast<arith::MulIOp>(innerOp).getLhs(),
            cast<arith::MulIOp>(innerOp).getRhs(), op.getMask(), op.getVl());
        rewriter.replaceOp(op, result);
      } else if (isa<vector::LoadOp>(innerOp)) {
        vector::LoadOp loadOp = cast<vector::LoadOp>(innerOp);
        // Prepare the MemRef descriptor for the `getStridedElementPtr`.
        // - Get the MemRef type of the load operation.
        // - Convert the MemRef type into LLVM struct type.
        // - Create UnrealizedConversionCastOp to provide the descriptor value.
        MemRefType memRefTy = loadOp.getMemRefType();
        Type structType = this->getTypeConverter()->convertType(memRefTy);
        Value memDesc = rewriter
                            .create<UnrealizedConversionCastOp>(
                                loc, structType, loadOp.getBase())
                            .getResult(0);
        // Prepare the integer indices for the `getStridedElementPtr`.
        // - Interate the indices of the load operation.
        // - Convert origin index type into integer type.
        // - Create UnrealizedConversionCastOp to provide the integer value.
        SmallVector<Value, 4> indices;
        for (Value idx : loadOp.getIndices()) {
          Type idxType = idx.getType();
          Type intType = this->getTypeConverter()->convertType(idxType);
          Value intIdx =
              rewriter.create<UnrealizedConversionCastOp>(loc, intType, idx)
                  .getResult(0);
          indices.push_back(intIdx);
        }
        // Prepare the data pointer for the VP load operation.
        // - Call the `getStridedElementPtr` with above descriptor and indices.
        Value dataPtr = this->getStridedElementPtr(loc, memRefTy, memDesc,
                                                   indices, rewriter);
        // Create VP load operation and replace the predication operation.
        // - Get the result type of the predication operation.
        // - Create VP load operation.
        // - Replace original predication operation.
        VectorType resultType = op.getResult().getType().cast<VectorType>();
        Value resultValue = rewriter.create<LLVM::VPLoadOp>(
            loc, resultType, dataPtr, op.getMask(), op.getVl());
        rewriter.replaceOp(op, resultValue);
      } else if (isa<vector::StoreOp>(innerOp)) {
        // The conversion to VP operation is similar to the load operation.
        // - Get MemRef descriptor.
        // - Get indices of the memory access.
        // - Get the data pointer.
        // - Create VP store operation and erase the predication operation.
        vector::StoreOp storeOp = cast<vector::StoreOp>(innerOp);
        Value valueToStore = storeOp.getValueToStore();
        MemRefType memRefTy = storeOp.getMemRefType();
        Type structType = this->getTypeConverter()->convertType(memRefTy);
        Value memDesc = rewriter
                            .create<UnrealizedConversionCastOp>(
                                loc, structType, storeOp.getBase())
                            .getResult(0);
        SmallVector<Value, 4> indices;
        for (Value idx : storeOp.getIndices()) {
          Type idxType = idx.getType();
          Type intType = this->getTypeConverter()->convertType(idxType);
          Value intIdx =
              rewriter.create<UnrealizedConversionCastOp>(loc, intType, idx)
                  .getResult(0);
          indices.push_back(intIdx);
        }
        Value dataPtr = this->getStridedElementPtr(loc, memRefTy, memDesc,
                                                   indices, rewriter);
        rewriter.create<LLVM::VPStoreOp>(loc, valueToStore, dataPtr,
                                         op.getMask(), op.getVl());
        rewriter.eraseOp(op);
      } else if (isa<vector::YieldOp>(innerOp)) {
        // Skip the YieldOp.
        continue;
      } else {
        // Unsupported inner operations.
        mlir::emitError(loc)
            << "unsupported inner operation " << innerOp.getName()
            << " of the predication operation.";
      }
    }
    return success();
  }
};

/// GetVL Operation Lowering Pass
//
//  GetVL Operation retrieves the max VL value based on the configurations of
//  SEW and LMUL. For example, returns four times the physical vl for element
//  type i32.
//  `%vl = vector.get_vl i32, 4 : index`
//
//  Lowering strategy:
//  - Validate the legality of LMUL and SEW.
//  - Map the LMUL Attribute to the LMUL numeric value in the RVV Dialect.
//  - Map the SEW Attribute to the SEW numeric value in the RVV Dialect.
//  - Lower to the RVV SetVL Operation by setting a large value to obtain the
//  max VL.
class VectorExpGetVLLowering
    : public ConvertOpToLLVMPattern<vector_exp::GetVLOp> {
public:
  using ConvertOpToLLVMPattern<vector_exp::GetVLOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector_exp::GetVLOp op, vector_exp::GetVLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    /// Max SetVL
    IndexType indexType = IndexType::get(ctx);
    // TODO: Find a more reasonable way to obtain the max VL.
    auto maxVLVal = rewriter.create<arith::ConstantIndexOp>(loc, 4096);
    // Supported data types: i32, i64, f32, f64.
    // Type mapping relationship:
    // 8 bit - arith.constant 0
    // 16 bit - arith.constant 1
    // 32 bit - arith.constant 2
    // 64 bit - arith.constant 3
    int dtypeWidth = adaptor.getDtype().getIntOrFloatBitWidth();
    int dtypeConfig;
    switch (dtypeWidth) {
    case 8:
      dtypeConfig = 0;
      break;
    case 16:
      dtypeConfig = 1;
      break;
    case 32:
      dtypeConfig = 2;
      break;
    case 64:
      dtypeConfig = 3;
      break;
    default:
      // Handle unknown or unsupported data type width.
      emitError(loc, "Unsupported data type width");
      return failure();
    }
    auto sewVal = rewriter.create<arith::ConstantIndexOp>(loc, dtypeConfig);
    // Supported LMUL values: 1, 2, 4, 8.
    // LMUL = 1 - arith.constant 0
    // LMUL = 2 - arith.constant 1
    // LMUL = 4 - arith.constant 2
    // LMUL = 8 - arith.constant 3
    uint64_t lmulAttr = adaptor.getLmul().getZExtValue();
    int lmulConfig;
    switch (lmulAttr) {
    case 1:
      lmulConfig = 0;
      break;
    case 2:
      lmulConfig = 1;
      break;
    case 4:
      lmulConfig = 2;
      break;
    case 8:
      lmulConfig = 3;
      break;
    default:
      // Handle unknown or unsupported LMUL value.
      emitError(loc, "Unsupported LMUL value: " + std::to_string(lmulAttr));
      return failure();
    }
    auto lmulVal = rewriter.create<arith::ConstantIndexOp>(loc, lmulConfig);
    auto maxVLOp = rewriter.create<buddy::rvv::RVVSetVlOp>(
        loc, indexType, maxVLVal, sewVal, lmulVal);

    rewriter.replaceOp(op, maxVLOp);
    return success();
  }
};

/// SetVL Operation Lowering Pass
//
//  This implements one of the programming models for the SetVL abstraction: Set
//  vector length per loop iteration. Lowering strategy includes:
//  - Forward search for the configuration of get VL, generate RVV SetVL.
//  - Use a symbol table to maintain the mapping of original SSA values and
//  generated SSA values within the Region.
//  - Convert vector operations within the SetVL Region into RVV Dialect or LLVM
//  VP vector operations.
class VectorExpSetVLLowering
    : public ConvertOpToLLVMPattern<vector_exp::SetVLOp> {
public:
  using ConvertOpToLLVMPattern<vector_exp::SetVLOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector_exp::SetVLOp op, vector_exp::SetVLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    // Locate affine.min operation.
    Value avlValue = op.getVl();
    Operation *affineMin = avlValue.getDefiningOp();
    if (!affineMin || !isa<affine::AffineMinOp>(affineMin)) {
      op.emitError() << "Expected 'VL' to be produced by an 'affine.min' "
                        "operation, but it was not. "
                     << "Current operation is '"
                     << affineMin->getName().getStringRef() << "'.";
      return failure();
    }
    // Locate vector_exp.get_vl operation.
    vector_exp::GetVLOp getVLOp = nullptr;
    bool foundGetVLOp = false;
    for (auto operand : affineMin->getOperands()) {
      getVLOp = operand.getDefiningOp<vector_exp::GetVLOp>();
      if (getVLOp) {
        foundGetVLOp = true;
        break;
      }
    }
    if (!foundGetVLOp) {
      affineMin->emitError()
          << "No operand of 'affine.min' is produced by 'vector_exp::GetVLOp'.";
      return failure();
    }
    // Get SEW and LMUL from GetVL operation.
    // Supported data types: i32, i64, f32, f64.
    // Type mapping relationship:
    // 8 bit - arith.constant 0
    // 16 bit - arith.constant 1
    // 32 bit - arith.constant 2
    // 64 bit - arith.constant 3
    auto dtype = getVLOp.getDtype();
    int dtypeWidth = getVLOp.getDtype().getIntOrFloatBitWidth();
    int dtypeConfig;
    switch (dtypeWidth) {
    case 8:
      dtypeConfig = 0;
      break;
    case 16:
      dtypeConfig = 1;
      break;
    case 32:
      dtypeConfig = 2;
      break;
    case 64:
      dtypeConfig = 3;
      break;
    default:
      // Handle unknown or unsupported data type width.
      emitError(loc, "Unsupported data type width");
      return failure();
    }
    auto sewVal = rewriter.create<arith::ConstantIndexOp>(loc, dtypeConfig);
    // Supported LMUL values: 1, 2, 4, 8.
    // LMUL = 1 - arith.constant 0
    // LMUL = 2 - arith.constant 1
    // LMUL = 4 - arith.constant 2
    // LMUL = 8 - arith.constant 3
    uint64_t lmulAttr = getVLOp.getLmul().getZExtValue();
    int lmulConfig;
    switch (lmulAttr) {
    case 1:
      lmulConfig = 0;
      break;
    case 2:
      lmulConfig = 1;
      break;
    case 4:
      lmulConfig = 2;
      break;
    case 8:
      lmulConfig = 3;
      break;
    default:
      // Handle unknown or unsupported LMUL value.
      emitError(loc, "Unsupported LMUL value: " + std::to_string(lmulAttr));
      return failure();
    }
    auto lmulVal = rewriter.create<arith::ConstantIndexOp>(loc, lmulConfig);
    // Generate RVV SetVL operation.
    IndexType indexType = IndexType::get(ctx);
    rewriter.create<buddy::rvv::RVVSetVlOp>(loc, indexType, avlValue, sewVal,
                                            lmulVal);

    // Access the Region
    Region &region = op.getRegion();
    // Access the first block
    Block &block = region.front();
    // Construct Scalable Vector Type
    // For RVV, the shape of the dynamic vector type is related to LMUL and SEW.
    // Calculate scalable vector size according to the GetVL configuration.
    // Scalable Vector Size = (64 x LMUL) / SEW
    int scalableVectorSize = (64 * lmulAttr) / dtypeWidth;
    SmallVector<int64_t, 1> shape = {scalableVectorSize};
    SmallVector<bool, 1> scalableDim = {true};
    ShapedType scalableVectorType = VectorType::get(shape, dtype, scalableDim);

    // SetVL Region Symbol Table
    llvm::DenseMap<Value, Value> symbolTable;
    // Iterate over every ops in the SetVL region.
    for (Operation &innerOp : block) {
      if (isa<arith::AddIOp>(innerOp)) {
        arith::AddIOp addIOp = cast<arith::AddIOp>(innerOp);
        Value lhsOrigVal = addIOp.getLhs();
        Value lhsVal = symbolTable[lhsOrigVal];
        Value rhsOrigVal = addIOp.getRhs();
        Value rhsVal = symbolTable[rhsOrigVal];
        Value resultValue = rewriter.create<buddy::rvv::RVVAddOp>(
            loc, scalableVectorType, lhsVal, rhsVal, avlValue);
        symbolTable[innerOp.getResult(0)] = resultValue;
      } else if (isa<vector::LoadOp>(innerOp)) {
        vector::LoadOp loadOp = cast<vector::LoadOp>(innerOp);
        auto baseOp = loadOp.getBase();
        auto idx = loadOp.getIndices().front();
        Value resultValue = rewriter.create<buddy::rvv::RVVLoadOp>(
            loc, scalableVectorType, baseOp, idx, avlValue);
        symbolTable[innerOp.getResult(0)] = resultValue;
      } else if (isa<vector::StoreOp>(innerOp)) {
        vector::StoreOp storeOp = cast<vector::StoreOp>(innerOp);
        Value origValueToStore = storeOp.getValueToStore();
        Value valueToStore = symbolTable[origValueToStore];
        auto baseOp = storeOp.getBase();
        auto idx = storeOp.getIndices().front();
        rewriter.create<buddy::rvv::RVVStoreOp>(loc, valueToStore, baseOp, idx,
                                                avlValue);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateLowerVectorExpConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<VectorExpPredicationLowering,VectorExpSetVLLowering,
  VectorExpGetVLLowering>(converter);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// LowerVectorExpPass
//===----------------------------------------------------------------------===//

namespace {
class LowerVectorExpPass
    : public PassWrapper<LowerVectorExpPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorExpPass)
  LowerVectorExpPass() = default;
  LowerVectorExpPass(const LowerVectorExpPass &) {}

  StringRef getArgument() const final { return "lower-vector-exp"; }
  StringRef getDescription() const final {
    return "Lower Vector Experiment Dialect.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        arith::ArithDialect,
        buddy::vector_exp::VectorExpDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        buddy::rvv::RVVDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }
};
} // end anonymous namespace.

void LowerVectorExpPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      buddy::rvv::RVVDialect,
      LLVM::LLVMDialect
    >();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  populateLowerVectorExpConversionPatterns(converter, patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerLowerVectorExpPass() { PassRegistration<LowerVectorExpPass>(); }
} // namespace buddy
} // namespace mlir
