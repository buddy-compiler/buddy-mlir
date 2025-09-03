//====- VIRToVectorPass.cpp - VIR Dialect Lowering Pass -------------------===//
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
// This file defines the VIR dialect lowering pass that converts dynamic vector
// operations to fixed/scalable vector operations with loop-based iteration.
//
// The pass implements the following transformation:
// 1. Converts `vir.set_vl` regions to affine loops
// 2. Transforms dynamic vector operations to fixed/scalable vector operations
// 3. Handles tail loops for remaining elements using scalar operations
// 4. Supports both fixed and scalable vector types via pass options
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseMap.h"

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"
#include "VIR/VIRTypes.h"

using namespace mlir;
using namespace buddy;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

namespace {

// Constants for better readability
static constexpr int DEFAULT_VECTOR_WIDTH = 4;
static constexpr bool DEFAULT_USE_SCALABLE = false;

/// VIR SetVL Operation Lowering Pattern
///
/// This pattern implements the lowering approach for the VIR SetVL abstraction:
/// converting dynamic vector regions to loop iterations with vector operations.
///
/// The pattern performs the following transformations:
/// 1. Analyzes the dynamic vector region to determine anchor type.
/// 2. Creates vectorization loops with proper bounds calculation.
/// 3. Converts dynamic vector operations to fixed/scalable vector operations.
/// 4. Generates tail loops for handling remaining elements.
/// 5. Maintains proper iteration variable tracking between loops.
class VIRSetVLLowering : public ConvertOpToLLVMPattern<vir::SetVLOp> {
public:
  VIRSetVLLowering(LLVMTypeConverter &converter, bool useScalable,
                   int vectorWidth)
      : ConvertOpToLLVMPattern<vir::SetVLOp>(converter),
        useScalable(useScalable), vectorWidth(vectorWidth) {}

private:
  bool useScalable;
  int vectorWidth;

  LogicalResult
  matchAndRewrite(vir::SetVLOp op, vir::SetVLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    //===------------------------------------------------------------------===//
    // Step 1: Find the anchor type from dynamic vector types.
    //===------------------------------------------------------------------===//
    // Iterate through every operation, if any variable has !vir.vec<?xT> type
    // without scaling factor, then define the element type of this type as
    // anchor type.
    Type anchorType = nullptr;

    Region &region = op.getRegion();
    Block &block = region.front();

    for (Operation &innerOp : block) {
      // Check all operands of the operation.
      for (Value operand : innerOp.getOperands()) {
        Type operandType = operand.getType();
        if (auto dynVecType = dyn_cast<vir::DynamicVectorType>(operandType)) {
          // Check if there is a scaling factor.
          if (!dynVecType.getScalingFactor()) {
            // No scaling factor, define the element type as anchor type.
            anchorType = dynVecType.getElementType();
            break;
          }
        }
      }

      // Check all results of the operation.
      for (Value result : innerOp.getResults()) {
        Type resultType = result.getType();
        if (auto dynVecType = dyn_cast<vir::DynamicVectorType>(resultType)) {
          // Check if there is a scaling factor.
          if (!dynVecType.getScalingFactor()) {
            // No scaling factor, define the element type as anchor type.
            anchorType = dynVecType.getElementType();
            break;
          }
        }
      }

      // If anchor type is found, we can exit the loop early.
      if (anchorType) {
        break;
      }
    }

    // Check if anchor type was found
    if (!anchorType) {
      return op.emitError(
          "No anchor type found in the dynamic vector region. "
          "Ensure there are operations with !vir.vec<?xT> types "
          "without scaling factors.");
    }

    //===------------------------------------------------------------------===//
    // Step 2: Construct target vector type based on pass options.
    //===------------------------------------------------------------------===//

    Type targetVectorType;
    if (useScalable) {
      // Scalable Vector Type: e.g. `vector<[4]xf32>`.
      SmallVector<bool, 1> scalableDim = {true};
      targetVectorType =
          VectorType::get({vectorWidth}, anchorType, scalableDim);
    } else {
      // Fixed Vector Type: e.g. `vector<4xf32>`.
      targetVectorType = VectorType::get({vectorWidth}, anchorType);
    }

    //===------------------------------------------------------------------===//
    // Step 3: Calculate loop bounds and create vectorization loop.
    //===------------------------------------------------------------------===//
    // Get the vector length value from SetVLOp.
    Value vlValue = op.getVl();

    // Calculate the upper bound for vectorized loop.
    // vl_upbound = (vl_total - vl_step) + 1
    Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value vlUpboundPat = rewriter.create<arith::SubIOp>(loc, vlValue, vlStep);
    Value vlUpbound = rewriter.create<arith::AddIOp>(
        loc, vlUpboundPat, rewriter.create<arith::ConstantIndexOp>(loc, 1));

    // Create identity affine map (d0) -> d0 for dynamic bounds.
    MLIRContext *ctx = rewriter.getContext();
    AffineMap identityMap =
        AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)}, ctx);

    // Create constant 0 for lower bound.
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Create affine for loop with iteration variable.
    auto affineForOp = rewriter.create<affine::AffineForOp>(
        loc, ValueRange{zero}, identityMap, // Lower bound: 0
        ValueRange{vlUpbound}, identityMap, // Upper bound: vlUpbound
        vectorWidth,                        // Step size
        /*iterArgs=*/ValueRange{zero},      // Initial iteration value: 0
        [&](OpBuilder &rewriter, Location loc, Value iv, ValueRange iterArgs) {
          //===--------------------------------------------------------------===//
          // Step 4: Convert operations inside the dynamic vector region.
          //===--------------------------------------------------------------===//
          // Symbol table: maps dynamic vector values to fixed / scalable vector
          // values.
          DenseMap<Value, Value> symbolTable;

          // Pre-allocate symbol table capacity for better performance
          symbolTable.reserve(block.getOperations().size());

          // Iterate through all operations in the block.
          for (Operation &innerOp : block) {
            if (isa<vir::LoadOp>(innerOp)) {
              // Convert `vir::LoadOp` to `vector::LoadOp`.
              auto virLoadOp = cast<vir::LoadOp>(innerOp);
              auto base = virLoadOp.getBase();
              Value baseOffset;
              auto firstIndex = virLoadOp.getIndices().front();
              // Calculate base offset based on index value.
              if (auto constOp =
                      firstIndex.getDefiningOp<arith::ConstantIndexOp>()) {
                if (constOp.value() == 0) {
                  // If index is 0, use loop induction variable directly.
                  baseOffset = iv;
                } else {
                  // If index is non-zero constant, add it to loop induction
                  // variable.
                  baseOffset =
                      rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
                }
              } else {
                // If index is not a constant, add it to loop induction
                // variable.
                baseOffset =
                    rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
              }

              auto vectorLoadOp = rewriter.create<vector::LoadOp>(
                  loc, targetVectorType, base, baseOffset);
              // Record result mapping in symbol table.
              symbolTable[virLoadOp.getResult()] = vectorLoadOp.getResult();
            } else if (isa<vir::StoreOp>(innerOp)) {
              // Convert `vir::StoreOp` to `vector::StoreOp`.
              auto virStoreOp = cast<vir::StoreOp>(innerOp);
              auto valueToStoreOrig = virStoreOp.getValue();
              auto valueToStore = symbolTable[valueToStoreOrig];
              auto base = virStoreOp.getBase();
              Value baseOffset;
              auto firstIndex = virStoreOp.getIndices().front();
              // Calculate base offset based on index value.
              if (auto constOp =
                      firstIndex.getDefiningOp<arith::ConstantIndexOp>()) {
                if (constOp.value() == 0) {
                  // If index is 0, use loop induction variable directly.
                  baseOffset = iv;
                } else {
                  // If index is non-zero constant, add it to loop induction
                  // variable.
                  baseOffset =
                      rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
                }
              } else {
                // If index is not a constant, add it to loop induction
                // variable.
                baseOffset =
                    rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
              }

              rewriter.create<vector::StoreOp>(loc, valueToStore, base,
                                               baseOffset);
            } else if (isa<vir::ConstantOp>(innerOp)) {
              // Convert `vir::ConstantOp` to `arith::ConstantOp` with vector
              // type.
              auto virConstOp = cast<vir::ConstantOp>(innerOp);
              auto constValue = virConstOp.getValue();

              // Create vector constant using DenseElementsAttr
              auto vectorConstAttr = DenseElementsAttr::get(
                  cast<ShapedType>(targetVectorType), constValue);
              auto vectorConstOp = rewriter.create<arith::ConstantOp>(
                  loc, targetVectorType, vectorConstAttr);

              // Record result mapping in symbol table.
              symbolTable[virConstOp.getResult()] = vectorConstOp.getResult();
            } else if (isa<vir::BroadcastOp>(innerOp)) {
              // Convert `vir::BroadcastOp` to `vector::BroadcastOp`.
              auto virBroadcastOp = cast<vir::BroadcastOp>(innerOp);
              auto scalarValue = virBroadcastOp.getValue();

              // Create vector broadcast to convert scalar to vector
              auto vectorBroadcastOp = rewriter.create<vector::BroadcastOp>(
                  loc, targetVectorType, scalarValue);

              // Record result mapping in symbol table.
              symbolTable[virBroadcastOp.getResult()] =
                  vectorBroadcastOp.getResult();
            } else if (isa<vir::FMAOp>(innerOp)) {
              // Convert `vir::FMAOp` to `vector::FMAOp`.
              auto virFMAOp = cast<vir::FMAOp>(innerOp);
              auto lhsValue = symbolTable[virFMAOp.getLhs()];
              auto rhsValue = symbolTable[virFMAOp.getRhs()];
              auto accValue = symbolTable[virFMAOp.getAcc()];

              // Create vector FMA operation
              auto vectorFMAOp = rewriter.create<vector::FMAOp>(
                  loc, lhsValue, rhsValue, accValue);

              // Record result mapping in symbol table.
              symbolTable[virFMAOp.getResult()] = vectorFMAOp.getResult();
            } else {
              // Emit warning for unsupported operations.
              emitWarning(loc, "Unsupported operation: " +
                                   innerOp.getName().getStringRef());
            }
          }

          // Calculate next iteration value: `i_next = i + vl_step`.
          Value iNext = rewriter.create<arith::AddIOp>(loc, iv, vlStep);

          // Yield the next iteration value.
          rewriter.create<mlir::affine::AffineYieldOp>(loc, iNext);
        });

    // Get the final iteration value from the affine loop.
    Value finalIterValue = affineForOp.getResult(0);

    //===------------------------------------------------------------------===//
    // Step 5: Create tail loop for remaining elements.
    //===------------------------------------------------------------------===//
    // Process the remainder of the elements with scalar operations.
    rewriter.create<affine::AffineForOp>(
        loc, ValueRange{finalIterValue},
        identityMap,                      // Start from final iter value
        ValueRange{vlValue}, identityMap, // End at vlValue
        1,                                // Step by 1
        /*iterArgs=*/std::nullopt,
        [&](OpBuilder &rewriter, Location loc, Value iv, ValueRange iterArgs) {
          //===------------------------------------------------------------===//
          // Step 6: Convert operations to scalar operations for tail loop.
          //===------------------------------------------------------------===//
          // Symbol table: maps dynamic vector values to scalar values.
          DenseMap<Value, Value> symbolTable;

          // Pre-allocate symbol table capacity for better performance
          symbolTable.reserve(block.getOperations().size());

          for (Operation &innerOp : block) {
            if (isa<vir::LoadOp>(innerOp)) {
              // Convert `vir::LoadOp` to `memref::LoadOp` (scalar).
              auto virLoadOp = cast<vir::LoadOp>(innerOp);
              auto base = virLoadOp.getBase();
              Value baseOffset;
              auto firstIndex = virLoadOp.getIndices().front();

              // Calculate base offset based on index value.
              if (auto constOp =
                      firstIndex.getDefiningOp<arith::ConstantIndexOp>()) {
                if (constOp.value() == 0) {
                  baseOffset = iv;
                } else {
                  baseOffset =
                      rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
                }
              } else {
                baseOffset =
                    rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
              }

              auto memrefLoadOp =
                  rewriter.create<memref::LoadOp>(loc, base, baseOffset);
              symbolTable[virLoadOp.getResult()] = memrefLoadOp.getResult();
            } else if (isa<vir::StoreOp>(innerOp)) {
              // Convert `vir::StoreOp` to `memref::StoreOp` (scalar).
              auto virStoreOp = cast<vir::StoreOp>(innerOp);
              auto valueToStoreOrig = virStoreOp.getValue();
              auto valueToStore = symbolTable[valueToStoreOrig];
              auto base = virStoreOp.getBase();
              Value baseOffset;
              auto firstIndex = virStoreOp.getIndices().front();

              // Calculate base offset based on index value.
              if (auto constOp =
                      firstIndex.getDefiningOp<arith::ConstantIndexOp>()) {
                if (constOp.value() == 0) {
                  baseOffset = iv;
                } else {
                  baseOffset =
                      rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
                }
              } else {
                baseOffset =
                    rewriter.create<arith::AddIOp>(loc, iv, firstIndex);
              }

              rewriter.create<memref::StoreOp>(loc, valueToStore, base,
                                               baseOffset);
            } else if (isa<vir::ConstantOp>(innerOp)) {
              // Convert `vir::ConstantOp` to `arith::ConstantOp` (scalar).
              auto virConstOp = cast<vir::ConstantOp>(innerOp);
              auto constValue = virConstOp.getValue();

              // Create arith constant with the scalar value
              auto arithConstOp =
                  rewriter.create<arith::ConstantOp>(loc, constValue);

              // Record result mapping in symbol table.
              symbolTable[virConstOp.getResult()] = arithConstOp.getResult();
            } else if (isa<vir::BroadcastOp>(innerOp)) {
              // Convert `vir::BroadcastOp` to scalar value
              // (no broadcast needed in tail loop).
              auto virBroadcastOp = cast<vir::BroadcastOp>(innerOp);
              auto scalarValue = virBroadcastOp.getValue();

              // In tail loop, we just use the scalar value directly
              symbolTable[virBroadcastOp.getResult()] = scalarValue;
            } else if (isa<vir::FMAOp>(innerOp)) {
              // Convert `vir::FMAOp` to scalar FMA operation in tail loop.
              auto virFMAOp = cast<vir::FMAOp>(innerOp);
              auto lhsValue = symbolTable[virFMAOp.getLhs()];
              auto rhsValue = symbolTable[virFMAOp.getRhs()];
              auto accValue = symbolTable[virFMAOp.getAcc()];

              // Create scalar FMA operation using arith::AddFOp and
              // arith::MulFOp since we need to handle scalar operations in tail
              // loop
              auto mulResult =
                  rewriter.create<arith::MulFOp>(loc, lhsValue, rhsValue);
              auto addResult =
                  rewriter.create<arith::AddFOp>(loc, mulResult, accValue);

              // Record result mapping in symbol table.
              symbolTable[virFMAOp.getResult()] = addResult;
            } else {
              // Emit warning for unsupported operations.
              emitWarning(loc, "Unsupported operation: " +
                                   innerOp.getName().getStringRef());
            }
          }

          rewriter.create<affine::AffineYieldOp>(loc);
        });

    // Erase the original `SetVLOp`.
    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVIRToVectorConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           bool useScalable, int vectorWidth) {
  // clang-format off
  patterns.add<VIRSetVLLowering>(converter, useScalable, vectorWidth);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// VIRToVectorPass
//===----------------------------------------------------------------------===//

namespace {
class VIRToVectorPass
    : public PassWrapper<VIRToVectorPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VIRToVectorPass)
  VIRToVectorPass() = default;
  VIRToVectorPass(const VIRToVectorPass &) {}

  StringRef getArgument() const final { return "lower-vir-to-vector"; }
  StringRef getDescription() const final {
    return "Lower VIR Dialect to Vector Dialect.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        arith::ArithDialect,
        buddy::vir::VIRDialect,
        func::FuncDialect,
        memref::MemRefDialect,
        affine::AffineDialect,
        vector::VectorDialect,
        LLVM::LLVMDialect>();
    // clang-format on
  }

  // Pass options
  Option<bool> useScalable{
      *this, "use-scalable",
      llvm::cl::desc("Use scalable vectors instead of fixed vectors"),
      llvm::cl::init(DEFAULT_USE_SCALABLE)};

  Option<int> vectorWidth{
      *this, "vector-width",
      llvm::cl::desc("Vector width for fixed/scalable vectors (default: 4)"),
      llvm::cl::init(DEFAULT_VECTOR_WIDTH)};
};
} // end anonymous namespace.

void VIRToVectorPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMConversionTarget target(*context);
  // clang-format off
  target.addLegalDialect<
      arith::ArithDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      vector::VectorDialect,
      affine::AffineDialect,
      LLVM::LLVMDialect
    >();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  populateVIRToVectorConversionPatterns(converter, patterns, useScalable,
                                        vectorWidth);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerVIRToVectorPass() { PassRegistration<VIRToVectorPass>(); }
} // namespace buddy
} // namespace mlir
