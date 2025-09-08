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
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

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

  /// @brief Recursively search for anchor types within a Region and its nested
  /// Regions
  /// @param region The region to search for
  /// @param anchorType Used to store references of found anchor types
  /// @return If the anchor type is found, return true
  bool findAnchorTypeRecursive(Region &region, Type &anchorType) const {
    if (anchorType)
      return true; // If found, exit early

    for (Block &block : region) {
      for (Operation &op : block) {
        // Check operands and results to find dynamic vector types
        auto checkValue = [&](Value value) {
          if (auto dynVecType =
                  dyn_cast<vir::DynamicVectorType>(value.getType())) {
            if (!dynVecType.getScalingFactor()) {
              anchorType = dynVecType.getElementType();
              return true;
            }
          }
          return false;
        };

        for (Value operand : op.getOperands())
          if (checkValue(operand))
            return true;
        for (Value result : op.getResults())
          if (checkValue(result))
            return true;

        // Recursively enter any region included in this operation
        for (Region &innerRegion : op.getRegions()) {
          if (findAnchorTypeRecursive(innerRegion, anchorType))
            return true;
        }
      }
    }
    return false;
  }

  /// @brief Recursively reducing operations within a block
  /// @param builder OpBuilder for creating new operations
  /// @param loc The current location
  /// @param block  Block to lower
  /// @param symbolTable Symbol table, used to map old values to new values
  /// @param mainLoopIV Inductive variable (IV) of the main quantization loop
  /// @param targetVectorType The target vector type can be null ptr for tail
  /// loops
  /// @param isTailLoop Flag, indicating whether the tail loop is currently
  /// being processed (generating scalar code)
  void lowerBlock(OpBuilder &builder, Location loc, Block *block,
                  DenseMap<Value, Value> &symbolTable, Value mainLoopIV,
                  Type targetVectorType, bool isTailLoop) const {
    // find mapped local Value from symbolTable or global Value
    auto findValue = [&symbolTable](Value srcValue) {
      if (symbolTable.contains(srcValue)) {
        return symbolTable.lookup(srcValue);
      }
      return srcValue;
    };

    for (Operation &innerOp : block->without_terminator()) {
      llvm::TypeSwitch<Operation *>(&innerOp)
          .Case<vir::LoadOp>([&](vir::LoadOp op) {
            Value base = op.getBase();
            // the lastest index is dynamic and global
            Value lastIndex = op.getIndices().back();
            Value baseOffset =
                builder.create<arith::AddIOp>(loc, mainLoopIV, lastIndex);
            auto sourceIndices = op.getIndices();
            SmallVector<Value> destIndices{};
            // collect indices for load op
            for (size_t i = 0; i < sourceIndices.size() - 1; i++) {
              auto srcOp = sourceIndices[i];
              Value destOp = findValue(srcOp);
              destIndices.push_back(destOp);
            }
            destIndices.push_back(baseOffset);

            if (isTailLoop) {
              auto memrefLoadOp =
                  builder.create<memref::LoadOp>(loc, base, destIndices);
              symbolTable[op.getResult()] = memrefLoadOp.getResult();
            } else {
              auto vectorLoadOp = builder.create<vector::LoadOp>(
                  loc, targetVectorType, base, destIndices);
              symbolTable[op.getResult()] = vectorLoadOp.getResult();
            }
          })
          .Case<vir::StoreOp>([&](vir::StoreOp op) {
            Value valueToStore = findValue(op.getValue());
            if (!valueToStore) {
              op.emitError("value to store not found in symbol table");
              return;
            }
            Value base = op.getBase();
            // the lastest index is dynamic and global
            Value lastIndex = op.getIndices().back();
            Value baseOffset =
                builder.create<arith::AddIOp>(loc, mainLoopIV, lastIndex);
            auto sourceIndices = op.getIndices();
            SmallVector<Value> destIndices{};
            // collect indices for load op
            for (size_t i = 0; i < sourceIndices.size() - 1; i++) {
              auto srcOp = sourceIndices[i];
              Value destOp = findValue(srcOp);
              destIndices.push_back(destOp);
            }
            destIndices.push_back(baseOffset);

            if (isTailLoop) {
              builder.create<memref::StoreOp>(loc, valueToStore, base,
                                              destIndices);
            } else {
              builder.create<vector::StoreOp>(loc, valueToStore, base,
                                              destIndices);
            }
          })
          .Case<vir::ConstantOp>([&](vir::ConstantOp op) {
            auto constValue = op.getValue();
            if (isTailLoop) {
              auto arithConstOp =
                  builder.create<arith::ConstantOp>(loc, constValue);
              symbolTable[op.getResult()] = arithConstOp.getResult();
            } else {
              auto vectorConstAttr = DenseElementsAttr::get(
                  cast<ShapedType>(targetVectorType), constValue);
              auto vectorConstOp = builder.create<arith::ConstantOp>(
                  loc, targetVectorType, vectorConstAttr);
              symbolTable[op.getResult()] = vectorConstOp.getResult();
            }
          })
          .Case<vir::BroadcastOp>([&](vir::BroadcastOp op) {
            Value scalarValue = findValue(op.getValue());
            if (isTailLoop) {
              symbolTable[op.getResult()] = scalarValue;
            } else {
              auto vectorBroadcastOp = builder.create<vector::BroadcastOp>(
                  loc, targetVectorType, scalarValue);
              symbolTable[op.getResult()] = vectorBroadcastOp.getResult();
            }
          })
          .Case<vir::FMAOp>([&](vir::FMAOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            Value acc = findValue(op.getAcc());
            if (!lhs || !rhs || !acc) {
              op.emitError("FMA operands not found in symbol table");
              return;
            }
            if (isTailLoop) {
              auto mulResult = builder.create<arith::MulFOp>(loc, lhs, rhs);
              auto addResult =
                  builder.create<arith::AddFOp>(loc, mulResult, acc);
              symbolTable[op.getResult()] = addResult;
            } else {
              auto vectorFMAOp =
                  builder.create<vector::FMAOp>(loc, lhs, rhs, acc);
              symbolTable[op.getResult()] = vectorFMAOp.getResult();
            }
          })
          .Case<affine::AffineForOp>([&](affine::AffineForOp op) {
            // Find the corresponding values of the initial iteration parameters
            // of the loop in the symbol table
            SmallVector<Value> initialIterArgs;
            for (Value arg : op.getInits()) {
              Value mappedArg = findValue(arg);
              if (!mappedArg) {
                op.emitError("initial iter_arg not found in symbol table");
                return;
              }
              initialIterArgs.push_back(mappedArg);
            }

            // create a new affine.for
            auto newForOp = builder.create<affine::AffineForOp>(
                loc, ValueRange(op.getLowerBoundOperands()),
                op.getLowerBoundMap(), ValueRange(op.getUpperBoundOperands()),
                op.getUpperBoundMap(), op.getStep().getLimitedValue(),
                ValueRange(initialIterArgs),
                [&](OpBuilder &b, Location bodyLoc, Value iv,
                    ValueRange iterArgs) {
                  // Create a new internal symbol table within the loop body
                  DenseMap<Value, Value> innerSymbolTable(symbolTable);

                  // Mapping Inductive Variable
                  innerSymbolTable[op.getInductionVar()] = iv;

                  // Map loop iteration parameters
                  for (auto it : llvm::zip(op.getRegionIterArgs(), iterArgs)) {
                    innerSymbolTable[std::get<0>(it)] = std::get<1>(it);
                  }

                  // Recursive reduction of blocks inside the loop body
                  lowerBlock(b, bodyLoc, op.getBody(), innerSymbolTable,
                             mainLoopIV, targetVectorType, isTailLoop);

                  // Process the termination symbol (affine. field) of the loop
                  // body
                  Operation *terminator = op.getBody()->getTerminator();
                  if (auto yieldOp =
                          dyn_cast<affine::AffineYieldOp>(terminator)) {
                    SmallVector<Value> yieldOperands;
                    for (Value operand : yieldOp.getOperands()) {
                      Value mappedOperand = innerSymbolTable.lookup(operand);
                      if (!mappedOperand) {
                        terminator->emitError(
                            "yield operand not found in inner symbol table");
                        return;
                      }
                      yieldOperands.push_back(mappedOperand);
                    }
                    b.create<affine::AffineYieldOp>(bodyLoc, yieldOperands);
                  }
                });

            // Register the results of the newly created loop in the external
            // symbol table
            for (auto it : llvm::zip(op.getResults(), newForOp.getResults())) {
              symbolTable[std::get<0>(it)] = std::get<1>(it);
            }
          })
          .Case<affine::AffineYieldOp, vector::YieldOp>([&](Operation *) {
            // During recursive descent, these terminators are handled by their
            // parent operations (such as affine. for). We will ignore them
            // directly here.
          })
          .Case<memref::LoadOp>([&](memref::LoadOp op) {
            // This operation is scalar loading and should be copied as is, but
            // its operands may need to be remapped from the symbol table.
            Value base = findValue(op.getMemRef());
            if (!base)
              base = op.getMemRef(); // If not mapped, use the original value

            SmallVector<Value> indices;
            for (Value index : op.getIndices()) {
              Value mappedIndex = findValue(index);
              indices.push_back(mappedIndex);
            }

            auto newLoadOp = builder.create<memref::LoadOp>(loc, base, indices);
            symbolTable[op.getResult()] = newLoadOp.getResult();
          })
          .Default([&](Operation *op) {
            emitWarning(loc, "Unsupported operation during lowering: " +
                                 op->getName().getStringRef());
          });
    }
  }

  LogicalResult
  matchAndRewrite(vir::SetVLOp op, vir::SetVLOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    //===------------------------------------------------------------------===//
    // Step 1: Recursive search for anchor type.
    //===------------------------------------------------------------------===//
    Type anchorType = nullptr;
    Region &region = op.getRegion();
    if (!findAnchorTypeRecursive(region, anchorType) || !anchorType) {
      return op.emitError(
          "No anchor type found in the dynamic vector region. "
          "Ensure there are operations with !vir.vec<?xT> types "
          "without scaling factors.");
    }

    //===------------------------------------------------------------------===//
    // Step 2: Construct the target vector type.
    //===------------------------------------------------------------------===//
    Type targetVectorType;
    if (useScalable) {
      SmallVector<bool, 1> scalableDim = {true};
      targetVectorType =
          VectorType::get({vectorWidth}, anchorType, scalableDim);
    } else {
      targetVectorType = VectorType::get({vectorWidth}, anchorType);
    }

    //===------------------------------------------------------------------===//
    // Step 3: Create a main vectorization loop.
    //===------------------------------------------------------------------===//
    Value vlValue = op.getVl();
    // Calculate the upper bound for vectorized loop.
    // vl_upbound = (vl_total - vl_step) + 1
    Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value vlUpboundPat = rewriter.create<arith::SubIOp>(loc, vlValue, vlStep);
    Value vlUpbound = rewriter.create<arith::AddIOp>(
        loc, vlUpboundPat, rewriter.create<arith::ConstantIndexOp>(loc, 1));

    // To avoid possible overflow calculations such as' (vl-1)/step * step+step
    // ', We directly set the upper bound of the loop to vl, and then process
    // the remainder in the tail loop. The number of iterations in the main loop
    // is vl / step
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto mainloop = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/ValueRange{zero}, rewriter.getDimIdentityMap(),
        /*upperBound=*/ValueRange{vlUpbound}, rewriter.getDimIdentityMap(),
        vectorWidth,
        /*iterArgs=*/std::nullopt,
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          DenseMap<Value, Value> symbolTable;
          // Initiate recursive descent process on top-level blocks (vectorized
          // mode)
          lowerBlock(b, bodyLoc, &region.front(), symbolTable, iv,
                     targetVectorType, /*isTailLoop=*/false);
          b.create<affine::AffineYieldOp>(bodyLoc);
        });

    //===------------------------------------------------------------------===//
    // Step 4: Create a tail loop to process the remaining elements.
    //===------------------------------------------------------------------===//
    // The starting point of the tail loop is where the main loop ends.
    // vl - (vl % step)
    Value vlRem = rewriter.create<arith::RemSIOp>(loc, vlValue, vlStep);
    Value tailStart = rewriter.create<arith::SubIOp>(loc, vlValue, vlRem);

    rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/ValueRange{tailStart}, rewriter.getDimIdentityMap(),
        /*upperBound=*/ValueRange{vlValue}, rewriter.getDimIdentityMap(),
        /*step=*/1,
        /*iterArgs=*/std::nullopt,
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          DenseMap<Value, Value> symbolTable;
          // Initiate recursive descent process on top-level blocks (scalar
          // mode)
          lowerBlock(b, bodyLoc, &region.front(), symbolTable, iv,
                     /*targetVectorType=*/nullptr, /*isTailLoop=*/true);
          b.create<affine::AffineYieldOp>(bodyLoc);
        });

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
