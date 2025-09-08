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
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cstdint>

// Include ARM SEV header for detecting register base width
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

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

VectorType vectorTypeHelper(Region &srcRegion, Type anchorType,
                            bool useScalable, bool verbose) {
#define VERBOSE_ANALYSIS_INFO(info)                                            \
  if (verbose) {                                                               \
    llvm::outs() << info;                                                      \
  }
  int vectorRegBitWidth = 1;
  int totalVectorRegs = 1;
  llvm::StringMap<bool> Features = llvm::sys::getHostCPUFeatures();

  VERBOSE_ANALYSIS_INFO("--- Checking x86/x64 Features ---\n");
  if (Features.count("avx2") && Features["avx2"]) {
    vectorRegBitWidth = 256;
    totalVectorRegs = 16;
    VERBOSE_ANALYSIS_INFO("[+] AVX2 is supported.\n");
  } else {
    VERBOSE_ANALYSIS_INFO("[-] AVX2 is NOT supported.\n");
  }

  if (Features.count("avx512f") && Features["avx512f"]) {
    vectorRegBitWidth = 512;
    totalVectorRegs = 32;
    VERBOSE_ANALYSIS_INFO("[+] AVX-512F is supported.\n");
  } else {
    VERBOSE_ANALYSIS_INFO("[-] AVX-512F is NOT supported.\n");
  }

  VERBOSE_ANALYSIS_INFO("\n--- Checking AArch64 Features ---\n");
  if (Features.count("sve") && Features["sve"]) {
    VERBOSE_ANALYSIS_INFO("[+] SVE is supported.\n");
#if defined(__ARM_FEATURE_SVE)
    uint64_t bytes = svcntb();
    vectorRegBitWidth = bytes * 8;
    totalVectorRegs = 32;
#else
    assert(
        false &&
        "LLVM detected ARM SVE, but MACRO __ARM_FEATURE_SVE is not defined.");
#endif
  } else {
    VERBOSE_ANALYSIS_INFO("[-] SVE is NOT supported.\n");
  }

  Block &block = srcRegion.front();
  auto &operations = block.getOperations();

  VERBOSE_ANALYSIS_INFO(
      "\n--- Running vectorTypeHelper Dataflow Analysis ---\n");

  // Data Flow analysis
  // USE, DEF, IN, OUT sets
  DenseMap<Operation *, DenseSet<Value>> useSets, defSets, inSets, outSets;

  // calculate USE and DEF sets
  for (Operation &op : operations) {
    for (Value result : op.getResults()) {
      if (auto st = dyn_cast<ShapedType>(result.getType());
          st && st.hasRank()) {
        defSets[&op].insert(result);
      }
    }
    for (Value operand : op.getOperands()) {
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->getParentRegion() == &srcRegion) {
        if (auto st = dyn_cast<ShapedType>(operand.getType());
            st && st.hasRank()) {
          useSets[&op].insert(operand);
        }
      }
    }
  }

  auto printValueSet = [&verbose](llvm::StringRef name,
                                  const DenseSet<Value> &valueSet) {
    VERBOSE_ANALYSIS_INFO("    " << name << " { ");
    for (Value val : valueSet) {
      if (verbose) {
        val.print(llvm::outs(), OpPrintingFlags().printGenericOpForm());
      }
      VERBOSE_ANALYSIS_INFO(" ");
    }
    VERBOSE_ANALYSIS_INFO("}\n");
  };

  bool changed = true;
  int iteration = 1;
  while (changed) {
    VERBOSE_ANALYSIS_INFO("\n--- Analysis Iteration " << iteration++
                                                      << " ---\n");
    changed = false;

    // backward iteration
    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
      Operation &op = *it;

      VERBOSE_ANALYSIS_INFO("Analyzing Op: ");
      if (verbose) {
        op.print(llvm::outs(), OpPrintingFlags().printGenericOpForm());
      }
      VERBOSE_ANALYSIS_INFO("\n");

      DenseSet<Value> oldIn = inSets[&op];

      // OUT(i) = ∪_{j ∈ Succ(i)} IN(j)
      DenseSet<Value> outSet;
      Operation *nextOp = op.getNextNode();
      if (nextOp) {
        outSet = inSets[nextOp];
      }
      outSets[&op] = outSet;
      printValueSet("OUT:", outSets[&op]);
      printValueSet("DEF:", defSets[&op]);
      printValueSet("USE:", useSets[&op]);

      // IN(i) = (OUT(i) - DEF(i)) ∪ USE(i)
      DenseSet<Value> inSet = outSet;
      for (Value defValue : defSets[&op]) {
        inSet.erase(defValue);
      }
      inSet.insert(useSets[&op].begin(), useSets[&op].end());
      inSets[&op] = inSet;
      printValueSet("New IN:", inSets[&op]);

      if (inSets[&op] != oldIn) {
        changed = true;
      }
    }
  }

  VERBOSE_ANALYSIS_INFO("\n--- Dataflow Analysis Converged ---\n");

  // Req_Group = max_i ( |IN(i) ∪ OUT(i)| )
  unsigned int maxActiveVectors = 0;
  for (Operation &op : operations) {
    DenseSet<Value> liveAcross = inSets[&op];
    liveAcross.insert(outSets[&op].begin(), outSets[&op].end());

    if (liveAcross.size() > maxActiveVectors) {
      maxActiveVectors = liveAcross.size();
    }
  }

  int reqGroup = (maxActiveVectors == 0) ? 1 : maxActiveVectors;
  VERBOSE_ANALYSIS_INFO("Final Max Active Vectors (Req_Group): " << reqGroup
                                                                 << "\n");

  VERBOSE_ANALYSIS_INFO("\n--- Calculating Final Vector Type ---\n");

  unsigned elementBitWidth = anchorType.getIntOrFloatBitWidth();
  if (elementBitWidth == 0) {
    srcRegion.getParentOp()->emitError(
        "Anchor type is not a valid integer or float type.");
    return nullptr;
  }
  VERBOSE_ANALYSIS_INFO("Element Bit Width (Width_Ele): " << elementBitWidth
                                                          << "\n");

  // Size_Group = floor(Num_Reg / Req_Group)
  int sizeGroup = totalVectorRegs / reqGroup;
  if (sizeGroup == 0) {
    emitWarning(srcRegion.getLoc(),
                "High register pressure detected. Calculated vector size might "
                "be suboptimal.");
    sizeGroup = 1;
  }
  VERBOSE_ANALYSIS_INFO("Calculated Size_Group = floor("
                        << totalVectorRegs << " / " << reqGroup
                        << ") = " << sizeGroup << "\n");

  // Size_vec = (Size_Group * Size_Reg) / Width_Ele
  int64_t sizeVec = (sizeGroup * vectorRegBitWidth) / elementBitWidth;
  VERBOSE_ANALYSIS_INFO("Calculated Size_vec = ("
                        << sizeGroup << " * " << vectorRegBitWidth << ") / "
                        << elementBitWidth << " = " << sizeVec << "\n");

  return VectorType::get({sizeVec}, anchorType);
#undef VERBOSE_ANALYSIS_INFO
} // namespace

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
                   int vectorWidth, bool verbose, bool useCustomVecWid)
      : ConvertOpToLLVMPattern<vir::SetVLOp>(converter),
        useScalable(useScalable), vectorWidth(vectorWidth), verbose(verbose),
        useCustomVecWid(useCustomVecWid) {}

private:
  bool useScalable;
  int vectorWidth;
  bool verbose;
  bool useCustomVecWid;

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
    int vectorWid = vectorWidth; // Dynamic defined vector width
    VectorType targetVectorType;
    if (useCustomVecWid) {
      if (verbose) {
        llvm::outs() << "Use user custom vector width = " << vectorWidth
                     << "\n";
      }
      if (useScalable) {
        // Scalable Vector Type: e.g. `vector<[4]xf32>`.
        SmallVector<bool, 1> scalableDim = {true};
        targetVectorType =
            VectorType::get({vectorWidth}, anchorType, scalableDim);
      } else {
        // Fixed Vector Type: e.g. `vector<4xf32>`.
        targetVectorType = VectorType::get({vectorWidth}, anchorType);
      }
    } else {
      targetVectorType =
          vectorTypeHelper(region, anchorType, useScalable, verbose);
      vectorWid = int(targetVectorType.getShape()[0]);
    }

    //===------------------------------------------------------------------===//
    // Step 3: Calculate loop bounds and create vectorization loop.
    //===------------------------------------------------------------------===//
    // Get the vector length value from SetVLOp.
    Value vlValue = op.getVl();

    // Calculate the upper bound for vectorized loop.
    // vl_upbound = (vl_total - vl_step) + 1
    Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWid);
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
        vectorWid,                          // Step size
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
                                           bool useScalable, int vectorWidth,
                                           bool verbose, bool useCustomVecWid) {
  // clang-format off
  patterns.add<VIRSetVLLowering>(converter, useScalable, vectorWidth, verbose, useCustomVecWid);
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

  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print register analysis information"),
                       llvm::cl::init(false)};
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
  // use custome vector width
  bool useCustomVecWid = vectorWidth.getNumOccurrences() > 0;
  populateVIRToVectorConversionPatterns(converter, patterns, useScalable,
                                        vectorWidth, verbose, useCustomVecWid);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerVIRToVectorPass() { PassRegistration<VIRToVectorPass>(); }
} // namespace buddy
} // namespace mlir
