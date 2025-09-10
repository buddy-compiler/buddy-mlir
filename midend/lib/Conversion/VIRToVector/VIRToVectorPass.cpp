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

  /// Recursively search for anchor type within a given Region and its nested
  Type findAnchorTypeRecursive(Region &region) const {
    // Check operands and results to find dynamic vector types
    auto checkValue = [&](Value value) {
      auto dynVecType = dyn_cast<vir::DynamicVectorType>(value.getType());
      if (dynVecType) {
        if (!dynVecType.getScalingFactor()) {
          return dynVecType.getElementType();
        }
      }
      return Type{};
    };
    Type anchorType;
    for (Block &block : region) {
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          anchorType = checkValue(result);
          if (anchorType) {
            return anchorType;
          }
        }
        // Recursively enter any region included in this operation
        for (Region &innerRegion : op.getRegions()) {
          anchorType = findAnchorTypeRecursive(innerRegion);
          if (anchorType) {
            return anchorType;
          }
        }
      }
    }
    return Type{};
  }

  /// Recursively lower operations within a given block
  void lowerBlock(OpBuilder &builder, Location loc, Block *block,
                  DenseMap<Value, Value> &virSymbolTable, Value mainLoopIV,
                  Type targetVectorType, bool isTailLoop) const {
    // find mapped local Value from virSymbolTable or global Value
    auto findValue = [&virSymbolTable](Value srcValue) {
      if (virSymbolTable.contains(srcValue)) {
        return virSymbolTable.lookup(srcValue);
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
            for (size_t i = 0, e = sourceIndices.size() - 1; i < e; ++i) {
              auto srcOp = sourceIndices[i];
              Value destOp = findValue(srcOp);
              destIndices.push_back(destOp);
            }
            destIndices.push_back(baseOffset);

            if (isTailLoop) {
              auto memrefLoadOp =
                  builder.create<memref::LoadOp>(loc, base, destIndices);
              virSymbolTable[op.getResult()] = memrefLoadOp.getResult();
            } else {
              auto vectorLoadOp = builder.create<vector::LoadOp>(
                  loc, targetVectorType, base, destIndices);
              virSymbolTable[op.getResult()] = vectorLoadOp.getResult();
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
            for (size_t i = 0, e = sourceIndices.size() - 1; i < e; ++i) {
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
              virSymbolTable[op.getResult()] = arithConstOp.getResult();
            } else {
              auto vectorConstAttr = DenseElementsAttr::get(
                  cast<ShapedType>(targetVectorType), constValue);
              auto vectorConstOp = builder.create<arith::ConstantOp>(
                  loc, targetVectorType, vectorConstAttr);
              virSymbolTable[op.getResult()] = vectorConstOp.getResult();
            }
          })
          .Case<vir::BroadcastOp>([&](vir::BroadcastOp op) {
            Value scalarValue = findValue(op.getValue());
            if (isTailLoop) {
              virSymbolTable[op.getResult()] = scalarValue;
            } else {
              auto vectorBroadcastOp = builder.create<vector::BroadcastOp>(
                  loc, targetVectorType, scalarValue);
              virSymbolTable[op.getResult()] = vectorBroadcastOp.getResult();
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
              virSymbolTable[op.getResult()] = addResult;
            } else {
              auto vectorFMAOp =
                  builder.create<vector::FMAOp>(loc, lhs, rhs, acc);
              virSymbolTable[op.getResult()] = vectorFMAOp.getResult();
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
                  DenseMap<Value, Value> innerSymbolTable(virSymbolTable);

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
              virSymbolTable[std::get<0>(it)] = std::get<1>(it);
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
            virSymbolTable[op.getResult()] = newLoadOp.getResult();
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
    Region &region = op.getRegion();
    Type anchorType = findAnchorTypeRecursive(region);
    if (!anchorType) {
      return op.emitError(
          "No anchor type found in the dynamic vector region. "
          "Ensure there are operations with !vir.vec<?xT> types "
          "without scaling factors.");
    }

    //===------------------------------------------------------------------===//
    // Step 2: Construct the target vector type.
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
    // Step 3: Create a main vectorization loop.
    //===------------------------------------------------------------------===//
    Value vlValue = op.getVl();
    // Calculate the upper bound for vectorized loop.
    // vl_upbound = (vl_total - vl_step) + 1
    Value vlStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWid);
    Value vlUpboundPat = rewriter.create<arith::SubIOp>(loc, vlValue, vlStep);
    Value vlUpbound = rewriter.create<arith::AddIOp>(
        loc, vlUpboundPat, rewriter.create<arith::ConstantIndexOp>(loc, 1));

    // To avoid possible overflow calculations such as' (vl-1)/step * step+step
    // ', We directly set the upper bound of the loop to vl, and then process
    // the remainder in the tail loop. The number of iterations in the main loop
    // is vl / step
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Create affine for loop with iteration variable.
    auto mainloop = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/ValueRange{zero}, rewriter.getDimIdentityMap(),
        /*upperBound=*/ValueRange{vlUpbound}, rewriter.getDimIdentityMap(),
        vectorWid,
        /*iterArgs=*/std::nullopt,
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
          //===--------------------------------------------------------------===//
          // Step 4: Convert operations inside the dynamic vector region.
          //===--------------------------------------------------------------===//
          // Symbol table: maps dynamic vector values to fixed / scalable vector
          // values.
          DenseMap<Value, Value> virSymbolTable;
          // Initiate recursive descent process on top-level blocks (vectorized
          // mode)
          lowerBlock(b, bodyLoc, &region.front(), virSymbolTable, iv,
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
          DenseMap<Value, Value> virSymbolTable;
          // Initiate recursive descent process on top-level blocks (scalar
          // mode)
          lowerBlock(b, bodyLoc, &region.front(), virSymbolTable, iv,
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
