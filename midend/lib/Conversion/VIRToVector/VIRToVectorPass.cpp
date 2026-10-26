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

#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cstdint>

// Include ARM SVE header for detecting register base width
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

// This pass vectorizes along the memref's most-minor dimension, so the
// vector.load/vector.store fast path is only valid when that specific stride is
// statically known to be 1.
static bool hasStaticallyUnitStrideInVectorizedDim(Value base) {
  auto memrefTy = dyn_cast<MemRefType>(base.getType());
  if (!memrefTy || memrefTy.getRank() == 0)
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefTy.getStridesAndOffset(strides, offset)) || strides.empty())
    return false;

  return strides.back() == 1;
}

// Downstream LLVM lowering may pack vector<...xi1> memory operations, which
// breaks round-tripping through scalar i1 loads/stores. Keep i1 memref accesses
// scalarized even inside the vectorized loop.
static bool shouldScalarizeI1VectorMemoryOp(VectorType vecTy) {
  return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getElementType().isInteger(1);
}

static Value buildScalarizedI1VectorLoad(OpBuilder &builder, Location loc,
                                         VectorType vecTy, Value base,
                                         ArrayRef<Value> baseIndices) {
  assert(!baseIndices.empty() && "expected at least one memref index");
  auto zero = arith::ConstantOp::create(
      builder, loc, vecTy,
      DenseElementsAttr::get(vecTy,
                             builder.getZeroAttr(vecTy.getElementType())));
  Value result = zero.getResult();
  for (int64_t lane = 0, e = vecTy.getNumElements(); lane < e; ++lane) {
    auto laneIdx = arith::ConstantIndexOp::create(builder, loc, lane);
    SmallVector<Value> laneIndices(baseIndices.begin(), baseIndices.end());
    laneIndices.back() =
        arith::AddIOp::create(builder, loc, laneIndices.back(), laneIdx);
    Value scalar = memref::LoadOp::create(builder, loc, base, laneIndices);
    result = vector::InsertOp::create(builder, loc, scalar, result,
                                      OpFoldResult(laneIdx));
  }
  return result;
}

static void buildScalarizedI1VectorStore(OpBuilder &builder, Location loc,
                                         Value vectorValue, Value base,
                                         ArrayRef<Value> baseIndices) {
  assert(!baseIndices.empty() && "expected at least one memref index");
  auto vecTy = cast<VectorType>(vectorValue.getType());
  for (int64_t lane = 0, e = vecTy.getNumElements(); lane < e; ++lane) {
    auto laneIdx = arith::ConstantIndexOp::create(builder, loc, lane);
    SmallVector<Value> laneIndices(baseIndices.begin(), baseIndices.end());
    laneIndices.back() =
        arith::AddIOp::create(builder, loc, laneIndices.back(), laneIdx);
    Value scalar = vector::ExtractOp::create(builder, loc, vectorValue,
                                             OpFoldResult(laneIdx));
    memref::StoreOp::create(builder, loc, scalar, base, laneIndices);
  }
}

/// The vector type of a dynamic vector region from its register pressure
/// (Req_Group, the most dynamic vectors live at once) and the vector
/// registers of the target: with `rvvVlen` (bits, > 0), a RISC-V Vector
/// target with 32 registers of that VLEN, whatever the host; otherwise the
/// host's registers. With `useScalable` (RVV only), the vector is scalable
/// and `rvvVlen` is the minimum VLEN.
VectorType vectorTypeHelper(Region &srcRegion, Type anchorType,
                            bool useScalable, bool verbose, int rvvVlen) {
#define VERBOSE_ANALYSIS_INFO(info)                                            \
  if (verbose) {                                                               \
    llvm::outs() << info;                                                      \
  }
  // Defaults when no target-specific features are detected (e.g. RISC-V or
  // hosts not enumerated below). A width of 1 made sizeVec zero for f32
  // (VectorType::get asserts); 128-bit is a reasonable generic SIMD nominal.
  int vectorRegBitWidth = 128;
  int totalVectorRegs = 16;
  if (rvvVlen > 0) {
    // A RISC-V Vector target (e.g. a cross compilation): the host is not
    // looked at.
    vectorRegBitWidth = rvvVlen;
    totalVectorRegs = 32;
    VERBOSE_ANALYSIS_INFO("--- RISC-V Vector target: VLEN " << rvvVlen
                                                            << " ---\n");
  } else {
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
  }

  VERBOSE_ANALYSIS_INFO(
      "\n--- Running vectorTypeHelper Liveness Analysis ---\n");

  // Req_Group = max_i |IN(i) U OUT(i)|, counting the dynamic vectors (the
  // vectors of the size chosen here) at every op of the region, those of
  // nested loops included. Liveness gives the values live at an op of a
  // block: those defined at or before it and used at or after it, in the
  // block or in its nested regions (IN U OUT). At an op in a loop, the
  // values live across the loop in the enclosing blocks are live too, but
  // not the loop's results and the initial values that die at the loop:
  // those are the iteration arguments of its body.
  //
  // On RVV, a broadcast scalar that only feeds the scalar operand of
  // arithmetic (vfmul.vf, vfmacc.vf, ...) stays in a scalar register and is
  // not counted. Such an instruction has one scalar operand: if the other
  // operand (the other multiplicand of an fma) is a broadcast too, both are
  // counted.
  Operation *setVL = srcRegion.getParentOp();
  Liveness liveness(setVL);
  auto isScalarOperand = [](Value v) {
    if (!v.getDefiningOp<vir::BroadcastOp>())
      return false;
    return llvm::all_of(v.getUses(), [](OpOperand &use) {
      Operation *user = use.getOwner();
      unsigned n = use.getOperandNumber();
      // a multiplicand of an fma, not the addend, or an operand of a binary
      // op
      bool binary =
          isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
              arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
              arith::MinNumFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp>(
              user);
      if (!(isa<vir::FMAOp>(user) && n < 2) && !binary)
        return false;
      Value other = user->getOperand(1 - n);
      return !other.getDefiningOp<vir::BroadcastOp>();
    });
  };
  auto isDynamic = [&](Value v) {
    return isa<vir::DynamicVectorType>(v.getType()) &&
           !(rvvVlen > 0 && isScalarOperand(v));
  };
  unsigned maxActiveVectors = 0;
  srcRegion.walk([&](Operation *op) {
    // an op with regions is measured at the ops of its blocks
    if (op->getNumRegions() != 0)
      return;
    DenseSet<Value> live;
    for (Value v :
         liveness.getLiveness(op->getBlock())->currentlyLiveValues(op))
      if (isDynamic(v))
        live.insert(v);
    for (Operation *loop = op->getParentOp(); loop != setVL;
         loop = loop->getParentOp()) {
      const LivenessBlockInfo *info = liveness.getLiveness(loop->getBlock());
      for (Value v : info->currentlyLiveValues(loop)) {
        if (!isDynamic(v) || v.getDefiningOp() == loop)
          continue;
        bool usedInside = llvm::any_of(v.getUsers(), [&](Operation *user) {
          return loop->isProperAncestor(user);
        });
        if (llvm::is_contained(loop->getOperands(), v) && !usedInside &&
            liveness.isDeadAfter(v, loop))
          continue;
        live.insert(v);
      }
    }
    if (verbose) {
      llvm::outs() << "  " << live.size() << " live at ";
      op->print(llvm::outs(), OpPrintingFlags().skipRegions());
      llvm::outs() << "\n";
    }
    maxActiveVectors = std::max<unsigned>(maxActiveVectors, live.size());
  });

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

  int64_t sizeVec;
  if (rvvVlen > 0) {
    // RVV: LMUL = min(8, 2^floor(log2(Size_Group))), and a register group
    // of LMUL registers holds LMUL * VLEN / SEW elements.
    int64_t lmul = std::min<int64_t>(8, llvm::bit_floor(unsigned(sizeGroup)));
    sizeVec = lmul * rvvVlen / elementBitWidth;
    VERBOSE_ANALYSIS_INFO("Calculated LMUL = " << lmul << ", Size_vec = ("
                                               << lmul << " * " << rvvVlen
                                               << ") / " << elementBitWidth
                                               << " = " << sizeVec << "\n");
  } else {
    // Size_vec = 2^floor(log2(Size_Group * Size_Reg / Width_Ele))
    sizeVec = llvm::bit_floor(uint64_t(sizeGroup) * vectorRegBitWidth /
                              elementBitWidth);
    VERBOSE_ANALYSIS_INFO("Calculated Size_vec = 2^floor(log2(("
                          << sizeGroup << " * " << vectorRegBitWidth << ") / "
                          << elementBitWidth << ")) = " << sizeVec << "\n");
  }

  // no wider than a constant vl: the lanes past it would compute nothing
  if (std::optional<int64_t> vl =
          getConstantIntValue(cast<vir::SetVLOp>(setVL).getVl());
      vl && *vl > 0 && sizeVec > *vl) {
    sizeVec = llvm::bit_floor(uint64_t(*vl));
    VERBOSE_ANALYSIS_INFO("Clamped to the constant vl "
                          << *vl << ": Size_vec = " << sizeVec << "\n");
  }

  if (sizeVec < 1) {
    emitWarning(srcRegion.getLoc(),
                "Calculated vector lane count is zero; clamping to 1. "
                "Consider passing -lower-vir-to-vector=vector-width=N.");
    sizeVec = 1;
  }

  if (useScalable) {
    // RVV: vscale = VLEN / 64, so vector<[N]xT> is the register group of
    // LMUL = N * SEW / 64 registers whatever the VLEN; at the minimum VLEN
    // it holds sizeVec = N * rvvVlen / 64 lanes. N >= 1 (a fractional LMUL
    // for SEW < 64 when sizeVec was clamped below that).
    int64_t n = std::max<int64_t>(1, sizeVec * 64 / rvvVlen);
    VERBOSE_ANALYSIS_INFO("Scalable: N = max(1, ("
                          << sizeVec << " * 64) / " << rvvVlen << ") = " << n
                          << ", vector<[" << n << "]x" << anchorType << ">\n");
    return VectorType::get({n}, anchorType, /*scalableDims=*/{true});
  }
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
                   int vectorWidth, bool verbose, bool useCustomVecWid,
                   int rvvVlen)
      : ConvertOpToLLVMPattern<vir::SetVLOp>(converter),
        useScalable(useScalable), vectorWidth(vectorWidth), verbose(verbose),
        useCustomVecWid(useCustomVecWid), rvvVlen(rvvVlen) {}

private:
  bool useScalable;
  int vectorWidth;
  bool verbose;
  bool useCustomVecWid;
  // VLEN of a RISC-V Vector target in bits, 0 for the host
  int rvvVlen;

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
                  ArrayRef<Value> leadingIVs, Type targetVectorType,
                  bool isTailLoop) const {
    // find mapped local Value from virSymbolTable or global Value
    auto findValue = [&virSymbolTable](Value srcValue) {
      if (virSymbolTable.contains(srcValue)) {
        return virSymbolTable.lookup(srcValue);
      }
      return srcValue;
    };
    auto deriveVectorTypeFor = [&](Type virTy) -> FailureOr<VectorType> {
      auto dynVecTy = dyn_cast<vir::DynamicVectorType>(virTy);
      if (!dynVecTy)
        return failure();
      auto fallbackVecTy = dyn_cast<VectorType>(targetVectorType);
      if (!fallbackVecTy)
        return failure();
      return VectorType::get(fallbackVecTy.getShape(),
                             dynVecTy.getElementType(),
                             fallbackVecTy.getScalableDims());
    };
    auto remapResultType = [&](Type ty, ArrayRef<Value> operands) -> Type {
      auto dyn = dyn_cast<vir::DynamicVectorType>(ty);
      if (!dyn)
        return ty;
      Type elem = dyn.getElementType();
      for (Value operand : operands) {
        if (auto vecTy = dyn_cast<VectorType>(operand.getType())) {
          return VectorType::get(vecTy.getShape(), elem,
                                 vecTy.getScalableDims());
        }
      }
      return elem;
    };
    auto cloneMappedOp = [&](Operation *op) {
      SmallVector<Value> operands;
      operands.reserve(op->getNumOperands());
      for (Value operand : op->getOperands())
        operands.push_back(findValue(operand));

      SmallVector<Type> resultTypes;
      resultTypes.reserve(op->getNumResults());
      for (Type ty : op->getResultTypes())
        resultTypes.push_back(remapResultType(ty, operands));

      OperationState state(loc, op->getName().getStringRef());
      state.addOperands(operands);
      state.addTypes(resultTypes);
      state.addAttributes(op->getAttrs());
      state.addSuccessors(op->getSuccessors());
      Operation *newOp = builder.create(state);
      for (auto [oldResult, newResult] :
           llvm::zip(op->getResults(), newOp->getResults()))
        virSymbolTable[oldResult] = newResult;
    };

    for (Operation &innerOp : block->without_terminator()) {
      llvm::TypeSwitch<Operation *>(&innerOp)
          .Case<memref::ExpandShapeOp, memref::SubViewOp, memref::TransposeOp>(
              [&](Operation *op) { cloneMappedOp(op); })
          .Case<vir::LoadOp>([&](vir::LoadOp op) {
            Value base = findValue(op.getBase());
            SmallVector<Value> destIndices{};
            auto sourceIndices = op.getIndices();

            // Indices convention:
            // - If indices are provided, treat the last index as the dynamic
            //   "vectorized" dimension offset and add mainLoopIV.
            // - If indices are empty, assume all leading indices are 0 and the
            //   last index is implicitly 0 (i.e. base of the vectorized dim).
            if (sourceIndices.empty()) {
              auto memrefTy = dyn_cast<MemRefType>(base.getType());
              if (!memrefTy || memrefTy.getRank() == 0) {
                op.emitError(
                    "expected ranked memref base for implicit indices");
                return;
              }
              int64_t rank = memrefTy.getRank();
              for (int64_t i = 0; i < rank - 1; ++i) {
                if (static_cast<size_t>(i) < leadingIVs.size()) {
                  destIndices.push_back(leadingIVs[i]);
                } else {
                  destIndices.push_back(
                      arith::ConstantIndexOp::create(builder, loc, 0));
                }
              }
              destIndices.push_back(mainLoopIV);
            } else {
              // the last index is dynamic and global
              Value lastIndex = findValue(sourceIndices.back());
              Value baseOffset =
                  arith::AddIOp::create(builder, loc, mainLoopIV, lastIndex);
              // collect indices for load op
              for (size_t i = 0, e = sourceIndices.size() - 1; i < e; ++i) {
                Value destOp = findValue(sourceIndices[i]);
                destIndices.push_back(destOp);
              }
              destIndices.push_back(baseOffset);
            }

            if (isTailLoop) {
              auto memrefLoadOp =
                  memref::LoadOp::create(builder, loc, base, destIndices);
              virSymbolTable[op.getResult()] = memrefLoadOp.getResult();
            } else {
              auto vecTyOr = deriveVectorTypeFor(op.getResult().getType());
              if (failed(vecTyOr)) {
                op.emitError("failed to derive vector type for vir.load");
                return;
              }
              if (shouldScalarizeI1VectorMemoryOp(*vecTyOr)) {
                virSymbolTable[op.getResult()] = buildScalarizedI1VectorLoad(
                    builder, loc, *vecTyOr, base, destIndices);
                return;
              }
              if (hasStaticallyUnitStrideInVectorizedDim(base)) {
                auto vectorLoadOp = vector::LoadOp::create(
                    builder, loc, *vecTyOr, base, destIndices);
                virSymbolTable[op.getResult()] = vectorLoadOp.getResult();
              } else {
                auto padding = arith::ConstantOp::create(
                    builder, loc,
                    builder.getZeroAttr((*vecTyOr).getElementType()));
                auto transferReadOp = vector::TransferReadOp::create(
                    builder, loc, *vecTyOr, base, destIndices, padding);
                virSymbolTable[op.getResult()] = transferReadOp.getResult();
              }
            }
          })
          .Case<vir::StoreOp>([&](vir::StoreOp op) {
            Value valueToStore = findValue(op.getValue());
            if (!valueToStore) {
              op.emitError("value to store not found in symbol table");
              return;
            }
            Value base = findValue(op.getBase());
            SmallVector<Value> destIndices{};
            auto sourceIndices = op.getIndices();
            if (sourceIndices.empty()) {
              auto memrefTy = dyn_cast<MemRefType>(base.getType());
              if (!memrefTy || memrefTy.getRank() == 0) {
                op.emitError(
                    "expected ranked memref base for implicit indices");
                return;
              }
              int64_t rank = memrefTy.getRank();
              for (int64_t i = 0; i < rank - 1; ++i) {
                if (static_cast<size_t>(i) < leadingIVs.size()) {
                  destIndices.push_back(leadingIVs[i]);
                } else {
                  destIndices.push_back(
                      arith::ConstantIndexOp::create(builder, loc, 0));
                }
              }
              destIndices.push_back(mainLoopIV);
            } else {
              Value lastIndex = findValue(sourceIndices.back());
              Value baseOffset =
                  arith::AddIOp::create(builder, loc, mainLoopIV, lastIndex);
              // collect indices for store op
              for (size_t i = 0, e = sourceIndices.size() - 1; i < e; ++i) {
                Value destOp = findValue(sourceIndices[i]);
                destIndices.push_back(destOp);
              }
              destIndices.push_back(baseOffset);
            }

            if (isTailLoop) {
              memref::StoreOp::create(builder, loc, valueToStore, base,
                                      destIndices);
            } else {
              if (auto vecTy = dyn_cast<VectorType>(valueToStore.getType());
                  shouldScalarizeI1VectorMemoryOp(vecTy)) {
                buildScalarizedI1VectorStore(builder, loc, valueToStore, base,
                                             destIndices);
                return;
              }
              if (hasStaticallyUnitStrideInVectorizedDim(base)) {
                vector::StoreOp::create(builder, loc, valueToStore, base,
                                        destIndices);
              } else {
                vector::TransferWriteOp::create(builder, loc, valueToStore,
                                                base, destIndices);
              }
            }
          })
          .Case<vir::ConstantOp>([&](vir::ConstantOp op) {
            auto constValue = op.getValue();
            if (isTailLoop) {
              auto arithConstOp =
                  arith::ConstantOp::create(builder, loc, constValue);
              virSymbolTable[op.getResult()] = arithConstOp.getResult();
            } else {
              auto vecTyOr = deriveVectorTypeFor(op.getResult().getType());
              if (failed(vecTyOr)) {
                op.emitError("failed to derive vector type for vir.constant");
                return;
              }
              auto vectorConstAttr = DenseElementsAttr::get(
                  cast<ShapedType>(*vecTyOr), constValue);
              auto vectorConstOp = arith::ConstantOp::create(
                  builder, loc, *vecTyOr, vectorConstAttr);
              virSymbolTable[op.getResult()] = vectorConstOp.getResult();
            }
          })
          .Case<vir::BroadcastOp>([&](vir::BroadcastOp op) {
            Value scalarValue = findValue(op.getValue());
            if (isTailLoop) {
              virSymbolTable[op.getResult()] = scalarValue;
            } else {
              auto vecTyOr = deriveVectorTypeFor(op.getResult().getType());
              if (failed(vecTyOr)) {
                op.emitError("failed to derive vector type for vir.broadcast");
                return;
              }
              auto vectorBroadcastOp = vector::BroadcastOp::create(
                  builder, loc, *vecTyOr, scalarValue);
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
              auto mulResult = arith::MulFOp::create(builder, loc, lhs, rhs);
              auto addResult =
                  arith::AddFOp::create(builder, loc, mulResult, acc);
              virSymbolTable[op.getResult()] = addResult;
            } else {
              auto vectorFMAOp =
                  vector::FMAOp::create(builder, loc, lhs, rhs, acc);
              virSymbolTable[op.getResult()] = vectorFMAOp.getResult();
            }
          })
          .Case<vir::ReduceOp>([&](vir::ReduceOp op) {
            Value input = findValue(op.getInput());
            Value acc = findValue(op.getAcc());
            if (!input || !acc) {
              op.emitError("reduce operands not found in symbol table");
              return;
            }

            auto kindAttr = op->getAttrOfType<StringAttr>("kind");
            if (!kindAttr) {
              op.emitError("missing 'kind' attribute on vir.reduce");
              return;
            }
            StringRef kind = kindAttr.getValue();

            if (isTailLoop) {
              Value out;
              if (kind == "add") {
                out = arith::AddFOp::create(builder, loc, input, acc);
              } else if (kind == "maxnum") {
                out = arith::MaxNumFOp::create(builder, loc, input, acc);
              } else {
                op.emitError(
                    "unsupported vir.reduce kind (expected add/maxnum)");
                return;
              }
              virSymbolTable[op.getResult()] = out;
              return;
            }

            auto vecTy = dyn_cast<VectorType>(input.getType());
            if (!vecTy) {
              op.emitError("expected vector input for non-tail vir.reduce");
              return;
            }

            vector::CombiningKind combineKind;
            if (kind == "add") {
              combineKind = vector::CombiningKind::ADD;
            } else if (kind == "maxnum") {
              combineKind = vector::CombiningKind::MAXNUMF;
            } else {
              op.emitError("unsupported vir.reduce kind (expected add/maxnum)");
              return;
            }

            auto reduced = vector::ReductionOp::create(
                builder, loc, combineKind, input, acc,
                op.getReassoc() ? arith::FastMathFlags::reassoc
                                : arith::FastMathFlags::none);
            virSymbolTable[op.getResult()] = reduced.getResult();
          })
          .Case<vir::SelectOp>([&](vir::SelectOp op) {
            Value cond = findValue(op.getCondition());
            Value tVal = findValue(op.getTrueValue());
            Value fVal = findValue(op.getFalseValue());
            if (!cond || !tVal || !fVal) {
              op.emitError("select operands not found in symbol table");
              return;
            }
            if (isTailLoop) {
              auto newOp =
                  arith::SelectOp::create(builder, loc, cond, tVal, fVal);
              virSymbolTable[op.getResult()] = newOp.getResult();
            } else {
              auto newOp = LLVM::SelectOp::create(builder, loc, tVal.getType(),
                                                  cond, tVal, fVal);
              virSymbolTable[op.getResult()] = newOp.getResult();
            }
          })
          .Case<vir::ExtFOp>([&](vir::ExtFOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::ExtFOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<vir::TruncFOp>([&](vir::TruncFOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::TruncFOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
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
            auto newForOp = affine::AffineForOp::create(
                builder, loc, ValueRange(op.getLowerBoundOperands()),
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
                             mainLoopIV, leadingIVs, targetVectorType,
                             isTailLoop);

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
                    affine::AffineYieldOp::create(b, bodyLoc, yieldOperands);
                  }
                });

            // Register the results of the newly created loop in the external
            // symbol table
            for (auto it : llvm::zip(op.getResults(), newForOp.getResults())) {
              virSymbolTable[std::get<0>(it)] = std::get<1>(it);
            }
          })
          .Case<scf::ForOp>([&](scf::ForOp op) {
            // as affine.for: the body lowered with the iteration arguments
            // and the results of the loop mapped
            SmallVector<Value> initialIterArgs;
            for (Value arg : op.getInitArgs())
              initialIterArgs.push_back(findValue(arg));
            auto newForOp = scf::ForOp::create(
                builder, loc, findValue(op.getLowerBound()),
                findValue(op.getUpperBound()), findValue(op.getStep()),
                initialIterArgs,
                [&](OpBuilder &b, Location bodyLoc, Value iv,
                    ValueRange iterArgs) {
                  DenseMap<Value, Value> innerSymbolTable(virSymbolTable);
                  innerSymbolTable[op.getInductionVar()] = iv;
                  for (auto [arg, newArg] :
                       llvm::zip(op.getRegionIterArgs(), iterArgs))
                    innerSymbolTable[arg] = newArg;
                  lowerBlock(b, bodyLoc, op.getBody(), innerSymbolTable,
                             mainLoopIV, leadingIVs, targetVectorType,
                             isTailLoop);
                  SmallVector<Value> yieldOperands;
                  for (Value operand :
                       op.getBody()->getTerminator()->getOperands()) {
                    auto it = innerSymbolTable.find(operand);
                    yieldOperands.push_back(
                        it != innerSymbolTable.end() ? it->second : operand);
                  }
                  scf::YieldOp::create(b, bodyLoc, yieldOperands);
                },
                op.getUnsignedCmp());
            for (auto [result, newResult] :
                 llvm::zip(op.getResults(), newForOp.getResults()))
              virSymbolTable[result] = newResult;
          })
          .Case<affine::AffineYieldOp, vector::YieldOp, scf::YieldOp>(
              [&](Operation *) {
                // During recursive descent, these terminators are handled by
                // their parent operations (such as affine. for). We will ignore
                // them directly here.
              })
          .Case<arith::ConstantOp>([&](arith::ConstantOp op) {
            auto newConst =
                arith::ConstantOp::create(builder, loc, op.getValue());
            virSymbolTable[op.getResult()] = newConst.getResult();
          })
          .Case<memref::DimOp>([&](memref::DimOp op) {
            Value src = findValue(op.getSource());
            Value idx = findValue(op.getIndex());
            auto newDim = memref::DimOp::create(builder, loc, src, idx);
            virSymbolTable[op.getResult()] = newDim.getResult();
          })
          .Case<memref::TransposeOp>([&](memref::TransposeOp op) {
            Value src = findValue(op.getIn());
            auto newOp = memref::TransposeOp::create(
                builder, loc, op.getResult().getType(), src,
                AffineMapAttr::get(op.getPermutation()));
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<memref::ExpandShapeOp>([&](memref::ExpandShapeOp op) {
            Value src = findValue(op.getSrc());
            SmallVector<OpFoldResult> remappedOutputShape;
            remappedOutputShape.reserve(op.getOutputShape().size());
            for (OpFoldResult ofr : op.getOutputShape()) {
              if (auto v = dyn_cast<Value>(ofr)) {
                remappedOutputShape.push_back(findValue(v));
              } else {
                remappedOutputShape.push_back(cast<Attribute>(ofr));
              }
            }
            auto newOp = memref::ExpandShapeOp::create(
                builder, loc, op.getResultType(), src,
                op.getReassociationIndices(), remappedOutputShape);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<memref::SubViewOp>([&](memref::SubViewOp op) {
            Value src = findValue(op.getSource());

            auto remapMixed = [&](ArrayRef<OpFoldResult> in) {
              SmallVector<OpFoldResult> out;
              out.reserve(in.size());
              for (OpFoldResult ofr : in) {
                if (auto v = dyn_cast<Value>(ofr)) {
                  out.push_back(findValue(v));
                } else {
                  out.push_back(cast<Attribute>(ofr));
                }
              }
              return out;
            };

            SmallVector<OpFoldResult> offsets =
                remapMixed(op.getMixedOffsets());
            SmallVector<OpFoldResult> sizes = remapMixed(op.getMixedSizes());
            SmallVector<OpFoldResult> strides =
                remapMixed(op.getMixedStrides());

            auto newOp = memref::SubViewOp::create(
                builder, loc, cast<MemRefType>(op.getResult().getType()), src,
                offsets, sizes, strides);
            virSymbolTable[op.getResult()] = newOp.getResult();
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

            auto newLoadOp =
                memref::LoadOp::create(builder, loc, base, indices);
            virSymbolTable[op.getResult()] = newLoadOp.getResult();
          })
          .Case<memref::StoreOp>([&](memref::StoreOp op) {
            Value base = findValue(op.getMemRef());
            if (!base)
              base = op.getMemRef();
            Value value = findValue(op.getValue());
            if (!value)
              value = op.getValue();

            SmallVector<Value> indices;
            for (Value index : op.getIndices()) {
              Value mappedIndex = findValue(index);
              indices.push_back(mappedIndex);
            }
            memref::StoreOp::create(builder, loc, value, base, indices);
          })
          .Case<arith::AddFOp>([&](arith::AddFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::AddFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::CmpFOp>([&](arith::CmpFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::CmpFOp::create(builder, loc, op.getPredicate(),
                                               lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::CmpIOp::create(builder, loc, op.getPredicate(),
                                               lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::SelectOp>([&](arith::SelectOp op) {
            Value cond = findValue(op.getCondition());
            Value tVal = findValue(op.getTrueValue());
            Value fVal = findValue(op.getFalseValue());
            auto newOp =
                arith::SelectOp::create(builder, loc, cond, tVal, fVal);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::AddIOp>([&](arith::AddIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::AddIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::SubFOp>([&](arith::SubFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::SubFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::SubIOp>([&](arith::SubIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::SubIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MulFOp>([&](arith::MulFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MulFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MulIOp>([&](arith::MulIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MulIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::AndIOp>([&](arith::AndIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::AndIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::OrIOp>([&](arith::OrIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::OrIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::XOrIOp>([&](arith::XOrIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::XOrIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ShLIOp>([&](arith::ShLIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::ShLIOp::create(builder, loc, lhs, rhs,
                                               op.getOverflowFlags());
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ShRUIOp>([&](arith::ShRUIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::ShRUIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ShRSIOp>([&](arith::ShRSIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::ShRSIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::DivFOp>([&](arith::DivFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::DivFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::DivSIOp>([&](arith::DivSIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::DivSIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::DivUIOp>([&](arith::DivUIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::DivUIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::RemSIOp>([&](arith::RemSIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::RemSIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::RemUIOp>([&](arith::RemUIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::RemUIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MaximumFOp>([&](arith::MaximumFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MaximumFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MinimumFOp>([&](arith::MinimumFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MinimumFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MaxNumFOp>([&](arith::MaxNumFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MaxNumFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MinNumFOp>([&](arith::MinNumFOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MinNumFOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MaxSIOp>([&](arith::MaxSIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MaxSIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MinSIOp>([&](arith::MinSIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MinSIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MaxUIOp>([&](arith::MaxUIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MaxUIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::MinUIOp>([&](arith::MinUIOp op) {
            Value lhs = findValue(op.getLhs());
            Value rhs = findValue(op.getRhs());
            auto newOp = arith::MinUIOp::create(builder, loc, lhs, rhs);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::NegFOp>([&](arith::NegFOp op) {
            Value operand = findValue(op.getOperand());
            auto newOp = arith::NegFOp::create(builder, loc, operand);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ExtSIOp>([&](arith::ExtSIOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::ExtSIOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ExtUIOp>([&](arith::ExtUIOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::ExtUIOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::TruncIOp>([&](arith::TruncIOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::TruncIOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::ExtFOp>([&](arith::ExtFOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::ExtFOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::TruncFOp>([&](arith::TruncFOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::TruncFOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::SIToFPOp>([&](arith::SIToFPOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::SIToFPOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::UIToFPOp>([&](arith::UIToFPOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::UIToFPOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::FPToSIOp>([&](arith::FPToSIOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::FPToSIOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::FPToUIOp>([&](arith::FPToUIOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::FPToUIOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<arith::BitcastOp>([&](arith::BitcastOp op) {
            Value in = findValue(op.getIn());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = arith::BitcastOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<math::ExpOp>([&](math::ExpOp op) {
            Value in = findValue(op.getOperand());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = math::ExpOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<math::SqrtOp>([&](math::SqrtOp op) {
            Value in = findValue(op.getOperand());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto newOp = math::SqrtOp::create(builder, loc, outTy, in);
            virSymbolTable[op.getResult()] = newOp.getResult();
          })
          .Case<math::RsqrtOp>([&](math::RsqrtOp op) {
            Value in = findValue(op.getOperand());
            Type outTy = op.getType();
            if (auto dyn = dyn_cast<vir::DynamicVectorType>(outTy)) {
              Type elem = dyn.getElementType();
              if (auto vecTy = dyn_cast<VectorType>(in.getType())) {
                outTy = VectorType::get(vecTy.getShape(), elem,
                                        vecTy.getScalableDims());
              } else {
                outTy = elem;
              }
            }
            auto sqrt = math::SqrtOp::create(builder, loc, outTy, in);
            Type elemTy = outTy;
            if (auto vecTy = dyn_cast<VectorType>(outTy))
              elemTy = vecTy.getElementType();
            auto oneAttr = builder.getFloatAttr(elemTy, 1.0);
            Value one = arith::ConstantOp::create(builder, loc, oneAttr);
            if (isa<VectorType>(outTy))
              one = vector::BroadcastOp::create(builder, loc, outTy, one);
            auto div = arith::DivFOp::create(builder, loc, one, sqrt);
            virSymbolTable[op.getResult()] = div.getResult();
          })
          .Case<math::LogOp, math::AbsFOp, math::CeilOp, math::FloorOp,
                math::RoundOp, math::TanhOp, math::ErfOp, math::PowFOp>(
              [&](Operation *op) { cloneMappedOp(op); })
          .Default([&](Operation *op) {
            emitWarning(loc, "Unsupported operation during lowering: " +
                                 op->getName().getStringRef());
          });
    }
  }

  /// The pieces a dynamic vector of a set_vl with results is lowered to:
  /// `numVectors` vectors of `width` lanes, then `numScalars` scalars.
  struct Pieces {
    int64_t width, numVectors, numScalars;
    int64_t size() const { return numVectors + numScalars; }
    // the number of lanes, vl
    int64_t length() const { return numVectors * width + numScalars; }
    bool isVector(int64_t p) const { return p < numVectors; }
    // the first lane of piece p
    int64_t offset(int64_t p) const {
      return isVector(p) ? p * width : numVectors * width + p - numVectors;
    }
    Type type(Type elem, int64_t p) const {
      return isVector(p) ? Type(VectorType::get({width}, elem)) : elem;
    }
  };
  using PieceMap = DenseMap<Value, SmallVector<Value, 4>>;

  /// Lowers the ops of `block` but its terminator for a set_vl with results:
  /// each op on dynamic vectors applies to every piece, a vir.reduce folds
  /// the pieces in order and loops carry all pieces of their dynamic vector
  /// values. `map` maps a value of the region to its pieces (one for a
  /// value that is not a dynamic vector).
  LogicalResult lowerBlockInPieces(OpBuilder &b, Location loc, Block *block,
                                   PieceMap &map, const Pieces &pieces) const {
    auto isDyn = [](Type t) { return isa<vir::DynamicVectorType>(t); };
    auto elemOf = [](Type t) {
      return cast<vir::DynamicVectorType>(t).getElementType();
    };
    // the pieces of a value (a value from above the region is its own
    // piece)
    auto get = [&](Value v) -> SmallVector<Value, 4> {
      auto it = map.find(v);
      if (it != map.end())
        return it->second;
      return {v};
    };
    auto one = [&](Value v) { return get(v).front(); };
    // the memref indices of piece p of the access at `indices`
    auto pieceIndices = [&](Value base, ValueRange indices, int64_t p) {
      SmallVector<Value> out;
      Value off = arith::ConstantIndexOp::create(b, loc, pieces.offset(p));
      if (indices.empty()) {
        int64_t rank = cast<MemRefType>(base.getType()).getRank();
        for (int64_t i = 0; i + 1 < rank; ++i)
          out.push_back(arith::ConstantIndexOp::create(b, loc, 0));
        out.push_back(off);
        return out;
      }
      for (Value index : indices.drop_back())
        out.push_back(one(index));
      out.push_back(arith::AddIOp::create(b, loc, one(indices.back()), off));
      return out;
    };
    // flattens the pieces of `values`; `counts` gets the number of each
    auto flatten = [&](ValueRange values, SmallVectorImpl<Value> &flat,
                       SmallVectorImpl<int64_t> *counts) {
      for (Value v : values) {
        SmallVector<Value, 4> ps = get(v);
        flat.append(ps.begin(), ps.end());
        if (counts)
          counts->push_back(ps.size());
      }
    };
    auto regroup = [&](ValueRange olds, ValueRange flat,
                       ArrayRef<int64_t> counts) {
      int64_t at = 0;
      for (auto [old, count] : llvm::zip(olds, counts)) {
        map[old] = SmallVector<Value, 4>(flat.slice(at, count));
        at += count;
      }
    };

    for (Operation &inner : block->without_terminator()) {
      Operation *op = &inner;
      for (Value v : op->getOperands()) {
        // (get() gives a value from above the region as one piece)
        if (isDyn(v.getType()) &&
            (!map.contains(v) || map[v].size() != size_t(pieces.size())))
          return op->emitError("a dynamic vector used in a vir.set_vl with "
                               "results must be defined in its region");
      }
      for (Type t : op->getResultTypes()) {
        auto dyn = dyn_cast<vir::DynamicVectorType>(t);
        if (dyn && (dyn.getShape().size() != 1 || dyn.getScalingFactor()))
          return op->emitError("a vir.set_vl with results supports only "
                               "!vir.vec<?xT> vectors");
      }

      if (auto load = dyn_cast<vir::LoadOp>(op)) {
        Value base = one(load.getBase());
        Type elem = elemOf(load.getType());
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          SmallVector<Value> idx = pieceIndices(base, load.getIndices(), p);
          if (!pieces.isVector(p)) {
            out.push_back(memref::LoadOp::create(b, loc, base, idx));
            continue;
          }
          auto vecTy = cast<VectorType>(pieces.type(elem, p));
          if (shouldScalarizeI1VectorMemoryOp(vecTy))
            out.push_back(
                buildScalarizedI1VectorLoad(b, loc, vecTy, base, idx));
          else if (hasStaticallyUnitStrideInVectorizedDim(base))
            out.push_back(vector::LoadOp::create(b, loc, vecTy, base, idx));
          else
            out.push_back(vector::TransferReadOp::create(
                b, loc, vecTy, base, idx,
                arith::ConstantOp::create(b, loc, b.getZeroAttr(elem))));
        }
        map[load.getResult()] = out;
      } else if (auto store = dyn_cast<vir::StoreOp>(op)) {
        Value base = one(store.getBase());
        SmallVector<Value, 4> values = get(store.getValue());
        for (int64_t p = 0; p < pieces.size(); ++p) {
          SmallVector<Value> idx = pieceIndices(base, store.getIndices(), p);
          if (!pieces.isVector(p))
            memref::StoreOp::create(b, loc, values[p], base, idx);
          else if (shouldScalarizeI1VectorMemoryOp(
                       cast<VectorType>(values[p].getType())))
            buildScalarizedI1VectorStore(b, loc, values[p], base, idx);
          else if (hasStaticallyUnitStrideInVectorizedDim(base))
            vector::StoreOp::create(b, loc, values[p], base, idx);
          else
            vector::TransferWriteOp::create(b, loc, values[p], base, idx);
        }
      } else if (auto from = dyn_cast<vir::FromVectorOp>(op)) {
        Value src = one(from.getSource());
        auto srcTy = cast<VectorType>(src.getType());
        if (srcTy.isScalable() || srcTy.getNumElements() != pieces.length())
          return op->emitError("vir.from_vector needs a fixed vector of vl "
                               "lanes");
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          int64_t off = pieces.offset(p);
          if (!pieces.isVector(p))
            out.push_back(vector::ExtractOp::create(b, loc, src, off));
          else if (pieces.width == srcTy.getNumElements())
            out.push_back(src);
          else
            out.push_back(vector::ExtractStridedSliceOp::create(
                b, loc, src, ArrayRef<int64_t>{off},
                ArrayRef<int64_t>{pieces.width}, ArrayRef<int64_t>{1}));
        }
        map[from.getResult()] = out;
      } else if (auto cst = dyn_cast<vir::ConstantOp>(op)) {
        Type elem = elemOf(cst.getType());
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          if (!pieces.isVector(p)) {
            out.push_back(arith::ConstantOp::create(b, loc, cst.getValue()));
            continue;
          }
          auto vecTy = cast<ShapedType>(pieces.type(elem, p));
          out.push_back(arith::ConstantOp::create(
              b, loc, vecTy, DenseElementsAttr::get(vecTy, cst.getValue())));
        }
        map[cst.getResult()] = out;
      } else if (auto bcast = dyn_cast<vir::BroadcastOp>(op)) {
        Value scalar = one(bcast.getValue());
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p)
          out.push_back(
              pieces.isVector(p)
                  ? Value(vector::BroadcastOp::create(
                        b, loc, pieces.type(scalar.getType(), p), scalar))
                  : scalar);
        map[bcast.getResult()] = out;
      } else if (auto fma = dyn_cast<vir::FMAOp>(op)) {
        SmallVector<Value, 4> lhs = get(fma.getLhs()), rhs = get(fma.getRhs()),
                              acc = get(fma.getAcc()), out;
        for (int64_t p = 0; p < pieces.size(); ++p)
          out.push_back(
              pieces.isVector(p)
                  ? Value(vector::FMAOp::create(b, loc, lhs[p], rhs[p], acc[p]))
                  : Value(math::FmaOp::create(b, loc, lhs[p], rhs[p], acc[p])));
        map[fma.getResult()] = out;
      } else if (auto reduce = dyn_cast<vir::ReduceOp>(op)) {
        StringRef kind = reduce.getKind();
        if (kind != "add" && kind != "maxnum")
          return op->emitError(
              "unsupported vir.reduce kind (expected add/maxnum)");
        auto combineKind = kind == "add" ? vector::CombiningKind::ADD
                                         : vector::CombiningKind::MAXNUMF;
        auto fastmath = reduce.getReassoc() ? arith::FastMathFlags::reassoc
                                            : arith::FastMathFlags::none;
        Value acc = one(reduce.getAcc());
        for (Value piece : get(reduce.getInput())) {
          if (isa<VectorType>(piece.getType()))
            acc = vector::ReductionOp::create(b, loc, combineKind, piece, acc,
                                              fastmath);
          else if (kind == "add")
            acc = arith::AddFOp::create(b, loc, acc, piece);
          else
            acc = arith::MaxNumFOp::create(b, loc, acc, piece);
        }
        map[reduce.getResult()] = {acc};
      } else if (auto select = dyn_cast<vir::SelectOp>(op)) {
        SmallVector<Value, 4> cond = get(select.getCondition()),
                              tv = get(select.getTrueValue()),
                              fv = get(select.getFalseValue()), out;
        for (int64_t p = 0; p < pieces.size(); ++p)
          out.push_back(arith::SelectOp::create(b, loc, cond[p], tv[p], fv[p]));
        map[select.getResult()] = out;
      } else if (isa<vir::ExtFOp, vir::TruncFOp>(op)) {
        Type elem = elemOf(op->getResult(0).getType());
        SmallVector<Value, 4> in = get(op->getOperand(0)), out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          Type t = pieces.type(elem, p);
          out.push_back(isa<vir::ExtFOp>(op)
                            ? Value(arith::ExtFOp::create(b, loc, t, in[p]))
                            : Value(arith::TruncFOp::create(b, loc, t, in[p])));
        }
        map[op->getResult(0)] = out;
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        SmallVector<Value> inits;
        SmallVector<int64_t> counts;
        flatten(forOp.getInitArgs(), inits, &counts);
        LogicalResult status = success();
        auto newFor = scf::ForOp::create(
            b, loc, one(forOp.getLowerBound()), one(forOp.getUpperBound()),
            one(forOp.getStep()), inits,
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange args) {
              map[forOp.getInductionVar()] = {iv};
              regroup(forOp.getRegionIterArgs(), args, counts);
              status =
                  lowerBlockInPieces(bb, bodyLoc, forOp.getBody(), map, pieces);
              SmallVector<Value> yields;
              flatten(forOp.getBody()->getTerminator()->getOperands(), yields,
                      nullptr);
              scf::YieldOp::create(bb, bodyLoc, yields);
            },
            forOp.getUnsignedCmp());
        if (failed(status))
          return failure();
        regroup(forOp.getResults(), newFor.getResults(), counts);
      } else if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        SmallVector<Value> inits, lbs, ubs;
        SmallVector<int64_t> counts;
        flatten(forOp.getInits(), inits, &counts);
        flatten(forOp.getLowerBoundOperands(), lbs, nullptr);
        flatten(forOp.getUpperBoundOperands(), ubs, nullptr);
        LogicalResult status = success();
        auto newFor = affine::AffineForOp::create(
            b, loc, lbs, forOp.getLowerBoundMap(), ubs,
            forOp.getUpperBoundMap(), forOp.getStep().getLimitedValue(), inits,
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange args) {
              map[forOp.getInductionVar()] = {iv};
              regroup(forOp.getRegionIterArgs(), args, counts);
              status =
                  lowerBlockInPieces(bb, bodyLoc, forOp.getBody(), map, pieces);
              SmallVector<Value> yields;
              flatten(forOp.getBody()->getTerminator()->getOperands(), yields,
                      nullptr);
              affine::AffineYieldOp::create(bb, bodyLoc, yields);
            });
        if (failed(status))
          return failure();
        regroup(forOp.getResults(), newFor.getResults(), counts);
      } else if (op->getNumRegions() != 0) {
        return op->emitError("unsupported op with regions in a vir.set_vl "
                             "with results");
      } else {
        bool onDyn = llvm::any_of(op->getOperandTypes(), isDyn) ||
                     llvm::any_of(op->getResultTypes(), isDyn);
        if (onDyn &&
            !isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
          return op->emitError("unsupported op on dynamic vectors in a "
                               "vir.set_vl with results");
        // an elementwise op on every piece, or any other op once
        int64_t n = onDyn ? pieces.size() : 1;
        SmallVector<SmallVector<Value, 4>> results(op->getNumResults());
        for (int64_t p = 0; p < n; ++p) {
          OperationState state(loc, op->getName());
          for (Value v : op->getOperands())
            state.addOperands(isDyn(v.getType()) ? get(v)[p] : one(v));
          for (Type t : op->getResultTypes())
            state.addTypes(isDyn(t) ? pieces.type(elemOf(t), p) : t);
          state.addAttributes(op->getAttrs());
          Operation *newOp = b.create(state);
          for (auto [i, r] : llvm::enumerate(newOp->getResults()))
            results[i].push_back(r);
        }
        for (auto [old, rs] : llvm::zip(op->getResults(), results))
          map[old] = rs;
      }
    }
    return success();
  }

  /// Lowers a set_vl with results: its vl must be a constant, and its
  /// region computes on whole dynamic vectors, each lowered to the pieces of
  /// `Pieces` (no strip-mined loop), so that values can flow from one
  /// piece to the next and out of the region.
  LogicalResult lowerWithResults(vir::SetVLOp op,
                                 ConversionPatternRewriter &rewriter,
                                 Type anchorType) const {
    std::optional<int64_t> vl = getConstantIntValue(op.getVl());
    if (!vl)
      return op.emitError("a vir.set_vl with results needs a constant vl");
    if (useScalable)
      return op.emitError("a vir.set_vl with results needs fixed vectors");
    int64_t width = useCustomVecWid
                        ? vectorWidth
                        : vectorTypeHelper(op.getRegion(), anchorType, false,
                                           verbose, rvvVlen)
                              .getShape()[0];
    Pieces pieces{width, *vl / width, *vl % width};
    // keep the lowered code small: vl over a few vectors
    if (pieces.size() > 64)
      return op.emitError("a vir.set_vl with results lowers to more than 64 "
                          "pieces; increase vector-width");
    PieceMap map;
    Block &block = op.getRegion().front();
    if (failed(lowerBlockInPieces(rewriter, op.getLoc(), &block, map, pieces)))
      return failure();
    SmallVector<Value> results;
    for (Value v : block.getTerminator()->getOperands()) {
      auto it = map.find(v);
      results.push_back(it != map.end() ? it->second.front() : v);
    }
    rewriter.replaceOp(op, results);
    return success();
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
    if (op.getNumResults() != 0)
      return lowerWithResults(op, rewriter, anchorType);
    WalkResult fromVector = region.walk([](vir::FromVectorOp from) {
      from.emitError("vir.from_vector needs a vir.set_vl with results");
      return WalkResult::interrupt();
    });
    if (fromVector.wasInterrupted())
      return failure();

    //===------------------------------------------------------------------===//
    // Step 2: Construct the target vector type.
    //===------------------------------------------------------------------===//

    // If the anchor dynamic vector has leading static dims (e.g.
    // !vir.vec<4x?xf32>), we materialize explicit outer loops for those dims
    // and lower the region body per-slice. This matches the current codegen
    // strategy that only vectorizes along the last (dynamic) dimension.
    SmallVector<int64_t> leadingStaticDims;
    for (Block &blk : region) {
      for (Operation &inner : blk) {
        for (Value result : inner.getResults()) {
          auto dynVecTy = dyn_cast<vir::DynamicVectorType>(result.getType());
          if (!dynVecTy)
            continue;
          ArrayRef<int64_t> shape = dynVecTy.getShape();
          if (shape.size() < 2)
            continue;
          // Only support static leading dims for now.
          bool ok = true;
          for (int64_t d : shape.drop_back()) {
            if (ShapedType::isDynamic(d)) {
              ok = false;
              break;
            }
          }
          if (!ok)
            continue;
          leadingStaticDims.assign(shape.begin(), shape.end() - 1);
          break;
        }
        if (!leadingStaticDims.empty())
          break;
      }
      if (!leadingStaticDims.empty())
        break;
    }

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
          vectorTypeHelper(region, anchorType, useScalable, verbose, rvvVlen);
      vectorWid = int(targetVectorType.getShape()[0]);
    }

    // A memref of i1 holds a byte per element, a vector of i1 a bit per
    // lane: fixed i1 accesses are scalarized per lane, which a scalable
    // vector (N * vscale lanes) cannot be yet.
    if (targetVectorType.isScalable()) {
      WalkResult i1Access = region.walk([](Operation *access) {
        Value base;
        if (auto load = dyn_cast<vir::LoadOp>(access))
          base = load.getBase();
        else if (auto store = dyn_cast<vir::StoreOp>(access))
          base = store.getBase();
        if (!base ||
            !cast<MemRefType>(base.getType()).getElementType().isInteger(1))
          return WalkResult::advance();
        access->emitError("a memref of i1 needs fixed vectors");
        return WalkResult::interrupt();
      });
      if (i1Access.wasInterrupted())
        return failure();
    }

    //===------------------------------------------------------------------===//
    // Step 3: Create a main vectorization loop.
    //===------------------------------------------------------------------===//
    // If the anchor dynamic vector has leading static dims (e.g.
    // !vir.vec<4x?xf32>), we materialize explicit outer loops for those dims
    // and lower the region body per-slice. The innermost loop still
    // vectorizes along the last (dynamic) dimension.
    auto emitVectorAndTail = [&](OpBuilder &b, ArrayRef<Value> leadingIVs) {
      Value vlValue = op.getVl();
      Value vlStep = arith::ConstantIndexOp::create(b, loc, vectorWid);
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      Value tailStart;
      if (cast<VectorType>(targetVectorType).isScalable()) {
        // A scalable vector has vectorWid * vscale lanes: the step, and with
        // it the end of the main loop, are runtime values.
        Value step = arith::MulIOp::create(
            b, loc, vlStep, vector::VectorScaleOp::create(b, loc));
        Value vlRem = arith::RemSIOp::create(b, loc, vlValue, step);
        tailStart = arith::SubIOp::create(b, loc, vlValue, vlRem);
        scf::ForOp::create(
            b, loc, zero, tailStart, step, /*initArgs=*/ValueRange{},
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange) {
              DenseMap<Value, Value> virSymbolTable;
              lowerBlock(bb, bodyLoc, &region.front(), virSymbolTable, iv,
                         leadingIVs, targetVectorType, /*isTailLoop=*/false);
              scf::YieldOp::create(bb, bodyLoc);
            });
      } else {
        Value vlUpboundPat = arith::SubIOp::create(b, loc, vlValue, vlStep);
        Value vlUpbound = arith::AddIOp::create(
            b, loc, vlUpboundPat, arith::ConstantIndexOp::create(b, loc, 1));

        affine::AffineForOp::create(
            b, loc, /*lowerBound=*/ValueRange{zero},
            rewriter.getDimIdentityMap(),
            /*upperBound=*/ValueRange{vlUpbound}, rewriter.getDimIdentityMap(),
            vectorWid,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange) {
              //===------------------------------------------------------------===//
              // Step 4: Convert operations inside the dynamic vector region.
              //===------------------------------------------------------------===//
              DenseMap<Value, Value> virSymbolTable;
              lowerBlock(bb, bodyLoc, &region.front(), virSymbolTable, iv,
                         leadingIVs, targetVectorType, /*isTailLoop=*/false);
              affine::AffineYieldOp::create(bb, bodyLoc);
            });

        //===----------------------------------------------------------------===//
        // Step 5: Create a tail loop to process the remaining elements.
        //===----------------------------------------------------------------===//
        Value vlRem = arith::RemSIOp::create(b, loc, vlValue, vlStep);
        tailStart = arith::SubIOp::create(b, loc, vlValue, vlRem);
      }

      auto lowerTail = [&](OpBuilder &bb, Location bodyLoc, Value iv) {
        DenseMap<Value, Value> virSymbolTable;
        lowerBlock(bb, bodyLoc, &region.front(), virSymbolTable, iv, leadingIVs,
                   /*targetVectorType=*/Type{},
                   /*isTailLoop=*/true);
      };
      if (cast<VectorType>(targetVectorType).isScalable()) {
        // Inside the loops over leading static dims, the scalable tailStart
        // is not a valid affine symbol (its remsi divisor is not provably
        // nonzero), so the tail loop is an scf.for as well.
        scf::ForOp::create(
            b, loc, tailStart, vlValue,
            arith::ConstantIndexOp::create(b, loc, 1),
            /*initArgs=*/ValueRange{},
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange) {
              lowerTail(bb, bodyLoc, iv);
              scf::YieldOp::create(bb, bodyLoc);
            });
      } else {
        affine::AffineForOp::create(
            b, loc, /*lowerBound=*/ValueRange{tailStart},
            rewriter.getDimIdentityMap(),
            /*upperBound=*/ValueRange{vlValue}, rewriter.getDimIdentityMap(),
            /*step=*/1,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange) {
              lowerTail(bb, bodyLoc, iv);
              affine::AffineYieldOp::create(bb, bodyLoc);
            });
      }
    };

    if (leadingStaticDims.empty()) {
      emitVectorAndTail(rewriter, /*leadingIVs=*/{});
    } else {
      // Build nested loops over the leading static dims.
      SmallVector<Value> ivs;
      std::function<void(unsigned, OpBuilder &)> emitOuter = [&](unsigned dim,
                                                                 OpBuilder &b) {
        if (dim == leadingStaticDims.size()) {
          emitVectorAndTail(b, ivs);
          return;
        }
        int64_t ub = leadingStaticDims[dim];
        Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
        Value cub = arith::ConstantIndexOp::create(b, loc, ub);
        affine::AffineForOp::create(
            b, loc, ValueRange{c0}, rewriter.getDimIdentityMap(),
            ValueRange{cub}, rewriter.getDimIdentityMap(), /*step=*/1,
            /*iterArgs=*/ValueRange{},
            [&](OpBuilder &bb, Location bodyLoc, Value iv, ValueRange) {
              ivs.push_back(iv);
              emitOuter(dim + 1, bb);
              ivs.pop_back();
              affine::AffineYieldOp::create(bb, bodyLoc);
            });
      };
      emitOuter(/*dim=*/0, rewriter);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVIRToVectorConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           bool useScalable, int vectorWidth,
                                           bool verbose, bool useCustomVecWid,
                                           int rvvVlen) {
  // clang-format off
  patterns.add<VIRSetVLLowering>(converter, useScalable, vectorWidth, verbose, useCustomVecWid, rvvVlen);
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
        math::MathDialect,
        memref::MemRefDialect,
        affine::AffineDialect,
        scf::SCFDialect,
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

  Option<int> rvvVlen{
      *this, "rvv-vlen",
      llvm::cl::desc("VLEN in bits of a RISC-V Vector target: without "
                     "vector-width, the vector size comes from its 32 "
                     "registers of VLEN bits instead of the host's; with "
                     "use-scalable, the minimum VLEN"),
      llvm::cl::init(0)};

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
      math::MathDialect,
      memref::MemRefDialect,
      vector::VectorDialect,
      affine::AffineDialect,
      scf::SCFDialect,
      LLVM::LLVMDialect
    >();
  // clang-format on
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  LLVMTypeConverter converter(context);
  RewritePatternSet patterns(context);
  // RVV: VLEN is a power of two of at least 32 bits
  if (rvvVlen != 0 &&
      (rvvVlen < 32 || !llvm::has_single_bit(unsigned(rvvVlen)))) {
    module.emitError("rvv-vlen must be a power of two >= 32, got ")
        << int(rvvVlen);
    return signalPassFailure();
  }
  // use custome vector width
  bool useCustomVecWid = vectorWidth.getNumOccurrences() > 0;
  // vscale depends on the target (VLEN / 64 on RVV, VL / 128 on SVE): the
  // size of a scalable vector is only chosen for RVV
  if (useScalable && !useCustomVecWid && rvvVlen == 0) {
    module.emitError("use-scalable without vector-width needs rvv-vlen");
    return signalPassFailure();
  }
  populateVIRToVectorConversionPatterns(converter, patterns, useScalable,
                                        vectorWidth, verbose, useCustomVecWid,
                                        rvvVlen);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerVIRToVectorPass() { PassRegistration<VIRToVectorPass>(); }
} // namespace buddy
} // namespace mlir
