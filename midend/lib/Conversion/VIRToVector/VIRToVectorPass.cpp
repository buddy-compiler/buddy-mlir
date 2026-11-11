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

#include <numeric>

#include "mlir/Analysis/AliasAnalysis.h"
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
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
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

/// Whether the rows of a strip-mined vir.set_vl may run in parallel.
static bool rowsAreIndependent(vir::SetVLOp op, size_t rows);

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
/// `scalar` broadcast to `vecTy`. The LLVM lowering of vector.broadcast
/// takes vector<[1]xT> for one lane (its splat shufflevector of mask [0]
/// folds away) and leaves the other vscale - 1 lanes poison: a broadcast to
/// vector<[2]xT> and its first half instead. The fold is
/// LLVM::ShuffleVectorOp::fold (LLVMDialect.cpp), which takes a vector of
/// one known-minimum element for a vector of one element; this can go once
/// it leaves scalable vectors alone.
static Value broadcastTo(OpBuilder &builder, Location loc, VectorType vecTy,
                         Value scalar) {
  if (vecTy.getRank() != 1 || vecTy.getShape()[0] != 1 || !vecTy.isScalable())
    return vector::BroadcastOp::create(builder, loc, vecTy, scalar);
  Value wide = vector::BroadcastOp::create(
      builder, loc, VectorType::get({2}, vecTy.getElementType(), {true}),
      scalar);
  return vector::ScalableExtractOp::create(builder, loc, vecTy, wide, 0);
}

/// The reduction of `vec` into `acc`. vector.reduction takes vector<[1]xT>
/// for a vector of one element: its canonicalization
/// (ElideSingleElementReduction, VectorOps.cpp) rewrites the reduction to
/// vector.extract of lane 0 and drops the other vscale - 1 lanes. The
/// reduction of a floating-point vector<[1]xT> is therefore built from the
/// LLVM intrinsics vector.reduction lowers to; this can go once the pattern
/// leaves scalable vectors alone.
static Value reduceVector(OpBuilder &builder, Location loc,
                          vector::CombiningKind kind, Value vec, Value acc,
                          arith::FastMathFlags fastmath) {
  auto vecTy = cast<VectorType>(vec.getType());
  Type elem = vecTy.getElementType();
  bool one = vecTy.getRank() == 1 && vecTy.getShape()[0] == 1 &&
             vecTy.isScalable() && isa<FloatType>(elem);
  if (one && (kind == vector::CombiningKind::ADD ||
              kind == vector::CombiningKind::MAXNUMF)) {
    bool reassoc =
        arith::bitEnumContainsAll(fastmath, arith::FastMathFlags::reassoc);
    auto fmf = LLVM::FastmathFlagsAttr::get(
        builder.getContext(),
        reassoc ? LLVM::FastmathFlags::reassoc : LLVM::FastmathFlags::none);
    if (kind == vector::CombiningKind::ADD)
      return LLVM::vector_reduce_fadd::create(builder, loc, elem, acc, vec,
                                              fmf);
    Value max = LLVM::vector_reduce_fmax::create(builder, loc, elem, vec, fmf);
    return arith::MaxNumFOp::create(builder, loc, max, acc);
  }
  return vector::ReductionOp::create(builder, loc, kind, vec, acc, fastmath);
}

static bool shouldScalarizeI1VectorMemoryOp(VectorType vecTy) {
  return vecTy && !vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getElementType().isInteger(1);
}

// A memref of i1 holds a byte per element, a vector of i1 a bit per lane:
// fixed i1 accesses are scalarized per lane. A scalable vector (N * vscale
// lanes) is read and written as the bytes of the memref, a vector<[N]xi8>
// at the address of the first element, truncated to / extended from the i1
// lanes: a plain load / store when the elements are contiguous (a static
// unit stride along the lanes), else a strided one (the VP intrinsics, all
// lanes on).
static bool isScalableI1(VectorType vecTy) {
  return vecTy && vecTy.isScalable() && vecTy.getRank() == 1 &&
         vecTy.getElementType().isInteger(1);
}

namespace {
/// A scalable access of a memref of i1: the address of its first element
/// and what a strided access needs.
struct I1Bytes {
  VectorType type; // vector<[N]xi8>
  Value pointer;   // of the first element
  bool contiguous;
  Value stride, mask, evl; // when not contiguous
};
} // namespace

static I1Bytes i1Bytes(OpBuilder &builder, Location loc, VectorType vecTy,
                       Value base, ArrayRef<Value> indices) {
  I1Bytes bytes;
  bytes.type = VectorType::get(vecTy.getShape(), builder.getI8Type(),
                               vecTy.getScalableDims());
  // the aligned pointer plus the linear index of the element, in bytes
  auto metadata = memref::ExtractStridedMetadataOp::create(builder, loc, base);
  Value linear = metadata.getOffset();
  for (auto [index, stride] : llvm::zip(indices, metadata.getStrides()))
    linear = arith::AddIOp::create(
        builder, loc, linear,
        arith::MulIOp::create(builder, loc, index, stride));
  Value address = arith::AddIOp::create(
      builder, loc,
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, base),
      linear);
  Type i64 = builder.getI64Type();
  Value bits = arith::IndexCastOp::create(builder, loc, i64, address);
  bytes.pointer = LLVM::IntToPtrOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()), bits);
  bytes.contiguous = hasStaticallyUnitStrideInVectorizedDim(base);
  if (!bytes.contiguous) {
    bytes.stride = arith::IndexCastOp::create(builder, loc, i64,
                                              metadata.getStrides().back());
    Value on =
        arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
    bytes.mask = broadcastTo(builder, loc, vecTy, on);
    Value lanes = arith::MulIOp::create(
        builder, loc,
        arith::ConstantIndexOp::create(builder, loc, vecTy.getShape()[0]),
        vector::VectorScaleOp::create(builder, loc));
    bytes.evl =
        arith::IndexCastOp::create(builder, loc, builder.getI32Type(), lanes);
  }
  return bytes;
}

static Value buildScalableI1VectorLoad(OpBuilder &builder, Location loc,
                                       VectorType vecTy, Value base,
                                       ArrayRef<Value> indices) {
  I1Bytes at = i1Bytes(builder, loc, vecTy, base, indices);
  Value bytes =
      at.contiguous
          ? Value(LLVM::LoadOp::create(builder, loc, at.type, at.pointer,
                                       /*alignment=*/1))
          : Value(LLVM::VPStridedLoadOp::create(
                builder, loc, at.type, at.pointer, at.stride, at.mask, at.evl));
  return arith::TruncIOp::create(builder, loc, vecTy, bytes);
}

static void buildScalableI1VectorStore(OpBuilder &builder, Location loc,
                                       Value vectorValue, Value base,
                                       ArrayRef<Value> indices) {
  auto vecTy = cast<VectorType>(vectorValue.getType());
  I1Bytes at = i1Bytes(builder, loc, vecTy, base, indices);
  Value bytes = arith::ExtUIOp::create(builder, loc, at.type, vectorValue);
  if (at.contiguous)
    LLVM::StoreOp::create(builder, loc, bytes, at.pointer, /*alignment=*/1);
  else
    LLVM::VPStridedStoreOp::create(builder, loc, bytes, at.pointer, at.stride,
                                   at.mask, at.evl);
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

/// The VLEN in bits of the host when it is a RISC-V CPU with V, else 0. The
/// fixed vectors of a host are sized as for an rvv-vlen of it.
static int hostRvvVlen() {
#if defined(__riscv)
  llvm::StringMap<bool> Features = llvm::sys::getHostCPUFeatures();
  if (Features.count("v") && Features["v"]) {
    unsigned long vlenb;
    // vlenb (CSR 0xc22) by number: the assembler knows its name only with
    // V, which buddy-opt need not be built with
    __asm__ volatile("csrr %0, 0xc22" : "=r"(vlenb));
    return int(vlenb * 8);
  }
#endif
  return 0;
}

/// The vector type of a dynamic vector region from its register pressure
/// (Req_Group, the most dynamic vectors live at once) and the vector
/// registers of the target: with `rvvVlen` (bits, > 0), a RISC-V Vector
/// target with 32 registers of that VLEN, whatever the host; otherwise the
/// host's registers. With `useScalable` (RVV only), the vector is scalable,
/// of the register group it is at VLEN `rvvVlen`. An index element has
/// `indexBitWidth` bits. `fits`, when given, is set to whether the vectors
/// live at once fit in the registers.
VectorType vectorTypeHelper(Region &srcRegion, Type anchorType,
                            bool useScalable, bool verbose, int rvvVlen,
                            unsigned indexBitWidth, int64_t rows = 1,
                            bool *fits = nullptr) {
#define VERBOSE_ANALYSIS_INFO(info)                                            \
  if (verbose) {                                                               \
    llvm::outs() << info;                                                      \
  }
  // Defaults when no target-specific features are detected (hosts not
  // enumerated below; a RISC-V host with V is an rvv-vlen, hostRvvVlen). A
  // width of 1 made sizeVec zero for f32 (VectorType::get asserts); 128-bit is
  // a reasonable generic SIMD nominal.
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
      totalVectorRegs = 32;
#if defined(__ARM_FEATURE_SVE)
      uint64_t bytes = svcntb();
      vectorRegBitWidth = bytes * 8;
#else
      // the vector length cannot be read without SVE code: the least, 128
      // bits (the fixed vectors are valid on any)
      vectorRegBitWidth = 128;
      VERBOSE_ANALYSIS_INFO("    built without SVE: its least vector "
                            "length, 128 bits\n");
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
  // the live range in each block of its dynamic vectors, as
  // LivenessBlockInfo::currentlyLiveValues has them, found once per block
  using LiveRange = std::tuple<Value, Operation *, Operation *>;
  DenseMap<Block *, SmallVector<LiveRange>> liveRanges;
  auto liveRangesIn = [&](Block *block) -> ArrayRef<LiveRange> {
    auto [it, inserted] = liveRanges.try_emplace(block);
    if (!inserted)
      return it->second;
    const LivenessBlockInfo *info = liveness.getLiveness(block);
    auto add = [&](Value v) {
      if (!isDynamic(v))
        return;
      Operation *start = info->isLiveIn(v) || isa<BlockArgument>(v)
                             ? &block->front()
                             : block->findAncestorOpInBlock(*v.getDefiningOp());
      Operation *end =
          info->isLiveOut(v) ? &block->back() : info->getEndOperation(v, start);
      it->second.emplace_back(v, start, end);
    };
    for (Value arg : block->getArguments())
      add(arg);
    for (Value in : info->in())
      add(in);
    for (Operation &op : *block)
      for (Value result : op.getResults())
        add(result);
    return it->second;
  };
  // the values live across each loop, the same at all its ops
  DenseMap<Operation *, SmallVector<Value>> liveAcross;
  auto liveAcrossLoop = [&](Operation *loop) -> ArrayRef<Value> {
    auto [it, inserted] = liveAcross.try_emplace(loop);
    if (!inserted)
      return it->second;
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
      it->second.push_back(v);
    }
    return it->second;
  };
  unsigned maxActiveVectors = 0;
  srcRegion.walk([&](Operation *op) {
    // an op with regions is measured at the ops of its blocks
    if (op->getNumRegions() != 0)
      return;
    DenseSet<Value> live;
    for (auto [v, start, end] : liveRangesIn(op->getBlock()))
      if (!op->isBeforeInBlock(start) && !end->isBeforeInBlock(op))
        live.insert(v);
    for (Operation *loop = op->getParentOp(); loop != setVL;
         loop = loop->getParentOp())
      live.insert_range(liveAcrossLoop(loop));
    if (verbose) {
      llvm::outs() << "  " << live.size() << " live at ";
      op->print(llvm::outs(), OpPrintingFlags().skipRegions());
      llvm::outs() << "\n";
    }
    maxActiveVectors = std::max<unsigned>(maxActiveVectors, live.size());
  });

  int reqGroup = (maxActiveVectors == 0) ? 1 : maxActiveVectors;
  if (fits)
    *fits = reqGroup <= totalVectorRegs;
  VERBOSE_ANALYSIS_INFO("Final Max Active Vectors (Req_Group): " << reqGroup
                                                                 << "\n");

  VERBOSE_ANALYSIS_INFO("\n--- Calculating Final Vector Type ---\n");

  // index as the LLVM lowering has it
  unsigned elementBitWidth =
      anchorType.isIndex() ? indexBitWidth : anchorType.getIntOrFloatBitWidth();
  // an i1 lane is a byte of its memref (and of the vector<[N]xi8> a scalable
  // access reads): the vectors of a region of i1 are sized for bytes
  if (elementBitWidth == 1)
    elementBitWidth = 8;
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

  // no wider than a dynamic vector of a constant vl (`rows` rows of vl
  // lanes): the lanes past it would compute nothing
  if (std::optional<int64_t> vl =
          getConstantIntValue(cast<vir::SetVLOp>(setVL).getVl());
      vl && *vl > 0 && sizeVec > *vl * rows) {
    sizeVec = llvm::bit_floor(uint64_t(*vl * rows));
    VERBOSE_ANALYSIS_INFO(
        "Clamped to the constant vl "
        << *vl
        << (rows > 1 ? " x " + std::to_string(rows) + " rows" : std::string())
        << ": Size_vec = " << sizeVec << "\n");
  }

  if (sizeVec < 1) {
    emitWarning(srcRegion.getLoc(),
                "Calculated vector lane count is zero; clamping to 1. "
                "Consider passing -lower-vir-to-vector=vector-width=N.");
    sizeVec = 1;
  }

  if (useScalable) {
    // RVV: vscale = VLEN / 64, so vector<[N]xT> is the register group of
    // LMUL = N * SEW / 64 registers whatever the VLEN; at VLEN rvvVlen
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
                   int rvvVlen, bool rvvVlenExact)
      : ConvertOpToLLVMPattern<vir::SetVLOp>(converter),
        useScalable(useScalable), vectorWidth(vectorWidth), verbose(verbose),
        useCustomVecWid(useCustomVecWid), rvvVlen(rvvVlen),
        rvvVlenExact(rvvVlenExact) {}

private:
  bool useScalable;
  int vectorWidth;
  bool verbose;
  bool useCustomVecWid;
  // VLEN of a RISC-V Vector target in bits, 0 for the host
  int rvvVlen;
  // rvvVlen is the VLEN of the target, not only its minimum (use-scalable
  // without rvv-vlen)
  bool rvvVlenExact;
  // set when lowerBlock meets an op it does not lower
  mutable bool unsupportedOp = false;

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
    // The memref indices of an access of `vecType` at `indices` (none: all
    // 0), as for vector.load: the lanes are along the last dim of the
    // memref, from the main loop IV on, and the rows of a multi-dimensional
    // dynamic vector along the dims before it, from the leading IVs on.
    auto accessIndices = [&](Value base, ValueRange indices, Type vecType) {
      int64_t rank = cast<MemRefType>(base.getType()).getRank();
      int64_t rows =
          cast<vir::DynamicVectorType>(vecType).getShape().size() - 1;
      // (checked before the lowering: the dynamic vectors are of the region,
      // their rows those of the leading IVs)
      assert(rows == 0 || rows == int64_t(leadingIVs.size()));
      SmallVector<Value> out;
      for (int64_t i = 0; i < rank; ++i) {
        Value iv = i == rank - 1          ? mainLoopIV
                   : i >= rank - 1 - rows ? leadingIVs[i - (rank - 1 - rows)]
                                          : Value();
        if (indices.empty())
          out.push_back(
              iv ? iv : Value(arith::ConstantIndexOp::create(builder, loc, 0)));
        else if (iv)
          out.push_back(
              arith::AddIOp::create(builder, loc, iv, findValue(indices[i])));
        else
          out.push_back(findValue(indices[i]));
      }
      return out;
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
            SmallVector<Value> destIndices =
                accessIndices(base, op.getIndices(), op.getType());

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
              if (isScalableI1(*vecTyOr)) {
                virSymbolTable[op.getResult()] = buildScalableI1VectorLoad(
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
            SmallVector<Value> destIndices =
                accessIndices(base, op.getIndices(), op.getValue().getType());

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
              if (isScalableI1(dyn_cast<VectorType>(valueToStore.getType()))) {
                buildScalableI1VectorStore(builder, loc, valueToStore, base,
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
              virSymbolTable[op.getResult()] =
                  broadcastTo(builder, loc, *vecTyOr, scalarValue);
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
            // One rounding in every lane: the tail as the vector.
            if (isTailLoop) {
              virSymbolTable[op.getResult()] =
                  math::FmaOp::create(builder, loc, lhs, rhs, acc);
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

            virSymbolTable[op.getResult()] =
                reduceVector(builder, loc, combineKind, input, acc,
                             op.getReassoc() ? arith::FastMathFlags::reassoc
                                             : arith::FastMathFlags::none);
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
              one = broadcastTo(builder, loc, cast<VectorType>(outTy), one);
            auto div = arith::DivFOp::create(builder, loc, one, sqrt);
            virSymbolTable[op.getResult()] = div.getResult();
          })
          .Case<math::LogOp, math::AbsFOp, math::CeilOp, math::FloorOp,
                math::RoundOp, math::TanhOp, math::ErfOp, math::PowFOp>(
              [&](Operation *op) { cloneMappedOp(op); })
          .Default([&](Operation *op) {
            // dropping it would drop its effect: fail the lowering (once
            // per region, which is lowered once per chunk)
            if (!unsupportedOp)
              op->emitError("unsupported operation in a strip-mined "
                            "vir.set_vl: ")
                  << op->getName();
            unsupportedOp = true;
          });
    }
  }

  /// The pieces a dynamic vector of a set_vl with results is lowered to:
  /// `numVectors` vectors of `lanes` lanes, then `numScalars` scalars. A
  /// dynamic vector of static layout `layout` (e.g. [2] for
  /// !vir.vec<2x?xT>) has its rows of vl lanes one after the other, row
  /// major, in rows * vl lanes. The vectors are vector<widthxT> (lanes =
  /// width), or with `scalable` vector<[width]xT>, of width * vscale lanes
  /// at the vscale of rvv-vlen, the VLEN of the target.
  struct Pieces {
    int64_t width, numVectors, numScalars;
    bool scalable = false;
    int64_t lanes = width;
    SmallVector<int64_t> layout;
    int64_t vl = 0;
    int64_t size() const { return numVectors + numScalars; }
    // the number of lanes, rows * vl
    int64_t length() const { return numVectors * lanes + numScalars; }
    bool isVector(int64_t p) const { return p < numVectors; }
    // the first lane of piece p
    int64_t fixedOffset(int64_t p) const {
      return isVector(p) ? p * lanes : numVectors * lanes + p - numVectors;
    }
    Value offset(OpBuilder &b, Location loc, int64_t p) const {
      return arith::ConstantIndexOp::create(b, loc, fixedOffset(p));
    }
    Type type(Type elem, int64_t p) const {
      return isVector(p) ? Type(VectorType::get({width}, elem, {scalable}))
                         : elem;
    }
  };
  using PieceMap = DenseMap<Value, SmallVector<Value, 4>>;

  /// Lowers the ops of `block` but its terminator for a set_vl with results:
  /// each op on dynamic vectors applies to every piece, a vir.reduce folds
  /// the pieces in order (with reassoc, its vector pieces added lane by lane
  /// first) and loops carry all pieces of their dynamic vector values. `map`
  /// maps a value of the region to its pieces (one for a value that is not a
  /// dynamic vector).
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
    // The memref and indices of the first lane of an access at `indices`
    // (none: all 0). The rows of a dynamic vector are the dims before the
    // last one of the memref, as for vector.load: the memref must hold them
    // one after the other, and these dims are collapsed into the last one.
    auto access = [&](Operation *op, Value base, ValueRange indices)
        -> FailureOr<std::pair<Value, SmallVector<Value>>> {
      auto type = cast<MemRefType>(base.getType());
      int64_t rank = type.getRank(), k = pieces.layout.size();
      SmallVector<Value> idx;
      for (int64_t i = 0; i < rank; ++i)
        idx.push_back(indices.empty()
                          ? Value(arith::ConstantIndexOp::create(b, loc, 0))
                          : one(indices[i]));
      if (k == 0)
        return std::make_pair(base, idx);
      SmallVector<ReassociationIndices> groups;
      for (int64_t i = 0; i + k + 1 < rank; ++i)
        groups.push_back({i});
      ReassociationIndices rows;
      for (int64_t i = std::max<int64_t>(0, rank - k - 1); i < rank; ++i)
        rows.push_back(i);
      groups.push_back(rows);
      // the rows from the first one on: the dims after the first one of the
      // layout match it, the last dim has vl elements
      ArrayRef<int64_t> shape = type.getShape();
      SmallVector<int64_t> inner(ArrayRef(pieces.layout).drop_front());
      inner.push_back(pieces.vl);
      if (rank < k + 1 || shape.take_back(k) != ArrayRef(inner) ||
          !memref::CollapseShapeOp::isGuaranteedCollapsible(type, groups))
        return op->emitError("the rows of a multi-dimensional dynamic vector "
                             "must be one after the other in memory");
      Value flat = memref::CollapseShapeOp::create(b, loc, base, groups);
      // the index of the first lane in the collapsed dim
      Value index = idx[rows.front()];
      for (int64_t i : ArrayRef(rows).drop_front()) {
        index = arith::MulIOp::create(
            b, loc, index, arith::ConstantIndexOp::create(b, loc, shape[i]));
        index = arith::AddIOp::create(b, loc, index, idx[i]);
      }
      idx.resize(rows.front());
      idx.push_back(index);
      return std::make_pair(flat, idx);
    };
    // the memref indices of piece p of an access at `start`
    auto pieceIndices = [&](ArrayRef<Value> start, int64_t p) {
      SmallVector<Value> out(start.drop_back());
      out.push_back(arith::AddIOp::create(b, loc, start.back(),
                                          pieces.offset(b, loc, p)));
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
        if (dyn && (dyn.getShape().drop_back() != ArrayRef(pieces.layout) ||
                    dyn.getScalingFactor()))
          return op->emitError("the dynamic vectors of a vir.set_vl with "
                               "results must have the same static layout and "
                               "no scaling factor");
      }

      if (auto load = dyn_cast<vir::LoadOp>(op)) {
        auto at = access(op, one(load.getBase()), load.getIndices());
        if (failed(at))
          return failure();
        Value base = at->first;
        Type elem = elemOf(load.getType());
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          SmallVector<Value> idx = pieceIndices(at->second, p);
          if (!pieces.isVector(p)) {
            out.push_back(memref::LoadOp::create(b, loc, base, idx));
            continue;
          }
          auto vecTy = cast<VectorType>(pieces.type(elem, p));
          if (pieces.scalable) {
            if (!hasStaticallyUnitStrideInVectorizedDim(base))
              return op->emitError("a vir.set_vl with results on scalable "
                                   "vectors needs unit-stride accesses");
            out.push_back(
                isScalableI1(vecTy)
                    ? buildScalableI1VectorLoad(b, loc, vecTy, base, idx)
                    : Value(vector::LoadOp::create(b, loc, vecTy, base, idx)));
          } else if (shouldScalarizeI1VectorMemoryOp(vecTy))
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
        auto at = access(op, one(store.getBase()), store.getIndices());
        if (failed(at))
          return failure();
        Value base = at->first;
        SmallVector<Value, 4> values = get(store.getValue());
        for (int64_t p = 0; p < pieces.size(); ++p) {
          SmallVector<Value> idx = pieceIndices(at->second, p);
          if (!pieces.isVector(p))
            memref::StoreOp::create(b, loc, values[p], base, idx);
          else if (pieces.scalable) {
            if (!hasStaticallyUnitStrideInVectorizedDim(base))
              return op->emitError("a vir.set_vl with results on scalable "
                                   "vectors needs unit-stride accesses");
            if (isScalableI1(cast<VectorType>(values[p].getType())))
              buildScalableI1VectorStore(b, loc, values[p], base, idx);
            else
              vector::StoreOp::create(b, loc, values[p], base, idx);
          } else if (shouldScalarizeI1VectorMemoryOp(
                         cast<VectorType>(values[p].getType())))
            buildScalarizedI1VectorStore(b, loc, values[p], base, idx);
          else if (hasStaticallyUnitStrideInVectorizedDim(base))
            vector::StoreOp::create(b, loc, values[p], base, idx);
          else
            vector::TransferWriteOp::create(b, loc, values[p], base, idx);
        }
      } else if (auto shuffle = dyn_cast<vir::ShuffleOp>(op)) {
        // Row r of the result is row s = mask[r] of v1 then v2.
        ArrayRef<int64_t> mask = shuffle.getMask();
        int64_t rows = pieces.layout.front();
        int64_t rowLanes =
            pieces.vl * llvm::product_of(ArrayRef(pieces.layout).drop_front());
        if (!pieces.scalable) {
          // fixed: one vector of all the lanes, a vector.shuffle
          if (pieces.size() != 1 || !pieces.isVector(0))
            return op->emitError("vir.shuffle needs a dynamic vector in one "
                                 "vector of the target");
          Value v1 = one(shuffle.getV1()), v2 = one(shuffle.getV2());
          SmallVector<int64_t> lanes;
          for (int64_t s : mask)
            for (int64_t c = 0; c < rowLanes; ++c)
              lanes.push_back(s < rows
                                  ? s * rowLanes + c
                                  : pieces.width + (s - rows) * rowLanes + c);
          map[shuffle.getResult()] = {
              vector::ShuffleOp::create(b, loc, v1, v2, lanes)};
          continue;
        }
        // Scalable: every lane is known (vscale = rvv-vlen / 64). Lane l of
        // the result is lane (s mod rows) * rowLanes + l mod rowLanes of v1
        // (s < rows) or v2, s = mask[l / rowLanes]. A piece of the result is
        // made of runs of lanes from one piece of v1 or v2: a whole piece is
        // that piece; a run of whole blocks of vscale lanes is taken and put
        // in place (vector.scalable.extract / insert, register moves); other
        // runs are slid into place and selected from their first lane on
        // (the runs come in order: the next ones set the lanes after).
        SmallVector<Value, 4> s1 = get(shuffle.getV1()),
                              s2 = get(shuffle.getV2()), out;
        int64_t vscale = pieces.lanes / pieces.width;
        auto pieceOf = [&](int64_t lane) {
          int64_t vectorLanes = pieces.numVectors * pieces.lanes;
          return lane < vectorLanes ? lane / pieces.lanes
                                    : pieces.numVectors + lane - vectorLanes;
        };
        // src slid down by `shift` lanes (up when negative): lane i is lane
        // i + shift of src, the lanes shifted in are poison
        auto slide = [&](Value src, int64_t shift) -> Value {
          if (shift == 0)
            return src;
          auto vecTy = cast<VectorType>(src.getType());
          Value by = arith::ConstantOp::create(
              b, loc, b.getI32IntegerAttr(std::abs(shift)));
          // the intrinsic takes LLVM types: index as an integer
          VectorType llvmTy = vecTy;
          if (vecTy.getElementType().isIndex()) {
            llvmTy = vecTy.clone(
                b.getIntegerType(getTypeConverter()->getIndexTypeBitwidth()));
            src = arith::IndexCastOp::create(b, loc, llvmTy, src);
          }
          Value none = ub::PoisonOp::create(b, loc, llvmTy);
          Value moved =
              LLVM::CallIntrinsicOp::create(
                  b, loc, llvmTy,
                  b.getStringAttr(shift > 0 ? "llvm.vector.splice.left"
                                            : "llvm.vector.splice.right"),
                  shift > 0 ? ValueRange{src, none, by}
                            : ValueRange{none, src, by})
                  .getResult(0);
          return llvmTy != vecTy
                     ? Value(arith::IndexCastOp::create(b, loc, vecTy, moved))
                     : moved;
        };
        // `res` with its lanes from `from` on those of `value`
        auto selectFrom = [&](Value res, int64_t from, Value value) -> Value {
          if (!res)
            return value;
          auto vecTy = cast<VectorType>(value.getType());
          Value before = vector::CreateMaskOp::create(
              b, loc, VectorType::get(vecTy.getShape(), b.getI1Type(), {true}),
              ValueRange{arith::ConstantIndexOp::create(b, loc, from)});
          return arith::SelectOp::create(b, loc, before, res, value);
        };
        for (int64_t p = 0; p < pieces.size(); ++p) {
          int64_t first = pieces.fixedOffset(p);
          int64_t n = pieces.isVector(p) ? pieces.lanes : 1;
          Type elem = elemOf(shuffle.getType());
          Value res;
          for (int64_t l = 0; l < n;) {
            int64_t lane = first + l, s = mask[lane / rowLanes];
            int64_t c = lane % rowLanes;
            int64_t from = s % rows * rowLanes + c, q = pieceOf(from);
            int64_t a = from - pieces.fixedOffset(q);
            int64_t qn = pieces.isVector(q) ? pieces.lanes : 1;
            int64_t len = std::min({n - l, rowLanes - c, qn - a});
            Value src = (s < rows ? s1 : s2)[q];
            if (!pieces.isVector(p)) {
              // a scalar: lane a of a vector, slid down to lane 0
              res = pieces.isVector(q)
                        ? Value(vector::ExtractOp::create(b, loc, slide(src, a),
                                                          ArrayRef<int64_t>{0}))
                        : src;
            } else if (!pieces.isVector(q)) {
              res = selectFrom(
                  res, l,
                  broadcastTo(b, loc, cast<VectorType>(pieces.type(elem, p)),
                              src));
            } else if (len == n) {
              res = src;
            } else if (l % vscale == 0 && a % vscale == 0 &&
                       len % vscale == 0) {
              // blocks of g * vscale lanes, g the largest power of two that
              // divides the positions, the length and the width
              int64_t g = std::gcd(std::gcd(l / vscale, a / vscale),
                                   std::gcd(len / vscale, pieces.width));
              g = int64_t(llvm::bit_floor(uint64_t(g & -g)));
              if (!res)
                res = ub::PoisonOp::create(b, loc, pieces.type(elem, p));
              auto subTy = VectorType::get({g}, elem, {true});
              for (int64_t k = 0; k < len / vscale; k += g) {
                Value sub = vector::ScalableExtractOp::create(
                    b, loc, subTy, src, a / vscale + k);
                res = vector::ScalableInsertOp::create(b, loc, sub, res,
                                                       l / vscale + k);
              }
            } else {
              res = selectFrom(res, l, slide(src, a - l));
            }
            l += len;
          }
          out.push_back(res);
        }
        map[shuffle.getResult()] = out;
      } else if (auto from = dyn_cast<vir::FromVectorOp>(op)) {
        Value src = one(from.getSource());
        auto srcTy = cast<VectorType>(src.getType());
        if (pieces.scalable)
          return op->emitError("vir.from_vector needs fixed vectors");
        if (srcTy.isScalable() || srcTy.getNumElements() != pieces.length())
          return op->emitError("vir.from_vector needs a fixed vector of vl "
                               "lanes");
        SmallVector<Value, 4> out;
        for (int64_t p = 0; p < pieces.size(); ++p) {
          int64_t off = pieces.fixedOffset(p);
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
          out.push_back(pieces.isVector(p)
                            ? broadcastTo(b, loc,
                                          cast<VectorType>(
                                              pieces.type(scalar.getType(), p)),
                                          scalar)
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
        auto combine = [&](Value x, Value y) -> Value {
          if (kind == "add")
            return arith::AddFOp::create(b, loc, x, y);
          return arith::MaxNumFOp::create(b, loc, x, y);
        };
        SmallVector<Value, 4> in = get(reduce.getInput());
        // With reassoc, the vector pieces are combined lane by lane first,
        // as LLVM splits the reduction of a vector wider than its registers:
        // widened to a power of two of pieces (the added ones neutral), its
        // halves added, until one piece is left, then reduced. One reduction
        // instead of one per piece, and the bits of the reduction of the
        // whole vector when there are no scalar pieces.
        if (reduce.getReassoc() && pieces.numVectors > 1) {
          SmallVector<Value, 4> vecs(in.begin(),
                                     in.begin() + pieces.numVectors);
          while (vecs.size() > 1) {
            // piece i with piece i + half, if any
            size_t half = llvm::PowerOf2Ceil(vecs.size()) / 2;
            SmallVector<Value, 4> next;
            for (size_t i = 0; i < half; ++i)
              next.push_back(i + half < vecs.size()
                                 ? combine(vecs[i], vecs[i + half])
                                 : vecs[i]);
            vecs = next;
          }
          in.erase(in.begin() + 1, in.begin() + pieces.numVectors);
          in.front() = vecs.front();
        }
        Value acc = one(reduce.getAcc());
        for (Value piece : in)
          acc = isa<VectorType>(piece.getType())
                    ? reduceVector(b, loc, combineKind, piece, acc, fastmath)
                    : combine(acc, piece);
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
    // the static layout of the dynamic vectors (checked on each op)
    SmallVector<int64_t> layout;
    op.getRegion().walk([&](Operation *inner) {
      for (Type t : inner->getResultTypes())
        if (auto dyn = dyn_cast<vir::DynamicVectorType>(t)) {
          layout.assign(dyn.getShape().begin(), dyn.getShape().end() - 1);
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    // a dynamic vector holds its rows one after the other
    int64_t lanes = *vl * llvm::product_of(layout);
    int64_t width =
        useCustomVecWid
            ? vectorWidth
            : vectorTypeHelper(op.getRegion(), anchorType, useScalable, verbose,
                               rvvVlen,
                               getTypeConverter()->getIndexTypeBitwidth(),
                               llvm::product_of(layout))
                  .getShape()[0];
    // Scalable vectors have width * vscale lanes at the vscale of the
    // target, rvv-vlen / 64: the code is for that VLEN only. Without it, a
    // vector<[N]xT> has more lanes than the ones it is cut for on a larger
    // VLEN: no code is valid on all of them.
    if (useScalable && !rvvVlenExact)
      return op.emitError("a vir.set_vl with results on scalable vectors "
                          "needs rvv-vlen, the exact VLEN");
    int64_t vscale = useScalable ? rvvVlen / 64 : 1;
    // A vir.shuffle of fixed vectors moves lanes within one vector: each
    // dynamic vector is one vector of all its lanes, whatever the register
    // pressure. On scalable vectors, every lane is known, so the pieces are
    // sized as for any region (the shuffle moves lanes between them).
    SmallVector<vir::ShuffleOp> shuffles;
    op.getRegion().walk([&](vir::ShuffleOp s) { shuffles.push_back(s); });
    if (!shuffles.empty()) {
      if (lanes <= 0)
        return op.emitError("vir.shuffle needs a dynamic vector with a "
                            "positive number of lanes");
      if (!useScalable && !useCustomVecWid)
        width = lanes;
      // RVV: vector<[N]xT> is a group of N * SEW / 64 registers, of at most
      // 8; each shuffle has its element type (vir.extf / truncf, wider than
      // the anchor's the width comes from; index as the LLVM lowering has
      // it)
      int64_t sew = 0;
      for (vir::ShuffleOp s : shuffles) {
        Type elem = s.getType().getElementType();
        sew = std::max<int64_t>(
            sew, elem.isIndex() ? getTypeConverter()->getIndexTypeBitwidth()
                                : elem.getIntOrFloatBitWidth());
      }
      // scalable: groups of at most 8 for every shuffle, whole vectors only
      // where they can be (no lanes left as scalars, which a shuffle moves
      // one by one)
      while (useScalable && !useCustomVecWid && width > 1 &&
             (width * sew > 8 * 64 || lanes % (width * vscale) != 0))
        width /= 2;
      // (vector-width can ask for more)
      if (useScalable)
        for (vir::ShuffleOp s : shuffles) {
          Type elem = s.getType().getElementType();
          int64_t sew = elem.isIndex()
                            ? getTypeConverter()->getIndexTypeBitwidth()
                            : elem.getIntOrFloatBitWidth();
          if (width * sew > 8 * 64)
            return s.emitError("vir.shuffle needs register groups of at "
                               "most 8, vector-width ")
                   << width << " gives " << width * sew / 64 << " registers";
        }
      if (verbose && !useScalable && !useCustomVecWid)
        llvm::outs() << "vir.shuffle: one vector of " << lanes
                     << " lanes, width " << width << "\n";
    }
    // whole vectors, then the lanes left as scalars: no lane is off
    int64_t step = width * vscale;
    Pieces pieces{width, lanes / step, lanes % step, useScalable, step};
    pieces.layout = layout;
    pieces.vl = *vl;
    // keep the lowered code small: vl over a few vectors
    if (pieces.size() > 64)
      return op.emitError("a vir.set_vl with results lowers to more than 64 "
                          "pieces; increase vector-width");
    Location loc = op.getLoc();
    PieceMap map;
    Block &block = op.getRegion().front();
    if (failed(lowerBlockInPieces(rewriter, loc, &block, map, pieces)))
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
    unsupportedOp = false;
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
    WalkResult shuffle = region.walk([](vir::ShuffleOp op) {
      op.emitError("vir.shuffle needs a vir.set_vl with results");
      return WalkResult::interrupt();
    });
    if (shuffle.wasInterrupted())
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
    // The dynamic vectors are the values of the region (each lowered to a
    // vector of the main loop and a scalar of the tail loop), and the loops
    // over the rows are those of one static layout.
    auto badLayout = [&](Type t) {
      auto dyn = dyn_cast<vir::DynamicVectorType>(t);
      return dyn && dyn.getShape().size() > 1 &&
             dyn.getShape().drop_back() != ArrayRef(leadingStaticDims);
    };
    WalkResult layouts = region.walk([&](Operation *inner) {
      for (Value v : inner->getOperands())
        if (isa<vir::DynamicVectorType>(v.getType()) &&
            !region.isAncestor(v.getParentRegion())) {
          inner->emitError("a dynamic vector used in a vir.set_vl must be "
                           "defined in its region");
          return WalkResult::interrupt();
        }
      // (a loop-carried value has the type of the loop result)
      if (llvm::any_of(inner->getResultTypes(), badLayout)) {
        inner->emitError("the multi-dimensional dynamic vectors of a "
                         "vir.set_vl must have the same static layout");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (layouts.wasInterrupted())
      return failure();

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
          vectorTypeHelper(region, anchorType, useScalable, verbose, rvvVlen,
                           getTypeConverter()->getIndexTypeBitwidth());
      vectorWid = int(targetVectorType.getShape()[0]);
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

      // The lanes past the main loop, fewer than its step, first in vectors
      // of half its width, then a quarter, ...: each at most once, in the
      // order of the lanes, with no mask; the lanes left go to the scalar
      // tail loop. Fixed vectors need a constant vl (the bounds of the affine
      // tail loop stay affine); scalable ones test the rest at run time,
      // their lanes counted at the VLEN of rvv-vlen, else the least (128).
      //
      // A part has at least 512 bits: a heuristic from measurements, not a
      // cost model. On the f32 YOLOv8 (16 f32 per part) on three RISC-V
      // cores, parts of 2 to 8 lanes cost about as much as the scalar lanes
      // they replace, plus their code and vtype changes (VLEN 128: 2.5%
      // slower than the scalar tail; from 16 lanes on, the same), while
      // parts of 16 lanes and more were the gain on VLEN 256 and 1024.
      // 512 bits are 64 i8, 32 f16, 8 f64 lanes: other element types,
      // instructions and cores may have another best size, for a target
      // cost model to give.
      {
        auto vecTy = cast<VectorType>(targetVectorType);
        bool scalable = vecTy.isScalable();
        std::optional<int64_t> constVl = getConstantIntValue(vlValue);
        Type elem = vecTy.getElementType();
        int64_t elemBits = elem.isIndex()
                               ? getTypeConverter()->getIndexTypeBitwidth()
                               : elem.getIntOrFloatBitWidth();
        // (an i1 lane is a byte, as for the size of the vectors)
        elemBits = std::max<int64_t>(elemBits, 8);
        int64_t lanesPerWidth =
            scalable ? (rvvVlen > 0 ? rvvVlen : 128) / 64 : 1;
        auto bigEnough = [&](int64_t w) {
          return w * lanesPerWidth * elemBits >= 512;
        };
        auto lowerPart = [&](OpBuilder &bb, Location bodyLoc, Value at,
                             int64_t w) {
          DenseMap<Value, Value> virSymbolTable;
          lowerBlock(bb, bodyLoc, &region.front(), virSymbolTable, at,
                     leadingIVs,
                     VectorType::get({w}, vecTy.getElementType(),
                                     vecTy.getScalableDims()),
                     /*isTailLoop=*/false);
        };
        if (scalable) {
          Value vscale = vector::VectorScaleOp::create(b, loc);
          for (int64_t w = vectorWid / 2; w >= 1 && bigEnough(w); w /= 2) {
            Value step = arith::MulIOp::create(
                b, loc, arith::ConstantIndexOp::create(b, loc, w), vscale);
            Value rest = arith::SubIOp::create(b, loc, vlValue, tailStart);
            Value fits = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::sge, rest, step);
            Value at = tailStart;
            scf::IfOp::create(b, loc, fits,
                              [&](OpBuilder &bb, Location bodyLoc) {
                                lowerPart(bb, bodyLoc, at, w);
                                scf::YieldOp::create(bb, bodyLoc);
                              });
            tailStart = arith::SelectOp::create(
                b, loc, fits, arith::AddIOp::create(b, loc, tailStart, step),
                tailStart);
          }
        } else if (constVl) {
          int64_t at = *constVl - *constVl % vectorWid;
          for (int64_t w = vectorWid / 2; w >= 2 && bigEnough(w); w /= 2)
            if (*constVl - at >= w) {
              lowerPart(b, loc, arith::ConstantIndexOp::create(b, loc, at), w);
              at += w;
            }
          tailStart = arith::ConstantIndexOp::create(b, loc, at);
        }
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
    } else if (rowsAreIndependent(op, leadingStaticDims.size())) {
      // One loop over the leading static dims, in parallel: each row writes
      // elements of its own and reads no element another row writes.
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
      SmallVector<Value> lbs, ubs, steps;
      for (int64_t ub : leadingStaticDims) {
        lbs.push_back(c0);
        ubs.push_back(arith::ConstantIndexOp::create(rewriter, loc, ub));
        steps.push_back(c1);
      }
      scf::ParallelOp::create(rewriter, loc, lbs, ubs, steps,
                              [&](OpBuilder &b, Location, ValueRange ivs) {
                                emitVectorAndTail(b, SmallVector<Value>(ivs));
                              });
    } else {
      // Nested serial loops over the leading static dims.
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

    if (unsupportedOp)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void populateVIRToVectorConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           bool useScalable, int vectorWidth,
                                           bool verbose, bool useCustomVecWid,
                                           int rvvVlen, bool rvvVlenExact) {
  // clang-format off
  patterns.add<VIRSetVLLowering>(converter, useScalable, vectorWidth, verbose, useCustomVecWid, rvvVlen, rvvVlenExact);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Unroll-and-jam in strip-mined regions
//===----------------------------------------------------------------------===//
//
// In a vir.set_vl without results, a loop whose iterations are independent
// and share a vector load of an inner loop (the output pixels of a row of a
// convolution: the filter vector of each (kh, kw, c)) runs `factor`
// iterations jammed together: their inner loops as one, the shared load
// once. The factor is the one whose vectors, sized for the register pressure
// as the region is lowered (vectorTypeHelper), compute the most lanes per
// shared load.

namespace {
/// The most iterations jammed together, a bound on the code size.
constexpr int64_t kMaxJamFactor = 8;

/// A memory access in a loop: the memref, the indices of a known load or
/// store (else none), the rank of the vector it accesses (0: a scalar), and
/// whether it writes.
struct JamAccess {
  Value base;
  SmallVector<Value> indices;
  int64_t vectorRank;
  bool write;
};
} // namespace

/// The buffer a memref is a view of.
static Value underlyingBuffer(Value v) {
  while (auto view = v.getDefiningOp<ViewLikeOpInterface>())
    v = view.getViewSource();
  return v;
}

/// Whether two memrefs cannot overlap: by the alias analysis, or as views of
/// different buffers that are both allocated, or one allocated and the other
/// memory that existed before the allocation (an argument of the function, a
/// global). A buffer of another origin, e.g. the result of a call, may be an
/// allocated one.
static bool areDisjoint(Value a, Value b, AliasAnalysis &aa) {
  if (aa.alias(a, b).isNo())
    return true;
  Value ra = underlyingBuffer(a), rb = underlyingBuffer(b);
  if (ra == rb)
    return false;
  auto allocated = [](Value v) {
    auto iface = v.getDefiningOp<MemoryEffectOpInterface>();
    return iface && iface.getEffectOnValue<MemoryEffects::Allocate>(v);
  };
  auto existing = [](Value v) {
    if (v.getDefiningOp<memref::GetGlobalOp>())
      return true;
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner()->isEntryBlock() &&
           isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
  };
  return (allocated(ra) && (allocated(rb) || existing(rb))) ||
         (allocated(rb) && existing(ra));
}

/// The memory accesses of the ops in `body`. Fails on another memory effect,
/// or an op whose effects are unknown (a VIR op other than vir.load and
/// vir.store has none).
static LogicalResult collectAccesses(Block &body,
                                     SmallVectorImpl<JamAccess> &accesses) {
  WalkResult walk = body.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return isa<vir::VIRDialect>(op->getDialect()) && !isa<vir::SetVLOp>(op)
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    for (MemoryEffects::EffectInstance &effect : effects) {
      Value base = effect.getValue();
      bool write = isa<MemoryEffects::Write>(effect.getEffect());
      if (!base || !isa<MemRefType>(base.getType()) ||
          !(write || isa<MemoryEffects::Read>(effect.getEffect())))
        return WalkResult::interrupt();
      JamAccess access{base, {}, 0, write};
      if (auto load = dyn_cast<vir::LoadOp>(op)) {
        access.indices = llvm::to_vector(load.getIndices());
        access.vectorRank = load.getResult().getType().getShape().size();
      } else if (auto store = dyn_cast<vir::StoreOp>(op)) {
        access.indices = llvm::to_vector(store.getIndices());
        access.vectorRank = store.getValue().getType().getShape().size();
      } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
        access.indices = llvm::to_vector(load.getIndices());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        access.indices = llvm::to_vector(store.getIndices());
      }
      accesses.push_back(std::move(access));
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

/// Whether no two elements of a memref of `type` share an address: the
/// identity layout, or static strides and sizes whose dims nest (sorted by
/// stride, each stride at least the extent of the dims below it).
static bool hasDistinctElements(MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  SmallVector<std::pair<int64_t, int64_t>> dims; // (stride, size)
  for (auto [stride, size] : llvm::zip(strides, type.getShape())) {
    if (size == 1)
      continue;
    if (ShapedType::isDynamic(stride) || ShapedType::isDynamic(size) ||
        stride <= 0)
      return false;
    dims.push_back({stride, size});
  }
  llvm::sort(dims);
  int64_t extent = 1;
  for (auto [stride, size] : dims) {
    if (stride < extent)
      return false;
    extent = stride * size;
  }
  return true;
}

/// Whether the rows of the dynamic vectors of `op` (`rows` leading static
/// dims) may run in parallel: each write is a vir.store of a vector with
/// these rows, to a memref whose elements are distinct, so that each row
/// writes elements of its own; and every access to a memref that may
/// overlap a written one is at the same base, indices and rank, so that a
/// row reads only elements it writes. A scalar memref.store, executed once
/// per row at the same address, keeps the rows serial.
static bool rowsAreIndependent(vir::SetVLOp op, size_t rows) {
  SmallVector<JamAccess> accesses;
  if (failed(collectAccesses(op.getRegion().front(), accesses)))
    return false;
  AliasAnalysis aa(op);
  for (const JamAccess &w : accesses) {
    if (!w.write)
      continue;
    if (w.vectorRank != int64_t(rows) + 1 ||
        !hasDistinctElements(cast<MemRefType>(w.base.getType())))
      return false;
    for (const JamAccess &a : accesses)
      if (!areDisjoint(a.base, w.base, aa) &&
          (a.base != w.base || a.indices != w.indices ||
           a.vectorRank != w.vectorRank))
        return false;
  }
  return true;
}

/// Whether `v` may differ from one iteration of `loop` to the next. The
/// inner loops have the same bounds in every iteration (isJammable); their
/// iteration arguments, and the results of other ops with regions, are
/// taken to differ.
static bool dependsOnIteration(Value v, scf::ForOp loop,
                               DenseMap<Value, bool> &memo) {
  if (loop.isDefinedOutsideOfLoop(v))
    return false;
  if (v == loop.getInductionVar())
    return true;
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  bool depends;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto inner = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    depends = !inner || arg != inner.getInductionVar();
  } else {
    Operation *def = v.getDefiningOp();
    depends = def->getNumRegions() != 0 ||
              llvm::any_of(def->getOperands(), [&](Value operand) {
                return dependsOnIteration(operand, loop, memo);
              });
  }
  memo[v] = depends;
  return depends;
}

/// Whether the iterations of `loop` may run jammed together, and share a
/// vector load: the loop has no results and a constant step; its inner
/// loops have the same bounds in every iteration; each iteration writes
/// elements of its own (at indices with the induction variable, not in the
/// dims a vector spans), which every access of a memref that may overlap
/// uses too; and a vector load of an inner loop is the same in every
/// iteration.
static bool isJammable(scf::ForOp loop, AliasAnalysis &aa) {
  if (loop.getNumResults() != 0 || !getConstantIntValue(loop.getStep()))
    return false;
  bool hasInnerLoop = false, invariantBounds = true;
  loop.getBody()->walk([&](scf::ForOp inner) {
    hasInnerLoop = true;
    for (Value v :
         {inner.getLowerBound(), inner.getUpperBound(), inner.getStep()})
      invariantBounds &= loop.isDefinedOutsideOfLoop(v);
  });
  if (!hasInnerLoop || !invariantBounds)
    return false;

  SmallVector<JamAccess> accesses;
  if (failed(collectAccesses(*loop.getBody(), accesses)))
    return false;
  Value iv = loop.getInductionVar();
  for (const JamAccess &w : accesses) {
    if (!w.write)
      continue;
    ArrayRef<Value> own = ArrayRef(w.indices).drop_back(
        std::min<size_t>(w.vectorRank, w.indices.size()));
    if (!llvm::is_contained(own, iv) ||
        !hasDistinctElements(cast<MemRefType>(w.base.getType())))
      return false;
    for (const JamAccess &a : accesses)
      if (!areDisjoint(a.base, w.base, aa) &&
          (a.base != w.base || a.indices != w.indices ||
           a.vectorRank != w.vectorRank))
        return false;
  }

  DenseMap<Value, bool> memo;
  WalkResult shared = loop.getBody()->walk([&](vir::LoadOp load) {
    if (load->getParentOp() == loop ||
        llvm::any_of(load->getOperands(), [&](Value v) {
          return dependsOnIteration(v, loop, memo);
        }))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return shared.wasInterrupted();
}

/// Emits the ops of `body` but its terminator for `maps.size()` iterations
/// jammed together, `maps[k]` mapping the values of the body to those of the
/// k-th: an inner scf.for once, carrying the iteration arguments of all; an
/// op without regions or writes that computes the same in all, once; any
/// other op for each.
static void emitJammed(OpBuilder &b, Block &body,
                       MutableArrayRef<IRMapping> maps) {
  for (Operation &op : body.without_terminator()) {
    if (auto inner = dyn_cast<scf::ForOp>(&op)) {
      unsigned n = inner.getNumRegionIterArgs();
      SmallVector<Value> inits;
      for (IRMapping &m : maps)
        for (Value v : inner.getInitArgs())
          inits.push_back(m.lookupOrDefault(v));
      auto jammed = scf::ForOp::create(
          b, inner.getLoc(), inner.getLowerBound(), inner.getUpperBound(),
          inner.getStep(), inits,
          [&](OpBuilder &ib, Location loc, Value iv, ValueRange args) {
            SmallVector<IRMapping> innerMaps(maps.begin(), maps.end());
            for (auto [k, m] : llvm::enumerate(innerMaps)) {
              m.map(inner.getInductionVar(), iv);
              m.map(inner.getRegionIterArgs(), args.slice(k * n, n));
            }
            emitJammed(ib, *inner.getBody(), innerMaps);
            SmallVector<Value> yields;
            for (IRMapping &m : innerMaps)
              for (Value v : inner.getBody()->getTerminator()->getOperands())
                yields.push_back(m.lookupOrDefault(v));
            scf::YieldOp::create(ib, loc, yields);
          });
      for (auto [k, m] : llvm::enumerate(maps))
        m.map(inner.getResults(), jammed.getResults().slice(k * n, n));
      continue;
    }
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    bool shared =
        op.getNumRegions() == 0 &&
        !(iface && iface.hasEffect<MemoryEffects::Write>()) &&
        llvm::all_of(op.getOperands(), [&](Value v) {
          return llvm::all_of(maps, [&](IRMapping &m) {
            return m.lookupOrDefault(v) == maps.front().lookupOrDefault(v);
          });
        });
    if (!shared) {
      for (IRMapping &m : maps)
        b.clone(op, m);
      continue;
    }
    Operation *clone = b.clone(op, maps.front());
    for (IRMapping &m : maps.drop_front())
      m.map(op.getResults(), clone->getResults());
  }
}

/// Jams `factor` iterations of `loop` together (isJammable); the iterations
/// left, fewer than `factor`, stay in `loop`.
static void jamLoop(scf::ForOp loop, int64_t factor) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  int64_t step = *getConstantIntValue(loop.getStep());
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
  Value block = arith::ConstantIndexOp::create(b, loc, step * factor);
  // the iterations: max(ub - lb, 0) ceildiv step
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value trips = arith::MaxSIOp::create(
      b, loc, arith::SubIOp::create(b, loc, ub, lb), zero);
  if (step != 1)
    trips = arith::DivSIOp::create(
        b, loc,
        arith::AddIOp::create(b, loc, trips,
                              arith::ConstantIndexOp::create(b, loc, step - 1)),
        arith::ConstantIndexOp::create(b, loc, step));
  // lb + (trips - trips % factor) * step
  Value jammed = arith::SubIOp::create(
      b, loc, trips,
      arith::RemSIOp::create(b, loc, trips,
                             arith::ConstantIndexOp::create(b, loc, factor)));
  Value mainUb = arith::AddIOp::create(
      b, loc, lb,
      step == 1
          ? jammed
          : Value(arith::MulIOp::create(
                b, loc, jammed, arith::ConstantIndexOp::create(b, loc, step))));
  scf::ForOp::create(
      b, loc, lb, mainUb, block, ValueRange{},
      [&](OpBuilder &mb, Location l, Value iv, ValueRange) {
        SmallVector<IRMapping> maps(factor);
        maps[0].map(loop.getInductionVar(), iv);
        for (int64_t k = 1; k < factor; ++k)
          maps[k].map(
              loop.getInductionVar(),
              arith::AddIOp::create(
                  mb, l, iv, arith::ConstantIndexOp::create(mb, l, k * step))
                  .getResult());
        emitJammed(mb, *loop.getBody(), maps);
        scf::YieldOp::create(mb, l);
      });
  loop.getLowerBoundMutable().assign(mainUb);
}

/// The element type of the first dynamic vector of a region, as
/// VIRSetVLLowering finds its anchor type.
static Type anchorTypeOf(Region &region) {
  Type anchor;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Value result : op->getResults())
      if (auto dyn = dyn_cast<vir::DynamicVectorType>(result.getType());
          dyn && !dyn.getScalingFactor()) {
        anchor = dyn.getElementType();
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return anchor;
}

/// Jams the loops of the strip-mined regions of `module` (see above). The
/// vectors are sized as vectorTypeHelper sizes them with these arguments.
static void jamStripMinedLoops(ModuleOp module, bool useScalable, int rvvVlen,
                               unsigned indexBitWidth) {
  AliasAnalysis aa(module);
  SmallVector<std::pair<vir::SetVLOp, scf::ForOp>> loops;
  module.walk([&](vir::SetVLOp setVL) {
    if (setVL->getNumResults() != 0)
      return;
    setVL.getRegion().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      if (!isJammable(loop, aa))
        return WalkResult::advance();
      loops.emplace_back(setVL, loop);
      return WalkResult::skip();
    });
  });
  for (auto [setVL, loop] : loops) {
    Type anchor = anchorTypeOf(setVL.getRegion());
    if (!anchor)
      continue;
    // the factor with the most lanes per shared load (the least of equals),
    // its vectors live at once in the registers
    // (no more than the iterations, when they are known)
    int64_t maxFactor = kMaxJamFactor;
    if (std::optional<APInt> trips = loop.getStaticTripCount())
      maxFactor = std::clamp<int64_t>(trips->getZExtValue(), 1, kMaxJamFactor);
    int64_t best = 1, bestLanes = 0;
    for (int64_t factor = 1; factor <= maxFactor; ++factor) {
      OpBuilder b(setVL);
      IRMapping m;
      Operation *trial = b.clone(*setVL, m);
      if (factor > 1)
        jamLoop(cast<scf::ForOp>(m.lookup(loop.getOperation())), factor);
      bool fits = false;
      VectorType type;
      {
        ScopedDiagnosticHandler quiet(module.getContext(),
                                      [](Diagnostic &) { return success(); });
        type = vectorTypeHelper(trial->getRegion(0), anchor, useScalable,
                                /*verbose=*/false, rvvVlen, indexBitWidth,
                                /*rows=*/1, &fits);
      }
      trial->erase();
      if (!type)
        break;
      int64_t lanes = factor * type.getShape()[0];
      if (fits && lanes > bestLanes) {
        best = factor;
        bestLanes = lanes;
      }
    }
    if (best > 1)
      jamLoop(loop, best);
  }
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
        ub::UBDialect,
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
                     "registers of VLEN bits instead of the host's (on a "
                     "RISC-V host with V, those of its VLEN); with "
                     "use-scalable, a vir.set_vl with results is lowered "
                     "for this VLEN (vscale = rvv-vlen / 64) and is only "
                     "valid on it. use-scalable without rvv-vlen (and "
                     "vector-width) is for any RVV VLEN: the vectors are "
                     "sized for VLEN 128, strip-mined regions are valid on "
                     "any VLEN, a vir.set_vl with results is rejected"),
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
      ub::UBDialect,
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
  // RVV: vscale = VLEN / 64, at least 1 for scalable vectors
  if (useScalable && rvvVlen != 0 && rvvVlen < 64) {
    module.emitError("rvv-vlen must be at least 64 with use-scalable, got ")
        << int(rvvVlen);
    return signalPassFailure();
  }
  // use custome vector width
  bool useCustomVecWid = vectorWidth.getNumOccurrences() > 0;
  // rvv-vlen is the exact VLEN; use-scalable without it is for any RVV
  // VLEN (vscale = VLEN / 64), whose vectors are sized for the least, 128
  bool exactVlen = rvvVlen != 0;
  int vlen = rvvVlen;
  if (useScalable && !useCustomVecWid && !exactVlen)
    vlen = 128;
  // Fixed vectors for the host: on a RISC-V host with V, those of its VLEN,
  // as for rvv-vlen
  if (!useScalable && !useCustomVecWid && !exactVlen) {
    vlen = hostRvvVlen();
    if (verbose && vlen)
      llvm::outs() << "--- RISC-V Vector host: VLEN " << vlen << " ---\n";
  }
  // the loops of strip-mined regions jammed for the vectors sized for the
  // register pressure (not for a given vector-width)
  if (!useCustomVecWid)
    jamStripMinedLoops(module, useScalable, vlen,
                       converter.getIndexTypeBitwidth());
  populateVIRToVectorConversionPatterns(converter, patterns, useScalable,
                                        vectorWidth, verbose, useCustomVecWid,
                                        vlen, exactVlen);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
  // a vir.set_vl left (its lowering reported why) fails the pass
  if (module.walk([](vir::SetVLOp) { return WalkResult::interrupt(); })
          .wasInterrupted())
    signalPassFailure();
}

namespace mlir {
namespace buddy {
void registerVIRToVectorPass() { PassRegistration<VIRToVectorPass>(); }
} // namespace buddy
} // namespace mlir
