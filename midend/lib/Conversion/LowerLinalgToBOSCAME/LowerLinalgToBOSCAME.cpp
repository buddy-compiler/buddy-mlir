//=== LowerLinalgToBOSCAME.cpp - Linalg to BOSCAME Dialect Lowering Pass --===//
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
// This file defines Linalg dialect lowering pass to BOSCAME dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/Transforms/AMEValueEmitter.h"
#include "Dialect/BOSCAME/Transforms/FPGAAMETarget.h"

using namespace mlir;
using namespace buddy::boscame;

namespace {

enum class ElementwiseKind {
  Add,
  Sub,
  Mul,
  MinS,
  MinU,
  MaxS,
  MaxU,
  FAdd,
  FSub,
  FMul,
  FMax,
};

enum class BroadcastKind {
  Row,
  Column,
  Element,
};

enum class BroadcastRegisterKind {
  TileA,
  TileB,
  Accumulation,
};

static unsigned getElementBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

// The element-width table lives in the shared AME emitter so that the upstream
// pathway and the FPGA pathway cannot drift apart.
static LogicalResult getMsetTypeImm(Type elementType, int64_t &imm) {
  FailureOr<int64_t> value = ame::getUpstreamMsetTypeImm(elementType);
  if (failed(value))
    return failure();
  imm = *value;
  return success();
}

static bool isParallel(utils::IteratorType type) {
  return type == utils::IteratorType::parallel;
}

static bool isReduction(utils::IteratorType type) {
  return type == utils::IteratorType::reduction;
}

static bool hasPureBufferOperands(linalg::GenericOp op) {
  return op.hasPureBufferSemantics() && op.getNumDpsInits() == 1;
}

static bool isAffineMap(AffineMap map, unsigned numDims,
                        ArrayRef<unsigned> dimPositions) {
  if (map.getNumDims() != numDims || map.getNumSymbols() != 0 ||
      map.getNumResults() != dimPositions.size())
    return false;

  MLIRContext *ctx = map.getContext();
  for (auto [result, dimPos] : llvm::zip_equal(map.getResults(), dimPositions))
    if (result != getAffineDimExpr(dimPos, ctx))
      return false;
  return true;
}

static bool isIdentity2D(AffineMap map) {
  return isAffineMap(map, 2, ArrayRef<unsigned>{0, 1});
}

static bool isIdentityND(AffineMap map, unsigned rank) {
  SmallVector<unsigned> dims;
  dims.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dims.push_back(i);
  return isAffineMap(map, rank, dims);
}

static bool isGenericMatmulLike(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3 || !isParallel(iterators[0]) ||
      !isParallel(iterators[1]) || !isReduction(iterators[2]))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !isAffineMap(maps[0], 3, ArrayRef<unsigned>{0, 2}) ||
      !isAffineMap(maps[1], 3, ArrayRef<unsigned>{2, 1}) ||
      !isAffineMap(maps[2], 3, ArrayRef<unsigned>{0, 1}))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *addOp = yieldOp.getValues()[0].getDefiningOp();
  if (!addOp)
    return false;

  StringRef addName = addOp->getName().getStringRef();
  if (addName != "arith.addi" && addName != "arith.addf")
    return false;

  Value lhs = addOp->getOperand(0);
  Value rhs = addOp->getOperand(1);
  Operation *mulOp = lhs.getDefiningOp();
  Value accumulator = rhs;
  if (!mulOp) {
    mulOp = rhs.getDefiningOp();
    accumulator = lhs;
  }
  if (!mulOp || accumulator != body.getArgument(2))
    return false;

  StringRef mulName = mulOp->getName().getStringRef();
  if (mulName != "arith.muli" && mulName != "arith.mulf")
    return false;

  return (mulOp->getOperand(0) == body.getArgument(0) &&
          mulOp->getOperand(1) == body.getArgument(1)) ||
         (mulOp->getOperand(0) == body.getArgument(1) &&
          mulOp->getOperand(1) == body.getArgument(0));
}

static bool matchElementwiseKind(linalg::GenericOp op, ElementwiseKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 2)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, isIdentity2D))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getNumOperands() != 2)
    return false;

  bool directOrder = opDef->getOperand(0) == body.getArgument(0) &&
                     opDef->getOperand(1) == body.getArgument(1);
  bool swappedOrder = opDef->getOperand(0) == body.getArgument(1) &&
                      opDef->getOperand(1) == body.getArgument(0);
  if (!directOrder && !swappedOrder)
    return false;

  StringRef name = opDef->getName().getStringRef();
  if (name == "arith.addi")
    kind = ElementwiseKind::Add;
  else if (name == "arith.subi" && directOrder)
    kind = ElementwiseKind::Sub;
  else if (name == "arith.muli")
    kind = ElementwiseKind::Mul;
  else if (name == "arith.minsi")
    kind = ElementwiseKind::MinS;
  else if (name == "arith.minui")
    kind = ElementwiseKind::MinU;
  else if (name == "arith.maxsi")
    kind = ElementwiseKind::MaxS;
  else if (name == "arith.maxui")
    kind = ElementwiseKind::MaxU;
  else if (name == "arith.addf")
    kind = ElementwiseKind::FAdd;
  else if (name == "arith.subf" && directOrder)
    kind = ElementwiseKind::FSub;
  else if (name == "arith.mulf")
    kind = ElementwiseKind::FMul;
  else if (name == "arith.maximumf" || name == "arith.maxnumf")
    kind = ElementwiseKind::FMax;
  else
    return false;

  return true;
}

static bool matchUnarySquare(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() < 2 || !llvm::all_of(iterators, isParallel))
    return false;

  unsigned rank = iterators.size();
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentityND(maps[0], rank) ||
      !isIdentityND(maps[1], rank))
    return false;

  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;

  Operation *opDef = yieldOp.getValues()[0].getDefiningOp();
  if (!opDef || opDef->getName().getStringRef() != "math.fpowi" ||
      opDef->getNumOperands() != 2 ||
      opDef->getOperand(0) != body.getArgument(0))
    return false;

  APInt exponent;
  if (!matchPattern(opDef->getOperand(1), m_ConstantInt(&exponent)))
    return false;
  return exponent.getSExtValue() == 2;
}

static bool matchTranspose(linalg::GenericOp op) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isAffineMap(maps[0], 2, ArrayRef<unsigned>{1, 0}) ||
      !isIdentity2D(maps[1]))
    return false;

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool matchBroadcast(linalg::GenericOp op, BroadcastKind &kind) {
  if (!hasPureBufferOperands(op) || op.getNumDpsInputs() != 1)
    return false;

  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 2 || !llvm::all_of(iterators, isParallel))
    return false;

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (maps.size() != 2 || !isIdentity2D(maps[1]))
    return false;

  AffineMap inputMap = maps[0];
  if (inputMap.getNumDims() != 2 || inputMap.getNumSymbols() != 0)
    return false;

  if (inputMap.getNumResults() == 0) {
    kind = BroadcastKind::Element;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{1})) {
    kind = BroadcastKind::Row;
  } else if (isAffineMap(inputMap, 2, ArrayRef<unsigned>{0})) {
    kind = BroadcastKind::Column;
  } else {
    return false;
  }

  Block &body = op.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues()[0] == body.getArgument(0);
}

static bool isLowerableGeneric(linalg::GenericOp op) {
  ElementwiseKind elementwiseKind;
  BroadcastKind broadcastKind;
  return isGenericMatmulLike(op) || matchElementwiseKind(op, elementwiseKind) ||
         matchUnarySquare(op) || matchTranspose(op) ||
         matchBroadcast(op, broadcastKind);
}

//===----------------------------------------------------------------------===//
// FPGA (Qwen3) target analysis
//
// These matchers are pure analysis: they decide whether the FPGA fast path
// may claim a matmul, and they are used both by the dispatcher and by the
// conversion legality contract so that the two can never disagree.
//===----------------------------------------------------------------------===//

static bool hasSupportedRowMajor2DLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.size() == 2 &&
         (ShapedType::isDynamic(strides.front()) || strides.front() > 0) &&
         strides.back() == 1;
}

// Triton materializes a logical [K, N] tile loaded from a row-major [N, full-K]
// weight as a zero-copy memref with strides [1, full-K]. The AME mlbe path can
// consume that physical [N, full-K] layout directly.
static bool hasSupportedTransposed2DLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.size() == 2 && strides.front() == 1 && strides.back() > 0;
}

static bool hasSupportedBLayout(MemRefType type) {
  return hasSupportedRowMajor2DLayout(type) ||
         hasSupportedTransposed2DLayout(type);
}

static constexpr StringLiteral kTritonConsumerFenceAttr =
    "bosc_ame.triton_consumer_fence";

struct QwenDirectCMatch {
  memref::AllocOp temporaryAlloc;
  linalg::FillOp zeroFill;
  memref::CopyOp copyToFinal;
  memref::DeallocOp temporaryDealloc;
  Value finalOutput;
};

struct TritonI8DotCastMatch {
  memref::AllocOp integerAlloc;
  Operation *integerInitialization;
  linalg::GenericOp castToF32;
  memref::AllocOp floatAlloc;
  memref::AllocOp zeroTemplateAlloc;
  linalg::FillOp zeroTemplateFill;
};

// Triton defines an integer dot as i8 x i8 -> i32 even when its result is
// immediately converted to fp32.  The FPGA AME instead keeps the exact i32
// accumulator internally and converts it to fp32 at msce32.m.  For K <= 1024
// the worst-case signed-i8 dot does not exceed 2^24, so this fusion is exact.
static FailureOr<TritonI8DotCastMatch>
matchTritonI8DotCast(linalg::MatmulOp op) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1 || op.getCast() != linalg::TypeFn::cast_signed ||
      op.hasUserDefinedMaps())
    return failure();

  Value A = op.getDpsInputOperand(0)->get();
  Value B = op.getDpsInputOperand(1)->get();
  Value integerC = op.getDpsInitOperand(0)->get();
  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto integerCType = dyn_cast<MemRefType>(integerC.getType());
  if (!AType || !BType || !integerCType || AType.getRank() != 2 ||
      BType.getRank() != 2 || integerCType.getRank() != 2 ||
      !AType.hasStaticShape() || !BType.hasStaticShape() ||
      !integerCType.hasStaticShape())
    return failure();

  int64_t dimM = AType.getDimSize(0);
  int64_t dimK = AType.getDimSize(1);
  int64_t dimN = BType.getDimSize(1);
  if (dimM <= 0 || dimN <= 0 || dimK <= 0 || dimK % 16 != 0 || dimK > 1024 ||
      BType.getDimSize(0) != dimK ||
      integerCType.getShape() != ArrayRef<int64_t>({dimM, dimN}) ||
      !AType.getElementType().isSignlessInteger(8) ||
      !BType.getElementType().isSignlessInteger(8) ||
      !integerCType.getElementType().isSignlessInteger(32) ||
      !hasSupportedRowMajor2DLayout(AType) || !hasSupportedBLayout(BType))
    return failure();

  auto integerAlloc = integerC.getDefiningOp<memref::AllocOp>();
  if (!integerAlloc || integerAlloc->getBlock() != op->getBlock() ||
      !integerAlloc->isBeforeInBlock(op))
    return failure();

  Operation *integerInitialization = nullptr;
  linalg::GenericOp castToF32;
  memref::AllocOp zeroTemplateAlloc;
  linalg::FillOp zeroTemplateFill;
  for (Operation *user : integerC.getUsers()) {
    if (user == op.getOperation())
      continue;
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (integerInitialization ||
          fill.getDpsInitOperand(0)->get() != integerC ||
          !matchPattern(fill.getDpsInputOperand(0)->get(), m_Zero()))
        return failure();
      integerInitialization = fill;
      continue;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      // A direct zero buffer may also seed earlier group-dot destinations.
      // Those are reads and do not change the value seen by this matmul.
      if (copy.getSource() == integerC) {
        if (copy->getBlock() != op->getBlock() || !copy->isBeforeInBlock(op))
          return failure();
        continue;
      }
      if (integerInitialization || copy.getTarget() != integerC)
        return failure();

      Value zeroTemplate = copy.getSource();
      auto zeroTemplateType = dyn_cast<MemRefType>(zeroTemplate.getType());
      zeroTemplateAlloc = zeroTemplate.getDefiningOp<memref::AllocOp>();
      if (!zeroTemplateType || zeroTemplateType != integerCType ||
          !zeroTemplateAlloc || zeroTemplateAlloc->getBlock() != op->getBlock())
        return failure();

      // Multiple static Triton group dots reuse one zero tensor. Bufferization
      // copies it into fresh destinations and may use the template itself for
      // the last dot. Only users before this copy matter: they must initialize
      // the template or read it through another copy, never modify it.
      for (Operation *templateUser : zeroTemplate.getUsers()) {
        if (auto templateFill = dyn_cast<linalg::FillOp>(templateUser)) {
          if (zeroTemplateFill ||
              templateFill.getDpsInitOperand(0)->get() != zeroTemplate ||
              !matchPattern(templateFill.getDpsInputOperand(0)->get(),
                            m_Zero()))
            return failure();
          zeroTemplateFill = templateFill;
          continue;
        }
        if (templateUser == copy.getOperation())
          continue;
        if (templateUser->getBlock() != copy->getBlock())
          return failure();
        if (copy->isBeforeInBlock(templateUser))
          continue;
        auto earlierCopy = dyn_cast<memref::CopyOp>(templateUser);
        if (!earlierCopy || earlierCopy.getSource() != zeroTemplate)
          return failure();
      }
      if (!zeroTemplateFill ||
          !zeroTemplateAlloc->isBeforeInBlock(zeroTemplateFill) ||
          !zeroTemplateFill->isBeforeInBlock(copy))
        return failure();
      integerInitialization = copy;
      continue;
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (castToF32 || generic.getNumDpsInputs() != 1 ||
          generic.getNumDpsInits() != 1 ||
          generic.getDpsInputOperand(0)->get() != integerC)
        return failure();
      castToF32 = generic;
      continue;
    }
    return failure();
  }

  if (!integerInitialization || !castToF32 ||
      castToF32->getBlock() != op->getBlock() ||
      !op->isBeforeInBlock(castToF32))
    return failure();
  if (isa<memref::CopyOp>(integerInitialization)) {
    if (integerAlloc->getNextNode() != integerInitialization ||
        integerInitialization->getNextNode() != op.getOperation())
      return failure();
  } else if (!integerAlloc->isBeforeInBlock(integerInitialization) ||
             !integerInitialization->isBeforeInBlock(op)) {
    return failure();
  }

  Value floatC = castToF32.getDpsInitOperand(0)->get();
  auto floatCType = dyn_cast<MemRefType>(floatC.getType());
  auto floatAlloc = floatC.getDefiningOp<memref::AllocOp>();
  if (!floatCType || !floatAlloc || floatCType.getRank() != 2 ||
      floatCType.getShape() != integerCType.getShape() ||
      !floatCType.getElementType().isF32() ||
      !hasSupportedRowMajor2DLayout(floatCType) ||
      floatAlloc->getBlock() != op->getBlock() ||
      !floatAlloc->isBeforeInBlock(castToF32))
    return failure();

  for (Operation *user : floatC.getUsers()) {
    if (user == castToF32.getOperation())
      continue;
    if (user->getBlock() != op->getBlock() || !castToF32->isBeforeInBlock(user))
      return failure();
  }

  if (castToF32.getNumLoops() != 2 || castToF32.getNumParallelLoops() != 2)
    return failure();
  for (AffineMap map : castToF32.getIndexingMapsArray())
    if (!map.isIdentity())
      return failure();

  Block &body = castToF32.getRegion().front();
  if (body.getNumArguments() != 2)
    return failure();
  auto cast = dyn_cast<arith::SIToFPOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!cast || !yield || cast->getNextNode() != yield.getOperation() ||
      cast.getIn() != body.getArgument(0) || yield.getNumOperands() != 1 ||
      yield.getValues().front() != cast.getOut())
    return failure();

  return TritonI8DotCastMatch{integerAlloc,      integerInitialization,
                              castToF32,         floatAlloc,
                              zeroTemplateAlloc, zeroTemplateFill};
}

// Match the bufferization shapes emitted for Qwen3 matmuls:
//
//   %tmp = memref.alloc
//   linalg.fill 0 -> %tmp
//   linalg.matmul A, B -> %tmp
//   <pure address calculation for %final>
//   memref.copy %tmp, %final  // optional; Buddy Frontend uses %tmp directly
//   memref.dealloc %tmp  // optional
//
// The strict use and ordering checks are what make it safe to compute directly
// into either %final or the original destination. Any less constrained matmul
// remains legal and is handled by the later linalg-to-VIR pipeline.
static FailureOr<QwenDirectCMatch> matchQwenDirectCMatmul(linalg::MatmulOp op) {
  if (!op.hasPureBufferSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1)
    return failure();

  Value A = op.getDpsInputOperand(0)->get();
  Value B = op.getDpsInputOperand(1)->get();
  Value temporaryC = op.getDpsInitOperand(0)->get();

  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto temporaryCType = dyn_cast<MemRefType>(temporaryC.getType());
  if (!AType || !BType || !temporaryCType)
    return failure();

  if (AType.getRank() != 2 || BType.getRank() != 2 ||
      temporaryCType.getRank() != 2)
    return failure();

  if (!AType.hasStaticShape() || !BType.hasStaticShape() ||
      !temporaryCType.hasStaticShape())
    return failure();

  int64_t dimM = AType.getDimSize(0);
  int64_t dimK = AType.getDimSize(1);
  int64_t dimN = BType.getDimSize(1);
  if (dimM <= 0 || dimN <= 0 || dimK <= 0 || dimK % 16 != 0 ||
      BType.getDimSize(0) != dimK || temporaryCType.getDimSize(0) != dimM ||
      temporaryCType.getDimSize(1) != dimN)
    return failure();

  if (!AType.getElementType().isSignlessInteger(8) ||
      !BType.getElementType().isSignlessInteger(8) ||
      !temporaryCType.getElementType().isF32())
    return failure();

  if (op.getCast() != linalg::TypeFn::cast_signed || op.hasUserDefinedMaps())
    return failure();

  // A dynamic leading stride is read from the descriptor. The inner stride
  // must be proven unit: selecting an FPGA target is not a layout assertion.
  if (!hasSupportedRowMajor2DLayout(AType) || !hasSupportedBLayout(BType) ||
      !hasSupportedRowMajor2DLayout(temporaryCType))
    return failure();

  auto temporaryAlloc = temporaryC.getDefiningOp<memref::AllocOp>();
  if (!temporaryAlloc || temporaryAlloc->getBlock() != op->getBlock() ||
      !temporaryAlloc->isBeforeInBlock(op))
    return failure();

  linalg::FillOp zeroFill;
  memref::CopyOp copyToFinal;
  memref::DeallocOp temporaryDealloc;
  bool sawMatmul = false;
  bool hasOtherConsumer = false;

  for (Operation *user : temporaryC.getUsers()) {
    if (user == op.getOperation()) {
      if (sawMatmul)
        return failure();
      sawMatmul = true;
      continue;
    }

    if (auto fillOp = dyn_cast<linalg::FillOp>(user)) {
      if (zeroFill || fillOp.getDpsInitOperand(0)->get() != temporaryC ||
          !matchPattern(fillOp.getDpsInputOperand(0)->get(), m_PosZeroFloat()))
        return failure();
      zeroFill = fillOp;
      continue;
    }

    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      // Triton accumulates all quantization groups in the first dot buffer
      // and copies that fully scaled result to the ABI output much later.
      // That copy is a downstream read, not the direct-C copy shape below.
      if (op->hasAttr(kTritonConsumerFenceAttr)) {
        if (copyOp.getSource() != temporaryC ||
            copyOp->getBlock() != op->getBlock() ||
            !op->isBeforeInBlock(copyOp))
          return failure();
        continue;
      }
      if (copyToFinal || copyOp.getSource() != temporaryC)
        return failure();
      copyToFinal = copyOp;
      continue;
    }

    if (auto deallocOp = dyn_cast<memref::DeallocOp>(user)) {
      if (temporaryDealloc || deallocOp.getMemref() != temporaryC)
        return failure();
      temporaryDealloc = deallocOp;
      continue;
    }

    // Buddy Frontend keeps the matmul destination as the SSA buffer consumed
    // by following elementwise operations.  Such uses are safe when they are
    // in the same block and occur after the matmul.
    if (user->getBlock() != op->getBlock() || !op->isBeforeInBlock(user))
      return failure();
    hasOtherConsumer = true;
  }

  if (!sawMatmul || !zeroFill)
    return failure();
  if (zeroFill->getBlock() != op->getBlock() ||
      zeroFill->getNextNode() != op.getOperation())
    return failure();

  if (temporaryDealloc && (temporaryDealloc->getBlock() != op->getBlock() ||
                           !op->isBeforeInBlock(temporaryDealloc)))
    return failure();

  if (!copyToFinal)
    return QwenDirectCMatch{temporaryAlloc, zeroFill, copyToFinal,
                            temporaryDealloc, temporaryC};

  // KeepTemporary permits downstream consumers; CopyElision must not erase
  // an allocation that is still read, written or escaped after the copy.
  if (hasOtherConsumer)
    return failure();

  if (copyToFinal->getBlock() != op->getBlock() ||
      !op->isBeforeInBlock(copyToFinal))
    return failure();

  if (temporaryDealloc && !copyToFinal->isBeforeInBlock(temporaryDealloc))
    return failure();

  Value finalOutput = copyToFinal.getTarget();
  auto finalType = dyn_cast<MemRefType>(finalOutput.getType());
  if (!finalType || finalOutput == temporaryC || !finalOutput.hasOneUse() ||
      *finalOutput.getUsers().begin() != copyToFinal.getOperation())
    return failure();

  if (finalType.getRank() != temporaryCType.getRank() ||
      finalType.getShape() != temporaryCType.getShape() ||
      finalType.getElementType() != temporaryCType.getElementType() ||
      finalType.getMemorySpace() != temporaryCType.getMemorySpace() ||
      !hasSupportedRowMajor2DLayout(finalType))
    return failure();

  AliasAnalysis aliasAnalysis(op->getParentOp());
  if (!aliasAnalysis.alias(finalOutput, A).isNo() ||
      !aliasAnalysis.alias(finalOutput, B).isNo())
    return failure();

  // Recompute at the original copy point instead of moving %final's defining
  // operations earlier. This preserves the time at which final C becomes
  // visible. Requiring the intervening operations to be pure guarantees that
  // A, B, and all memory are unchanged between the old matmul and copy.
  for (Operation *between = op->getNextNode(); between != copyToFinal;
       between = between->getNextNode()) {
    if (!between || between->getNumRegions() != 0 || !isPure(between))
      return failure();
  }

  return QwenDirectCMatch{temporaryAlloc, zeroFill, copyToFinal,
                          temporaryDealloc, finalOutput};
}

class TritonI8DotCastToF32Matmul : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<TritonI8DotCastMatch> match = matchTritonI8DotCast(op);
    if (failed(match))
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value oldFloatC = match->castToF32.getDpsInitOperand(0)->get();
    auto floatType = cast<MemRefType>(oldFloatC.getType());
    Location loc = op.getLoc();

    // Preserve the original dot's execution point and redirect all consumers
    // of the later sitofp buffer to this exact-fp32 AME destination.
    rewriter.setInsertionPoint(op);
    Value floatC = memref::AllocOp::create(rewriter, loc, floatType);
    Value zero = arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    linalg::FillOp::create(rewriter, loc, zero, floatC);
    auto floatMatmul = linalg::MatmulOp::create(rewriter, loc, ValueRange{A, B},
                                                ValueRange{floatC});
    floatMatmul.setCast(linalg::TypeFn::cast_signed);
    floatMatmul->setAttr(kTritonConsumerFenceAttr, rewriter.getUnitAttr());

    oldFloatC.replaceAllUsesWith(floatC);
    rewriter.eraseOp(match->castToF32);
    rewriter.eraseOp(match->floatAlloc);
    rewriter.eraseOp(op);

    auto eraseZeroBufferIfDead = [&](memref::AllocOp alloc,
                                     linalg::FillOp fill) {
      if (alloc && fill && alloc.getResult().hasOneUse() &&
          *alloc.getResult().getUsers().begin() == fill.getOperation()) {
        rewriter.eraseOp(fill);
        rewriter.eraseOp(alloc);
      }
    };
    if (isa<memref::CopyOp>(match->integerInitialization)) {
      rewriter.eraseOp(match->integerInitialization);
      rewriter.eraseOp(match->integerAlloc);
      eraseZeroBufferIfDead(match->zeroTemplateAlloc, match->zeroTemplateFill);
    } else {
      eraseZeroBufferIfDead(match->integerAlloc,
                            cast<linalg::FillOp>(match->integerInitialization));
    }
    return success();
  }
};


static Value createDim(PatternRewriter &rewriter, Location loc, Value memref,
                       int64_t dim) {
  return memref::DimOp::create(rewriter, loc, memref, dim);
}

static Value createIndexMin(PatternRewriter &rewriter, Location loc,
                            Value bound, Value iv, int64_t step) {
  Value remain = arith::SubIOp::create(rewriter, loc, bound, iv);
  Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
  Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                    remain, stepVal);
  return arith::SelectOp::create(rewriter, loc, cmp, remain, stepVal);
}

static Value castIndexToI64(PatternRewriter &rewriter, Location loc,
                            Value value) {
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                    value);
}

static Value createByteStride(PatternRewriter &rewriter, Location loc,
                              Value memref, unsigned dim = 0) {
  return ame::createByteStride(rewriter, loc, memref, dim);
}

static Value createSubView(PatternRewriter &rewriter, Location loc,
                           Value source, ArrayRef<Value> offsets,
                           ArrayRef<Value> sizes) {
  SmallVector<OpFoldResult> offsetResults;
  SmallVector<OpFoldResult> sizeResults;
  for (Value offset : offsets)
    offsetResults.push_back(offset);
  for (Value size : sizes)
    sizeResults.push_back(size);
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  return memref::SubViewOp::create(rewriter, loc, source, offsetResults,
                                   sizeResults, strides);
}

static LogicalResult createMSetTypeAndTiles(PatternRewriter &rewriter,
                                            Operation *anchor, Location loc,
                                            Type elementType, Value currM,
                                            Value currN, Value currK = {}) {
  int64_t msetTypeImm = 0;
  if (failed(getMsetTypeImm(elementType, msetTypeImm)))
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME element type");

  MSettypeiOp::create(rewriter, loc, rewriter.getI64Type(), msetTypeImm);
  MSettilemOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currM));
  MSettilenOp::create(rewriter, loc, rewriter.getI64Type(),
                      castIndexToI64(rewriter, loc, currN));
  if (currK)
    MSettilekOp::create(rewriter, loc, rewriter.getI64Type(),
                        castIndexToI64(rewriter, loc, currK));
  return success();
}

static FailureOr<Value> createLoadA(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlae8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlae16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlae32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlae64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME A load type");
}

static FailureOr<Value> createLoadB(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlbe8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlbe16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlbe32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlbe64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME B load type");
}

static FailureOr<Value> createLoadC(PatternRewriter &rewriter,
                                    Operation *anchor, Location loc,
                                    Type elementType, Value source,
                                    Value stride) {
  Type tileType = VectorType::get({4, 4}, elementType);
  if (elementType.isInteger(8))
    return Mlce8mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    return Mlce16mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(32) || elementType.isF32())
    return Mlce32mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else if (elementType.isInteger(64) || elementType.isF64())
    return Mlce64mOp::create(rewriter, loc, tileType, source, stride).getRes();
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C load type");
}

static LogicalResult createStoreC(PatternRewriter &rewriter, Operation *anchor,
                                  Location loc, Type elementType, Value src,
                                  Value dest, Value stride) {
  if (elementType.isInteger(8))
    Msce8mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    Msce16mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(32) || elementType.isF32())
    Msce32mOp::create(rewriter, loc, src, dest, stride);
  else if (elementType.isInteger(64) || elementType.isF64())
    Msce64mOp::create(rewriter, loc, src, dest, stride);
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME C store type");
  return success();
}

static FailureOr<Value> createMatmul(PatternRewriter &rewriter,
                                     Operation *anchor, Location loc,
                                     Type lhsType, Type resultType, Value acc,
                                     Value lhs, Value rhs) {
  Type tileType = VectorType::get({4, 4}, resultType);
  if (resultType.isInteger(32) && lhsType.isInteger(32))
    return MmaWmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(16))
    return MwmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(32) && lhsType.isInteger(8))
    return MqmaBmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(16) && lhsType.isInteger(16))
    return MmaHmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (resultType.isInteger(64) && lhsType.isInteger(64))
    return MmaDwmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if ((lhsType.isF16() || lhsType.isBF16()) && resultType.isF32())
    return MfwmaHfmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF32())
    return MfmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF32() && resultType.isF64())
    return MfwmaFmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else if (lhsType.isF64() && resultType.isF64())
    return MfmaDmmOp::create(rewriter, loc, tileType, acc, lhs, rhs).getRes();
  else
    return rewriter.notifyMatchFailure(
        anchor, "unsupported BOSCAME matmul instruction type");
}

static Value createMatrixOp(PatternRewriter &rewriter, Location loc,
                            StringRef name, Type resultType,
                            ValueRange operands) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

static FailureOr<Value> createElementwise(PatternRewriter &rewriter,
                                          Operation *anchor, Location loc,
                                          ElementwiseKind kind,
                                          Type elementType, Value lhs,
                                          Value rhs) {
  StringRef suffix;
  if (elementType.isInteger(8))
    suffix = ".b.mm";
  else if (elementType.isInteger(16))
    suffix = ".h.mm";
  else if (elementType.isInteger(32))
    suffix = ".w.mm";
  else if (elementType.isInteger(64))
    suffix = ".dw.mm";
  else if (elementType.isF16() || elementType.isBF16())
    suffix = ".hf.mm";
  else if (elementType.isF32())
    suffix = ".f.mm";
  else if (elementType.isF64())
    suffix = ".d.mm";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME elementwise type");

  StringRef mnemonic;
  switch (kind) {
  case ElementwiseKind::Add:
    mnemonic = "madd";
    break;
  case ElementwiseKind::Sub:
    mnemonic = "msub";
    break;
  case ElementwiseKind::Mul:
    mnemonic = "mmul";
    break;
  case ElementwiseKind::MinS:
    mnemonic = "mmin";
    break;
  case ElementwiseKind::MinU:
    mnemonic = "mminu";
    break;
  case ElementwiseKind::MaxS:
    mnemonic = "mmax";
    break;
  case ElementwiseKind::MaxU:
    mnemonic = "mmaxu";
    break;
  case ElementwiseKind::FAdd:
    mnemonic = "mfadd";
    break;
  case ElementwiseKind::FSub:
    mnemonic = "mfsub";
    break;
  case ElementwiseKind::FMul:
    mnemonic = "mfmul";
    break;
  case ElementwiseKind::FMax:
    mnemonic = "mfmax";
    break;
  }
  std::string name = "bosc_ame." + mnemonic.str() + suffix.str();
  return createMatrixOp(rewriter, loc, name, lhs.getType(), {lhs, rhs});
}

static FailureOr<Value> createBroadcast(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastKind kind,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  if (kind == BroadcastKind::Row) {
    std::string name = "bosc_ame.mbc" + registerSuffix.str() + "r.m";
    return createMatrixOp(rewriter, loc, name, src.getType(), {src});
  }

  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME broadcast type");

  std::string name = "bosc_ame.mbc" + registerSuffix.str() +
                     (kind == BroadcastKind::Column ? "ce" : "ee") +
                     bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static FailureOr<Value> createTranspose(PatternRewriter &rewriter,
                                        Operation *anchor, Location loc,
                                        BroadcastRegisterKind regKind,
                                        Type elementType, Value src) {
  StringRef registerSuffix = regKind == BroadcastRegisterKind::TileA   ? "a"
                             : regKind == BroadcastRegisterKind::TileB ? "b"
                                                                       : "c";
  StringRef bitWidth;
  if (elementType.isInteger(8))
    bitWidth = "8";
  else if (elementType.isInteger(16) || elementType.isF16() ||
           elementType.isBF16())
    bitWidth = "16";
  else if (elementType.isInteger(32) || elementType.isF32())
    bitWidth = "32";
  else if (elementType.isInteger(64) || elementType.isF64())
    bitWidth = "64";
  else
    return rewriter.notifyMatchFailure(anchor,
                                       "unsupported BOSCAME transpose type");
  std::string name =
      "bosc_ame.mt" + registerSuffix.str() + "e" + bitWidth.str() + ".m";
  return createMatrixOp(rewriter, loc, name, src.getType(), {src});
}

static LogicalResult lowerMatmulLike(Operation *anchor,
                                     PatternRewriter &rewriter, Location loc,
                                     Value A, Value B, Value C) {
  auto AType = dyn_cast<MemRefType>(A.getType());
  auto BType = dyn_cast<MemRefType>(B.getType());
  auto CType = dyn_cast<MemRefType>(C.getType());
  if (!AType || !BType || !CType)
    return rewriter.notifyMatchFailure(anchor, "expected memref operands");

  Type elemTypeA = AType.getElementType();
  Type elemTypeB = BType.getElementType();
  Type elemTypeC = CType.getElementType();
  if (elemTypeA != elemTypeB)
    return rewriter.notifyMatchFailure(
        anchor, "matmul lhs and rhs element types differ");

  int64_t tileM = 4, tileN = 4, tileK = 4;
  if (elemTypeA.isInteger(8) && elemTypeC.isInteger(32))
    tileK = 16;
  else if ((elemTypeA.isF16() || elemTypeA.isBF16()) && elemTypeC.isF32())
    tileK = 8;
  else if (elemTypeA.isInteger(16) && elemTypeC.isInteger(32))
    tileK = 8;
  else if (elemTypeA.isInteger(32) && elemTypeC.isInteger(32))
    tileK = 4;
  else if (elemTypeA.isF32() && (elemTypeC.isF32() || elemTypeC.isF64()))
    tileK = 4;
  else if (elemTypeA.isInteger(64) && elemTypeC.isInteger(64))
    tileK = 2;
  else if (elemTypeA.isF64() && elemTypeC.isF64())
    tileK = 2;
  else
    return rewriter.notifyMatchFailure(anchor, "unsupported matmul precision");

  Value dimM = createDim(rewriter, loc, A, 0);
  Value dimK = createDim(rewriter, loc, A, 1);
  Value dimN = createDim(rewriter, loc, B, 1);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
  Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);
  Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

  auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
  rewriter.setInsertionPointToStart(loopM.getBody());
  Value ivM = loopM.getInductionVar();

  auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
  rewriter.setInsertionPointToStart(loopN.getBody());
  Value ivN = loopN.getInductionVar();

  auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK);
  rewriter.setInsertionPointToStart(loopK.getBody());
  Value ivK = loopK.getInductionVar();

  Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
  Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);
  Value currK = createIndexMin(rewriter, loc, dimK, ivK, tileK);

  Value subA = createSubView(rewriter, loc, A, {ivM, ivK}, {currM, currK});
  Value subB = createSubView(rewriter, loc, B, {ivK, ivN}, {currK, currN});
  Value subC = createSubView(rewriter, loc, C, {ivM, ivN}, {currM, currN});

  Value strideA = createByteStride(rewriter, loc, subA);
  Value strideB = createByteStride(rewriter, loc, subB);
  Value strideC = createByteStride(rewriter, loc, subC);

  if (failed(createMSetTypeAndTiles(rewriter, anchor, loc, elemTypeA, currM,
                                    currN, currK)))
    return failure();
  FailureOr<Value> tileA =
      createLoadA(rewriter, anchor, loc, elemTypeA, subA, strideA);
  FailureOr<Value> tileB =
      createLoadB(rewriter, anchor, loc, elemTypeB, subB, strideB);
  FailureOr<Value> acc =
      createLoadC(rewriter, anchor, loc, elemTypeC, subC, strideC);
  if (failed(tileA) || failed(tileB) || failed(acc))
    return failure();
  FailureOr<Value> result = createMatmul(rewriter, anchor, loc, elemTypeA,
                                         elemTypeC, *acc, *tileA, *tileB);
  if (failed(result) || failed(createStoreC(rewriter, anchor, loc, elemTypeC,
                                            *result, subC, strideC)))
    return failure();

  rewriter.setInsertionPointAfter(loopM);
  return success();
}

/// Which pathway owns one `linalg.matmul`.
enum class MatmulLoweringPlan {
  /// The upstream value-semantics matmul helper can lower it.
  Generic,
  /// The Qwen3 FPGA AME fast path claims it.
  FpgaFastPath,
  /// Deliberately left for the later CPU/VIR pipeline.
  LeaveForFallback,
};

/// Pure analysis shared by the dispatcher and the conversion legality contract.
///
/// Keeping one classifier means the target never declares an operation illegal
/// that no pattern can handle, and never leaves a claimed candidate legal.
static MatmulLoweringPlan classifyMatmul(linalg::MatmulOp op,
                                         AmeTargetProfile profile,
                                         bool tritonW8A8FastPath) {
  if (profile != AmeTargetProfile::Qwen3Fpga)
    return MatmulLoweringPlan::Generic;

  if (op->hasAttr(kTritonConsumerFenceAttr) ||
      (tritonW8A8FastPath && succeeded(matchTritonI8DotCast(op))) ||
      succeeded(matchQwenDirectCMatmul(op)))
    return MatmulLoweringPlan::FpgaFastPath;

  return MatmulLoweringPlan::LeaveForFallback;
}

/// Lower one strict Qwen3 direct-C matmul with the FPGA AME schedule.
///
/// The i32 accumulator is a single loop-carried SSA chain:
///
///   %acc0 = bosc_ame.mlce32.m %c, %strideC        // accumulator seed
///   %acc1 = scf.for %k ... iter_args(%acc = %acc0) {
///     %a = bosc_ame.mlae8.m %subA, %strideA
///     %b = bosc_ame.mlbe8.m | mlbte8.m %subB, %strideB
///     %next = bosc_ame.mqma.b.mm %acc, %a, %b
///     scf.yield %next
///   }
///   bosc_ame.msce32.m %acc1, %c, %strideC           // i32 -> f32 in hardware
///
/// Making the accumulation an SSA chain is what keeps the "the accumulator
/// stays resident across every K tile" invariant reviewable: reloading the fp32
/// result between K tiles would reinterpret its IEEE-754 bits as an integer
/// accumulator, which is a silent wrong-answer bug.
///
/// The wide 2A4B / 1A8B schedules (eight accumulator chains) are not part of
/// this first migration step; the tiled schedule below is the correctness
/// baseline they will be built on.
static LogicalResult lowerFpgaMatmul(linalg::MatmulOp op,
                                     PatternRewriter &rewriter,
                                     AmeTargetProfile profile,
                                     bool tritonW8A8FastPath,
                                     QwenDirectCMatch &directC) {
  Location loc = op.getLoc();
  Value A = op.getDpsInputOperand(0)->get();
  Value B = op.getDpsInputOperand(1)->get();
  Value C = directC.finalOutput;

  auto AType = cast<MemRefType>(A.getType());
  auto BType = cast<MemRefType>(B.getType());
  auto CType = cast<MemRefType>(C.getType());

  Type i8Type = rewriter.getI8Type();
  Type i32Type = rewriter.getI32Type();

  // Target capability check: the FPGA datapath used here is i8 x i8 -> i32 with
  // an fp32 memory side for the accumulator load/store pair.
  if (!isFpgaMmaSupported(i8Type, i32Type) ||
      !isFpgaAccumulatorMemorySupported(i32Type, CType.getElementType())) {
    op.emitOpError() << "unsupported Qwen3 FPGA AME datapath: i8 x i8 -> i32 "
                        "accumulator with "
                     << CType.getElementType() << " accumulator memory";
    return failure();
  }

  // Triton materializes a logical [K, N] tile loaded from a row-major
  // [N, full-K] weight as a zero-copy memref with strides [1, full-K]; the AME
  // mlbe path consumes that physical layout directly.
  const bool transposedB = !hasSupportedRowMajor2DLayout(BType) &&
                           hasSupportedTransposed2DLayout(BType);

  const bool copiedDestination = static_cast<bool>(directC.copyToFinal);
  if (copiedDestination) {
    // Insert the replacement at the old copy point. The final output and its
    // address calculation already dominate this location, and final C is not
    // made visible earlier than in the original program.
    rewriter.setInsertionPoint(directC.copyToFinal);
    Value zero = directC.zeroFill.getDpsInputOperand(0)->get();
    linalg::FillOp::create(rewriter, directC.zeroFill.getLoc(), zero, C);
  } else {
    // Buddy Frontend bufferization already exposes the destination buffer to
    // downstream operations, so replace the matmul in place and retain the
    // original allocation/fill/lifetime.
    rewriter.setInsertionPoint(op);
  }

  constexpr int64_t tileM = 16;
  constexpr int64_t tileN = 16;
  constexpr int64_t tileK = 64;

  Value dimM = createDim(rewriter, loc, A, 0);
  Value dimK = createDim(rewriter, loc, A, 1);
  Value dimN = createDim(rewriter, loc, B, 1);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
  Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);
  Value stepK = arith::ConstantIndexOp::create(rewriter, loc, tileK);

  auto finishReplacement = [&]() {
    rewriter.eraseOp(op);
    if (copiedDestination) {
      if (directC.temporaryDealloc)
        rewriter.eraseOp(directC.temporaryDealloc);
      rewriter.eraseOp(directC.copyToFinal);
      rewriter.eraseOp(directC.zeroFill);
      rewriter.eraseOp(directC.temporaryAlloc);
    }
  };

  const int64_t staticM = AType.getDimSize(0);
  const int64_t staticK = AType.getDimSize(1);
  const int64_t staticN = BType.getDimSize(1);
  const bool use2A4B = tritonW8A8FastPath && transposedB && staticM == 32 &&
                       staticN % 64 == 0 && staticK % 64 == 0;
  const bool useDecodeWide = tritonW8A8FastPath && transposedB && staticM > 0 &&
                             staticM <= 16 && staticN % 64 == 0 &&
                             staticK % 64 == 0;

  if (use2A4B || useDecodeWide) {
    // Wide FPGA schedules.  Both keep the accumulator resident across the whole
    // K reduction, but use more than one accumulator chain:
    //
    //   2A4B        - two A tiles (rows 0..15 and 16..31) against one bank of
    //                 four B tiles, eight accumulators.
    //   1A8B/decode - one A tile against two batches of four B tiles; the B
    //                 bank is reused for the second batch.  N64 stays as the
    //                 exact four-chain fallback for the last half block.
    //
    // Every accumulator is an SSA value carried by the K loop, so the physical
    // slots that the old lane numbers encoded are dataflow now.
    Value strideA = ame::createByteStride(rewriter, loc, A, 0);
    // Logical B is [K, N] but the Triton view is physically [N, K].
    Value strideB = ame::createByteStride(rewriter, loc, B, 1);
    Value strideC = ame::createByteStride(rewriter, loc, C, 0);

    Type cMemoryType = CType.getElementType();
    constexpr int64_t bankN = 16;

    auto cIndex = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    auto cTile = [&](int64_t row, int64_t column, int64_t rows) {
      return createSubView(rewriter, loc, C, {cIndex(row), cIndex(column)},
                           {cIndex(rows), cIndex(bankN)});
    };
    auto aTile = [&](int64_t row, Value k, int64_t rows) {
      return createSubView(rewriter, loc, A, {cIndex(row), k},
                           {cIndex(rows), cIndex(tileK)});
    };
    auto bTile = [&](Value k, int64_t column) {
      return createSubView(rewriter, loc, B, {k, cIndex(column)},
                           {cIndex(tileK), cIndex(bankN)});
    };

    // Seed the accumulator chains from the (zero-filled) destination.
    auto seedAccumulator = [&](Value tile, unsigned slot) -> FailureOr<Value> {
      return ame::createLoadAccumulator(rewriter, loc, i32Type, cMemoryType,
                                        tile, strideC, op, slot);
    };

    // One activation tile against four B tiles feeding `accBase + 0..3`.
    auto emitFourWeights = [&](Value k, int64_t column, unsigned accBase,
                               Value activation, ArrayRef<Value> current)
        -> FailureOr<SmallVector<Value, 8>> {
      SmallVector<Value, 8> updated(current.begin(), current.end());
      // The load/MMA interleaving reproduces the validated schedule: the first
      // two B tiles are in flight before the first MMA, and every later load is
      // issued before the MMA that consumes the previous tile.
      FailureOr<Value> weight0 =
          ame::createLoadB(rewriter, loc, i8Type, bTile(k, column), strideB, op);
      FailureOr<Value> weight1 = ame::createLoadB(
          rewriter, loc, i8Type, bTile(k, column + bankN), strideB, op, 5);
      if (failed(weight0) || failed(weight1))
        return failure();

      FailureOr<Value> next0 = ame::createMma(
          rewriter, loc, updated[accBase + 0], activation, *weight0, op);
      if (failed(next0))
        return failure();
      updated[accBase + 0] = *next0;

      FailureOr<Value> weight2 = ame::createLoadB(
          rewriter, loc, i8Type, bTile(k, column + 2 * bankN), strideB, op, 6);
      if (failed(weight2))
        return failure();
      FailureOr<Value> next1 = ame::createMma(
          rewriter, loc, updated[accBase + 1], activation, *weight1, op);
      if (failed(next1))
        return failure();
      updated[accBase + 1] = *next1;

      FailureOr<Value> weight3 = ame::createLoadB(
          rewriter, loc, i8Type, bTile(k, column + 3 * bankN), strideB, op, 7);
      if (failed(weight3))
        return failure();
      FailureOr<Value> next2 = ame::createMma(
          rewriter, loc, updated[accBase + 2], activation, *weight2, op);
      if (failed(next2))
        return failure();
      updated[accBase + 2] = *next2;

      FailureOr<Value> next3 = ame::createMma(
          rewriter, loc, updated[accBase + 3], activation, *weight3, op);
      if (failed(next3))
        return failure();
      updated[accBase + 3] = *next3;
      return updated;
    };

    if (use2A4B) {
      Value tileMValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(16));
      if (failed(ame::configureTiles(rewriter, loc, tileMValue, cIndex(bankN),
                                     cIndex(tileK))))
        return failure();

      for (int64_t column = 0; column < staticN; column += 4 * bankN) {
        if (failed(ame::configureAccumulatorType(rewriter, loc, i32Type,
                                                profile, op)))
          return failure();
        SmallVector<Value, 8> seeds;
        for (unsigned lane = 0; lane < 8; ++lane) {
          int64_t row = lane < 4 ? 0 : 16;
          int64_t offset = column + static_cast<int64_t>(lane % 4) * bankN;
          FailureOr<Value> seed = seedAccumulator(cTile(row, offset, 16), lane);
          if (failed(seed))
            return failure();
          seeds.push_back(*seed);
        }

        if (failed(ame::configureMmaType(rewriter, loc, i8Type, profile, op)))
          return failure();

        auto kLoop = scf::ForOp::create(rewriter, loc, c0, dimK, stepK, seeds);
        rewriter.setInsertionPointToStart(kLoop.getBody());
        Value k = kLoop.getInductionVar();
        MutableArrayRef<BlockArgument> carried = kLoop.getRegionIterArgs();

        FailureOr<Value> activation0 =
            ame::createLoadA(rewriter, loc, i8Type, aTile(0, k, 16), strideA, op);
        FailureOr<Value> activation1 = ame::createLoadA(
            rewriter, loc, i8Type, aTile(16, k, 16), strideA, op, 2);
        if (failed(activation0) || failed(activation1))
          return failure();

        // Interleave the two A banks exactly like the validated schedule:
        // acc0..3 consume A rows 0..15, acc4..7 consume A rows 16..31, and the
        // four B registers are shared by both banks.
        SmallVector<Value, 8> updated(carried.begin(), carried.end());
        // The validated 2A4B schedule keeps two B tiles in flight and alternates
        // the two A banks: b4, b5, acc0+=a0*b4, b6, acc4+=a2*b4, b7,
        // acc1+=a0*b5, acc5+=a2*b5, acc2+=a0*b6, acc6+=a2*b6, acc3+=a0*b7,
        // acc7+=a2*b7.
        FailureOr<Value> weight0 =
            ame::createLoadB(rewriter, loc, i8Type, bTile(k, column), strideB, op);
        FailureOr<Value> weight1 = ame::createLoadB(
            rewriter, loc, i8Type, bTile(k, column + bankN), strideB, op, 5);
        if (failed(weight0) || failed(weight1))
          return failure();

        FailureOr<Value> mma0 = ame::createMma(rewriter, loc, updated[0],
                                               *activation0, *weight0, op);
        if (failed(mma0))
          return failure();
        updated[0] = *mma0;

        FailureOr<Value> weight2 = ame::createLoadB(
            rewriter, loc, i8Type, bTile(k, column + 2 * bankN), strideB, op, 6);
        if (failed(weight2))
          return failure();
        FailureOr<Value> mma4 = ame::createMma(rewriter, loc, updated[4],
                                               *activation1, *weight0, op);
        if (failed(mma4))
          return failure();
        updated[4] = *mma4;

        FailureOr<Value> weight3 = ame::createLoadB(
            rewriter, loc, i8Type, bTile(k, column + 3 * bankN), strideB, op, 7);
        if (failed(weight3))
          return failure();

        FailureOr<Value> mma1 = ame::createMma(rewriter, loc, updated[1],
                                               *activation0, *weight1, op);
        if (failed(mma1))
          return failure();
        updated[1] = *mma1;
        FailureOr<Value> mma5 = ame::createMma(rewriter, loc, updated[5],
                                               *activation1, *weight1, op);
        if (failed(mma5))
          return failure();
        updated[5] = *mma5;
        FailureOr<Value> mma2 = ame::createMma(rewriter, loc, updated[2],
                                               *activation0, *weight2, op);
        if (failed(mma2))
          return failure();
        updated[2] = *mma2;
        FailureOr<Value> mma6 = ame::createMma(rewriter, loc, updated[6],
                                               *activation1, *weight2, op);
        if (failed(mma6))
          return failure();
        updated[6] = *mma6;
        FailureOr<Value> mma3 = ame::createMma(rewriter, loc, updated[3],
                                               *activation0, *weight3, op);
        if (failed(mma3))
          return failure();
        updated[3] = *mma3;
        FailureOr<Value> mma7 = ame::createMma(rewriter, loc, updated[7],
                                               *activation1, *weight3, op);
        if (failed(mma7))
          return failure();
        updated[7] = *mma7;
        scf::YieldOp::create(rewriter, loc, updated);
        rewriter.setInsertionPointAfter(kLoop);

        if (failed(ame::configureAccumulatorType(rewriter, loc, i32Type, profile,
                                                 op)))
          return failure();
        for (unsigned lane = 0; lane < 8; ++lane) {
          int64_t row = lane < 4 ? 0 : 16;
          int64_t offset = column + static_cast<int64_t>(lane % 4) * bankN;
          if (failed(ame::createStoreAccumulator(
                  rewriter, loc, kLoop.getResults()[lane],
                  cTile(row, offset, 16), strideC, op)))
            return failure();
        }
      }

      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      finishReplacement();
      return success();
    }

    // Decode-style M <= 16 path: consume N128 with eight accumulator chains and
    // keep N64 as the exact four-chain fallback for the last half block.
    Value tileMValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(staticM));
    if (failed(ame::configureTiles(rewriter, loc, tileMValue, cIndex(bankN),
                                   cIndex(tileK))))
      return failure();

    for (int64_t column = 0; column < staticN;) {
      const int64_t banks = staticN - column >= 8 * bankN ? 2 : 1;
      const unsigned lanes = static_cast<unsigned>(banks) * 4;

      if (failed(ame::configureAccumulatorType(rewriter, loc, i32Type,
                                              profile, op)))
        return failure();

      SmallVector<Value, 8> seeds;
      for (unsigned lane = 0; lane < lanes; ++lane) {
        FailureOr<Value> seed =
            seedAccumulator(cTile(0, column + lane * bankN, staticM), lane);
        if (failed(seed))
          return failure();
        seeds.push_back(*seed);
      }

      if (failed(ame::configureMmaType(rewriter, loc, i8Type, profile, op)))
        return failure();

      auto kLoop = scf::ForOp::create(rewriter, loc, c0, dimK, stepK, seeds);
      rewriter.setInsertionPointToStart(kLoop.getBody());
      Value k = kLoop.getInductionVar();
      MutableArrayRef<BlockArgument> carried = kLoop.getRegionIterArgs();

      FailureOr<Value> activation = ame::createLoadA(
          rewriter, loc, i8Type, aTile(0, k, staticM), strideA, op);
      if (failed(activation))
        return failure();

      SmallVector<Value, 8> carriedValues(carried.begin(), carried.end());
      FailureOr<SmallVector<Value, 8>> updated =
          emitFourWeights(k, column, 0, *activation, carriedValues);
      if (failed(updated))
        return failure();
      if (banks == 2) {
        // The second batch reuses the same four B registers.
        FailureOr<SmallVector<Value, 8>> second =
            emitFourWeights(k, column + 4 * bankN, 4, *activation, *updated);
        if (failed(second))
          return failure();
        updated = *second;
      }
      scf::YieldOp::create(rewriter, loc, *updated);
      rewriter.setInsertionPointAfter(kLoop);

      if (failed(
              ame::configureAccumulatorType(rewriter, loc, i32Type, profile, op)))
        return failure();
      for (unsigned lane = 0; lane < lanes; ++lane) {
        if (failed(ame::createStoreAccumulator(
                rewriter, loc, kLoop.getResults()[lane],
                cTile(0, column + lane * bankN, staticM), strideC, op)))
          return failure();
      }
      column += static_cast<int64_t>(lanes) * bankN;
    }

    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
    finishReplacement();
    return success();
  }

  auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
  rewriter.setInsertionPointToStart(loopM.getBody());
  Value ivM = loopM.getInductionVar();

  auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
  rewriter.setInsertionPointToStart(loopN.getBody());
  Value ivN = loopN.getInductionVar();

  Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
  Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

  Value subC = createSubView(rewriter, loc, C, {ivM, ivN}, {currM, currN});
  Value strideC = ame::createByteStride(rewriter, loc, subC, 0);

  // --- Step 1: accumulator load (mtype = accumulator datapath) ---
  if (failed(ame::configureTiles(rewriter, loc, currM, currN)) ||
      failed(ame::configureAccumulatorType(rewriter, loc, i32Type, profile, op)))
    return failure();

  FailureOr<Value> accumulatorSeed =
      ame::createLoadAccumulator(rewriter, loc, i32Type, CType.getElementType(),
                                 subC, strideC, op);
  if (failed(accumulatorSeed))
    return failure();

  // --- Step 2: MMA for every K tile (mtype = A/B datapath) ---
  if (failed(ame::configureMmaType(rewriter, loc, i8Type, profile, op)))
    return failure();

  auto loopK = scf::ForOp::create(rewriter, loc, c0, dimK, stepK,
                                  ValueRange{*accumulatorSeed});
  rewriter.setInsertionPointToStart(loopK.getBody());
  Value ivK = loopK.getInductionVar();
  Value currK = createIndexMin(rewriter, loc, dimK, ivK, tileK);

  Value subA = createSubView(rewriter, loc, A, {ivM, ivK}, {currM, currK});
  Value subB = createSubView(rewriter, loc, B, {ivK, ivN}, {currK, currN});
  Value strideA = ame::createByteStride(rewriter, loc, subA, 0);
  Value strideB = ame::createByteStride(rewriter, loc, subB, transposedB ? 1 : 0);

  ame::configureTileK(rewriter, loc, currK);

  FailureOr<Value> activation =
      ame::createLoadA(rewriter, loc, i8Type, subA, strideA, op);
  FailureOr<Value> weight =
      transposedB
          ? ame::createLoadB(rewriter, loc, i8Type, subB, strideB, op)
          : ame::createLoadBTransposed(rewriter, loc, i8Type, subB, strideB, op);
  if (failed(activation) || failed(weight))
    return failure();

  FailureOr<Value> nextAccumulator =
      ame::createMma(rewriter, loc, loopK.getRegionIterArgs()[0], *activation,
                     *weight, op);
  if (failed(nextAccumulator))
    return failure();
  scf::YieldOp::create(rewriter, loc, ValueRange{*nextAccumulator});
  rewriter.setInsertionPointAfter(loopK);

  // --- Step 3: accumulator store after all K tiles ---
  if (failed(ame::configureAccumulatorType(rewriter, loc, i32Type, profile, op)) ||
      failed(ame::createStoreAccumulator(rewriter, loc, loopK.getResults()[0],
                                         subC, strideC, op)))
    return failure();

  rewriter.setInsertionPointAfter(loopM);
  LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);

  finishReplacement();
  return success();
}

/// NR exposes the integer accumulator without the v0.1 implicit f32 conversion.
/// Keep linalg's ordinary C += A * B semantics, including nonzero C and an
/// explicit subsequent sitofp. The NR mlbe8 instruction consumes physical
/// [N, K] rows; ordinary [K, N] weights need a small scalar packing buffer.
static bool isNrMatmul(linalg::MatmulOp op, bool transposeB) {
  if (!op.hasPureBufferSemantics() ||
      op.getCast() != linalg::TypeFn::cast_signed ||
      (op.hasUserDefinedMaps() && !transposeB))
    return false;
  auto a = dyn_cast<MemRefType>(op.getDpsInputOperand(0)->get().getType());
  auto b = dyn_cast<MemRefType>(op.getDpsInputOperand(1)->get().getType());
  auto c = dyn_cast<MemRefType>(op.getDpsInitOperand(0)->get().getType());
  if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.hasStaticShape() || !b.hasStaticShape() ||
      !c.hasStaticShape() || !a.getElementType().isSignlessInteger(8) ||
      !b.getElementType().isSignlessInteger(8) ||
      !c.getElementType().isSignlessInteger(32))
    return false;
  int64_t m = a.getDimSize(0), k = a.getDimSize(1);
  int64_t n = b.getDimSize(transposeB ? 0 : 1);
  return m > 0 && n > 0 && k > 0 &&
         b.getDimSize(transposeB ? 1 : 0) == k &&
         c.getShape() == ArrayRef<int64_t>({m, n}) &&
         hasSupportedRowMajor2DLayout(a) &&
         hasSupportedRowMajor2DLayout(c) &&
         (transposeB ? hasSupportedRowMajor2DLayout(b)
                     : hasSupportedBLayout(b));
}

class NrMatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  NrMatmulToBOSCAMELowering(MLIRContext *context, int64_t tileN)
      : OpRewritePattern<linalg::MatmulOp>(context), tileN(tileN) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    bool transposeB = isa<linalg::MatmulTransposeBOp>(op.getOperation());
    if (!isNrMatmul(op, transposeB))
      return failure();
    Location loc = op.getLoc();
    Value a = op.getDpsInputOperand(0)->get();
    Value b = op.getDpsInputOperand(1)->get();
    Value c = op.getDpsInitOperand(0)->get();
    auto bType = cast<MemRefType>(b.getType());
    bool physicalNK = transposeB ||
                      (!hasSupportedRowMajor2DLayout(bType) &&
                       hasSupportedTransposed2DLayout(bType));
    Type i8 = rewriter.getI8Type(), i32 = rewriter.getI32Type();
    constexpr AmeTargetProfile profile = AmeTargetProfile::NrFpga;
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    Value zero = index(0), one = index(1);
    Value stepM = index(16), stepN = index(tileN), stepK = index(64);
    Value dimM = createDim(rewriter, loc, a, 0);
    Value dimK = createDim(rewriter, loc, a, 1);
    Value dimN = createDim(rewriter, loc, b, transposeB ? 0 : 1);
    Value packedB;
    if (!physicalNK) {
      auto storage = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get({tileN, 64}, i8));
      storage->setAttr("alignment", rewriter.getI64IntegerAttr(64));
      packedB = storage;
    }

    // Drain preceding CPU writes before AME seeds its accumulator / tiles.
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
    auto mLoop = scf::ForOp::create(rewriter, loc, zero, dimM, stepM);
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value m = mLoop.getInductionVar();
    Value rows = createIndexMin(rewriter, loc, dimM, m, 16);
    auto nLoop = scf::ForOp::create(rewriter, loc, zero, dimN, stepN);
    rewriter.setInsertionPointToStart(nLoop.getBody());
    Value n = nLoop.getInductionVar();
    Value columns = createIndexMin(rewriter, loc, dimN, n, tileN);
    Value subC = createSubView(rewriter, loc, c, {m, n}, {rows, columns});
    Value strideC = ame::createByteStride(rewriter, loc, subC);
    if (failed(ame::configureTiles(rewriter, loc, rows, columns)) ||
        failed(ame::configureAccumulatorType(rewriter, loc, i32, profile, op)))
      return failure();
    FailureOr<Value> seed = ame::createLoadAccumulator(
        rewriter, loc, i32, i32, subC, strideC, op);
    if (failed(seed) ||
        failed(ame::configureMmaType(rewriter, loc, i8, profile, op)))
      return failure();
    auto kLoop = scf::ForOp::create(rewriter, loc, zero, dimK, stepK,
                                   ValueRange{*seed});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value k = kLoop.getInductionVar();
    Value depth = createIndexMin(rewriter, loc, dimK, k, 64);
    Value subA = createSubView(rewriter, loc, a, {m, k}, {rows, depth});
    Value subB, strideB;
    if (physicalNK) {
      subB = transposeB
                 ? createSubView(rewriter, loc, b, {n, k}, {columns, depth})
                 : createSubView(rewriter, loc, b, {k, n}, {depth, columns});
      strideB = ame::createByteStride(rewriter, loc, subB, transposeB ? 0 : 1);
    } else {
      // Complete the preceding B load before reusing its scratch, and publish
      // this tile after packing. Both loops stop at the real tail dimensions.
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      auto packN = scf::ForOp::create(rewriter, loc, zero, columns, one);
      rewriter.setInsertionPointToStart(packN.getBody());
      auto packK = scf::ForOp::create(rewriter, loc, zero, depth, one);
      rewriter.setInsertionPointToStart(packK.getBody());
      Value bn = arith::AddIOp::create(rewriter, loc, n, packN.getInductionVar());
      Value bk = arith::AddIOp::create(rewriter, loc, k, packK.getInductionVar());
      Value element = memref::LoadOp::create(rewriter, loc, b, ValueRange{bk, bn});
      memref::StoreOp::create(rewriter, loc, element, packedB,
                             ValueRange{packN.getInductionVar(),
                                        packK.getInductionVar()});
      rewriter.setInsertionPointAfter(packN);
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
      subB = packedB;
      strideB = castIndexToI64(rewriter, loc, stepK);
    }
    if (failed(ame::configureTileK(rewriter, loc, depth)))
      return failure();
    FailureOr<Value> tileA = ame::createLoadA(
        rewriter, loc, i8, subA, ame::createByteStride(rewriter, loc, subA), op);
    FailureOr<Value> tileB =
        ame::createLoadB(rewriter, loc, i8, subB, strideB, op);
    if (failed(tileA) || failed(tileB))
      return failure();
    FailureOr<Value> acc = ame::createMma(rewriter, loc,
        kLoop.getRegionIterArgs()[0], *tileA, *tileB, op);
    if (failed(acc))
      return failure();
    scf::YieldOp::create(rewriter, loc, ValueRange{*acc});
    rewriter.setInsertionPointAfter(kLoop);
    if (failed(ame::configureAccumulatorType(rewriter, loc, i32, profile, op)) ||
        failed(ame::createStoreAccumulator(rewriter, loc, kLoop.getResult(0),
                                          subC, strideC, op)))
      return failure();
    rewriter.setInsertionPointAfter(mLoop);
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst);
    rewriter.eraseOp(op);
    return success();
  }

private:
  // The v0.5 direct-memory AME contract supports N up to 64. This controls
  // both the logical tile and the optional [N, K] packing buffer; it does not
  // change the source/output row strides or reset the accumulator between K
  // chunks. Keep 16 as the default until wider schedules are board-validated.
  int64_t tileN;
};

class MatmulToBOSCAMELowering : public OpRewritePattern<linalg::MatmulOp> {
public:
  MatmulToBOSCAMELowering(MLIRContext *context, AmeTargetProfile profile,
                          bool tritonW8A8FastPath)
      : OpRewritePattern<linalg::MatmulOp>(context), profile(profile),
        tritonW8A8FastPath(tritonW8A8FastPath) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    switch (classifyMatmul(op, profile, tritonW8A8FastPath)) {
    case MatmulLoweringPlan::LeaveForFallback:
      // Legal by contract: a later CPU/VIR lowering owns this operation.
      return rewriter.notifyMatchFailure(
          op, "matmul is left for the CPU/VIR pipeline on this AME target");
    case MatmulLoweringPlan::FpgaFastPath: {
      FailureOr<QwenDirectCMatch> directC = matchQwenDirectCMatmul(op);
      if (failed(directC)) {
        if (op->hasAttr(kTritonConsumerFenceAttr))
          op.emitOpError() << "fused Triton W8A8 matmul is not a safe Qwen3 "
                              "FPGA direct-C operation";
        return failure();
      }
      return lowerFpgaMatmul(op, rewriter, profile, tritonW8A8FastPath,
                             *directC);
    }
    case MatmulLoweringPlan::Generic:
      break;
    }

    if (!op.hasPureBufferSemantics())
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

private:
  AmeTargetProfile profile;
  bool tritonW8A8FastPath;
};

class GenericMatmulToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!isGenericMatmulLike(op))
      return failure();

    Value A = op.getDpsInputOperand(0)->get();
    Value B = op.getDpsInputOperand(1)->get();
    Value C = op.getDpsInitOperand(0)->get();

    if (failed(lowerMatmulLike(op, rewriter, op.getLoc(), A, B, C)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class GenericElementwiseToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    ElementwiseKind kind;
    if (!matchElementwiseKind(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto lhsType = dyn_cast<MemRefType>(lhs.getType());
    auto rhsType = dyn_cast<MemRefType>(rhs.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (lhsType.getElementType() != rhsType.getElementType() ||
        lhsType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op,
                                         "elementwise operand types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subLhs =
        createSubView(rewriter, loc, lhs, {ivM, ivN}, {currM, currN});
    Value subRhs =
        createSubView(rewriter, loc, rhs, {ivM, ivN}, {currM, currN});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideLhs = createByteStride(rewriter, loc, subLhs);
    Value strideRhs = createByteStride(rewriter, loc, subRhs);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> lhsTile =
        createLoadC(rewriter, op, loc, elementType, subLhs, strideLhs);
    FailureOr<Value> rhsTile =
        createLoadC(rewriter, op, loc, elementType, subRhs, strideRhs);
    if (failed(lhsTile) || failed(rhsTile))
      return failure();
    FailureOr<Value> result = createElementwise(
        rewriter, op, loc, kind, elementType, *lhsTile, *rhsTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericUnarySquareToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchUnarySquare(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != outType.getRank() ||
        inputType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "expected same-rank memrefs with rank >= 2");

    if (inputType.getElementType() != outType.getElementType())
      return rewriter.notifyMatchFailure(op, "square operand types differ");

    Type elementType = outType.getElementType();
    if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32() &&
        !elementType.isF64())
      return rewriter.notifyMatchFailure(
          op, "math.fpowi square expects floating point element type");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    unsigned rank = outType.getRank();
    unsigned matrixDimM = rank - 2;
    unsigned matrixDimN = rank - 1;

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    SmallVector<Value> outerIvs;
    SmallVector<scf::ForOp> outerLoops;
    for (unsigned dim = 0; dim < matrixDimM; ++dim) {
      Value dimSize = createDim(rewriter, loc, out, dim);
      auto loop = scf::ForOp::create(rewriter, loc, c0, dimSize, c1);
      outerLoops.push_back(loop);
      rewriter.setInsertionPointToStart(loop.getBody());
      outerIvs.push_back(loop.getInductionVar());
    }

    Value dimM = createDim(rewriter, loc, out, matrixDimM);
    Value dimN = createDim(rewriter, loc, out, matrixDimN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    SmallVector<Value> offsets(outerIvs.begin(), outerIvs.end());
    offsets.push_back(ivM);
    offsets.push_back(ivN);

    SmallVector<Value> sizes(outerIvs.size(), c1);
    sizes.push_back(currM);
    sizes.push_back(currN);

    Value subInput = createSubView(rewriter, loc, input, offsets, sizes);
    Value subOut = createSubView(rewriter, loc, out, offsets, sizes);

    Value strideInput = createByteStride(rewriter, loc, subInput, matrixDimM);
    Value strideOut = createByteStride(rewriter, loc, subOut, matrixDimM);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createElementwise(rewriter, op, loc, ElementwiseKind::FMul, elementType,
                          *inputTile, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    if (!outerLoops.empty())
      rewriter.setInsertionPointAfter(outerLoops.front());
    else
      rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericTransposeToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchTranspose(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || inputType.getRank() != 2 ||
        outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "transpose element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;
    for (int64_t dim : outType.getShape()) {
      if (ShapedType::isDynamic(dim) || dim % tileM != 0)
        return rewriter.notifyMatchFailure(
            op, "BOSCAME transpose requires full 4x4 square tiles");
    }

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value subInput =
        createSubView(rewriter, loc, input, {ivN, ivM}, {currN, currM});
    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});

    Value strideInput = createByteStride(rewriter, loc, subInput);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile =
        createLoadC(rewriter, op, loc, elementType, subInput, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result =
        createTranspose(rewriter, op, loc, BroadcastRegisterKind::Accumulation,
                        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

class GenericBroadcastToBOSCAMELowering
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    BroadcastKind kind;
    if (!matchBroadcast(op, kind))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getDpsInputOperand(0)->get();
    Value out = op.getDpsInitOperand(0)->get();

    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outType = dyn_cast<MemRefType>(out.getType());
    if (!inputType || !outType || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected memref operands");

    Type elementType = outType.getElementType();
    if (inputType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(op, "broadcast element types differ");

    constexpr int64_t tileM = 4;
    constexpr int64_t tileN = 4;

    Value dimM = createDim(rewriter, loc, out, 0);
    Value dimN = createDim(rewriter, loc, out, 1);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value stepM = arith::ConstantIndexOp::create(rewriter, loc, tileM);
    Value stepN = arith::ConstantIndexOp::create(rewriter, loc, tileN);

    auto loopM = scf::ForOp::create(rewriter, loc, c0, dimM, stepM);
    rewriter.setInsertionPointToStart(loopM.getBody());
    Value ivM = loopM.getInductionVar();

    auto loopN = scf::ForOp::create(rewriter, loc, c0, dimN, stepN);
    rewriter.setInsertionPointToStart(loopN.getBody());
    Value ivN = loopN.getInductionVar();

    Value currM = createIndexMin(rewriter, loc, dimM, ivM, tileM);
    Value currN = createIndexMin(rewriter, loc, dimN, ivN, tileN);

    Value broadcastSource = input;
    if (kind == BroadcastKind::Row) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "row broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivN}, {currN});
    } else if (kind == BroadcastKind::Column) {
      if (inputType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "column broadcast expects rank-1 input");
      broadcastSource = createSubView(rewriter, loc, input, {ivM}, {currM});
    } else if (inputType.getRank() != 0) {
      return rewriter.notifyMatchFailure(
          op, "element broadcast expects rank-0 input");
    }

    Value subOut =
        createSubView(rewriter, loc, out, {ivM, ivN}, {currM, currN});
    Value strideInput = createByteStride(rewriter, loc, broadcastSource);
    Value strideOut = createByteStride(rewriter, loc, subOut);

    if (failed(createMSetTypeAndTiles(rewriter, op, loc, elementType, currM,
                                      currN)))
      return failure();
    FailureOr<Value> inputTile = createLoadC(rewriter, op, loc, elementType,
                                             broadcastSource, strideInput);
    if (failed(inputTile))
      return failure();
    FailureOr<Value> result = createBroadcast(
        rewriter, op, loc, kind, BroadcastRegisterKind::Accumulation,
        elementType, *inputTile);
    if (failed(result) || failed(createStoreC(rewriter, op, loc, elementType,
                                              *result, subOut, strideOut)))
      return failure();

    rewriter.setInsertionPointAfter(loopM);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class LowerLinalgToBOSCAMEPass
    : public PassWrapper<LowerLinalgToBOSCAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToBOSCAMEPass)

  StringRef getArgument() const final { return "lower-linalg-to-boscame"; }
  StringRef getDescription() const final {
    return "Lower linalg dialect operations to BOSCAME dialect operations.";
  }

  LowerLinalgToBOSCAMEPass() = default;
  LowerLinalgToBOSCAMEPass(const LowerLinalgToBOSCAMEPass &) {}

  /// AME hardware contract.  Empty means "use the bosc_ame.target module
  /// attribute, or the upstream default".
  Option<std::string> ameTarget{
      *this, "target",
      llvm::cl::desc("AME hardware contract: 'upstream' (default) or "
                     "'nr-fpga' (NH/RA) or 'qwen3-fpga' (legacy). Must agree "
                     "with the bosc_ame.target "
                     "module attribute when both are present."),
      llvm::cl::init("")};

  Option<bool> tritonW8A8FastPath{
      *this, "triton-w8a8-fast-path",
      llvm::cl::desc("Fuse exact Triton i8 dot + sitofp and use the Qwen FPGA "
                     "AME schedule for transposed weights (requires "
                     "target=qwen3-fpga)"),
      llvm::cl::init(false)};

  Option<int64_t> nrTileN{
      *this, "nr-tile-n",
      llvm::cl::desc("NR FPGA INT8 matmul N tile: 16 (default), 32, or 64; "
                     "wider tiles require target=nr-fpga"),
      llvm::cl::init(16)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BOSCAMEDialect, arith::ArithDialect, linalg::LinalgDialect,
                LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerLinalgToBOSCAMEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  FailureOr<AmeTargetProfile> profile =
      resolveAmeTarget(module, ameTarget.getValue());
  if (failed(profile)) {
    signalPassFailure();
    return;
  }

  if (nrTileN != 16 && nrTileN != 32 && nrTileN != 64) {
    module.emitError("nr-tile-n must be 16, 32, or 64");
    signalPassFailure();
    return;
  }
  if (nrTileN != 16 && *profile != AmeTargetProfile::NrFpga) {
    module.emitError("non-default nr-tile-n requires target=nr-fpga");
    signalPassFailure();
    return;
  }

  // Record a non-default decision on the module so that every later stage (the
  // BOSCAME export, the LLVM target feature, a following pass that resolves the
  // target with no option of its own) sees the same contract instead of
  // re-deciding.  The default profile is left implicit so that the upstream IR
  // text stays byte-for-byte identical.
  if (*profile != AmeTargetProfile::Upstream)
    module->setAttr(
        kAmeTargetAttrName,
        StringAttr::get(context, stringifyAmeTargetProfile(*profile)));

  // The FPGA convention is only defined for the W8A8 datapath; anything else in
  // the module has to be diagnosed here rather than silently mis-mapped.
  if (isFpgaTarget(*profile) &&
      failed(verifyFpgaAmeCapabilities(module))) {
    signalPassFailure();
    return;
  }

  // `triton-w8a8-fast-path` selects the Triton fusion and the FPGA schedule; it
  // must never be the thing that silently switches the hardware encoding.
  // Without the FPGA target there is no i8 x i8 -> f32 AME datapath at all.
  if (tritonW8A8FastPath && *profile != AmeTargetProfile::Qwen3Fpga) {
    module.emitError()
        << "triton-w8a8-fast-path requires bosc_ame.target = \"qwen3-fpga\" "
           "(the upstream pathway has no i8 x i8 -> f32 AME matmul)";
    signalPassFailure();
    return;
  }

  // Materialize every eligible Triton i32-dot/sitofp fusion before dialect
  // conversion. Keeping this as a distinct greedy phase guarantees that all
  // newly created f32 matmuls are visible to the AME conversion, including
  // the first of several static quantization groups.
  if (tritonW8A8FastPath) {
    RewritePatternSet fusionPatterns(context);
    fusionPatterns.add<TritonI8DotCastToF32Matmul>(context);
    if (failed(applyPatternsGreedily(module, std::move(fusionPatterns)))) {
      signalPassFailure();
      return;
    }
  }

  RewritePatternSet patterns(context);
  ConversionTarget conversionTarget(*context);
  conversionTarget.addLegalDialect<BOSCAMEDialect, arith::ArithDialect,
                                   linalg::LinalgDialect, LLVM::LLVMDialect,
                                   memref::MemRefDialect, scf::SCFDialect>();

  if (*profile == AmeTargetProfile::NrFpga) {
    patterns.add<NrMatmulToBOSCAMELowering>(context, nrTileN.getValue());
    conversionTarget.addDynamicallyLegalOp<linalg::MatmulOp>(
        [](linalg::MatmulOp op) {
          return !isNrMatmul(
              op, isa<linalg::MatmulTransposeBOp>(op.getOperation()));
        });
  } else if (*profile == AmeTargetProfile::Qwen3Fpga) {
    // FPGA contract: this pass only owns the operations the FPGA emitter can
    // actually lower.  The upstream Generic* lowerings are deliberately NOT
    // registered here, because they program `bosc_ame.msettypei` with a raw
    // element width, which is the wrong hardware encoding for this target.
    // Everything unclaimed stays legal and is left to the CPU/VIR pipeline,
    // which is why the legality contract below is dynamic instead of a blanket
    // `addIllegalOp<linalg::MatmulOp>()`.
    patterns.add<MatmulToBOSCAMELowering>(context, *profile,
                                          tritonW8A8FastPath);
    conversionTarget.addDynamicallyLegalOp<linalg::MatmulOp>(
        [&](linalg::MatmulOp op) {
          return classifyMatmul(op, *profile, tritonW8A8FastPath) ==
                 MatmulLoweringPlan::LeaveForFallback;
        });
  } else {
    // Default contract: byte-for-byte the upstream/main behaviour, including
    // the generic elementwise/broadcast/transpose/unary-square lowerings.
    patterns.add<MatmulToBOSCAMELowering>(context, *profile,
                                          tritonW8A8FastPath);
    patterns.add<
        GenericMatmulToBOSCAMELowering, GenericElementwiseToBOSCAMELowering,
        GenericUnarySquareToBOSCAMELowering, GenericTransposeToBOSCAMELowering,
        GenericBroadcastToBOSCAMELowering>(context);
    conversionTarget.addIllegalOp<linalg::MatmulOp>();
    conversionTarget.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return !isLowerableGeneric(op); });
  }

  if (failed(applyPartialConversion(module, conversionTarget,
                                    std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  if (*profile == AmeTargetProfile::Qwen3Fpga) {
    // The direct-C rewrite removes the temporary allocation, its zero fill and
    // the copy; collect any allocation that became unused as a result.
    SmallVector<memref::AllocOp> deadAllocs;
    module.walk([&](memref::AllocOp allocOp) {
      if (allocOp->use_empty())
        deadAllocs.push_back(allocOp);
    });
    for (memref::AllocOp allocOp : deadAllocs)
      allocOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace buddy {
void registerLowerLinalgToBOSCAMEPass() {
  PassRegistration<LowerLinalgToBOSCAMEPass>();
}
} // namespace buddy
} // namespace mlir
