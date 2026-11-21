//===- LowerLinalgToAME.cpp - Linalg to ZTT v0.6 AME lowering
//----------------===//
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

#include "Conversion/LowerLinalgToAME/LowerLinalgToAME.h"
#include "Dialect/AME/AMEDialect.h"
#include "Dialect/AME/AMEOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/RISCVXAME.h"

using namespace mlir;
using namespace ::buddy::ame;

namespace {

/// Only this tuple has the same wrapping arithmetic as arith.muli/addi without
/// an additional floating-point, widening, packing or saturation contract.
static bool isSupported(linalg::MatmulOp op) {
  auto m = getAffineDimExpr(0, op.getContext());
  auto n = getAffineDimExpr(1, op.getContext());
  auto k = getAffineDimExpr(2, op.getContext());
  SmallVector<AffineMap> maps = {AffineMap::get(3, 0, {m, k}, op.getContext()),
                                 AffineMap::get(3, 0, {k, n}, op.getContext()),
                                 AffineMap::get(3, 0, {m, n}, op.getContext())};
  if (op.getNumResults() || op.getIndexingMapsArray() != maps)
    return false;
  return llvm::all_of(op->getOperandTypes(), [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    if (!memref || memref.getRank() != 2 ||
        !memref.getElementType().isSignlessInteger(32))
      return false;
    Attribute space = memref.getMemorySpace();
    auto integerSpace = dyn_cast_if_present<IntegerAttr>(space);
    return !space || (integerSpace && integerSpace.getValue().isZero());
  });
}

static bool isSupportedMatmul(linalg::GenericOp op) {
  if (op.getNumResults() || op.getInputs().size() != 2 ||
      op.getOutputs().size() != 1 ||
      op.getIteratorTypesArray() !=
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel,
                                           utils::IteratorType::reduction})
    return false;
  auto m = getAffineDimExpr(0, op.getContext());
  auto n = getAffineDimExpr(1, op.getContext());
  auto k = getAffineDimExpr(2, op.getContext());
  SmallVector<AffineMap> maps = {AffineMap::get(3, 0, {m, k}, op.getContext()),
                                 AffineMap::get(3, 0, {k, n}, op.getContext()),
                                 AffineMap::get(3, 0, {m, n}, op.getContext())};
  if (op.getIndexingMapsArray() != maps ||
      !llvm::all_of(op->getOperandTypes(), [](Type type) {
        auto memref = dyn_cast<MemRefType>(type);
        if (!memref || memref.getRank() != 2 ||
            !memref.getElementType().isSignlessInteger(32))
          return false;
        Attribute space = memref.getMemorySpace();
        auto integerSpace = dyn_cast_if_present<IntegerAttr>(space);
        return !space || (integerSpace && integerSpace.getValue().isZero());
      }))
    return false;
  Block &body = op.getRegion().front();
  if (body.getOperations().size() != 3)
    return false;
  auto mul = dyn_cast<arith::MulIOp>(body.front());
  auto add = dyn_cast<arith::AddIOp>(*std::next(body.begin()));
  auto operandsMatch = [](Operation *operation, Value lhs, Value rhs) {
    return (operation->getOperand(0) == lhs &&
            operation->getOperand(1) == rhs) ||
           (operation->getOperand(0) == rhs && operation->getOperand(1) == lhs);
  };
  return mul && add &&
         operandsMatch(mul, body.getArgument(0), body.getArgument(1)) &&
         operandsMatch(add, mul.getResult(), body.getArgument(2)) &&
         body.getTerminator()->getOperand(0) == add.getResult();
}

/// Stage through contiguous tiles instead of deriving raw addresses from input
/// memrefs. Scalar load/store honor their offsets and arbitrary strided
/// layouts, and padding prevents all matrix memory operations from touching a
/// tail.
class MatmulEmitter {
public:
  MatmulEmitter(IRRewriter &rewriter, linalg::MatmulOp op, unsigned side)
      : b(rewriter), loc(op.getLoc()), a(op.getInputs()[0]),
        rhs(op.getInputs()[1]), c(op.getOutputs()[0]), side(side) {}

  MatmulEmitter(IRRewriter &rewriter, linalg::GenericOp op, unsigned side)
      : b(rewriter), loc(op.getLoc()), a(op.getInputs()[0]),
        rhs(op.getInputs()[1]), c(op.getOutputs()[0]), side(side) {}

  void emit() {
    c0 = arith::ConstantIndexOp::create(b, loc, 0);
    c1 = arith::ConstantIndexOp::create(b, loc, 1);
    tileSide = arith::ConstantIndexOp::create(b, loc, side);
    zero = arith::ConstantIntOp::create(b, loc, 0, 32);
    m = memref::DimOp::create(b, loc, a, 0);
    n = memref::DimOp::create(b, loc, rhs, 1);
    k = memref::DimOp::create(b, loc, a, 1);
    Value zero64 = arith::ConstantIntOp::create(b, loc, 0, 64);
    Value one64 = arith::ConstantIntOp::create(b, loc, 1, 64);
    Value owner = AmeAcquireOp::create(b, loc, b.getI64Type(), zero64);
    Value bit = arith::AndIOp::create(b, loc, owner, one64);
    Value owned =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, bit, zero64);
    auto dispatch = scf::IfOp::create(b, loc, owned, true);
    b.setInsertionPointToStart(dispatch.thenBlock());
    emitAME();
    b.setInsertionPointToStart(dispatch.elseBlock());
    emitScalar();
    b.setInsertionPointAfter(dispatch);
  }

private:
  IRRewriter &b;
  Location loc;
  Value a, rhs, c;
  unsigned side;
  Value c0, c1, tileSide, zero, m, n, k;

  void loop(Value bound, Value step, function_ref<void(Value)> body) {
    auto op = scf::ForOp::create(b, loc, c0, bound, step);
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
    b.setInsertionPointAfter(op);
  }

  Value add(Value lhs, Value rhs) {
    return arith::AddIOp::create(b, loc, lhs, rhs);
  }

  Value inBounds(Value row, Value col, Value rows, Value cols) {
    Value r =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, row, rows);
    Value s =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, col, cols);
    return arith::AndIOp::create(b, loc, r, s);
  }

  void pack(Value source, Value tile, Value rowBase, Value colBase, Value rows,
            Value cols) {
    loop(tileSide, c1, [&](Value i) {
      loop(tileSide, c1, [&](Value j) {
        Value row = add(rowBase, i), col = add(colBase, j);
        auto select = scf::IfOp::create(b, loc, TypeRange{b.getI32Type()},
                                        inBounds(row, col, rows, cols), true);
        b.setInsertionPointToStart(select.thenBlock());
        Value value =
            memref::LoadOp::create(b, loc, source, ValueRange{row, col});
        scf::YieldOp::create(b, loc, value);
        b.setInsertionPointToStart(select.elseBlock());
        scf::YieldOp::create(b, loc, zero);
        b.setInsertionPointAfter(select);
        memref::StoreOp::create(b, loc, select.getResult(0), tile,
                                ValueRange{i, j});
      });
    });
  }

  void unpack(Value tile, Value rowBase, Value colBase) {
    loop(tileSide, c1, [&](Value i) {
      loop(tileSide, c1, [&](Value j) {
        Value row = add(rowBase, i), col = add(colBase, j);
        auto store = scf::IfOp::create(b, loc, inBounds(row, col, m, n), false);
        b.setInsertionPointToStart(store.thenBlock());
        Value value = memref::LoadOp::create(b, loc, tile, ValueRange{i, j});
        memref::StoreOp::create(b, loc, value, c, ValueRange{row, col});
        b.setInsertionPointAfter(store);
      });
    });
  }

  Value pointer(Value tile) {
    // These are our own contiguous, zero-offset stack tiles, not input memrefs.
    Value address =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, tile);
    Value integer = arith::IndexCastOp::create(b, loc, b.getI64Type(), address);
    return LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext()), integer);
  }

  Value configure(Type type, Value descriptor, bool accumulator = false) {
    Value seed = LLVM::UndefOp::create(b, loc, type);
    if (accumulator)
      return AsettypOp::create(b, loc, type, seed, descriptor);
    return MsettypOp::create(b, loc, type, seed, descriptor);
  }

  void emitAME() {
    // Bound stack usage even when the original matmul is nested in a loop.
    Value savedStack = LLVM::StackSaveOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext()));
    auto tileType = MemRefType::get({side, side}, b.getI32Type());
    Value aTile = memref::AllocaOp::create(b, loc, tileType);
    Value bTile = memref::AllocaOp::create(b, loc, tileType);
    Value cTile = memref::AllocaOp::create(b, loc, tileType);
    Value aPtr = pointer(aTile), bPtr = pointer(bTile), cPtr = pointer(cTile);
    auto matrix = LLVM::LLVMTargetExtType::get(
        b.getContext(), "riscv.ztt.matrix", {b.getI32Type()}, {side, side});
    auto accumulator = LLVM::LLVMTargetExtType::get(
        b.getContext(), "riscv.ztt.acc", {b.getI32Type()}, {side, side});
    // ZTT v0.6 signed, non-saturating i32 descriptor. Modular arithmetic is
    // bit-identical to i32 arith.muli/addi, including negative values/overflow.
    Value descriptor = arith::ConstantIntOp::create(b, loc, 0x40000020, 64);
    loop(m, tileSide, [&](Value row) {
      loop(n, tileSide, [&](Value col) {
        pack(c, cTile, row, col, m, n);
        Value cConfig = configure(matrix, descriptor);
        Value initialC = MlsRmOp::create(b, loc, matrix, cConfig, cPtr);
        Value accConfig = configure(accumulator, descriptor, true);
        Value initial =
            MmovAMOp::create(b, loc, accumulator, accConfig, initialC);
        auto reduction =
            scf::ForOp::create(b, loc, c0, k, tileSide, ValueRange{initial});
        b.setInsertionPointToStart(reduction.getBody());
        Value kk = reduction.getInductionVar();
        pack(a, aTile, row, kk, m, k);
        pack(rhs, bTile, kk, col, k, n);
        Value aConfig = configure(matrix, descriptor);
        Value bConfig = configure(matrix, descriptor);
        Value lhs = MlsRmOp::create(b, loc, matrix, aConfig, aPtr);
        Value right = MlsRmOp::create(b, loc, matrix, bConfig, bPtr);
        Value result = Mmulacc2dOp::create(
            b, loc, accumulator, reduction.getRegionIterArgs()[0], lhs, right);
        scf::YieldOp::create(b, loc, result);
        b.setInsertionPointAfter(reduction);
        Value outputConfig = configure(matrix, descriptor);
        Value output = MmovMAOp::create(b, loc, matrix, outputConfig,
                                        reduction.getResult(0));
        MssRmOp::create(b, loc, output, cPtr);
        unpack(cTile, row, col);
      });
    });
    LLVM::StackRestoreOp::create(b, loc, savedStack);
    AmeReleaseOp::create(b, loc);
  }

  void emitScalar() {
    loop(m, c1, [&](Value i) {
      loop(n, c1, [&](Value j) {
        Value initial = memref::LoadOp::create(b, loc, c, ValueRange{i, j});
        auto reduction =
            scf::ForOp::create(b, loc, c0, k, c1, ValueRange{initial});
        b.setInsertionPointToStart(reduction.getBody());
        Value kk = reduction.getInductionVar();
        Value lhs = memref::LoadOp::create(b, loc, a, ValueRange{i, kk});
        Value right = memref::LoadOp::create(b, loc, rhs, ValueRange{kk, j});
        Value product = arith::MulIOp::create(b, loc, lhs, right);
        Value sum = add(reduction.getRegionIterArgs()[0], product);
        scf::YieldOp::create(b, loc, sum);
        b.setInsertionPointAfter(reduction);
        memref::StoreOp::create(b, loc, reduction.getResult(0), c,
                                ValueRange{i, j});
      });
    });
  }
};

/// A generic is a computation region, not an opcode. Match the iteration
/// space, every indexing map, and every scalar operation before changing it.
/// Projected permutations cover copies, transposes and broadcasts as well as
/// arithmetic regions, without assuming contiguous input/output storage.
static bool isPointwise(linalg::GenericOp op) {
  if (op.getNumResults() || op.getOutputs().size() != 1 ||
      op.getNumLoops() < 1 || op.getNumLoops() > 2 ||
      !op.hasPureBufferSemantics() ||
      !llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      }))
    return false;
  auto maps = op.getIndexingMapsArray();
  if (!maps.back().isPermutation())
    return false;
  for (auto [type, map] : llvm::zip(op->getOperandTypes(), maps)) {
    // Linalg also permits scalar ins operands, whose indexing maps are empty.
    if (type.isSignlessInteger(32) && map.getNumResults() == 0)
      continue;
    auto memref = dyn_cast<MemRefType>(type);
    if (!memref || !memref.getElementType().isSignlessInteger(32) ||
        !map.isProjectedPermutation())
      return false;
    Attribute space = memref.getMemorySpace();
    auto integerSpace = dyn_cast_if_present<IntegerAttr>(space);
    if (space && (!integerSpace || !integerSpace.getValue().isZero()))
      return false;
  }
  Block &body = op.getRegion().front();
  for (Operation &scalar : body.without_terminator()) {
    if (isa<arith::ConstantIntOp, arith::ConstantOp>(scalar)) {
      if (!scalar.getResult(0).getType().isSignlessInteger(32))
        return false;
    } else if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
                    arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
                    arith::ShRUIOp, arith::MinSIOp, arith::MaxSIOp,
                    arith::MinUIOp, arith::MaxUIOp>(scalar)) {
      return false;
    }
    if (scalar.getNumResults() != 1 ||
        !scalar.getResult(0).getType().isSignlessInteger(32) ||
        !llvm::all_of(scalar.getOperandTypes(),
                      [](Type type) { return type.isSignlessInteger(32); }))
      return false;
  }
  return llvm::all_of(body.getTerminator()->getOperandTypes(),
                      [](Type type) { return type.isSignlessInteger(32); });
}

class PointwiseEmitter {
public:
  PointwiseEmitter(IRRewriter &rewriter, linalg::GenericOp op, unsigned side)
      : b(rewriter), op(op), loc(op.getLoc()), side(side),
        output(op.getOutputs().front()) {
    // Express all inputs in output-coordinate order. This also handles a
    // permuted output map, independently of the order of the parallel loops.
    AffineMap inverse = inversePermutation(op.getIndexingMapsArray().back());
    for (AffineMap map : op.getIndexingMapsArray())
      maps.push_back(map.compose(inverse));
  }

  void emit() {
    c0 = arith::ConstantIndexOp::create(b, loc, 0);
    c1 = arith::ConstantIndexOp::create(b, loc, 1);
    tileSide = arith::ConstantIndexOp::create(b, loc, side);
    zero = arith::ConstantIntOp::create(b, loc, 0, 32);
    rows = memref::DimOp::create(b, loc, output, 0);
    cols = op.getNumLoops() == 2
               ? Value(memref::DimOp::create(b, loc, output, 1))
               : c1;
    matrix = LLVM::LLVMTargetExtType::get(b.getContext(), "riscv.ztt.matrix",
                                          {b.getI32Type()}, {side, side});
    signedType = arith::ConstantIntOp::create(b, loc, 0x40000020, 64);
    unsignedType = arith::ConstantIntOp::create(b, loc, 0x20, 64);
    Value zero64 = arith::ConstantIntOp::create(b, loc, 0, 64);
    Value one64 = arith::ConstantIntOp::create(b, loc, 1, 64);
    Value owner = AmeAcquireOp::create(b, loc, b.getI64Type(), zero64);
    Value bit = arith::AndIOp::create(b, loc, owner, one64);
    Value owned =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, bit, zero64);
    auto dispatch = scf::IfOp::create(b, loc, owned, true);
    b.setInsertionPointToStart(dispatch.thenBlock());
    emitAME();
    b.setInsertionPointToStart(dispatch.elseBlock());
    emitScalar();
    b.setInsertionPointAfter(dispatch);
  }

private:
  IRRewriter &b;
  linalg::GenericOp op;
  Location loc;
  unsigned side;
  Value output, c0, c1, tileSide, zero, rows, cols;
  Value signedType, unsignedType;
  Type matrix;
  SmallVector<AffineMap> maps;

  void loop(Value bound, Value step, function_ref<void(Value)> body) {
    auto loop = scf::ForOp::create(b, loc, c0, bound, step);
    b.setInsertionPointToStart(loop.getBody());
    body(loop.getInductionVar());
    b.setInsertionPointAfter(loop);
  }

  Value add(Value lhs, Value rhs) {
    return arith::AddIOp::create(b, loc, lhs, rhs);
  }

  Value inBounds(Value row, Value col) {
    Value r =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, row, rows);
    Value c =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, col, cols);
    return arith::AndIOp::create(b, loc, r, c);
  }

  SmallVector<Value> indices(AffineMap map, Value row, Value col) {
    SmallVector<Value> result;
    for (AffineExpr expr : map.getResults())
      result.push_back(cast<AffineDimExpr>(expr).getPosition() == 0 ? row
                                                                    : col);
    return result;
  }

  Value configure(Value value, bool isUnsigned = false) {
    return MsettypOp::create(b, loc, matrix, value,
                             isUnsigned ? unsignedType : signedType);
  }

  Value seed(bool isUnsigned = false) {
    return configure(LLVM::UndefOp::create(b, loc, matrix), isUnsigned);
  }

  Value broadcast(Value scalar) {
    Value wide = arith::ExtSIOp::create(b, loc, b.getI64Type(), scalar);
    // The third operand supplies the source type explicitly. No amestype CSR
    // is needed, unlike the arithmetic .ew.x instruction forms.
    return MbcastMXOp::create(b, loc, matrix, seed(), wide, signedType);
  }

  /// Pack one input tile. For a transpose the staging tile is in source order
  /// and mls.cm performs the transpose. A projected single axis is staged only
  /// in row/column zero and then broadcast by the corresponding instruction.
  Value load(Value source, AffineMap map, Value rowBase, Value colBase,
             Value tile, Value ptr) {
    if (source.getType().isSignlessInteger(32))
      return broadcast(source);
    if (map.getNumResults() == 0)
      return broadcast(memref::LoadOp::create(b, loc, source, ValueRange{}));
    bool transpose = map.getNumResults() == 2 &&
                     cast<AffineDimExpr>(map.getResult(0)).getPosition() == 1;
    bool singleAxis = map.getNumResults() == 1;
    bool rowAxis = cast<AffineDimExpr>(map.getResult(0)).getPosition() == 0;
    loop(tileSide, c1, [&](Value i) {
      loop(tileSide, c1, [&](Value j) {
        Value row = add(rowBase, i), col = add(colBase, j);
        Value valid = inBounds(row, col);
        if (singleAxis) {
          // For a one-dimensional output, cols is one and the useful values
          // occupy column zero, exactly as for a column broadcast in 2-D.
          Value first = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                              rowAxis ? j : i, c0);
          valid = arith::AndIOp::create(b, loc, valid, first);
        }
        auto select =
            scf::IfOp::create(b, loc, TypeRange{b.getI32Type()}, valid, true);
        b.setInsertionPointToStart(select.thenBlock());
        Value value =
            memref::LoadOp::create(b, loc, source, indices(map, row, col));
        scf::YieldOp::create(b, loc, value);
        b.setInsertionPointToStart(select.elseBlock());
        scf::YieldOp::create(b, loc, zero);
        b.setInsertionPointAfter(select);
        memref::StoreOp::create(b, loc, select.getResult(0), tile,
                                transpose ? ValueRange{j, i}
                                          : ValueRange{i, j});
      });
    });
    Value result = transpose
                       ? Value(MlsCmOp::create(b, loc, matrix, seed(), ptr))
                       : Value(MlsRmOp::create(b, loc, matrix, seed(), ptr));
    if (singleAxis) {
      Value index = arith::ConstantIntOp::create(b, loc, 0, 64);
      result = rowAxis ? Value(McolbcastEwXOp::create(b, loc, matrix, seed(),
                                                      index, result))
                       : Value(MrowbcastEwXOp::create(b, loc, matrix, seed(),
                                                      index, result));
    }
    return result;
  }

  Value arithmetic(Operation &scalar, Value lhs, Value rhs) {
    bool isUnsigned = isa<arith::MinUIOp, arith::MaxUIOp>(scalar);
    if (isUnsigned) {
      // msettyp clears data. Convert into fresh non-saturating destinations:
      // same-width modular conversion preserves every i32 bit, even when
      // signedness changes. Restore signedness for subsequent region ops.
      lhs = MconvEwOp::create(b, loc, matrix, seed(true), lhs);
      rhs = MconvEwOp::create(b, loc, matrix, seed(true), rhs);
    }
    Value dest = seed(isUnsigned), result;
#define AME_BINARY(SCALAR, MATRIX)                                             \
  if (isa<arith::SCALAR>(scalar))                                              \
  result = MATRIX::create(b, loc, matrix, dest, lhs, rhs)
    AME_BINARY(AddIOp, MaddEwOp);
    // ZTT subtraction is ms2 - ms1, whereas arith.subi is lhs - rhs.
    if (isa<arith::SubIOp>(scalar))
      result = MsubEwOp::create(b, loc, matrix, dest, rhs, lhs);
    AME_BINARY(MulIOp, MmulEwOp);
    AME_BINARY(AndIOp, MandEwOp);
    AME_BINARY(OrIOp, MorEwOp);
    AME_BINARY(XOrIOp, MxorEwOp);
    AME_BINARY(ShLIOp, MsllEwOp);
    AME_BINARY(ShRSIOp, MsraEwOp);
    AME_BINARY(ShRUIOp, MsrlEwOp);
    AME_BINARY(MinSIOp, MminEwOp);
    AME_BINARY(MaxSIOp, MmaxEwOp);
    AME_BINARY(MinUIOp, MminEwOp);
    AME_BINARY(MaxUIOp, MmaxEwOp);
#undef AME_BINARY
    return isUnsigned ? Value(MconvEwOp::create(b, loc, matrix, seed(), result))
                      : result;
  }

  void emitAME() {
    Value saved = LLVM::StackSaveOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext()));
    Value tile = memref::AllocaOp::create(
        b, loc, MemRefType::get({side, side}, b.getI32Type()));
    Value address =
        memref::ExtractAlignedPointerAsIndexOp::create(b, loc, tile);
    Value integer = arith::IndexCastOp::create(b, loc, b.getI64Type(), address);
    Value ptr = LLVM::IntToPtrOp::create(
        b, loc, LLVM::LLVMPointerType::get(b.getContext()), integer);
    loop(rows, tileSide, [&](Value rowBase) {
      loop(cols, tileSide, [&](Value colBase) {
        IRMapping values;
        Block &body = op.getRegion().front();
        for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
          // outs often refers to uninitialized storage. Only read it if the
          // computation actually uses the corresponding region argument.
          if (!arg.use_empty())
            values.map(arg, load(op->getOperand(index), maps[index], rowBase,
                                 colBase, tile, ptr));
        }
        auto get = [&](Value value) {
          if (Value mapped = values.lookupOrNull(value))
            return mapped;
          Value mapped = broadcast(value); // An external scalar capture.
          values.map(value, mapped);
          return mapped;
        };
        for (Operation &scalar : body.without_terminator()) {
          Value result;
          if (isa<arith::ConstantOp>(scalar)) {
            Operation *constant = b.clone(scalar);
            result = broadcast(constant->getResult(0));
          } else {
            result = arithmetic(scalar, get(scalar.getOperand(0)),
                                get(scalar.getOperand(1)));
          }
          values.map(scalar.getResult(0), result);
        }
        Value result = get(body.getTerminator()->getOperand(0));
        MssRmOp::create(b, loc, result, ptr);
        loop(tileSide, c1, [&](Value i) {
          loop(tileSide, c1, [&](Value j) {
            Value row = add(rowBase, i), col = add(colBase, j);
            auto store = scf::IfOp::create(b, loc, inBounds(row, col), false);
            b.setInsertionPointToStart(store.thenBlock());
            Value value =
                memref::LoadOp::create(b, loc, tile, ValueRange{i, j});
            memref::StoreOp::create(b, loc, value, output,
                                    indices(maps.back(), row, col));
            b.setInsertionPointAfter(store);
          });
        });
      });
    });
    LLVM::StackRestoreOp::create(b, loc, saved);
    AmeReleaseOp::create(b, loc);
  }

  void emitScalar() {
    loop(rows, c1, [&](Value row) {
      loop(cols, c1, [&](Value col) {
        IRMapping values;
        Block &body = op.getRegion().front();
        for (auto [index, arg] : llvm::enumerate(body.getArguments()))
          if (!arg.use_empty()) {
            Value source = op->getOperand(index);
            values.map(arg, source.getType().isSignlessInteger(32)
                                ? source
                                : Value(memref::LoadOp::create(
                                      b, loc, source,
                                      indices(maps[index], row, col))));
          }
        for (Operation &scalar : body.without_terminator())
          b.clone(scalar, values);
        Value result =
            values.lookupOrDefault(body.getTerminator()->getOperand(0));
        memref::StoreOp::create(b, loc, result, output,
                                indices(maps.back(), row, col));
      });
    });
  }
};

class LowerLinalgToAMEPass
    : public PassWrapper<LowerLinalgToAMEPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToAMEPass)
  LowerLinalgToAMEPass() = default;
  LowerLinalgToAMEPass(const LowerLinalgToAMEPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "lower-linalg-to-ame"; }
  StringRef getDescription() const final {
    return "Lower bufferized i32 Linalg matmul and generic regions to ZTT v0.6 "
           "AME";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMEDialect, arith::ArithDialect, func::FuncDialect,
                    LLVM::LLVMDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (target != "generic" && target != "gem5") {
      getOperation().emitError("unsupported AME target: ")
          << target << "; expected generic or gem5";
      return signalPassFailure();
    }
    unsigned side = llvm::RISCV::getXAMEProfile(target == "gem5").TileSide;
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) {
      if (isSupported(op))
        matmuls.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (linalg::MatmulOp op : matmuls) {
      rewriter.setInsertionPoint(op);
      MatmulEmitter(rewriter, op, side).emit();
      rewriter.eraseOp(op);
    }
    SmallVector<linalg::GenericOp> generics;
    getOperation().walk([&](linalg::GenericOp op) {
      if (isSupportedMatmul(op) || isPointwise(op))
        generics.push_back(op);
    });
    for (linalg::GenericOp op : generics) {
      rewriter.setInsertionPoint(op);
      if (isSupportedMatmul(op))
        MatmulEmitter(rewriter, op, side).emit();
      else
        PointwiseEmitter(rewriter, op, side).emit();
      rewriter.eraseOp(op);
    }
  }

private:
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("XAME profile: generic (8x8) or gem5 (4x4)"),
      llvm::cl::init("generic")};
};

} // namespace

std::unique_ptr<Pass> mlir::buddy::createLowerLinalgToAMEPass() {
  return std::make_unique<LowerLinalgToAMEPass>();
}

void mlir::buddy::registerLowerLinalgToAMEPass() {
  PassRegistration<LowerLinalgToAMEPass>();
}
