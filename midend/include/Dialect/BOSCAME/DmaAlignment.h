//===- DmaAlignment.h - Provable 64 B DMA endpoint alignment ---===//
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
// Where the effective address of a DMA endpoint comes from, and what the
// compiler has to be able to *prove* about it.
//
// `bosc_ame.dma.start` requires 64 B aligned effective addresses:
// alignedPointer + descriptorOffset * elementBytes. The root is a fixed SPM
// bank, an explicitly aligned allocation, or a DDR argument covered by the
// module ABI. Descriptor offsets are in elements; dynamic offsets require an
// SSA divisibility proof. A subview adds offsets times the source strides,
// whereas reinterpret_cast replaces the descriptor offset.
//
// The third one is a *`memref.view` byte shift*: `memref.view %root[%shift]` is
// the only operation in the view chain that moves the base pointer while
// leaving the descriptor offset at zero, which is exactly what a regular
// dynamic address (a row of a tile, a K sub-block) needs.  A `view` shift is
// therefore not a property of the type; the endpoint can only be accepted when
// every shift on the chain is either static and aligned or a runtime value the
// compiler can prove to be a multiple of the 64 B granularity.
//
// `isProvablyMultipleOf` is that proof, kept deliberately small: it walks the
// integer expression that produces the shift and only accepts constructions
// whose divisibility follows from the operation itself (constants, addition,
// multiplication by a divisible factor, widening casts, `select`/`min`/`max` of
// divisible values and an `scf.for` induction variable whose lower bound and
// step are divisible).  Anything else - a raw function argument, a load, a
// division - is *not* provable. The experimental BOSCAME path then fails;
// it never silently falls back to CPU DDR accesses.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H
#define BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H

#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>
#include <optional>

namespace buddy {
namespace boscame {
namespace spmdma {

inline bool isDescriptorOffsetAligned(mlir::Value value, int64_t factor);
inline bool isProvablyMultipleOf(mlir::Value value, int64_t factor);

struct StaticPointerOrigin {
  mlir::Value root;
  int64_t byteShift = 0;
};

// Separate the aligned-pointer origin from the descriptor offset. Only
// memref.view shifts the pointer; reinterpret_cast replaces its descriptor
// offset and subview computes that offset using source strides. Those offsets
// are queried separately on the final value, never accumulated here.
inline std::optional<StaticPointerOrigin>
getStaticPointerOrigin(mlir::Value value) {
  if (!value)
    return std::nullopt;
  StaticPointerOrigin origin{value, 0};
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *defining = origin.root.getDefiningOp()) {
    if (!visited.insert(defining).second)
      return std::nullopt;
    if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(defining)) {
      auto shift = mlir::getConstantIntValue(view.getByteShift());
      if (!shift ||
          llvm::AddOverflow(origin.byteShift, *shift, origin.byteShift))
        return std::nullopt;
      origin.root = view.getSource();
    } else if (auto sub = llvm::dyn_cast<mlir::memref::SubViewOp>(defining)) {
      origin.root = sub.getSource();
    } else if (auto transpose =
                   llvm::dyn_cast<mlir::memref::TransposeOp>(defining)) {
      // A transpose only changes the descriptor's shape/strides; it does not
      // shift the aligned base pointer.
      origin.root = transpose.getIn();
    } else if (auto cast =
                   llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining)) {
      origin.root = cast.getSource();
    } else if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(defining)) {
      origin.root = cast.getSource();
    } else if (auto cast =
                   llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(defining)) {
      origin.root = cast.getSource();
    } else if (auto collapse =
                   llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining)) {
      origin.root = collapse.getSrc();
    } else if (auto expand =
                   llvm::dyn_cast<mlir::memref::ExpandShapeOp>(defining)) {
      origin.root = expand.getSrc();
    } else if (auto reshape =
                   llvm::dyn_cast<mlir::memref::ReshapeOp>(defining)) {
      // memref.reshape requires identity layouts and preserves the base.
      origin.root = reshape.getSource();
    } else if (auto metadata =
                   llvm::dyn_cast<mlir::memref::ExtractStridedMetadataOp>(
                       defining)) {
      if (origin.root != metadata.getBaseBuffer())
        return std::nullopt;
      origin.root = metadata.getSource();
    } else {
      break;
    }
  }
  return origin;
}

inline bool isAffineMultipleOf(mlir::AffineExpr expr,
                               mlir::affine::AffineApplyOp apply,
                               int64_t factor) {
  if (factor == 1)
    return true;
  if (auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
    return constant.getValue() % factor == 0;
  if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expr))
    return isProvablyMultipleOf(apply.getDimOperands()[dim.getPosition()],
                                factor);
  if (auto symbol = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr))
    return isProvablyMultipleOf(apply.getSymbolOperands()[symbol.getPosition()],
                                factor);
  auto binary = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binary)
    return false;
  if (expr.getKind() == mlir::AffineExprKind::Add)
    return isAffineMultipleOf(binary.getLHS(), apply, factor) &&
           isAffineMultipleOf(binary.getRHS(), apply, factor);
  if (expr.getKind() == mlir::AffineExprKind::Mul) {
    if (auto constant =
            llvm::dyn_cast<mlir::AffineConstantExpr>(binary.getRHS()))
      return isAffineMultipleOf(binary.getLHS(), apply,
                                factor / std::gcd(factor, constant.getValue()));
    if (auto constant =
            llvm::dyn_cast<mlir::AffineConstantExpr>(binary.getLHS()))
      return isAffineMultipleOf(binary.getRHS(), apply,
                                factor / std::gcd(factor, constant.getValue()));
    return isAffineMultipleOf(binary.getLHS(), apply, factor) ||
           isAffineMultipleOf(binary.getRHS(), apply, factor);
  }
  return false;
}

// True when `value` is provably an exact multiple of `factor`.
//
// Never optimistic: a false result rejects DMA, while a wrong true result
// would emit a misaligned hardware transfer.
inline bool isProvablyMultipleOf(mlir::Value value, int64_t factor) {
  if (factor <= 0 || !value)
    return false;
  if (factor == 1)
    return true;

  // An `scf.for` induction variable is `lowerBound + i * step`, so it is a
  // multiple of `factor` when both are.
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!forOp || forOp.getInductionVar() != value)
      return false;
    return isProvablyMultipleOf(forOp.getLowerBound(), factor) &&
           isProvablyMultipleOf(forOp.getStep(), factor);
  }

  mlir::Operation *defining = value.getDefiningOp();
  if (!defining)
    return false;

  if (auto apply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(defining))
    return isAffineMultipleOf(apply.getAffineMap().getResult(0), apply, factor);
  if (auto metadata =
          llvm::dyn_cast<mlir::memref::ExtractStridedMetadataOp>(defining)) {
    if (value == metadata.getOffset())
      return isDescriptorOffsetAligned(metadata.getSource(), factor);
    auto type = llvm::cast<mlir::MemRefType>(metadata.getSource().getType());
    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    if (mlir::succeeded(type.getStridesAndOffset(strides, offset)))
      for (auto [result, stride] : llvm::zip(metadata.getStrides(), strides))
        if (value == result)
          return stride != mlir::ShapedType::kDynamic && stride % factor == 0;
    return false;
  }

  if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(defining)) {
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt() % factor == 0;
    return false;
  }
  if (auto bitAnd = llvm::dyn_cast<mlir::arith::AndIOp>(defining)) {
    // Clearing the low bits proves divisibility when the requested factor is
    // a power of two.  For other factors, this bit-mask shape is insufficient.
    if ((factor & (factor - 1)) != 0)
      return false;
    auto clearsLowBits = [&](mlir::Value mask) {
      auto constant = getConstantIntValue(mask);
      return constant && *constant % factor == 0;
    };
    return clearsLowBits(bitAnd.getLhs()) || clearsLowBits(bitAnd.getRhs());
  }
  if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(defining))
    return isProvablyMultipleOf(add.getLhs(), factor) &&
           isProvablyMultipleOf(add.getRhs(), factor);
  if (auto sub = llvm::dyn_cast<mlir::arith::SubIOp>(defining))
    return isProvablyMultipleOf(sub.getLhs(), factor) &&
           isProvablyMultipleOf(sub.getRhs(), factor);
  if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(defining)) {
    if (isProvablyMultipleOf(mul.getLhs(), factor) ||
        isProvablyMultipleOf(mul.getRhs(), factor))
      return true;
    // `a * c` is a multiple of `factor` when the constant factor divides it
    // and the other operand carries the remaining divisibility: a row offset
    // that is a 16-element multiple of i32 is a 64-byte multiple.
    if (std::optional<int64_t> constant = getConstantIntValue(mul.getRhs()))
      if (*constant != 0) {
        if (*constant % factor == 0)
          return true;
        if (factor % *constant == 0)
          return isProvablyMultipleOf(mul.getLhs(), factor / *constant);
      }
    if (std::optional<int64_t> constant = getConstantIntValue(mul.getLhs()))
      if (*constant != 0) {
        if (*constant % factor == 0)
          return true;
        if (factor % *constant == 0)
          return isProvablyMultipleOf(mul.getRhs(), factor / *constant);
      }
    return false;
  }
  if (auto indexCast = llvm::dyn_cast<mlir::arith::IndexCastOp>(defining))
    return isProvablyMultipleOf(indexCast.getIn(), factor);
  if (auto ext = llvm::dyn_cast<mlir::arith::ExtSIOp>(defining))
    return isProvablyMultipleOf(ext.getIn(), factor);
  if (auto ext = llvm::dyn_cast<mlir::arith::ExtUIOp>(defining))
    return isProvablyMultipleOf(ext.getIn(), factor);
  if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(defining))
    return isProvablyMultipleOf(select.getTrueValue(), factor) &&
           isProvablyMultipleOf(select.getFalseValue(), factor);
  if (auto min = llvm::dyn_cast<mlir::arith::MinSIOp>(defining))
    return isProvablyMultipleOf(min.getLhs(), factor) &&
           isProvablyMultipleOf(min.getRhs(), factor);
  if (auto min = llvm::dyn_cast<mlir::arith::MinUIOp>(defining))
    return isProvablyMultipleOf(min.getLhs(), factor) &&
           isProvablyMultipleOf(min.getRhs(), factor);
  if (auto max = llvm::dyn_cast<mlir::arith::MaxSIOp>(defining))
    return isProvablyMultipleOf(max.getLhs(), factor) &&
           isProvablyMultipleOf(max.getRhs(), factor);
  if (auto max = llvm::dyn_cast<mlir::arith::MaxUIOp>(defining))
    return isProvablyMultipleOf(max.getLhs(), factor) &&
           isProvablyMultipleOf(max.getRhs(), factor);
  // A constant-folded `affine.apply` / `arith.muli` chain is covered above; a
  // division, a load, a call result and everything else is not provable.
  return false;
}

inline bool isProvablyMultipleOf(mlir::OpFoldResult value, int64_t factor) {
  if (factor <= 0)
    return false;
  if (auto constant = mlir::getConstantIntValue(value))
    return *constant % factor == 0;
  return isProvablyMultipleOf(llvm::dyn_cast<mlir::Value>(value), factor);
}

// A known residue is stronger than divisibility: it can prove that a short
// prefix makes both endpoints aligned. Unknown residues never authorize DMA.
inline std::optional<int64_t>
getKnownRemainder(mlir::Value value, int64_t modulus, unsigned depth = 0) {
  if (!value || modulus <= 0 || depth > 64)
    return std::nullopt;
  auto normalize = [=](int64_t n) {
    int64_t r = n % modulus;
    return r < 0 ? r + modulus : r;
  };
  if (auto constant = mlir::getConstantIntValue(value))
    return normalize(*constant);
  if (modulus == 1 || isProvablyMultipleOf(value, modulus))
    return 0;
  auto integerType = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if ((modulus & (modulus - 1)) ||
      (integerType && integerType.getWidth() < 6 &&
       (uint64_t(1) << integerType.getWidth()) < uint64_t(modulus)))
    return std::nullopt;
  auto next = [&](mlir::Value v) {
    return getKnownRemainder(v, modulus, depth + 1);
  };
  auto castRemainder = [&](mlir::Value input) -> std::optional<int64_t> {
    auto type = llvm::dyn_cast<mlir::IntegerType>(input.getType());
    // Sign/zero extension preserves only the low bits already present in the
    // source. In particular extui(-1 : i2) is 3, not 63 modulo 64.
    if ((modulus & (modulus - 1)) ||
        (type && type.getWidth() < 6 &&
         (uint64_t(1) << type.getWidth()) < uint64_t(modulus)))
      return std::nullopt;
    return next(input);
  };
  mlir::Operation *op = value.getDefiningOp();
  if (auto add = llvm::dyn_cast_or_null<mlir::arith::AddIOp>(op)) {
    auto x = next(add.getLhs()), y = next(add.getRhs());
    int64_t result;
    if (x && y && !llvm::AddOverflow(*x, *y, result))
      return normalize(result);
  } else if (auto sub = llvm::dyn_cast_or_null<mlir::arith::SubIOp>(op)) {
    auto x = next(sub.getLhs()), y = next(sub.getRhs());
    if (x && y)
      return normalize(*x - *y);
  } else if (auto mul = llvm::dyn_cast_or_null<mlir::arith::MulIOp>(op)) {
    auto x = next(mul.getLhs()), y = next(mul.getRhs());
    int64_t result;
    if ((x && *x == 0) || (y && *y == 0))
      return 0;
    if (x && y && !llvm::MulOverflow(*x, *y, result))
      return normalize(result);
  } else if (auto cast = llvm::dyn_cast_or_null<mlir::arith::IndexCastOp>(op)) {
    return castRemainder(cast.getIn());
  } else if (auto cast = llvm::dyn_cast_or_null<mlir::arith::ExtSIOp>(op)) {
    return castRemainder(cast.getIn());
  } else if (auto cast = llvm::dyn_cast_or_null<mlir::arith::ExtUIOp>(op)) {
    return castRemainder(cast.getIn());
  } else if (auto select = llvm::dyn_cast_or_null<mlir::arith::SelectOp>(op)) {
    auto x = next(select.getTrueValue()), y = next(select.getFalseValue());
    if (x && y && x == y)
      return x;
  }
  return std::nullopt;
}

inline std::optional<int64_t> getKnownRemainder(mlir::OpFoldResult value,
                                                int64_t modulus) {
  if (modulus <= 0)
    return std::nullopt;
  if (auto n = mlir::getConstantIntValue(value))
    return (*n % modulus + modulus) % modulus;
  return getKnownRemainder(llvm::dyn_cast<mlir::Value>(value), modulus);
}

// Static absolute descriptor offset in elements. A reinterpret_cast replaces
// the source offset; a subview adds offsets times the SOURCE strides.
inline std::optional<int64_t> getStaticDescriptorOffset(mlir::Value value) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return std::nullopt;
  llvm::SmallVector<int64_t> strides;
  int64_t offset;
  if (mlir::failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (offset != mlir::ShapedType::kDynamic)
    return offset;
  if (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>())
    return mlir::getConstantIntValue(cast.getMixedOffsets()[0]);
  if (auto sub = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto sourceOffset = getStaticDescriptorOffset(sub.getSource());
    auto sourceType = llvm::cast<mlir::MemRefType>(sub.getSource().getType());
    if (!sourceOffset ||
        mlir::failed(sourceType.getStridesAndOffset(strides, offset)))
      return std::nullopt;
    offset = *sourceOffset;
    for (auto [part, stride] : llvm::zip(sub.getMixedOffsets(), strides)) {
      auto constant = mlir::getConstantIntValue(part);
      int64_t delta;
      if (!constant || stride == mlir::ShapedType::kDynamic ||
          llvm::MulOverflow(*constant, stride, delta) ||
          llvm::AddOverflow(offset, delta, offset))
        return std::nullopt;
    }
    return offset;
  }
  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>())
    return getStaticDescriptorOffset(cast.getSource());
  if (auto transpose = value.getDefiningOp<mlir::memref::TransposeOp>())
    return getStaticDescriptorOffset(transpose.getIn());
  if (auto cast = value.getDefiningOp<mlir::memref::MemorySpaceCastOp>())
    return getStaticDescriptorOffset(cast.getSource());
  if (auto collapse = value.getDefiningOp<mlir::memref::CollapseShapeOp>())
    return getStaticDescriptorOffset(collapse.getSrc());
  if (auto expand = value.getDefiningOp<mlir::memref::ExpandShapeOp>())
    return getStaticDescriptorOffset(expand.getSrc());
  return std::nullopt;
}

inline bool isDescriptorOffsetAligned(mlir::Value value, int64_t factor) {
  if (factor == 1)
    return true;
  // `memref.view` changes the base pointer in bytes while leaving the
  // descriptor offset of its source unchanged.  The byte-shift alignment is
  // checked separately by the endpoint proof below.
  if (auto view = value.getDefiningOp<mlir::memref::ViewOp>())
    return isDescriptorOffsetAligned(view.getSource(), factor);
  if (auto offset = getStaticDescriptorOffset(value))
    return *offset % factor == 0;
  if (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>())
    return getKnownRemainder(cast.getMixedOffsets()[0], factor) ==
           std::optional<int64_t>(0);
  if (auto sub = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    if (!isDescriptorOffsetAligned(sub.getSource(), factor))
      return false;
    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    auto type = llvm::cast<mlir::MemRefType>(sub.getSource().getType());
    if (mlir::failed(type.getStridesAndOffset(strides, offset)))
      return false;
    for (auto [part, stride] : llvm::zip(sub.getMixedOffsets(), strides)) {
      if (stride == mlir::ShapedType::kDynamic ||
          !isProvablyMultipleOf(part, factor / std::gcd(factor, stride)))
        return false;
    }
    return true;
  }
  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>())
    return isDescriptorOffsetAligned(cast.getSource(), factor);
  if (auto transpose = value.getDefiningOp<mlir::memref::TransposeOp>())
    return isDescriptorOffsetAligned(transpose.getIn(), factor);
  if (auto cast = value.getDefiningOp<mlir::memref::MemorySpaceCastOp>())
    return isDescriptorOffsetAligned(cast.getSource(), factor);
  if (auto collapse = value.getDefiningOp<mlir::memref::CollapseShapeOp>())
    return isDescriptorOffsetAligned(collapse.getSrc(), factor);
  if (auto expand = value.getDefiningOp<mlir::memref::ExpandShapeOp>())
    return isDescriptorOffsetAligned(expand.getSrc(), factor);
  return false;
}

// Return whether an endpoint's effective address can be proven aligned,
// without emitting a diagnostic.  The planner uses this predicate while it
// is choosing between direct DDR and staged placement; an unproven view
// shift should simply keep the operand on the fallback path.  Diagnostics
// remain the responsibility of verifyEndpointPointerAlignment once a direct
// endpoint has actually been selected.
inline bool isEndpointPointerAlignmentProven(mlir::Value endpoint,
                                             int64_t alignmentBytes) {
  if (alignmentBytes <= 0)
    return false;
  auto type = llvm::dyn_cast<mlir::MemRefType>(endpoint.getType());
  unsigned width = type ? getSpmElementByteWidth(type.getElementType()) : 0;
  if (!width ||
      !isDescriptorOffsetAligned(
          endpoint, alignmentBytes / std::gcd(alignmentBytes, int64_t(width))))
    return false;

  mlir::Value current = endpoint;
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *defining = current.getDefiningOp()) {
    if (!visited.insert(defining).second)
      return false;
    if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(defining)) {
      if (!isProvablyMultipleOf(view.getByteShift(), alignmentBytes))
        return false;
      current = view.getSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto transpose = llvm::dyn_cast<mlir::memref::TransposeOp>(defining)) {
      current = transpose.getIn();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(defining)) {
      current = subview.getSource();
      continue;
    }
    if (auto reinterpret =
            llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining)) {
      current = reinterpret.getSource();
      continue;
    }
    if (auto collapse =
            llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining)) {
      current = collapse.getSrc();
      continue;
    }
    if (auto expand = llvm::dyn_cast<mlir::memref::ExpandShapeOp>(defining)) {
      current = expand.getSrc();
      continue;
    }
    if (auto metadata =
            llvm::dyn_cast<mlir::memref::ExtractStridedMetadataOp>(defining)) {
      current = metadata.getSource();
      continue;
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(defining))
      return alloc.getAlignment() &&
             *alloc.getAlignment() >= uint64_t(alignmentBytes) &&
             *alloc.getAlignment() % alignmentBytes == 0;
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocaOp>(defining))
      return alloc.getAlignment() &&
             *alloc.getAlignment() >= uint64_t(alignmentBytes) &&
             *alloc.getAlignment() % alignmentBytes == 0;
    if (auto global = llvm::dyn_cast<mlir::memref::GetGlobalOp>(defining)) {
      auto storage =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
              global, global.getNameAttr());
      return storage && storage.getAlignment() &&
             *storage.getAlignment() >= uint64_t(alignmentBytes) &&
             *storage.getAlignment() % alignmentBytes == 0;
    }
    // Function arguments and explicit SPM roots carry their alignment through
    // the ABI or the SPM view contract.  No further proof is available here.
    break;
  }
  return true;
}

inline std::optional<int64_t> getDescriptorOffsetRemainder(mlir::Value value,
                                                           int64_t modulus) {
  if (auto offset = getStaticDescriptorOffset(value))
    return (*offset % modulus + modulus) % modulus;
  if (auto cast = value.getDefiningOp<mlir::memref::ReinterpretCastOp>())
    return getKnownRemainder(cast.getMixedOffsets()[0], modulus);
  if (auto sub = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto result = getDescriptorOffsetRemainder(sub.getSource(), modulus);
    llvm::SmallVector<int64_t> strides;
    int64_t ignored;
    auto type = llvm::cast<mlir::MemRefType>(sub.getSource().getType());
    if (!result || mlir::failed(type.getStridesAndOffset(strides, ignored)))
      return std::nullopt;
    for (auto [part, stride] : llvm::zip(sub.getMixedOffsets(), strides)) {
      if (stride == mlir::ShapedType::kDynamic || stride < 0)
        return std::nullopt;
      int64_t factor = modulus / std::gcd(modulus, stride);
      auto remainder = getKnownRemainder(part, factor);
      if (!remainder)
        return std::nullopt;
      *result = (*result + *remainder * (stride % modulus)) % modulus;
    }
    return result;
  }
  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>())
    return getDescriptorOffsetRemainder(cast.getSource(), modulus);
  if (auto cast = value.getDefiningOp<mlir::memref::MemorySpaceCastOp>())
    return getDescriptorOffsetRemainder(cast.getSource(), modulus);
  if (auto transpose = value.getDefiningOp<mlir::memref::TransposeOp>())
    return getDescriptorOffsetRemainder(transpose.getIn(), modulus);
  if (auto collapse = value.getDefiningOp<mlir::memref::CollapseShapeOp>())
    return getDescriptorOffsetRemainder(collapse.getSrc(), modulus);
  if (auto expand = value.getDefiningOp<mlir::memref::ExpandShapeOp>())
    return getDescriptorOffsetRemainder(expand.getSrc(), modulus);
  return std::nullopt;
}

// Residue of the actual effective address, relative to a root whose alignment
// is independently checked by the ABI/placement verifier. No IR is created.
// Reinterpret casts replace descriptor offsets; only memref.view shifts the
// pointer. Keeping these components separate avoids counting offsets twice.
inline std::optional<int64_t> getEndpointByteRemainder(mlir::Value endpoint,
                                                       int64_t modulus = 64) {
  if (modulus <= 0 || modulus > 64)
    return std::nullopt;
  if (isEndpointPointerAlignmentProven(endpoint, modulus))
    return 0;
  auto type = llvm::dyn_cast<mlir::MemRefType>(endpoint.getType());
  unsigned width = type ? getSpmElementByteWidth(type.getElementType()) : 0;
  if (!width)
    return std::nullopt;
  auto offset = getDescriptorOffsetRemainder(
      endpoint, modulus / std::gcd(modulus, int64_t(width)));
  if (!offset)
    return std::nullopt;
  int64_t result = *offset * width % modulus;
  mlir::Value current = endpoint;
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *op = current.getDefiningOp()) {
    if (!visited.insert(op).second)
      return std::nullopt;
    if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(op)) {
      auto shift = getKnownRemainder(view.getByteShift(), modulus);
      if (!shift)
        return std::nullopt;
      result = (result + *shift) % modulus;
      current = view.getSource();
    } else if (auto sub = llvm::dyn_cast<mlir::memref::SubViewOp>(op)) {
      current = sub.getSource();
    } else if (auto cast =
                   llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
      current = cast.getSource();
    } else if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
      current = cast.getSource();
    } else if (auto cast =
                   llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(op)) {
      current = cast.getSource();
    } else if (auto transpose = llvm::dyn_cast<mlir::memref::TransposeOp>(op)) {
      current = transpose.getIn();
    } else if (auto collapse =
                   llvm::dyn_cast<mlir::memref::CollapseShapeOp>(op)) {
      current = collapse.getSrc();
    } else if (auto expand = llvm::dyn_cast<mlir::memref::ExpandShapeOp>(op)) {
      current = expand.getSrc();
    } else {
      break;
    }
  }
  return (llvm::isa<mlir::BlockArgument>(current) ||
          isEndpointPointerAlignmentProven(current, modulus))
             ? std::optional<int64_t>(result)
             : std::nullopt;
}

// Walk the view chain of a DMA endpoint and check that the base pointer it
// yields is `alignmentBytes` aligned, given that the storage root is.
//
// Only the operations that really move the base pointer need a proof:
// `memref.view` shifts the pointer by its `byte_shift` operand and keeps the
// descriptor offset at zero.  Everything else in the chain either keeps the
// pointer (`memref.cast`, `memref.memory_space_cast`) or encodes its offset in
// the descriptor (`memref.subview`/`memref.reinterpret_cast`). The descriptor
// proof above handles both static and provably aligned dynamic offsets.
inline mlir::LogicalResult
verifyEndpointPointerAlignment(mlir::Operation *op, mlir::Value endpoint,
                               llvm::StringRef what, int64_t alignmentBytes) {
  // Alignment applies to the effective address, not independently to a view's
  // byte shift and descriptor offset. A prefix may make their sum aligned.
  if (alignmentBytes > 0 && alignmentBytes <= 64 &&
      getEndpointByteRemainder(endpoint, alignmentBytes) ==
          std::optional<int64_t>(0))
    return mlir::success();
  auto type = llvm::dyn_cast<mlir::MemRefType>(endpoint.getType());
  unsigned width = type ? getSpmElementByteWidth(type.getElementType()) : 0;
  if (!width ||
      !isDescriptorOffsetAligned(
          endpoint, alignmentBytes / std::gcd(alignmentBytes, int64_t(width))))
    return op->emitOpError() << what
                             << " endpoint effective address cannot be proven "
                                "64 B aligned (descriptor offset in elements)";
  mlir::Value current = endpoint;
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *defining = current.getDefiningOp()) {
    if (!visited.insert(defining).second)
      return op->emitOpError()
             << what << " endpoint " << current
             << " has a cyclic view chain; the alignment of its effective "
                "address cannot be established";
    if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(defining)) {
      mlir::Value shift = view.getByteShift();
      std::optional<int64_t> constant = mlir::getConstantIntValue(shift);
      if (constant) {
        if (*constant % alignmentBytes != 0)
          return op->emitOpError()
                 << what << " endpoint " << current
                 << " shifts the base pointer by " << *constant
                 << " bytes, which is not a multiple of the " << alignmentBytes
                 << "-byte DMA granularity";
      } else if (!isProvablyMultipleOf(shift, alignmentBytes)) {
        return op->emitOpError()
               << what << " endpoint " << current
               << " shifts the base pointer by the runtime value " << shift
               << ", which cannot be proven to be a multiple of the "
               << alignmentBytes
               << "-byte DMA granularity; use an ordinary copy for this range "
                  "or make the shift provably aligned";
      }
      current = view.getSource();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto transpose = llvm::dyn_cast<mlir::memref::TransposeOp>(defining)) {
      current = transpose.getIn();
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::MemorySpaceCastOp>(defining)) {
      current = cast.getSource();
      continue;
    }
    if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(defining)) {
      current = subview.getSource();
      continue;
    }
    if (auto reinterpret =
            llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining)) {
      current = reinterpret.getSource();
      continue;
    }
    if (auto collapse =
            llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining)) {
      current = collapse.getSrc();
      continue;
    }
    if (auto expand = llvm::dyn_cast<mlir::memref::ExpandShapeOp>(defining)) {
      current = expand.getSrc();
      continue;
    }
    if (auto metadata =
            llvm::dyn_cast<mlir::memref::ExtractStridedMetadataOp>(defining)) {
      current = metadata.getSource();
      continue;
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(defining)) {
      if (!alloc.getAlignment() ||
          *alloc.getAlignment() < uint64_t(alignmentBytes) ||
          *alloc.getAlignment() % alignmentBytes != 0)
        return op->emitOpError()
               << what << " endpoint allocation must prove 64 B alignment";
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocaOp>(defining)) {
      if (!alloc.getAlignment() ||
          *alloc.getAlignment() < uint64_t(alignmentBytes) ||
          *alloc.getAlignment() % alignmentBytes != 0)
        return op->emitOpError()
               << what << " endpoint allocation must prove 64 B alignment";
    }
    if (auto global = llvm::dyn_cast<mlir::memref::GetGlobalOp>(defining)) {
      auto storage =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
              global, global.getNameAttr());
      if (!storage || !storage.getAlignment() ||
          *storage.getAlignment() < uint64_t(alignmentBytes) ||
          *storage.getAlignment() % alignmentBytes != 0)
        return op->emitOpError()
               << what << " endpoint global must prove 64 B alignment";
    }
    // `memref.alloc`/`memref.alloca` with an alignment attribute, a storage
    // root (block argument, `bosc_ame.spm.view`) or anything else: the root
    // alignment is the caller's/allocator's guarantee, not a view shift.
    break;
  }
  return mlir::success();
}

} // namespace spmdma
} // namespace boscame
} // namespace buddy

#endif // BUDDY_DIALECT_BOSCAME_DMAALIGNMENT_H
