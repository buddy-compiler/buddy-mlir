//===- ZAMESDialect.cpp - ZAMES Dialect Implementation -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include "ZAMES/ZAMESDialect.h"
#include "ZAMES/ZAMESOps.h"
#include "ZAMES/ZAMESTypes.h"

using namespace mlir;
using namespace buddy::zames;

#include "ZAMES/ZAMESDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ZAMES/ZAMES.cpp.inc"

namespace {

static Type expectedMemrefElementType(StringRef token, MLIRContext *ctx) {
  return llvm::StringSwitch<Type>(token)
      .Case("i4", IntegerType::get(ctx, 4, IntegerType::Signless))
      .Case("u4", IntegerType::get(ctx, 4, IntegerType::Unsigned))
      .Case("i8", IntegerType::get(ctx, 8, IntegerType::Signless))
      .Case("u8", IntegerType::get(ctx, 8, IntegerType::Unsigned))
      .Case("i32", IntegerType::get(ctx, 32, IntegerType::Signless))
      .Case("tf32", Float32Type::get(ctx))
      .Case("fp16", Float16Type::get(ctx))
      .Case("bf16", BFloat16Type::get(ctx))
      .Case("fp32", Float32Type::get(ctx))
      .Case("fp8e4m3", Float8E4M3FNType::get(ctx))
      .Case("fp8e5m2", Float8E5M2Type::get(ctx))
      .Default(Type());
}

static LogicalResult verifyTileType(Location loc, TileType tile) {
  return TileType::verify([&]() { return emitError(loc); }, tile.getKind(),
                          tile.getShape(), tile.getElement());
}

static LogicalResult verifyMemrefCompat(Location loc, StringRef token,
                                        Type memrefElemType) {
  Type expected = expectedMemrefElementType(token, loc.getContext());
  // Historical packed tokens have no one-to-one builtin storage type.
  if (expected && expected != memrefElemType)
    return emitError(loc) << "expected memref element type " << expected
                          << " for ZAMES element token " << token
                          << ", but got " << memrefElemType;
  return success();
}

static unsigned countDynamicDims(TileType tile) {
  return llvm::count_if(tile.getShape(), ShapedType::isDynamic);
}

static LogicalResult verifyDynamicSizeCount(Operation *op, unsigned expected,
                                            unsigned actual) {
  if (expected != actual)
    return op->emitError() << "expected " << expected
                           << " dynamic size operands, but got " << actual;
  return success();
}

// A dynamic extent is compatible with either a static or a dynamic extent.
static bool staticExtentsMatch(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

static LogicalResult verifyTileView(Operation *op, MemRefType view,
                                    TileType tile, bool transpose) {
  ArrayRef<int64_t> shape = tile.getShape();
  bool normal = staticExtentsMatch(view.getDimSize(0), shape[0]) &&
                staticExtentsMatch(view.getDimSize(1), shape[1]);
  bool transposed = staticExtentsMatch(view.getDimSize(0), shape[1]) &&
                    staticExtentsMatch(view.getDimSize(1), shape[0]);
  if (normal || (transpose && transposed))
    return success();
  return op->emitError() << "expected tile view shape to match tile shape "
                         << shape[0] << "x" << shape[1]
                         << (transpose ? " or its transpose" : "")
                         << ", but got " << view.getDimSize(0) << "x"
                         << view.getDimSize(1);
}

static LogicalResult verifyMemoryView(Operation *op, TileType tile,
                                      MemRefType view) {
  if (failed(verifyTileType(op->getLoc(), tile)))
    return failure();
  if (!view)
    return op->emitError("expected ranked memref tile view");
  if (view.getRank() != 2)
    return op->emitError() << "expected rank-2 tile view, but got rank "
                           << view.getRank();
  return verifyMemrefCompat(op->getLoc(), tile.getElement(),
                            view.getElementType());
}

} // namespace

void ZAMESDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ZAMES/ZAMES.cpp.inc"
      >();
  registerTypes();
}

LogicalResult LoadOp::verify() {
  TileType tile = getResult().getType();
  auto view = dyn_cast<MemRefType>(getView().getType());
  if (failed(verifyMemoryView(*this, tile, view)) ||
      failed(verifyTileView(*this, view, tile, getTranspose())))
    return failure();
  return verifyDynamicSizeCount(*this, countDynamicDims(tile),
                                getDynamicSizes().size());
}

LogicalResult StoreOp::verify() {
  TileType tile = getTile().getType();
  auto view = dyn_cast<MemRefType>(getView().getType());
  if (failed(verifyMemoryView(*this, tile, view)) ||
      failed(verifyTileView(*this, view, tile, getTranspose())))
    return failure();
  return verifyDynamicSizeCount(*this, countDynamicDims(tile),
                                getDynamicSizes().size());
}

LogicalResult MmaOp::verify() {
  TileType a = getA().getType();
  TileType b = getB().getType();
  TileType c = getC().getType();
  if (failed(verifyTileType(getLoc(), a)) ||
      failed(verifyTileType(getLoc(), b)) ||
      failed(verifyTileType(getLoc(), c)))
    return failure();
  if (!a.isA())
    return emitOpError("first operand must have kind a");
  if (!b.isB())
    return emitOpError("second operand must have kind b");
  if (!c.isC())
    return emitOpError("accumulator must have kind c");
  ArrayRef<int64_t> aShape = a.getShape();
  ArrayRef<int64_t> bShape = b.getShape();
  ArrayRef<int64_t> cShape = c.getShape();
  if (!staticExtentsMatch(aShape[1], bShape[1]))
    return emitOpError("A.K and B.K static dimensions must match");
  if (!staticExtentsMatch(aShape[0], cShape[0]))
    return emitOpError("A.M and C.M static dimensions must match");
  if (!staticExtentsMatch(bShape[0], cShape[1]))
    return emitOpError("B.N and C.N static dimensions must match");
  // Only A.M, B.N, and A.K determine the dynamic operand list, in M/N/K order.
  unsigned expectedDynamicSizes = ShapedType::isDynamic(aShape[0]) +
                                  ShapedType::isDynamic(bShape[0]) +
                                  ShapedType::isDynamic(aShape[1]);
  return verifyDynamicSizeCount(*this, expectedDynamicSizes,
                                getDynamicSizes().size());
}

LogicalResult MzeroOp::verify() {
  TileType tile = getResult().getType();
  if (failed(verifyTileType(getLoc(), tile)))
    return failure();
  return verifyDynamicSizeCount(*this, countDynamicDims(tile),
                                getDynamicSizes().size());
}
