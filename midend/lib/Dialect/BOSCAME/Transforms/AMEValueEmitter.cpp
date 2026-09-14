//===- AMEValueEmitter.cpp - SSA value-semantics AME building blocks ------===//
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

#include "Dialect/BOSCAME/Transforms/AMEValueEmitter.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace buddy::boscame;

namespace buddy {
namespace boscame {
namespace ame {

VectorType getMatrixTileType(MLIRContext *context, Type elementType) {
  return VectorType::get({4, 4}, elementType);
}

static unsigned getElementByteWidth(Type elementType) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  return std::max(1u, (bitWidth + 7) / 8);
}

/// Cast an integer/index value to i64; i64 values are passed through.
static Value castToI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(64))
    return value;
  return arith::IndexCastOp::create(builder, loc, builder.getI64Type(), value);
}

Value createByteStride(OpBuilder &builder, Location loc, Value memref,
                       unsigned dim) {
  auto memrefType = cast<MemRefType>(memref.getType());
  unsigned rank = memrefType.getRank();
  unsigned bytesPerElem = getElementByteWidth(memrefType.getElementType());
  Value bytes = arith::ConstantIndexOp::create(builder, loc, bytesPerElem);

  if (rank == 0)
    return arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                      bytes);

  // extract_strided_metadata results:
  //   [baseBuffer, offset, sizes..., strides...]
  auto metadata = memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  Value strideElem = metadata.getResult(2 + rank + dim);
  Value strideBytes = arith::MulIOp::create(builder, loc, strideElem, bytes);
  return arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                    strideBytes);
}

/// Emit one of the `ml{e,a,b,c}{8,16,32,64}.m`-style loads.
template <typename OpTy>
static FailureOr<Value> createLoadOp(OpBuilder &builder, Location loc,
                                     Type tileType, Value source,
                                     Value byteStride) {
  return OpTy::create(builder, loc, tileType, source, byteStride).getRes();
}

/// Emit one of the `m{qma,ma,...}*.mm` MMAs with a uniform call shape.
template <typename OpTy>
static FailureOr<Value> createWidenMmaOp(OpBuilder &builder, Location loc,
                                         Type tileType, Value acc, Value lhs,
                                         Value rhs) {
  return OpTy::create(builder, loc, tileType, acc, lhs, rhs).getRes();
}

/// Emit one of the `msce{8,16,32,64}.m` stores.
template <typename OpTy>
static LogicalResult createStoreOp(OpBuilder &builder, Location loc, Value src,
                                   Value dest, Value byteStride) {
  OpTy::create(builder, loc, src, dest, byteStride);
  return success();
}

/// Dispatch the load/store family on the datapath element width.
enum class DatapathKind { I8, I16, I32, I64, F16, F32, F64, Unsupported };

static DatapathKind classify(Type elementType) {
  if (elementType.isInteger(8))
    return DatapathKind::I8;
  if (elementType.isInteger(16))
    return DatapathKind::I16;
  if (elementType.isInteger(32))
    return DatapathKind::I32;
  if (elementType.isInteger(64))
    return DatapathKind::I64;
  if (elementType.isF16() || elementType.isBF16())
    return DatapathKind::F16;
  if (elementType.isF32())
    return DatapathKind::F32;
  if (elementType.isF64())
    return DatapathKind::F64;
  return DatapathKind::Unsupported;
}

/// Report an unsupported instruction/type combination.  The emitter is only
/// reached after the caller's capability check, so an unsupported type here is
/// a real error rather than "this pattern did not match".
static FailureOr<Value> rejectUnsupported(Operation *anchor, StringRef what,
                                          Type elementType) {
  if (anchor)
    anchor->emitOpError()
        << "unsupported BOSCAME " << what << " element type " << elementType;
  return failure();
}

FailureOr<Value> createLoadA(OpBuilder &builder, Location loc, Type elementType,
                             Value source, Value byteStride,
                             Operation *anchor) {
  Type tileType = getMatrixTileType(builder.getContext(), elementType);
  switch (classify(elementType)) {
  case DatapathKind::I8:
    return createLoadOp<Mlae8mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I16:
  case DatapathKind::F16:
    return createLoadOp<Mlae16mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I32:
  case DatapathKind::F32:
    return createLoadOp<Mlae32mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I64:
  case DatapathKind::F64:
    return createLoadOp<Mlae64mOp>(builder, loc, tileType, source, byteStride);
  default:
    return rejectUnsupported(anchor, "A load", elementType);
  }
}

FailureOr<Value> createLoadB(OpBuilder &builder, Location loc, Type elementType,
                             Value source, Value byteStride,
                             Operation *anchor) {
  Type tileType = getMatrixTileType(builder.getContext(), elementType);
  switch (classify(elementType)) {
  case DatapathKind::I8:
    return createLoadOp<Mlbe8mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I16:
  case DatapathKind::F16:
    return createLoadOp<Mlbe16mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I32:
  case DatapathKind::F32:
    return createLoadOp<Mlbe32mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I64:
  case DatapathKind::F64:
    return createLoadOp<Mlbe64mOp>(builder, loc, tileType, source, byteStride);
  default:
    return rejectUnsupported(anchor, "B load", elementType);
  }
}

FailureOr<Value> createLoadBTransposed(OpBuilder &builder, Location loc,
                                       Type elementType, Value source,
                                       Value byteStride, Operation *anchor) {
  Type tileType = getMatrixTileType(builder.getContext(), elementType);
  switch (classify(elementType)) {
  case DatapathKind::I8:
    return createLoadOp<Mlbte8mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I16:
  case DatapathKind::F16:
    return createLoadOp<Mlbte16mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I32:
  case DatapathKind::F32:
    return createLoadOp<Mlbte32mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I64:
  case DatapathKind::F64:
    return createLoadOp<Mlbte64mOp>(builder, loc, tileType, source, byteStride);
  default:
    return rejectUnsupported(anchor, "transposed B load", elementType);
  }
}

FailureOr<Value> createLoadAccumulator(OpBuilder &builder, Location loc,
                                       Type accElementType,
                                       Type memoryElementType, Value source,
                                       Value byteStride, Operation *anchor) {
  Type tileType = getMatrixTileType(builder.getContext(), accElementType);
  switch (classify(accElementType)) {
  case DatapathKind::I8:
    return createLoadOp<Mlce8mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I16:
  case DatapathKind::F16:
    return createLoadOp<Mlce16mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I32:
  case DatapathKind::F32:
    return createLoadOp<Mlce32mOp>(builder, loc, tileType, source, byteStride);
  case DatapathKind::I64:
  case DatapathKind::F64:
    return createLoadOp<Mlce64mOp>(builder, loc, tileType, source, byteStride);
  default:
    return rejectUnsupported(anchor, "accumulator load", accElementType);
  }
}

FailureOr<Value> createMma(OpBuilder &builder, Location loc, Value acc,
                           Value lhs, Value rhs, Operation *anchor) {
  auto accType = dyn_cast<VectorType>(acc.getType());
  auto lhsType = dyn_cast<VectorType>(lhs.getType());
  if (!accType || !lhsType) {
    if (anchor)
      anchor->emitOpError() << "BOSCAME MMA expects matrix values, got "
                            << acc.getType() << " and " << lhs.getType();
    return failure();
  }

  Type accElementType = accType.getElementType();
  Type lhsElementType = lhsType.getElementType();
  Type tileType = getMatrixTileType(builder.getContext(), accElementType);

  if (accElementType.isInteger(32) && lhsElementType.isInteger(32))
    return createWidenMmaOp<MmaWmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (accElementType.isInteger(32) && lhsElementType.isInteger(16))
    return createWidenMmaOp<MwmaHmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (accElementType.isInteger(32) && lhsElementType.isInteger(8))
    return createWidenMmaOp<MqmaBmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (accElementType.isInteger(16) && lhsElementType.isInteger(16))
    return createWidenMmaOp<MmaHmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (accElementType.isInteger(64) && lhsElementType.isInteger(64))
    return createWidenMmaOp<MmaDwmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if ((lhsElementType.isF16() || lhsElementType.isBF16()) &&
      accElementType.isF32())
    return createWidenMmaOp<MfwmaHfmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (lhsElementType.isF32() && accElementType.isF32())
    return createWidenMmaOp<MfmaFmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (lhsElementType.isF32() && accElementType.isF64())
    return createWidenMmaOp<MfwmaFmmOp>(builder, loc, tileType, acc, lhs, rhs);
  if (lhsElementType.isF64() && accElementType.isF64())
    return createWidenMmaOp<MfmaDmmOp>(builder, loc, tileType, acc, lhs, rhs);

  if (anchor)
    anchor->emitOpError()
        << "unsupported BOSCAME matmul instruction type: lhs " << lhsElementType
        << ", accumulator " << accElementType;
  return failure();
}

LogicalResult createStoreAccumulator(OpBuilder &builder, Location loc,
                                     Value src, Value dest, Value byteStride,
                                     Operation *anchor) {
  auto srcType = dyn_cast<VectorType>(src.getType());
  if (!srcType) {
    if (anchor)
      anchor->emitOpError()
          << "accumulator store expects a matrix value, got " << src.getType();
    return failure();
  }

  switch (classify(srcType.getElementType())) {
  case DatapathKind::I8:
    return createStoreOp<Msce8mOp>(builder, loc, src, dest, byteStride);
  case DatapathKind::I16:
  case DatapathKind::F16:
    return createStoreOp<Msce16mOp>(builder, loc, src, dest, byteStride);
  case DatapathKind::I32:
  case DatapathKind::F32:
    return createStoreOp<Msce32mOp>(builder, loc, src, dest, byteStride);
  case DatapathKind::I64:
  case DatapathKind::F64:
    return createStoreOp<Msce64mOp>(builder, loc, src, dest, byteStride);
  default:
    if (anchor)
      anchor->emitOpError()
          << "unsupported BOSCAME accumulator store element type "
          << srcType.getElementType();
    return failure();
  }
}

LogicalResult configureTileK(OpBuilder &builder, Location loc, Value k) {
  MSettilekOp::create(builder, loc, builder.getI64Type(),
                      castToI64(builder, loc, k));
  return success();
}

LogicalResult configureTiles(OpBuilder &builder, Location loc, Value m, Value n,
                             Value k) {
  Type i64Type = builder.getI64Type();
  MSettilemOp::create(builder, loc, i64Type, castToI64(builder, loc, m));
  MSettilenOp::create(builder, loc, i64Type, castToI64(builder, loc, n));
  if (k)
    configureTileK(builder, loc, k);
  return success();
}

FailureOr<int64_t> getUpstreamMsetTypeImm(Type elementType) {
  if (elementType.isInteger(4))
    return 4;
  if (elementType.isInteger(8))
    return 8;
  if (elementType.isInteger(16) || elementType.isF16() ||
      elementType.isBF16())
    return 16;
  if (elementType.isInteger(32) || elementType.isF32())
    return 32;
  if (elementType.isInteger(64) || elementType.isF64())
    return 64;
  return failure();
}

/// Emit the `mtype` CSR write for one phase.
static FailureOr<int64_t> configureMtype(OpBuilder &builder, Location loc,
                                         Type elementType,
                                         AmeTargetProfile profile,
                                         FpgaMtypePhase phase,
                                         Operation *anchor) {
  Type i64Type = builder.getI64Type();

  if (profile == AmeTargetProfile::Upstream) {
    // Upstream/main pathway: raw element width, immediate form.
    FailureOr<int64_t> imm = getUpstreamMsetTypeImm(elementType);
    if (failed(imm)) {
      if (anchor)
        anchor->emitOpError()
            << "unsupported BOSCAME element type for msettypei: "
            << elementType;
      return failure();
    }
    MSettypeiOp::create(builder, loc, i64Type, *imm);
    return *imm;
  }

  // FPGA pathway: bit-field CSR written from a register.  The value must stay
  // a register-form write: it is the marker that tells the LLVM backend this
  // module uses the FPGA encoding rather than the raw element width.
  FailureOr<int64_t> imm = getFpgaMtypeImm(elementType, phase);
  if (failed(imm)) {
    if (anchor)
      anchor->emitOpError()
          << "unsupported FPGA AME mtype phase for element type "
          << elementType;
    return failure();
  }
  Value value = arith::ConstantOp::create(builder, loc, i64Type,
                                          builder.getI64IntegerAttr(*imm));
  MSettypeOp::create(builder, loc, i64Type, value);
  return *imm;
}

FailureOr<int64_t> configureMmaType(OpBuilder &builder, Location loc,
                                    Type elementType, AmeTargetProfile profile,
                                    Operation *anchor) {
  return configureMtype(builder, loc, elementType, profile,
                        FpgaMtypePhase::Mma, anchor);
}

FailureOr<int64_t> configureAccumulatorType(OpBuilder &builder, Location loc,
                                            Type accElementType,
                                            AmeTargetProfile profile,
                                            Operation *anchor) {
  return configureMtype(builder, loc, accElementType, profile,
                        FpgaMtypePhase::Accumulator, anchor);
}

} // namespace ame
} // namespace boscame
} // namespace buddy
