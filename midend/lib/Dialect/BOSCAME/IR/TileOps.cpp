//===- TileOps.cpp - SPM <-> AME tile operation contracts ------===//
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
// Verifiers and layout rules of the `bosc_ame.tile.*` operations.  Keeping the
// rules in one translation unit (rather than in each op verifier) is what lets
// the LLVM lowering reuse exactly the layout the verifier accepted: the
// lowering only re-derives the strides and the row stride, and never has to
// repeat a check that could drift from the verifier's.
//
//===----------------------------------------------------------------------===//

#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/Endpoint.h"
#include "Dialect/BOSCAME/MatrixTransfer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace buddy::boscame;

StringRef buddy::boscame::stringifyTileRole(TileRole role) {
  switch (role) {
  case TileRole::A:
    return "A";
  case TileRole::B:
    return "B";
  case TileRole::Accumulator:
    return "accumulator";
  }
  llvm_unreachable("unknown tile role");
}

std::optional<unsigned>
buddy::boscame::getTileElementByteWidth(TileRole role, Type elementType) {
  auto integer = dyn_cast<IntegerType>(elementType);
  if (!integer)
    return std::nullopt;
  switch (role) {
  case TileRole::A:
  case TileRole::B:
    // Signed i8 x i8 is the datapath this dialect implements today.  The AME
    // has wider tile loads/stores, but accepting them here would pick an mtype
    // phase and a register file that are not wired up.
    return integer.getWidth() == 8 ? std::optional<unsigned>(1) : std::nullopt;
  case TileRole::Accumulator:
    // Raw i32 accumulator memory: the exact representation the AME
    // accumulator stores.
    return integer.getWidth() == 32 ? std::optional<unsigned>(4) : std::nullopt;
  }
  llvm_unreachable("unknown tile role");
}

// Human-readable description of what a role accepts, for diagnostics.
static StringRef getTileRoleRequirement(TileRole role) {
  switch (role) {
  case TileRole::A:
    return "a memref<MxKxi8> with K contiguous (strides[1] == 1) and a row "
           "stride of at least K (strides[0] >= K)";
  case TileRole::B:
    return "a memref<KxNxi8> describing the physical NxK B buffer, i.e. with "
           "K contiguous inside one physical row (strides[0] == 1) and a row "
           "pitch of at least K (strides[1] >= K)";
  case TileRole::Accumulator:
    return "a memref<MxNxi32> with N contiguous (strides[1] == 1) and a row "
           "stride of at least N (strides[0] >= N)";
  }
  llvm_unreachable("unknown tile role");
}

static LogicalResult verifyDirectDdrProvenance(Operation *op, Value endpoint) {
  EndpointKind kind = classifyEndpoint(endpoint);
  if (kind == EndpointKind::PrivateDDRFrame)
    return verifyPrivateEndpoint(op, endpoint);
  if (kind != EndpointKind::ExternalDDR)
    return op->emitOpError(
        "direct DDR tile endpoint must trace to an external function entry "
        "argument or a managed private DDR frame");
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto abi = module ? module->getAttrOfType<StringAttr>(spmdma::kDmaAbiAttrName)
                    : StringAttr();
  if (!abi || (abi.getValue() != spmdma::kDmaAbiAlignedNonAliasing &&
               abi.getValue() != spmdma::kDmaAbiAligned))
    return op->emitOpError()
           << "direct DDR tile endpoint requires module attribute " << char(96)
           << spmdma::kDmaAbiAttrName << " = " << char(34)
           << spmdma::kDmaAbiAlignedNonAliasing << char(34) << char(96);
  return success();
}

std::optional<TileStorageLayout>
buddy::boscame::computeTileStorageLayout(MemRefType type, TileRole role) {
  if (!type || type.getRank() != 2 || !type.hasStaticShape())
    return std::nullopt;

  std::optional<unsigned> elementBytes =
      getTileElementByteWidth(role, type.getElementType());
  if (!elementBytes)
    return std::nullopt;

  llvm::SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (strides[0] == ShapedType::kDynamic || strides[1] == ShapedType::kDynamic)
    return std::nullopt;
  if ((offset != ShapedType::kDynamic && offset < 0) || strides[0] <= 0 ||
      strides[1] <= 0)
    return std::nullopt;

  TileStorageLayout layout;
  layout.shape = {type.getShape()[0], type.getShape()[1]};
  layout.strides = {strides[0], strides[1]};
  layout.elementBytes = *elementBytes;
  // A direct DDR descriptor may carry a runtime subview offset.  The caller's
  // aligned-pointer ABI supplies the alignment proof; keep the static layout
  // usable for stride checks and let tile lowering recover the dynamic offset
  // from the memref descriptor.
  if (offset == ShapedType::kDynamic)
    layout.byteOffset = 0;
  else if (llvm::MulOverflow(offset, layout.elementBytes, layout.byteOffset))
    return std::nullopt;

  // The hardware row stride is the *physical* row distance.  For B, whose
  // storage is `N x K` with K contiguous, that is the stride between two
  // physical rows of the staged buffer, i.e. the logical trailing stride.
  const int rowIndex = tileRowStrideIndex(role);
  const int innerIndex = 1 - rowIndex;
  // The inner dimension must be the contiguous one, and the row pitch must not
  // make two physical rows overlap.
  if (strides[innerIndex] != 1)
    return std::nullopt;
  if (strides[rowIndex] < layout.shape[innerIndex])
    return std::nullopt;
  if (llvm::MulOverflow(strides[rowIndex], layout.elementBytes,
                        layout.rowStrideBytes))
    return std::nullopt;
  // An actual one-row view never issues the unused next-row address. Use a
  // legal descriptor pitch without changing any accessed byte. Padded/multirow
  // views retain their original pitch and strict alignment requirement.
  if (layout.shape[rowIndex] == 1)
    layout.rowStrideBytes = llvm::alignTo(
        layout.shape[innerIndex] * layout.elementBytes, int64_t(64));
  return layout;
}

LogicalResult buddy::boscame::verifyTileStorageLayout(
    Operation *op, MemRefType type, TileRole role, TileStorageLayout &layout) {
  if (!type)
    return op->emitOpError() << "operand must be a ranked memref in "
                                "`#bosc_ame.spm<bank>`, got an "
                                "unranked memref";

  // AME tile loads/stores normally address the fixed SPM windows.  A placement
  // decision may explicitly select the external DDR descriptor instead.  The
  // opt-in is kept on each operation so ordinary clients (and compiler-owned
  // temporaries) retain the historical SPM contract.
  bool directDdr = op->hasAttr("bosc_ame.direct_ddr");
  if (!isa_and_nonnull<SPMAttr>(type.getMemorySpace()) && !directDdr)
    return op->emitOpError()
           << "operand " << type
           << " must carry the `#bosc_ame.spm<bank>` storage space; a DDR "
              "buffer has no SPM address for the AME "
           << stringifyTileRole(role) << " load/store";

  if (type.getRank() != 2)
    return op->emitOpError() << "operand " << type
                             << " must be a 2-D memref (the first version "
                                "supports static 2-D tiles only), got rank "
                             << type.getRank();

  if (!type.hasStaticShape())
    return op->emitOpError()
           << "operand " << type
           << " must have a static shape: the tile configuration is compiled "
              "into the instruction sequence";

  if (!getTileElementByteWidth(role, type.getElementType()))
    return op->emitOpError()
           << "operand element type " << type.getElementType() << " is not "
           << "supported for the " << stringifyTileRole(role)
           << " tile; the first version implements signed i8 A/B tiles and "
              "i32 accumulators";

  for (int64_t dim : type.getShape())
    if (dim < 1 || dim > kAmeMaxTileDim)
      return op->emitOpError() << "tile dimension " << dim << " of " << type
                               << " is outside the AME tile capacity [1, "
                               << kAmeMaxTileDim << "]";

  std::optional<TileStorageLayout> computed =
      computeTileStorageLayout(type, role);
  if (!computed)
    return op->emitOpError()
           << "operand " << type << " has a layout the AME "
           << stringifyTileRole(role)
           << " load/store cannot address; the hardware needs "
           << getTileRoleRequirement(role);

  layout = *computed;
  if (directDdr && layout.byteOffset % spmdma::kDmaAlignmentBytes != 0)
    return op->emitOpError()
           << "direct DDR " << stringifyTileRole(role) << " descriptor offset "
           << layout.byteOffset << " bytes is not 64-byte aligned";
  if (directDdr && layout.rowStrideBytes % spmdma::kDmaAlignmentBytes != 0)
    return op->emitOpError() << "direct DDR " << stringifyTileRole(role)
                             << " physical row stride " << layout.rowStrideBytes
                             << " bytes is not 64-byte aligned";
  return success();
}

LogicalResult buddy::boscame::verifyTileMemRefOperand(
    Operation *op, Value operand, TileRole role, TileStorageLayout &layout) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type)
    return op->emitOpError()
           << "operand " << operand << " must be a ranked memref, got "
           << operand.getType();
  if (failed(verifyTileStorageLayout(op, type, role, layout)))
    return failure();
  if (op->hasAttr("bosc_ame.direct_ddr")) {
    if (failed(verifyDirectDdrProvenance(op, operand)) ||
        failed(spmdma::verifyEndpointPointerAlignment(
            op, operand, "direct DDR tile", spmdma::kDmaAlignmentBytes)))
      return failure();
  }
  return success();
}

LogicalResult
buddy::boscame::verifyTileMatrixValue(Operation *op, Value value, TileRole role,
                                      std::array<int64_t, 2> &shape) {
  auto vector = dyn_cast<VectorType>(value.getType());
  if (!vector)
    return op->emitOpError() << "matrix value " << value
                             << " must be a 2-D vector (BOSCAME_MatrixType), "
                                "got "
                             << value.getType();

  if (vector.getRank() != 2)
    return op->emitOpError()
           << "matrix value " << value << " must be a 2-D vector with the real "
           << "logical tile shape, got rank " << vector.getRank();

  if (vector.isScalable())
    return op->emitOpError()
           << "matrix value " << value
           << " must be a fixed 2-D vector at this level; the scalable "
              "register type is chosen by the lowering";

  if (!getTileElementByteWidth(role, vector.getElementType()))
    return op->emitOpError()
           << "matrix element type " << vector.getElementType() << " is not "
           << "supported for the " << stringifyTileRole(role)
           << " tile; the first version implements signed i8 A/B tiles and "
              "i32 accumulators";

  shape = {vector.getShape()[0], vector.getShape()[1]};
  for (int64_t dim : shape)
    if (dim < 1 || dim > kAmeMaxTileDim)
      return op->emitOpError()
             << "matrix tile dimension " << dim << " of " << value
             << " is outside the AME tile capacity [1, " << kAmeMaxTileDim
             << "]";
  return success();
}

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

// Resolve the complete logical endpoint against its declared SPM window,
// rather than just one register-sized tile. Checking the complete ranges
// prevents an early tile's store from destroying a later tile's source.
struct MatrixSpmRange {
  Value root;
  int64_t bank;
  int64_t begin;
  int64_t end;
};

static std::optional<MatrixSpmRange>
getMatrixSpmRange(Value endpoint, const matrixtransfer::Layout &layout) {
  auto type = dyn_cast<MemRefType>(endpoint.getType());
  auto bank = spmdma::getSpmBank(type);
  auto origin = spmdma::getStaticPointerOrigin(endpoint);
  auto window = origin ? origin->root.getDefiningOp<SpmViewOp>() : SpmViewOp();
  if (!bank || !origin || !window || origin->byteShift < 0 ||
      window.getBank() != *bank)
    return std::nullopt;
  auto windowBytes = spmdma::getContiguousZeroOffsetByteSize(window.getType());
  SmallVector<int64_t> strides;
  int64_t offset, begin, end, windowEnd;
  if (!windowBytes || failed(type.getStridesAndOffset(strides, offset)) ||
      offset == ShapedType::kDynamic || offset < 0 ||
      llvm::MulOverflow(offset, layout.elementBytes, begin) ||
      llvm::AddOverflow(begin, origin->byteShift, begin) ||
      llvm::AddOverflow(begin, static_cast<int64_t>(window.getOffset()),
                        begin) ||
      llvm::MulOverflow(layout.rows - 1, layout.pitch, end) ||
      llvm::AddOverflow(end, layout.rowBytes, end) ||
      llvm::AddOverflow(end, begin, end) ||
      llvm::AddOverflow(static_cast<int64_t>(window.getOffset()), *windowBytes,
                        windowEnd) ||
      begin < static_cast<int64_t>(window.getOffset()) || end > windowEnd ||
      begin < 0 || end > spmdma::kSpmBankCapacityBytes)
    return std::nullopt;
  return MatrixSpmRange{origin->root, *bank, begin, end};
}

LogicalResult MatrixCopyOp::verify() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  auto target = module ? module->getAttrOfType<StringAttr>("bosc_ame.target")
                       : StringAttr();
  if (!target || target.getValue() != "nr-fpga")
    return emitOpError("requires the explicit bosc_ame.target=nr-fpga "
                       "raw ACC32 memory contract");
  auto sourceType = dyn_cast<MemRefType>(getSource().getType());
  auto targetType = dyn_cast<MemRefType>(getTarget().getType());
  if (!sourceType || !targetType ||
      !matrixtransfer::isSupportedElementType(sourceType.getElementType()) ||
      !matrixtransfer::isSupportedElementType(targetType.getElementType()))
    return emitOpError(
        "supports only signless i8/i16/i32/i64 or f32/f64 logical "
        "storage through the raw ACC32 memory path");
  auto axis = matrixtransfer::getCommonUnitAxis(sourceType, targetType);
  if (!axis || sourceType.getShape() != targetType.getShape())
    return emitOpError("requires equally shaped static rank-2 endpoints of the "
                       "same element type with a common physical unit axis");
  if (getInstructionBits() != matrixtransfer::kInstructionBits ||
      getRegisterCount() != matrixtransfer::kRegisterCount ||
      getRegisterClass() != matrixtransfer::kRegisterClass ||
      getMtype() != matrixtransfer::kMtype)
    return emitOpError("supports only one raw ACC32 register with "
                       "instruction_bits=32, register_class=acc and "
                       "mtype=65602 on the NR target");

  auto sourceLayout = matrixtransfer::getLayout(sourceType, *axis);
  auto targetLayout = matrixtransfer::getLayout(targetType, *axis);
  auto sourceOrigin = matrixtransfer::getBlockOrigin(
      *sourceLayout, getSourceOffset(), getRows(), getRowBytes(),
      getSourcePitch());
  auto targetOrigin = matrixtransfer::getBlockOrigin(
      *targetLayout, getTargetOffset(), getRows(), getRowBytes(),
      getTargetPitch());
  if (!sourceOrigin || !targetOrigin || sourceOrigin != targetOrigin)
    return emitOpError("requires matching in-bounds row/column origins, exact "
                       "word-aligned ranges and pitches, at most 64 rows and "
                       "256 bytes per row; copying through pitch padding is "
                       "forbidden");

  bool sourceSpm = isa_and_nonnull<SPMAttr>(sourceType.getMemorySpace());
  bool targetSpm = isa_and_nonnull<SPMAttr>(targetType.getMemorySpace());
  if ((!sourceSpm && !targetSpm) ||
      (!sourceSpm && sourceType.getMemorySpace()) ||
      (!targetSpm && targetType.getMemorySpace()))
    return emitOpError("requires an SPM/external DDR boundary or two proven "
                       "independent SPM endpoints; physical TR/ACC values are "
                       "not memory endpoints");

  if (sourceSpm && targetSpm) {
    auto source = getMatrixSpmRange(getSource(), *sourceLayout);
    auto destination = getMatrixSpmRange(getTarget(), *targetLayout);
    if (!source || !destination)
      return emitOpError("internal matrix copy requires static endpoint ranges "
                         "bounded by their declared SPM windows and banks");
    if (source->root == destination->root ||
        (source->bank == destination->bank &&
         source->begin < destination->end && destination->begin < source->end))
      return emitOpError("internal matrix copy requires independent, "
                         "nonoverlapping complete SPM endpoint ranges");
    for (Value endpoint : {getSource(), getTarget()})
      if (failed(spmdma::verifyEndpointPointerAlignment(*this, endpoint,
                                                        "matrix copy", 4)))
        return failure();
    return success();
  }

  Value ddr = sourceSpm ? getTarget() : getSource();
  if (classifyEndpoint(ddr) == EndpointKind::PrivateDDRFrame) {
    if (failed(verifyPrivateEndpoint(*this, ddr)))
      return failure();
    for (Value endpoint : {getSource(), getTarget()})
      if (failed(spmdma::verifyEndpointPointerAlignment(*this, endpoint,
                                                        "matrix copy", 4)))
        return failure();
    return success();
  }
  auto root = dyn_cast_or_null<BlockArgument>(getEndpointRoot(ddr));
  auto rootType =
      root ? dyn_cast<BaseMemRefType>(root.getType()) : BaseMemRefType();
  auto function = root ? dyn_cast<func::FuncOp>(root.getOwner()->getParentOp())
                       : func::FuncOp();
  if (!root || !root.getOwner()->isEntryBlock() || !function || !rootType ||
      rootType.getMemorySpace() || !module ||
      !SymbolTable::symbolKnownUseEmpty(function, module))
    return emitOpError("DDR endpoint must trace to an external entry argument "
                       "whose caller alignment contract applies");
  auto abi = module->getAttrOfType<DictionaryAttr>("contract.abi");
  auto alignment =
      abi ? abi.getAs<IntegerAttr>("args_aligned_bytes") : IntegerAttr();
  auto legacy = module->getAttrOfType<StringAttr>(spmdma::kDmaAbiAttrName);
  bool aligned =
      alignment && alignment.getInt() >= 4 && alignment.getInt() % 4 == 0;
  if (!aligned && !abi && legacy)
    aligned = legacy.getValue() == spmdma::kDmaAbiAligned ||
              legacy.getValue() == spmdma::kDmaAbiAlignedNonAliasing;
  if (!aligned)
    return emitOpError("requires an entry ABI proving at least 4-byte root "
                       "alignment divisible by 4");
  for (Value endpoint : {getSource(), getTarget()})
    if (failed(spmdma::verifyEndpointPointerAlignment(*this, endpoint,
                                                      "matrix copy", 4)))
      return failure();
  return success();
}

// Shape and element type of the memory operand and of the matrix value must
// describe the same logical tile.
static LogicalResult verifyLoadShape(Operation *op, TileRole role,
                                     const TileStorageLayout &layout,
                                     ArrayRef<int64_t> shape) {
  if (layout.shape[0] != shape[0] || layout.shape[1] != shape[1])
    return op->emitOpError()
           << "the " << stringifyTileRole(role) << " memref shape ["
           << layout.shape[0] << ", " << layout.shape[1]
           << "] does not match the matrix shape [" << shape[0] << ", "
           << shape[1]
           << "]; logical and physical tile descriptions must agree";
  return success();
}

LogicalResult TileLoadAOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::A, layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::A, shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::A, layout, shape);
}

LogicalResult TileLoadBOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::B, layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::B, shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::B, layout, shape);
}

LogicalResult TileLoadAccOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getSource(), TileRole::Accumulator,
                                     layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::Accumulator, layout, shape);
}

LogicalResult TileZeroAccOp::verify() {
  std::array<int64_t, 2> shape;
  return verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                               shape);
}

LogicalResult TileStoreAccOp::verify() {
  TileStorageLayout layout;
  if (failed(verifyTileMemRefOperand(*this, getDest(), TileRole::Accumulator,
                                     layout)))
    return failure();
  std::array<int64_t, 2> shape;
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   shape)))
    return failure();
  return verifyLoadShape(*this, TileRole::Accumulator, layout, shape);
}

LogicalResult TileMmaOp::verify() {
  std::array<int64_t, 2> lhsShape, rhsShape, accShape, resultShape;
  if (failed(verifyTileMatrixValue(*this, getLhs(), TileRole::A, lhsShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getRhs(), TileRole::B, rhsShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getAcc(), TileRole::Accumulator,
                                   accShape)))
    return failure();
  if (failed(verifyTileMatrixValue(*this, getMatrix(), TileRole::Accumulator,
                                   resultShape)))
    return failure();

  // The accumulator dependency is explicit: C_in and C_out are the same value
  // type, so a chain of MMAs cannot silently change shape or element type.
  if (getAcc().getType() != getMatrix().getType())
    return emitOpError() << "the accumulator operand type "
                         << getAcc().getType() << " and the result type "
                         << getMatrix().getType()
                         << " must be identical; a tile.mma returns the "
                            "updated accumulator";

  if (accShape != resultShape)
    return emitOpError() << "accumulator shape [" << accShape[0] << ", "
                         << accShape[1] << "] and result shape ["
                         << resultShape[0] << ", " << resultShape[1]
                         << "] must agree";

  if (lhsShape[1] != rhsShape[0])
    return emitOpError() << "the inner dimension of A (K = " << lhsShape[1]
                         << ") and the outer dimension of B (K = "
                         << rhsShape[0] << ") must agree";

  if (accShape[0] != lhsShape[0] || accShape[1] != rhsShape[1])
    return emitOpError() << "C must be an M x N tile with M = " << lhsShape[0]
                         << " (from A) and N = " << rhsShape[1]
                         << " (from B), but has shape [" << accShape[0] << ", "
                         << accShape[1] << "]";

  return success();
}

//===----------------------------------------------------------------------===//
// Memory effects
//===----------------------------------------------------------------------===//

void MatrixCopyOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable());
}

void TileLoadAOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileLoadBOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileLoadAccOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

void TileStoreAccOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable());
}
