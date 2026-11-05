//===- MarkedMatmul.cpp - Buffer-form integer matmul to AME ---------------===//
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

#include "Conversion/BOSCAMEHelpers/BOSCAMEHelpers.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "Dialect/BOSCAME/TileOps.h"
#include "Dialect/BOSCAME/Transforms/BOSCAMETileLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

Value index(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value flatten(OpBuilder &builder, Location loc, Value buffer) {
  auto type = cast<MemRefType>(buffer.getType());
  auto flat = MemRefType::get({type.getNumElements()}, type.getElementType());
  return memref::CollapseShapeOp::create(
      builder, loc, flat, buffer, ArrayRef<ReassociationIndices>{{0, 1}});
}

Value typedSpm(OpBuilder &builder, Location loc, Value flat,
               ArrayRef<int64_t> shape, ArrayRef<int64_t> strides) {
  auto source = cast<MemRefType>(flat.getType());
  auto type =
      MemRefType::get(shape, source.getElementType(),
                      StridedLayoutAttr::get(builder.getContext(), 0, strides),
                      source.getMemorySpace());
  return memref::ReinterpretCastOp::create(builder, loc, type, flat, 0, shape,
                                           strides);
}

struct SpmSnapshot {
  Value ordinary;
  Value spmFlat;
  Value ordinaryFlat;
  int64_t count;
};

struct SpmVectorInterval {
  SmallVector<int64_t> strides;
  int64_t offset;
  int64_t begin;
  int64_t count;
};

// Packing may use scalar accesses only after an explicit vector transfer out
// of SPM. Keep the physical layout, including B's K-contiguous columns, so
// the existing logical-index packing also works for non-tile SPM inputs.
// The rounded vector interval must fit the actual spm.view allocation; a
// memref's logical shape alone does not authorize reading padded bytes.
FailureOr<SpmVectorInterval> getSpmVectorInterval(Operation *anchor,
                                                  Value source) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<int64_t> strides;
  int64_t ignoredOffset;
  auto offset = getStaticDescriptorOffset(source);
  if (!offset || *offset < 0 ||
      failed(type.getStridesAndOffset(strides, ignoredOffset)) ||
      llvm::any_of(
          strides,
          [](int64_t stride) {
            return stride <= 0 || ShapedType::isDynamic(stride);
          })) {
    anchor->emitError("AME SPM packing requires static positive strides and "
                      "a known nonnegative descriptor offset");
    return failure();
  }
  Value root = source;
  while (true) {
    if (auto op = root.getDefiningOp<memref::ReinterpretCastOp>())
      root = op.getSource();
    else if (auto op = root.getDefiningOp<memref::SubViewOp>())
      root = op.getSource();
    else if (auto op = root.getDefiningOp<memref::TransposeOp>())
      root = op.getIn();
    else if (auto op = root.getDefiningOp<memref::CastOp>())
      root = op.getSource();
    else if (auto op = root.getDefiningOp<memref::CollapseShapeOp>())
      root = op.getSrc();
    else if (auto op = root.getDefiningOp<memref::ExpandShapeOp>())
      root = op.getSrc();
    else
      break;
  }
  auto view = root.getDefiningOp<SpmViewOp>();
  auto bytes =
      view ? getContiguousZeroOffsetByteSize(cast<MemRefType>(view.getType()))
           : std::nullopt;
  unsigned width = getSpmElementByteWidth(type.getElementType());
  if (!bytes || !width ||
      cast<MemRefType>(root.getType()).getElementType() !=
          type.getElementType()) {
    anchor->emitError("AME SPM packing requires a bounded typed spm.view "
                      "without pointer-shifting or element-type casts");
    return failure();
  }
  int64_t span = 1;
  for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
    int64_t term;
    if (size <= 0 || llvm::MulOverflow(size - 1, stride, term) ||
        llvm::AddOverflow(span, term, span)) {
      anchor->emitError("AME SPM packing has an invalid physical extent");
      return failure();
    }
  }
  int64_t end;
  if (llvm::AddOverflow(*offset, span, end) || end > *bytes / width) {
    anchor->emitError("AME SPM packing leaves its spm.view allocation");
    return failure();
  }
  int64_t lanes = kDmaAlignmentBytes / width;
  int64_t begin = llvm::alignDown(*offset, lanes);
  end = llvm::alignTo(end, lanes);
  if (end > *bytes / width) {
    anchor->emitError("AME SPM packing needs complete 64 B vectors inside "
                      "its spm.view allocation");
    return failure();
  }
  int64_t count = end - begin;
  return SpmVectorInterval{strides, *offset, begin, count};
}

FailureOr<SpmSnapshot> snapshotSpm(OpBuilder &builder, Operation *anchor,
                                   Value source) {
  auto interval = getSpmVectorInterval(anchor, source);
  if (failed(interval))
    return failure();
  auto type = cast<MemRefType>(source.getType());
  auto &[strides, offset, begin, count] = *interval;
  Location loc = anchor->getLoc();
  auto spmType =
      MemRefType::get({count}, type.getElementType(),
                      StridedLayoutAttr::get(builder.getContext(), begin, {1}),
                      type.getMemorySpace());
  Value spmFlat = memref::ReinterpretCastOp::create(
      builder, loc, spmType, source, begin, ArrayRef<int64_t>{count},
      ArrayRef<int64_t>{1});
  Value ordinaryFlat = memref::AllocOp::create(
      builder, loc, MemRefType::get({count}, type.getElementType()),
      builder.getI64IntegerAttr(64));
  emitDdrPolicy(builder, loc, true);
  if (failed(copyVectors(builder, loc, spmFlat, ordinaryFlat, count)))
    return failure();
  auto ordinaryType = MemRefType::get(
      type.getShape(), type.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), offset - begin, strides));
  Value ordinary = memref::ReinterpretCastOp::create(
      builder, loc, ordinaryType, ordinaryFlat, offset - begin, type.getShape(),
      strides);
  return SpmSnapshot{ordinary, spmFlat, ordinaryFlat, count};
}

// All packing accesses ordinary memory, so scalar packing is safe. Its index
// callback preserves the logical [K,N] B indexing irrespective of its strides.
template <typename Copy>
void copyRectangle(OpBuilder &builder, Location loc, int64_t rows,
                   int64_t columns, Copy copy) {
  OpBuilder::InsertionGuard outer(builder);
  Value zero = index(builder, loc, 0), one = index(builder, loc, 1);
  auto rowLoop =
      scf::ForOp::create(builder, loc, zero, index(builder, loc, rows), one);
  builder.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop =
      scf::ForOp::create(builder, loc, zero, index(builder, loc, columns), one);
  builder.setInsertionPointToStart(colLoop.getBody());
  copy(rowLoop.getInductionVar(), colLoop.getInductionVar());
}

LogicalResult lowerMatmul(linalg::MatmulOp matmul, Workspace &workspace) {
  if (matmul.getNumResults() || matmul.getInputs().size() != 2 ||
      matmul.getOutputs().size() != 1)
    return matmul.emitError("marked AME matmul must have buffer semantics");
  Value a = matmul.getInputs()[0], b = matmul.getInputs()[1],
        c = matmul.getOutputs()[0];
  auto at = dyn_cast<MemRefType>(a.getType()),
       bt = dyn_cast<MemRefType>(b.getType()),
       ct = dyn_cast<MemRefType>(c.getType());
  if (!at || !bt || !ct || at.getRank() != 2 || bt.getRank() != 2 ||
      ct.getRank() != 2 || !at.hasStaticShape() || !bt.hasStaticShape() ||
      !ct.hasStaticShape() || !at.getElementType().isInteger(8) ||
      !bt.getElementType().isInteger(8) || !ct.getElementType().isInteger(32))
    return matmul.emitError(
        "AME target requires static i8/i8/i32 rank-2 buffers");
  if (matmul.getCast() != linalg::TypeFn::cast_signed)
    return matmul.emitError("AME target requires signed integer extension");
  int64_t m = ct.getDimSize(0), n = ct.getDimSize(1), k = at.getDimSize(1);
  if (m <= 0 || n <= 0 || k <= 0 || at.getDimSize(0) != m ||
      bt.getDimSize(0) != k || bt.getDimSize(1) != n)
    return matmul.emitError("AME target has unsupported matmul dimensions");
  OpBuilder builder(matmul);
  Location loc = matmul.getLoc();
  constexpr int64_t tile = 64;
  bool oneTile = m <= tile && n <= tile && k == tile;
  bool directA = oneTile && getSpmBank(at) &&
                 computeTileStorageLayout(at, TileRole::A).has_value();
  bool directB = oneTile && getSpmBank(bt) &&
                 computeTileStorageLayout(bt, TileRole::B).has_value();
  if ((directA && failed(getSpmVectorInterval(matmul, a))) ||
      (directB && failed(getSpmVectorInterval(matmul, b))))
    return failure();
  if (getSpmBank(at) && !directA) {
    auto snapshot = snapshotSpm(builder, matmul, a);
    if (failed(snapshot))
      return failure();
    a = snapshot->ordinary;
  }
  if (getSpmBank(bt) && !directB) {
    auto snapshot = snapshotSpm(builder, matmul, b);
    if (failed(snapshot))
      return failure();
    b = snapshot->ordinary;
  }
  std::optional<SpmSnapshot> output;
  if (getSpmBank(ct)) {
    auto snapshot = snapshotSpm(builder, matmul, c);
    if (failed(snapshot))
      return failure();
    output = *snapshot;
    c = snapshot->ordinary;
  }
  auto alloc = [&](ArrayRef<int64_t> shape, Type element) -> Value {
    return memref::AllocOp::create(builder, loc,
                                   MemRefType::get(shape, element),
                                   builder.getI64IntegerAttr(64));
  };
  auto fillZero = [&](Value buffer) {
    Type element = cast<MemRefType>(buffer.getType()).getElementType();
    Value zero = arith::ConstantOp::create(builder, loc, element,
                                           builder.getIntegerAttr(element, 0));
    linalg::FillOp::create(builder, loc, ValueRange{zero}, ValueRange{buffer});
  };
  // Each static M/N tile gets its own windows. K blocks reuse the windows of
  // this access point only after the preceding values have been stored.
  for (int64_t mi = 0; mi < m; mi += tile) {
    for (int64_t ni = 0; ni < n; ni += tile) {
      int64_t tm = std::min(tile, m - mi), tn = std::min(tile, n - ni);
      int64_t pitch = llvm::alignTo(tn, int64_t(16));
      Value spmA, spmB;
      if (!directA) {
        auto allocated = workspace.allocate(
            builder, matmul, MemRefType::get({tm * tile}, builder.getI8Type()),
            0);
        if (failed(allocated))
          return failure();
        spmA = *allocated;
      }
      if (!directB) {
        auto allocated = workspace.allocate(
            builder, matmul, MemRefType::get({tn * tile}, builder.getI8Type()),
            1);
        if (failed(allocated))
          return failure();
        spmB = *allocated;
      }
      auto spmC = workspace.allocate(
          builder, matmul, MemRefType::get({tm * pitch}, builder.getI32Type()),
          0);
      if (failed(spmC))
        return failure();
      Value aPack = directA ? Value() : alloc({tm, tile}, builder.getI8Type());
      Value bPack = directB ? Value() : alloc({tn, tile}, builder.getI8Type());
      Value cPack = alloc({tm, pitch}, builder.getI32Type());
      Value aFlat = directA ? Value() : flatten(builder, loc, aPack),
            bFlat = directB ? Value() : flatten(builder, loc, bPack),
            cFlat = flatten(builder, loc, cPack);
      Value aTile =
          directA ? a : typedSpm(builder, loc, spmA, {tm, tile}, {tile, 1});
      Value bTile =
          directB ? b : typedSpm(builder, loc, spmB, {tile, tn}, {1, tile});
      Value cTile = typedSpm(builder, loc, *spmC, {tm, tn}, {pitch, 1});
      fillZero(cPack);
      // Seed exactly the standard matmul's C. MatmulConverter's original dot C
      // is added separately by Linalg and must not be added here again.
      copyRectangle(builder, loc, tm, tn, [&](Value i, Value j) {
        Value ci =
            arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
        Value cj =
            arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
        Value value =
            memref::LoadOp::create(builder, loc, c, ValueRange{ci, cj});
        memref::StoreOp::create(builder, loc, value, cPack, ValueRange{i, j});
      });
      emitDmaAndWait(builder, loc, cFlat, *spmC, tm * pitch * 4);
      for (int64_t ki = 0; ki < k; ki += tile) {
        int64_t tk = std::min(tile, k - ki);
        if (!directA) {
          fillZero(aPack);
          copyRectangle(builder, loc, tm, tk, [&](Value i, Value kk) {
            Value ai =
                arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
            Value ak = arith::AddIOp::create(builder, loc, kk,
                                             index(builder, loc, ki));
            Value value =
                memref::LoadOp::create(builder, loc, a, ValueRange{ai, ak});
            memref::StoreOp::create(builder, loc, value, aPack,
                                    ValueRange{i, kk});
          });
          emitDmaAndWait(builder, loc, aFlat, spmA, tm * tile);
        }
        if (!directB) {
          fillZero(bPack);
          copyRectangle(builder, loc, tn, tk, [&](Value j, Value kk) {
            Value bk = arith::AddIOp::create(builder, loc, kk,
                                             index(builder, loc, ki));
            Value bj =
                arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
            Value value =
                memref::LoadOp::create(builder, loc, b, ValueRange{bk, bj});
            memref::StoreOp::create(builder, loc, value, bPack,
                                    ValueRange{j, kk});
          });
          emitDmaAndWait(builder, loc, bFlat, spmB, tn * tile);
        }
        // All runtime calls finish before any AME value becomes live.
        auto accType = VectorType::get({tm, tn}, builder.getI32Type());
        Value acc = TileLoadAccOp::create(builder, loc, accType, cTile);
        Value ta = TileLoadAOp::create(
            builder, loc, VectorType::get({tm, tile}, builder.getI8Type()),
            aTile);
        Value tb = TileLoadBOp::create(
            builder, loc, VectorType::get({tile, tn}, builder.getI8Type()),
            bTile);
        Value result = TileMmaOp::create(builder, loc, accType, acc, ta, tb);
        TileStoreAccOp::create(builder, loc, result, cTile);
      }
      emitDdrPolicy(builder, loc, true);
      if (failed(copyVectors(builder, loc, *spmC, cFlat, tm * pitch)))
        return failure();
      copyRectangle(builder, loc, tm, tn, [&](Value i, Value j) {
        Value value =
            memref::LoadOp::create(builder, loc, cPack, ValueRange{i, j});
        Value ci =
            arith::AddIOp::create(builder, loc, i, index(builder, loc, mi));
        Value cj =
            arith::AddIOp::create(builder, loc, j, index(builder, loc, ni));
        memref::StoreOp::create(builder, loc, value, c, ValueRange{ci, cj});
      });
    }
  }
  if (output) {
    emitDdrPolicy(builder, loc, false);
    if (failed(copyVectors(builder, loc, output->ordinaryFlat, output->spmFlat,
                           output->count)))
      return failure();
  }
  matmul.erase();
  return success();
}

// Floating dots retain their standard Linalg arithmetic. Materialize only
// the consumer's SPM operands, so subsequent scalar loop lowering never
// accesses SPM. This runs after placement and therefore cannot promote the
// ordinary snapshots back into SPM.
LogicalResult materializeFloatingMatmul(linalg::MatmulOp matmul) {
  if (!matmul.hasPureBufferSemantics() || matmul.getInputs().size() != 2 ||
      matmul.getOutputs().size() != 1)
    return matmul.emitError("floating SPM matmul requires buffer semantics");
  for (Value operand : matmul.getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
        !isa<FloatType>(type.getElementType()) ||
        llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
      return matmul.emitError(
          "floating SPM matmul requires static positive rank-2 float buffers");
  }
  OpBuilder builder(matmul);
  std::optional<SpmSnapshot> output;
  for (unsigned number = 0; number < matmul.getNumOperands(); ++number) {
    Value operand = matmul.getOperand(number);
    if (!getSpmBank(operand.getType()))
      continue;
    auto snapshot = snapshotSpm(builder, matmul, operand);
    if (failed(snapshot))
      return failure();
    matmul->setOperand(number, snapshot->ordinary);
    if (number == matmul.getNumOperands() - 1)
      output = *snapshot;
  }
  if (output) {
    builder.setInsertionPointAfter(matmul);
    emitDdrPolicy(builder, matmul.getLoc(), false);
    if (failed(copyVectors(builder, matmul.getLoc(), output->ordinaryFlat,
                           output->spmFlat, output->count)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult lowerMarkedMatmuls(ModuleOp module) {
  SmallVector<linalg::MatmulOp> targets;
  SmallVector<linalg::MatmulOp> floating;
  module.walk([&](linalg::MatmulOp op) {
    if (op->hasAttr(kLowerToAme))
      targets.push_back(op);
    else if (isa<FloatType>(
                 getElementTypeOrSelf(op.getOutputs()[0].getType())) &&
             llvm::any_of(op.getOperandTypes(), [](Type type) {
               return getSpmBank(type).has_value();
             }))
      floating.push_back(op);
  });
  for (linalg::MatmulOp op : floating)
    if (failed(materializeFloatingMatmul(op)))
      return failure();
  if (targets.empty())
    return success();
  auto target = module->getAttrOfType<StringAttr>(kAmeTargetAttrName);
  if (target && target.getValue() != kAmeTargetNrFpga)
    return module.emitError(
        "marked AME matmul requires bosc_ame.target=nr-fpga");
  module->setAttr(kAmeTargetAttrName,
                  StringAttr::get(module.getContext(), kAmeTargetNrFpga));
  Workspace workspace;
  for (linalg::MatmulOp op : targets)
    if (failed(lowerMatmul(op, workspace)))
      return failure();
  return success();
}

} // namespace buddy::boscame::lowering
