//===- StructuredMemory.cpp - Rectangular DMA accesses --------------------===//
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
#include "Dialect/BOSCAME/DmaAlignment.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace buddy::boscame::spmdma;

namespace buddy::boscame::lowering {
namespace {

struct AccessPlan {
  int64_t fastAxis;
  int64_t rows;
  int64_t width;
  int64_t spmPitch;
  int64_t chunk;
  OpFoldResult validRows;
  OpFoldResult validWidth;
};

FailureOr<AccessPlan> planAccess(Operation *anchor, Value ptr,
                                 RankedTensorType tensor,
                                 ArrayRef<OpFoldResult> mask,
                                 OpBuilder &builder) {
  auto type = dyn_cast<MemRefType>(ptr.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type || type.getRank() != 2 || !tensor.hasStaticShape() ||
      type.getShape() != tensor.getShape() ||
      type.getElementType() != tensor.getElementType() ||
      failed(type.getStridesAndOffset(strides, offset)) ||
      (strides[0] != 1 && strides[1] != 1)) {
    anchor->emitError(
        "BOSCAME rank-2 access requires a static rectangle with a unit-stride "
        "physical axis; gather/scatter and split pointers are unsupported");
    return failure();
  }
  int64_t fast = strides[1] == 1 ? 1 : 0;
  int64_t width = tensor.getDimSize(fast), rows = tensor.getDimSize(1 - fast);
  unsigned bytes = getSpmElementByteWidth(tensor.getElementType());
  if (!bytes || width <= 0 || rows <= 0 || strides[1 - fast] < width) {
    anchor->emitError("BOSCAME rank-2 access has unsupported overlapping "
                      "layout or element type");
    return failure();
  }
  int64_t lanes = kDmaAlignmentBytes / bytes;
  OpFoldResult validRows =
      mask.empty() ? builder.getIndexAttr(rows) : mask[1 - fast];
  OpFoldResult validWidth =
      mask.empty() ? builder.getIndexAttr(width) : mask[fast];
  auto count = getConstantIntValue(validWidth);
  auto rowCount = getConstantIntValue(validRows);
  if ((rowCount && (*rowCount < 0 || *rowCount > rows)) ||
      (count && (*count < 0 || *count > width || *count % lanes)) ||
      (!count && (width % lanes || !isProvablyMultipleOf(validWidth, lanes)))) {
    anchor->emitError(
        "BOSCAME access requires a 64 B multiple of authorized DDR bytes; "
        "short stores and unprovable dynamic mask extents are unsupported");
    return failure();
  }
  return AccessPlan{fast,
                    rows,
                    width,
                    int64_t(llvm::alignTo(width, lanes)),
                    count && *count ? *count : lanes,
                    validRows,
                    validWidth};
}

Value slice1D(OpBuilder &builder, Location loc, Value flat, int64_t offset,
              int64_t count) {
  return memref::SubViewOp::create(
      builder, loc, flat, ArrayRef<OpFoldResult>{builder.getIndexAttr(offset)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(count)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
}

Value rowView(OpBuilder &builder, Location loc, Value ptr, int64_t fast,
              int64_t row, int64_t start, int64_t count) {
  SmallVector<OpFoldResult> offsets(2), sizes(2),
      strides(2, builder.getIndexAttr(1));
  offsets[fast] = builder.getIndexAttr(start);
  offsets[1 - fast] = builder.getIndexAttr(row);
  sizes[fast] = builder.getIndexAttr(count);
  sizes[1 - fast] = builder.getIndexAttr(1);
  auto type = memref::SubViewOp::inferRankReducedResultType(
      {count}, cast<MemRefType>(ptr.getType()), offsets, sizes, strides);
  return memref::SubViewOp::create(builder, loc, type, ptr, offsets, sizes,
                                   strides);
}

Value indexValue(OpBuilder &builder, Location loc, OpFoldResult extent) {
  if (auto constant = getConstantIntValue(extent))
    return arith::ConstantIndexOp::create(builder, loc, *constant);
  return cast<Value>(extent);
}

// Static SPM slices keep bank bounds exact. Dynamic masks guard each complete
// packet, so no excluded DDR byte is touched. No packet crosses a mask edge.
template <typename Emit>
LogicalResult emitPackets(OpBuilder &builder, Operation *anchor,
                          const AccessPlan &plan, Emit emit) {
  Location loc = anchor->getLoc();
  auto rows = getConstantIntValue(plan.validRows);
  auto width = getConstantIntValue(plan.validWidth);
  for (int64_t row = 0; row < (rows ? *rows : plan.rows); ++row) {
    for (int64_t start = 0; start < (width ? *width : plan.width);
         start += plan.chunk) {
      Value condition;
      if (!rows)
        condition = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::slt,
            arith::ConstantIndexOp::create(builder, loc, row),
            indexValue(builder, loc, plan.validRows));
      if (!width) {
        Value colCondition = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::slt,
            arith::ConstantIndexOp::create(builder, loc, start),
            indexValue(builder, loc, plan.validWidth));
        condition = condition ? arith::AndIOp::create(builder, loc, condition,
                                                      colCondition)
                                    .getResult()
                              : colCondition;
      }
      OpBuilder::InsertionGuard guard(builder);
      if (condition) {
        auto ifOp = scf::IfOp::create(builder, loc, condition, false);
        builder.setInsertionPointToStart(ifOp.thenBlock());
      }
      if (failed(emit(row, start)))
        return failure();
    }
  }
  return success();
}

} // namespace

FailureOr<Value> lowerRank2Load(OpBuilder &builder, Operation *anchor,
                                Value ptr, RankedTensorType type,
                                ArrayRef<OpFoldResult> mask, Value other,
                                Workspace &workspace) {
  auto plan = planAccess(anchor, ptr, type, mask, builder);
  if (failed(plan))
    return failure();
  Location loc = anchor->getLoc();
  auto spm = workspace.allocate(
      builder, anchor,
      MemRefType::get({plan->rows * plan->spmPitch}, type.getElementType()));
  if (failed(spm))
    return failure();
  if (other && failed(fillVectors(builder, loc, *spm, other)))
    return failure();
  if (failed(
          emitPackets(builder, anchor, *plan, [&](int64_t row, int64_t start) {
            Value ddr = rowView(builder, loc, ptr, plan->fastAxis, row, start,
                                plan->chunk);
            Value spmRow = slice1D(builder, loc, *spm,
                                   row * plan->spmPitch + start, plan->chunk);
            emitDmaAndWait(builder, loc, ddr, spmRow,
                           plan->chunk *
                               getSpmElementByteWidth(type.getElementType()));
            return success();
          })))
    return failure();
  SmallVector<int64_t> logicalStrides(2);
  logicalStrides[plan->fastAxis] = 1;
  logicalStrides[1 - plan->fastAxis] = plan->spmPitch;
  auto logicalType = MemRefType::get(
      type.getShape(), type.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), 0, logicalStrides),
      cast<MemRefType>((*spm).getType()).getMemorySpace());
  Value logical = memref::ReinterpretCastOp::create(
      builder, loc, logicalType, *spm, 0, type.getShape(), logicalStrides);
  return bufferization::ToTensorOp::create(builder, loc, type, logical, true,
                                           true)
      .getResult();
}

LogicalResult lowerRank2Store(OpBuilder &builder, Operation *anchor, Value ptr,
                              Value tensor, ArrayRef<OpFoldResult> mask,
                              Workspace &workspace) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto plan = planAccess(anchor, ptr, type, mask, builder);
  if (failed(plan))
    return failure();
  Location loc = anchor->getLoc();
  auto source = getStoreBuffer(builder, anchor, tensor);
  if (failed(source))
    return failure();
  auto sourceType = cast<MemRefType>(source->getType());
  if (getSpmBank(sourceType)) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(sourceType.getStridesAndOffset(strides, offset)) ||
        strides[plan->fastAxis] != 1)
      return anchor->emitError("BOSCAME direct SPM store needs a matching "
                               "unit-stride physical axis");
    return emitPackets(builder, anchor, *plan, [&](int64_t row, int64_t start) {
      Value ddr =
          rowView(builder, loc, ptr, plan->fastAxis, row, start, plan->chunk);
      Value spm = rowView(builder, loc, *source, plan->fastAxis, row, start,
                          plan->chunk);
      emitDmaAndWait(builder, loc, spm, ddr,
                     plan->chunk *
                         getSpmElementByteWidth(type.getElementType()));
      return success();
    });
  }
  Value ordinary = *source;
  auto spm = workspace.allocate(
      builder, anchor,
      MemRefType::get({plan->rows * plan->spmPitch}, type.getElementType()));
  if (failed(spm))
    return failure();
  return emitPackets(builder, anchor, *plan, [&](int64_t row, int64_t start) {
    Value ddr =
        rowView(builder, loc, ptr, plan->fastAxis, row, start, plan->chunk);
    Value spmRow =
        slice1D(builder, loc, *spm, row * plan->spmPitch + start, plan->chunk);
    Value source;
    if (plan->fastAxis == 1) {
      source = rowView(builder, loc, ordinary, 1, row, start, plan->chunk);
    } else {
      // Packing a strided tensor row may use scalar access in ordinary memory.
      // Only the subsequent complete vector transfer touches SPM.
      source = memref::AllocOp::create(
          builder, loc, MemRefType::get({plan->chunk}, type.getElementType()),
          builder.getI64IntegerAttr(64));
      for (int64_t k = 0; k < plan->chunk; ++k) {
        Value i = arith::ConstantIndexOp::create(builder, loc, start + k);
        Value j = arith::ConstantIndexOp::create(builder, loc, row);
        Value value =
            memref::LoadOp::create(builder, loc, ordinary, ValueRange{i, j});
        memref::StoreOp::create(
            builder, loc, value, source,
            ValueRange{arith::ConstantIndexOp::create(builder, loc, k)});
      }
    }
    emitDdrPolicy(builder, loc, false);
    if (failed(copyVectors(builder, loc, source, spmRow, plan->chunk)))
      return failure();
    emitDmaAndWait(builder, loc, spmRow, ddr,
                   plan->chunk * getSpmElementByteWidth(type.getElementType()));
    return success();
  });
}

} // namespace buddy::boscame::lowering
