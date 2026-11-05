//===- BOSCAMEOrderedMatmul.cpp - Ordered compatibility recipe ------------===//
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
// Expand a verified scalar matmul recipe into serial M/N/K loops, cloning the
// original payload and using one-element vectors for SPM element accesses.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEOrderedMatmul.h"
#include "Dialect/BOSCAME/SpmDma.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace {
struct ElementAccess {
  Value buffer;
  SmallVector<Value> indices;
};

ElementAccess unitStrideElement(OpBuilder &builder, Location loc, Value source,
                                ValueRange indices) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (succeeded(type.getStridesAndOffset(strides, offset))) {
    if (strides.back() == 1)
      return {source, llvm::to_vector(indices)};
    // A one-lane access has no vectorization order to change. Transposing
    // a column-major matrix and its indices preserves its exact address.
    if (type.getRank() == 2 && strides.front() == 1) {
      Value transposed = memref::TransposeOp::create(
          builder, loc, source,
          AffineMapAttr::get(AffineMap::getPermutationMap(
              ArrayRef<unsigned>{1, 0}, builder.getContext())));
      return {transposed, {indices[1], indices[0]}};
    }
  }

  // vector.load/store require a unit trailing stride even for vector<1>.
  // Rebuild only the addressed element, including the descriptor offset,
  // so general strided operands retain the scalar reference semantics.
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, source);
  Value address = metadata.getOffset();
  for (auto [index, stride] : llvm::zip(indices, metadata.getStrides())) {
    Value displacement =
        builder.createOrFold<arith::MulIOp>(loc, index, stride);
    address = builder.createOrFold<arith::AddIOp>(loc, address, displacement);
  }
  auto singletonType = MemRefType::get(
      {1}, type.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, {1}),
      type.getMemorySpace());
  Value singleton = memref::ReinterpretCastOp::create(
      builder, loc, singletonType, metadata.getBaseBuffer(), address,
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
  return {singleton, {arith::ConstantIndexOp::create(builder, loc, 0)}};
}

Value loadElement(OpBuilder &builder, Location loc, Value source,
                  ValueRange indices) {
  auto type = cast<MemRefType>(source.getType());
  if (!::buddy::boscame::spmdma::getSpmBank(type))
    return memref::LoadOp::create(builder, loc, source, indices);
  auto access = unitStrideElement(builder, loc, source, indices);
  Value vector = vector::LoadOp::create(
      builder, loc, VectorType::get({1}, type.getElementType()), access.buffer,
      access.indices);
  return vector::ExtractOp::create(builder, loc, vector, int64_t{0});
}

void storeElement(OpBuilder &builder, Location loc, Value value, Value target,
                  ValueRange indices) {
  auto type = cast<MemRefType>(target.getType());
  if (!::buddy::boscame::spmdma::getSpmBank(type)) {
    memref::StoreOp::create(builder, loc, value, target, indices);
    return;
  }
  Value vector = vector::BroadcastOp::create(
      builder, loc, VectorType::get({1}, type.getElementType()), value);
  auto access = unitStrideElement(builder, loc, target, indices);
  vector::StoreOp::create(builder, loc, vector, access.buffer, access.indices);
}
} // namespace

Operation *buddy::boscame::emitOrderedMatmul(linalg::MatmulOp matmul) {
  OpBuilder builder(matmul);
  Location loc = matmul.getLoc();
  Value a = matmul.getInputs()[0], b = matmul.getInputs()[1],
        c = matmul.getOutputs()[0];
  auto aType = cast<MemRefType>(a.getType());
  auto cType = cast<MemRefType>(c.getType());
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value mBound =
      arith::ConstantIndexOp::create(builder, loc, cType.getDimSize(0));
  Value nBound =
      arith::ConstantIndexOp::create(builder, loc, cType.getDimSize(1));
  Value kBound =
      arith::ConstantIndexOp::create(builder, loc, aType.getDimSize(1));
  auto mLoop = scf::ForOp::create(builder, loc, zero, mBound, one);
  builder.setInsertionPointToStart(mLoop.getBody());
  auto nLoop = scf::ForOp::create(builder, loc, zero, nBound, one);
  builder.setInsertionPointToStart(nLoop.getBody());
  auto kLoop = scf::ForOp::create(builder, loc, zero, kBound, one);
  builder.setInsertionPointToStart(kLoop.getBody());
  Value m = mLoop.getInductionVar(), n = nLoop.getInductionVar(),
        k = kLoop.getInductionVar();
  Value lhs = loadElement(builder, loc, a, ValueRange{m, k});
  Value rhs = loadElement(builder, loc, b, ValueRange{k, n});
  Value accumulator = loadElement(builder, loc, c, ValueRange{m, n});
  Block &payload = matmul.getRegion().front();
  IRMapping mapping;
  mapping.map(payload.getArguments(), ValueRange{lhs, rhs, accumulator});
  for (Operation &operation : payload.without_terminator())
    builder.clone(operation, mapping);
  auto yield = cast<linalg::YieldOp>(payload.getTerminator());
  storeElement(builder, loc, mapping.lookup(yield.getValues()[0]), c,
               ValueRange{m, n});
  return mLoop;
}
