//===-- VIROps.cpp - Dynamic Vector IR Operation Implementation -----------===//
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
// This file implements the operation set for the Dynamic Vector IR (VIR)
// dialect, including vector length management and dynamic vector operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "VIR/VIRDialect.h"
#include "VIR/VIROps.h"

#define GET_OP_CLASSES
#include "VIR/VIR.cpp.inc"

using namespace mlir;
using namespace buddy::vir;

LogicalResult SetVLOp::verify() {
  // the region has one block (SizedRegion<1>, verified before)
  if (getRegion().front().getNumArguments() != 0)
    return emitOpError("region must not have block arguments");
  if (getNumResults() != 0)
    return success();
  // Without results the region runs once per chunk of lanes, chunks the
  // lowering chooses, so its memory effects must not depend on them. Only
  // these ops may have one: VIR ops (vir.store writes its own lanes), reads,
  // scf.for / affine.for (their ops are checked; other ops with nested ops,
  // e.g. scf.if, are not lowered), and the memref.store of an accumulator:
  // the result of a vir.reduce whose accumulator is a memref.load, in the
  // region, of the same memref value and indices (each chunk combines its
  // lanes with what the chunks before stored). Any other write, allocation
  // or free, and an op of unknown memory effects, is rejected: whatever the
  // memory it reaches (a view made in the region may alias memory from
  // outside), and the lowering of the chunks does not handle it.
  Region &region = getRegion();
  auto reject = [&](Operation *op, const Twine &why) {
    op->emitOpError(why) << " in a vir.set_vl without results: the region "
                            "runs once per chunk of lanes, chunks the "
                            "lowering chooses";
    return WalkResult::interrupt();
  };
  WalkResult walk = region.walk([&](Operation *op) {
    if (isa<VIRDialect>(op->getDialect()))
      return WalkResult::advance();
    // the loops the lowering of the chunks lowers (their ops are checked)
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      StringRef name = op->getName().getStringRef();
      if (name == "scf.for" || name == "affine.for")
        return WalkResult::advance();
      return reject(op, "has nested ops the lowering does not handle");
    }
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return reject(op, "has unknown memory effects");
    SmallVector<MemoryEffects::EffectInstance> all;
    effects.getEffects(all);
    if (llvm::all_of(all, [](auto &e) {
          return isa<MemoryEffects::Read>(e.getEffect());
        }))
      return WalkResult::advance();
    auto store = dyn_cast<memref::StoreOp>(op);
    if (!store)
      return reject(op, "writes, allocates or frees memory");
    auto reduce = store.getValue().getDefiningOp<ReduceOp>();
    if (!reduce)
      return reject(op, "stores a value that is not the result of a "
                        "vir.reduce");
    auto load = reduce.getAcc().getDefiningOp<memref::LoadOp>();
    if (!load || !region.isAncestor(load->getParentRegion()) ||
        load.getMemRef() != store.getMemRef() ||
        load.getIndices() != store.getIndices())
      return reject(op, "stores a vir.reduce whose accumulator is not a "
                        "memref.load in the region of the element stored");
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

// One index per memref dim or none, and a dim of the memref per dim of the
// dynamic vector.
static LogicalResult verifyAccess(Operation *op, MemRefType memref,
                                  ValueRange indices, DynamicVectorType vec) {
  if (!indices.empty() && int64_t(indices.size()) != memref.getRank())
    return op->emitOpError("needs no index or one per memref dim, ")
           << memref.getRank();
  if (int64_t(vec.getShape().size()) > memref.getRank())
    return op->emitOpError("needs a memref of at least the rank of the "
                           "dynamic vector, ")
           << vec.getShape().size();
  return success();
}

LogicalResult LoadOp::verify() {
  return verifyAccess(*this, getBase().getType(), getIndices(), getType());
}

LogicalResult StoreOp::verify() {
  return verifyAccess(*this, getBase().getType(), getIndices(),
                      getValue().getType());
}

LogicalResult ReduceOp::verify() {
  Type elem = getInput().getType().getElementType();
  if (getAcc().getType() != elem || getResult().getType() != elem)
    return emitOpError("expects the accumulator and the result of the "
                       "element type of the input, ")
           << elem;
  StringRef kind = getKind();
  bool isFloat = isa<FloatType>(elem);
  bool isInt = isa<IntegerType>(elem);
  if (llvm::is_contained({"add", "mul"}, kind) && (isFloat || isInt))
    return success();
  if (llvm::is_contained(
          {"minsi", "minui", "maxsi", "maxui", "and", "or", "xor"}, kind))
    return isInt ? success()
                 : emitOpError("expects integers for kind ") << kind;
  if (llvm::is_contained({"minnumf", "maxnumf", "minimumf", "maximumf",
                          "minimumnumf", "maximumnumf", "maxnum"},
                         kind))
    return isFloat ? success()
                   : emitOpError("expects floats for kind ") << kind;
  return emitOpError("unknown kind ") << kind;
}

LogicalResult ShuffleOp::verify() {
  ArrayRef<int64_t> shape = getV1().getType().getShape();
  if (shape.size() < 2)
    return emitOpError("needs a dynamic vector with a leading static dim");
  int64_t rows = shape.front();
  if (int64_t(getMask().size()) != rows)
    return emitOpError("needs one mask index per row, ") << rows;
  for (int64_t index : getMask())
    if (index < 0 || index >= 2 * rows)
      return emitOpError("mask index ")
             << index << " is out of the " << 2 * rows << " rows";
  return success();
}
