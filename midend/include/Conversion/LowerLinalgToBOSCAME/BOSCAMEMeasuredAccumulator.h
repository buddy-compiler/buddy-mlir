//===- BOSCAMEMeasuredAccumulator.h - Measured accumulator source key -----===//
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
// Source-key matching for the first complete-adapter accumulator measurements.
// Profiles identify which HotAccumulator shape was timed; they are not a
// general cost model and do not by themselves authorize SPM window selection.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMEASUREDACCUMULATOR_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOSCAMEMEASUREDACCUMULATOR_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEHotAccumulator.h"
#include "llvm/ADT/STLExtras.h"

namespace buddy::boscame {

// Closed adapter epilogue shapes that have published timing evidence.
enum class MeasuredAccumulatorMode { Overwrite, NormalPostAdd };

// Source facts for the first complete-adapter measurements. These are not
// timings, a general cost model, or permission to select an SPM window.
struct MeasuredAccumulatorProfile {
  MeasuredAccumulatorMode mode;
  int64_t trips;
  int64_t pitch;
  HotAccumulator hot;
};

namespace measured_accumulator_detail {

inline bool constant(mlir::Value value, int64_t expected, unsigned width) {
  auto op = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto attr = op ? llvm::dyn_cast<mlir::IntegerAttr>(op.getValue())
                 : mlir::IntegerAttr();
  return attr && value.getType().isSignlessInteger(width) &&
         attr.getInt() == expected;
}

inline bool layout(mlir::Value value, llvm::ArrayRef<int64_t> shape,
                   llvm::ArrayRef<int64_t> expectedStrides, int64_t offset,
                   unsigned width) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(value.getType());
  llvm::SmallVector<int64_t> strides;
  int64_t actualOffset;
  return type && type.hasStaticShape() && !type.getMemorySpace() &&
         type.getElementType().isSignlessInteger(width) &&
         type.getShape() == shape &&
         mlir::succeeded(type.getStridesAndOffset(strides, actualOffset)) &&
         actualOffset == offset &&
         llvm::ArrayRef<int64_t>(strides) == expectedStrides;
}

inline bool allocation(mlir::Value value) {
  auto alloc = value.getDefiningOp<mlir::memref::AllocOp>();
  auto alignment = alloc ? alloc->getAttrOfType<mlir::IntegerAttr>("alignment")
                         : mlir::IntegerAttr();
  return alloc && !alloc->getNumOperands() && alignment &&
         alignment.getInt() == 64 && layout(value, {32, 64}, {64, 1}, 0, 32);
}

inline bool externalView(mlir::Value value, mlir::Value argument,
                         llvm::ArrayRef<int64_t> shape,
                         llvm::ArrayRef<int64_t> strides,
                         mlir::Value dynamicOffset, unsigned width) {
  auto view = value.getDefiningOp<mlir::memref::ReinterpretCastOp>();
  if (!view || view.getSource() != argument || view.getStaticSizes() != shape ||
      view.getStaticStrides() != strides ||
      !layout(value, shape, strides,
              dynamicOffset ? mlir::ShapedType::kDynamic : 0, width))
    return false;
  auto offsets = view.getMixedOffsets();
  if (offsets.size() != 1)
    return false;
  if (dynamicOffset)
    return llvm::dyn_cast<mlir::Value>(offsets[0]) == dynamicOffset;
  return mlir::getConstantIntValue(offsets[0]) == std::optional<int64_t>(0);
}

inline bool flattened(mlir::Value value, mlir::Value source) {
  auto collapse = value.getDefiningOp<mlir::memref::CollapseShapeOp>();
  if (!collapse || collapse.getSrc() != source ||
      !layout(value, {2048}, {1}, 0, 32))
    return false;
  auto groups = collapse.getReassociationIndices();
  return groups.size() == 1 &&
         llvm::ArrayRef<int64_t>(groups[0]) == llvm::ArrayRef<int64_t>({0, 1});
}

inline bool zeroFill(mlir::linalg::FillOp fill, mlir::Value storage) {
  auto contract =
      fill ? fill->getAttrOfType<mlir::DictionaryAttr>("contract.fill")
           : mlir::DictionaryAttr();
  auto padding =
      contract ? contract.getAs<mlir::BoolAttr>("padding") : mlir::BoolAttr();
  if (!fill || !fill.hasPureBufferSemantics() || fill->getNumResults() ||
      (padding && padding.getValue()) || fill.getInputs().size() != 1 ||
      fill.getOutputs().size() != 1 || fill.getOutputs()[0] != storage ||
      !constant(fill.getInputs()[0], 0, 32))
    return false;
  mlir::Block *body = fill.getBody();
  auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(body->getTerminator());
  return body->getNumArguments() == 2 && body->getOperations().size() == 1 &&
         yield && yield.getValues().size() == 1 &&
         yield.getValues()[0] == body->getArgument(0);
}

// Validate the named operation's actual body too: a generic-form named op
// must not smuggle a different payload into an otherwise matching profile.
inline bool signedMatmul(mlir::linalg::MatmulOp matmul, mlir::Value storage) {
  using namespace mlir;
  auto compute = matmul
                     ? matmul->getAttrOfType<DictionaryAttr>("contract.compute")
                     : DictionaryAttr();
  if (!matmul || !matmul.hasPureBufferSemantics() || matmul->getNumResults() ||
      (compute && compute.getAs<StringAttr>("order")) ||
      matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      matmul.getOutputs()[0] != storage ||
      matmul.getCast() != linalg::TypeFn::cast_signed ||
      matmul.getIndexingMapsArray() !=
          linalg::MatmulOp::getDefaultIndexingMaps(matmul.getContext()))
    return false;
  Block *body = matmul.getBody();
  if (body->getNumArguments() != 3 || body->getOperations().size() != 5 ||
      !body->getArgument(0).getType().isInteger(8) ||
      !body->getArgument(1).getType().isInteger(8) ||
      !body->getArgument(2).getType().isInteger(32))
    return false;
  auto it = body->begin();
  auto a = dyn_cast<arith::ExtSIOp>(*it++);
  auto b = dyn_cast<arith::ExtSIOp>(*it++);
  auto product = dyn_cast<arith::MulIOp>(*it++);
  auto sum = dyn_cast<arith::AddIOp>(*it++);
  auto yield = dyn_cast<linalg::YieldOp>(*it);
  return a && b && product && sum && yield &&
         a.getIn() == body->getArgument(0) && a.getType().isInteger(32) &&
         b.getIn() == body->getArgument(1) && b.getType().isInteger(32) &&
         product.getLhs() == a.getResult() &&
         product.getRhs() == b.getResult() &&
         product.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         sum.getLhs() == body->getArgument(2) &&
         sum.getRhs() == product.getResult() &&
         sum.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == sum.getResult();
}

inline bool moduloAdd(mlir::linalg::GenericOp generic, mlir::Value storage,
                      mlir::Value &old) {
  using namespace mlir;
  if (!generic || !generic.hasPureBufferSemantics() ||
      generic->getNumResults() || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic.getNumLoops() != 1 ||
      generic.getNumParallelLoops() != 1 ||
      generic.getIndexingMapsArray().size() != 3 ||
      !llvm::all_of(generic.getIndexingMapsArray(), [](AffineMap map) {
        return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
               map.isIdentity();
      }))
    return false;
  auto oldView =
      generic.getInputs()[1].getDefiningOp<memref::CollapseShapeOp>();
  if (!oldView)
    return false;
  old = oldView.getSrc();
  if (old == storage || !allocation(old) ||
      !flattened(generic.getInputs()[0], storage) ||
      !flattened(generic.getInputs()[1], old) ||
      !flattened(generic.getOutputs()[0], storage) ||
      generic.getInputs()[0] == generic.getOutputs()[0])
    return false;
  Block *body = generic.getBody();
  if (body->getNumArguments() != 3 || body->getOperations().size() != 2 ||
      !llvm::all_of(body->getArgumentTypes(),
                    [](Type type) { return type.isInteger(32); }))
    return false;
  auto sum = dyn_cast<arith::AddIOp>(body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->back());
  return sum && yield &&
         sum.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         sum.getLhs() == body->getArgument(0) &&
         sum.getRhs() == body->getArgument(1) &&
         yield.getValues().size() == 1 &&
         yield.getValues()[0] == sum.getResult();
}

} // namespace measured_accumulator_detail

// Match only the measured, closed two-observer adapter after ordinary-IR
// planning rewrites and before placement decisions. Never consult plan.*,
// schedule, cost or profile attributes, including when verifying a plan.
// Unknown work, alternate aliases, control flow and additional candidates
// deliberately fail to match instead of extrapolating isolated timings.
inline std::optional<MeasuredAccumulatorProfile>
analyzeMeasuredAccumulator(mlir::Value storage) {
  using namespace mlir;
  namespace detail = measured_accumulator_detail;
  auto hot = analyzeHotAccumulator(storage);
  auto allocation = storage.getDefiningOp<memref::AllocOp>();
  auto function =
      allocation ? allocation->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!hot || !function || !detail::allocation(storage) ||
      (hot->trips != 6 && hot->trips != 24) ||
      function.getBody().getBlocks().size() != 1 ||
      allocation->getBlock() != &function.getBody().front() ||
      function.getNumArguments() != 4 ||
      function.getFunctionType().getNumResults() != 0)
    return std::nullopt;
  for (unsigned i = 0; i != 4; ++i) {
    auto type = dyn_cast<UnrankedMemRefType>(function.getArgument(i).getType());
    if (!type || type.getMemorySpace() ||
        !type.getElementType().isSignlessInteger(i < 2 ? 8 : 32))
      return std::nullopt;
  }
  auto module = function->getParentOfType<ModuleOp>();
  auto abi = module ? module->getAttrOfType<DictionaryAttr>("contract.abi")
                    : DictionaryAttr();
  auto alignment =
      abi ? abi.getAs<IntegerAttr>("args_aligned_bytes") : IntegerAttr();
  auto disjoint = abi ? abi.getAs<BoolAttr>("args_disjoint") : BoolAttr();
  if (!alignment || alignment.getInt() != 64 || !disjoint ||
      !disjoint.getValue())
    return std::nullopt;

  linalg::MatmulOp matmul;
  for (Operation *user : storage.getUsers())
    if (auto candidate = dyn_cast<linalg::MatmulOp>(user))
      matmul = candidate;
  auto loop =
      matmul ? dyn_cast<scf::ForOp>(matmul->getParentOp()) : scf::ForOp();
  if (!loop || !detail::signedMatmul(matmul, storage) ||
      loop->getBlock() != allocation->getBlock() || loop.getNumResults() ||
      loop.getNumRegionIterArgs() || loop->hasAttr("unsignedCmp") ||
      !detail::constant(loop.getLowerBound(), 0, 32) ||
      !detail::constant(loop.getUpperBound(), hot->trips, 32) ||
      !detail::constant(loop.getStep(), 1, 32) ||
      loop.getBody()->getOperations().size() != 6)
    return std::nullopt;
  auto it = loop.getBody()->begin();
  auto multiply = dyn_cast<arith::MulIOp>(*it++);
  auto cast = dyn_cast<arith::IndexCastOp>(*it++);
  auto aView = dyn_cast<memref::ReinterpretCastOp>(*it++);
  auto bView = dyn_cast<memref::ReinterpretCastOp>(*it++);
  Operation *bodyMatmul = &*it++;
  auto yield = dyn_cast<scf::YieldOp>(*it);
  int64_t pitch = hot->trips * 128;
  if (!multiply || !cast || !aView || !bView || bodyMatmul != matmul ||
      !yield || yield->getNumOperands() ||
      multiply.getLhs() != loop.getInductionVar() ||
      !detail::constant(multiply.getRhs(), 128, 32) ||
      multiply.getOverflowFlags() != arith::IntegerOverflowFlags::none ||
      cast.getIn() != multiply.getResult() || !cast.getType().isIndex() ||
      !detail::externalView(aView.getResult(), function.getArgument(0),
                            {32, 128}, {pitch, 1}, cast.getResult(), 8) ||
      !detail::externalView(bView.getResult(), function.getArgument(1),
                            {128, 64}, {1, pitch}, cast.getResult(), 8) ||
      matmul.getInputs()[0] != aView.getResult() ||
      matmul.getInputs()[1] != bView.getResult())
    return std::nullopt;

  linalg::FillOp fill;
  linalg::GenericOp generic;
  SmallVector<memref::CopyOp> copies;
  memref::DeallocOp dealloc, oldDealloc;
  func::ReturnOp returnOp;
  for (Operation &op : function.getBody().front()) {
    if (auto candidate = dyn_cast<linalg::FillOp>(op)) {
      if (fill)
        return std::nullopt;
      fill = candidate;
    } else if (auto candidate = dyn_cast<linalg::GenericOp>(op)) {
      if (generic)
        return std::nullopt;
      generic = candidate;
    } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      copies.push_back(copy);
    } else if (auto candidate = dyn_cast<memref::DeallocOp>(op)) {
      if (candidate.getMemref() == storage) {
        if (dealloc)
          return std::nullopt;
        dealloc = candidate;
      } else {
        if (oldDealloc)
          return std::nullopt;
        oldDealloc = candidate;
      }
    } else if (auto candidate = dyn_cast<func::ReturnOp>(op)) {
      if (returnOp || candidate->getNumOperands())
        return std::nullopt;
      returnOp = candidate;
    }
  }
  bool postAdd = bool(generic);
  if (!detail::zeroFill(fill, storage) || !dealloc || !returnOp ||
      copies.size() != (postAdd ? 3u : 2u))
    return std::nullopt;
  auto cCopy = copies[postAdd ? 1 : 0];
  auto snapshotCopy = copies[postAdd ? 2 : 1];
  Value c = cCopy.getTarget(), snapshot = snapshotCopy.getTarget();
  if (cCopy.getSource() != storage || snapshotCopy.getSource() != storage ||
      !detail::externalView(c, function.getArgument(2), {32, 64}, {pitch, 1},
                            Value(), 32) ||
      !detail::externalView(snapshot, function.getArgument(3), {32, 64},
                            {64, 1}, Value(), 32))
    return std::nullopt;

  SmallVector<Operation *> events{allocation, fill, loop};
  llvm::DenseSet<Operation *> allowed{c.getDefiningOp(),
                                      snapshot.getDefiningOp()};
  if (postAdd) {
    Value old;
    if (!detail::moduloAdd(generic, storage, old) || !oldDealloc ||
        oldDealloc.getMemref() != old || copies[0].getSource() != c ||
        copies[0].getTarget() != old)
      return std::nullopt;
    events.append({old.getDefiningOp(), copies[0], generic, oldDealloc});
    allowed.insert(generic.getInputs()[0].getDefiningOp());
    allowed.insert(generic.getInputs()[1].getDefiningOp());
    allowed.insert(generic.getOutputs()[0].getDefiningOp());
  } else if (oldDealloc) {
    return std::nullopt;
  }
  events.append({cCopy, snapshotCopy, dealloc, returnOp});
  allowed.insert(events.begin(), events.end());
  unsigned next = 0;
  for (Operation &op : function.getBody().front()) {
    if (auto scalar = dyn_cast<arith::ConstantOp>(op)) {
      // Dead integer constants left by ordinary planning have no recipe cost.
      // Every live scalar must feed the closed graph reconstructed above.
      if (!isa<IntegerAttr>(scalar.getValue()) ||
          !llvm::all_of(scalar.getResult().getUsers(), [&](Operation *user) {
            return allowed.contains(user) || user == multiply;
          }))
        return std::nullopt;
      continue;
    }
    if (!allowed.contains(&op))
      return std::nullopt;
    if (next != events.size() && events[next] == &op)
      ++next;
    else if (llvm::is_contained(events, &op))
      return std::nullopt;
  }
  if (next != events.size())
    return std::nullopt;
  return MeasuredAccumulatorProfile{postAdd
                                        ? MeasuredAccumulatorMode::NormalPostAdd
                                        : MeasuredAccumulatorMode::Overwrite,
                                    hot->trips, pitch, *hot};
}

} // namespace buddy::boscame
#endif
