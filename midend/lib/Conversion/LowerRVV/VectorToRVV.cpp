#include "RVV/Transforms.h"

#include "RVV/RVVDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
bool isVector(Type type, Type element) {
  auto vector = dyn_cast<VectorType>(type);
  return vector && vector.getRank() == 1 && vector.getShape()[0] == 2 &&
         vector.isScalable() && vector.getElementType() == element;
}

Value prefixLength(Value mask) {
  auto vector = cast<VectorType>(mask.getType());
  if (vector.getRank() != 1 || vector.getShape()[0] != 2 ||
      !vector.isScalable())
    return {};
  if (auto create = mask.getDefiningOp<vector::CreateMaskOp>())
    return create.getOperand(0);
  auto compare = mask.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::ult)
    return {};
  auto step = compare.getLhs().getDefiningOp<vector::StepOp>();
  auto broadcast = compare.getRhs().getDefiningOp<vector::BroadcastOp>();
  if (!step || !broadcast || !broadcast.getSource().getType().isInteger(32))
    return {};
  return broadcast.getSource();
}

struct ActiveVector {
  Value mask;
  Value length;
};

struct VectorToRVVPass
    : public PassWrapper<VectorToRVVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorToRVVPass)
  StringRef getArgument() const final { return "vector-to-rvv"; }
  StringRef getDescription() const final {
    return "Lower prefix-masked FP32 vector kernels to Buddy RVV";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::rvv::RVVDialect, arith::ArithDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (!function->hasAttr("rvv.kernel"))
        continue;
      DenseMap<Value, ActiveVector> active;
      SmallVector<Operation *> operations;
      function.walk([&](Operation *op) { operations.push_back(op); });
      for (Operation *op : operations) {
        OpBuilder builder(op);
        auto loc = op->getLoc();
        auto f32 = builder.getF32Type();
        auto checkMemory = [&](Value base, ValueRange indices) {
          auto memref = cast<MemRefType>(base.getType());
          return memref.getRank() == 1 && indices.size() == 1 &&
                 memref.getElementType().isF32() &&
                 memref.getLayout().isIdentity();
        };
        if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
          Value requested = prefixLength(load.getMask());
          if (!requested || !isVector(load.getType(), f32) ||
              !checkMemory(load.getBase(), load.getIndices())) {
            load.emitError(
                "requires a prefix mask and contiguous 1D scalable FP32 load");
            return signalPassFailure();
          }
          auto index = builder.getIndexType();
          auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
          auto two = arith::ConstantIndexOp::create(builder, loc, 2);
          auto width = vector::VectorScaleOp::create(builder, loc);
          Value capacity = arith::MulIOp::create(builder, loc, width, two);
          Value length = requested;
          if (length.getType().isIndex()) {
            length = arith::MaxSIOp::create(builder, loc, length, zero);
          } else {
            length = arith::IndexCastUIOp::create(builder, loc, index, length);
          }
          length = arith::MinUIOp::create(builder, loc, length, capacity);
          auto sew = arith::ConstantIndexOp::create(builder, loc, 2);
          auto lmul = arith::ConstantIndexOp::create(builder, loc, 0);
          Value vl = ::buddy::rvv::RVVSetVlOp::create(builder, loc, index,
                                                      length, sew, lmul);
          Value result = ::buddy::rvv::RVVLoadOp::create(
              builder, loc, load.getType(), load.getBase(),
              load.getIndices()[0], vl);
          active[result] = {load.getMask(), vl};
          load.getResult().replaceAllUsesWith(result);
          load.erase();
          continue;
        }
        if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
          auto found = active.find(store.getValueToStore());
          if (found == active.end() || found->second.mask != store.getMask() ||
              !checkMemory(store.getBase(), store.getIndices())) {
            store.emitError(
                "requires the same prefix mask as its active vector input");
            return signalPassFailure();
          }
          ::buddy::rvv::RVVStoreOp::create(
              builder, loc, store.getValueToStore(), store.getBase(),
              store.getIndices()[0], found->second.length);
          store.erase();
          continue;
        }
        if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(
                op) &&
            isa<VectorType>(op->getResult(0).getType())) {
          auto lhs = active.find(op->getOperand(0));
          auto rhs = active.find(op->getOperand(1));
          Value right = op->getOperand(1);
          auto splat = right.getDefiningOp<vector::BroadcastOp>();
          bool scalarRight = splat && splat.getSource().getType().isF32() &&
                             isVector(right.getType(), f32);
          if (lhs == active.end() ||
              (!scalarRight &&
               (rhs == active.end() || lhs->second.mask != rhs->second.mask))) {
            op->emitError("requires active lhs and rhs with the same prefix "
                          "mask or an FP32 scalar broadcast");
            return signalPassFailure();
          }
          if (scalarRight)
            right = splat.getSource();
          OperationState state(loc, isa<arith::AddFOp>(op)   ? "rvv.fadd"
                                    : isa<arith::SubFOp>(op) ? "rvv.fsub"
                                    : isa<arith::MulFOp>(op) ? "rvv.fmul"
                                                             : "rvv.fdiv");
          state.addOperands({op->getOperand(0), right, lhs->second.length});
          state.addTypes(op->getResult(0).getType());
          Value result = builder.create(state)->getResult(0);
          active[result] = lhs->second;
          op->getResult(0).replaceAllUsesWith(result);
          op->erase();
        }
      }
      // Mask construction and passthrough constants are dead after lowering.
      bool erased;
      do {
        erased = false;
        SmallVector<Operation *> dead;
        function.walk([&](Operation *op) {
          if (op->getNumRegions() == 0 && op->getNumResults() &&
              op->use_empty() && isMemoryEffectFree(op))
            dead.push_back(op);
        });
        for (Operation *op : llvm::reverse(dead)) {
          op->erase();
          erased = true;
        }
      } while (erased);
      WalkResult result = function.walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() == "rvv" ||
            isa<vector::VectorScaleOp>(op))
          return WalkResult::advance();
        if (llvm::any_of(op->getOperandTypes(),
                         [](Type t) { return isa<VectorType>(t); }) ||
            llvm::any_of(op->getResultTypes(),
                         [](Type t) { return isa<VectorType>(t); })) {
          op->emitError("unsupported vector operation in rvv.kernel");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return signalPassFailure();
    }
  }
};
} // namespace

void mlir::buddy::registerVectorToRVVPass() {
  PassRegistration<VectorToRVVPass>();
}
