//====- AssignPhysicalBanksPass.cpp - Map virtual handles to physical banks -===//
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
// Assign physical banks for Bank-SSA ops.
// - Input ops: bb_bank_alloc/release/mvin/mvout/mul_warp16.
// - Output ops: bb_mset/mvin/mvout/mul_warp16.
// - Model: 16 banks, each 16KB. row*col consumes contiguous banks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace buddy;

namespace {

struct BankSlot {
  int64_t base = -1;
  int64_t row = 1;
  int64_t col = 1;
};

static uint64_t fieldBits(uint64_t val, int startBit, int endBit) {
  uint64_t width = endBit - startBit + 1;
  uint64_t mask = (1ULL << width) - 1;
  return (val & mask) << startBit;
}

static Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(v));
}

static buckyball::MsetOp createMset(OpBuilder &b, Location loc, uint64_t bankId,
                                    bool alloc, uint64_t row, uint64_t col) {
  auto op = b.create<buckyball::MsetOp>(loc, cstI64(b, loc, bankId));
  op->setAttr("alloc", b.getBoolAttr(alloc));
  op->setAttr("row", b.getI64IntegerAttr(row));
  op->setAttr("col", b.getI64IntegerAttr(col));
  return op;
}

class AssignPhysicalBanksPass
    : public PassWrapper<AssignPhysicalBanksPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignPhysicalBanksPass)
  AssignPhysicalBanksPass() = default;
  AssignPhysicalBanksPass(const AssignPhysicalBanksPass &) {}
  StringRef getArgument() const final { return "assign-physical-banks"; }
  StringRef getDescription() const final {
    return "Assign physical banks for bank-SSA ops.";
  }

  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Number of physical banks."),
                          llvm::cl::init(16)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder b(func.getContext());
    llvm::DenseMap<int64_t, BankSlot> vm;
    llvm::SmallVector<int8_t, 32> used(bankNum, 0);
    auto getConstI64 = [&](Value v) -> std::optional<int64_t> {
      auto c = v.getDefiningOp<arith::ConstantOp>();
      if (!c)
        return std::nullopt;
      auto a = dyn_cast<IntegerAttr>(c.getValue());
      if (!a)
        return std::nullopt;
      return a.getInt();
    };


    auto tryAlloc = [&](int64_t row, int64_t col) -> std::optional<int64_t> {
      int64_t need = row * col;
      for (int64_t s = 0; s + need <= bankNum; ++s) {
        bool ok = true;
        for (int64_t i = 0; i < need; ++i) {
          if (used[s + i]) {
            ok = false;
            break;
          }
        }
        if (!ok)
          continue;
        for (int64_t i = 0; i < need; ++i)
          used[s + i] = 1;
        return s;
      }
      return std::nullopt;
    };

    auto freeAlloc = [&](const BankSlot &s) {
      int64_t need = s.row * s.col;
      for (int64_t i = 0; i < need; ++i)
        used[s.base + i] = 0;
    };

    for (Block &blk : func.getBlocks()) {
      for (Operation &op : llvm::make_early_inc_range(blk.getOperations())) {
        b.setInsertionPoint(&op);
        Location loc = op.getLoc();

        if (auto a = dyn_cast<buckyball::BankAllocOp>(op)) {
          int64_t row = a.getRow();
          int64_t col = a.getCol();
          if (row <= 0 || col <= 0)
            return signalPassFailure();
          auto base = tryAlloc(row, col);
          if (!base) {
            op.emitError("assign-physical-banks: out of physical banks");
            return signalPassFailure();
          }
          Value bid = cstI64(b, loc, static_cast<uint64_t>(*base));
          createMset(b, loc, static_cast<uint64_t>(*base), true, row, col);
          vm[*base] = BankSlot{*base, row, col};
          a.getBank().replaceAllUsesWith(bid);
          op.erase();
          continue;
        }

        if (auto r = dyn_cast<buckyball::BankReleaseOp>(op)) {
          auto bid = getConstI64(r.getBank());
          if (!bid) {
            op.emitError("release expects constant bank id after assignment");
            return signalPassFailure();
          }
          auto it = vm.find(*bid);
          if (it == vm.end()) {
            op.emitError("release of unknown virtual bank handle");
            return signalPassFailure();
          }
          createMset(b, loc, static_cast<uint64_t>(it->second.base), false, 0, 0);
          freeAlloc(it->second);
          vm.erase(it);
          op.erase();
          continue;
        }

        if (auto mv = dyn_cast<buckyball::BankMvinOp>(op)) {
          b.create<buckyball::MvinOp>(loc, mv.getInput(), mv.getBank(), mv.getDepth(),
                                      mv.getStride());
          mv.getBankOut().replaceAllUsesWith(mv.getBank());
          op.erase();
          continue;
        }

        if (auto mv = dyn_cast<buckyball::BankMvoutOp>(op)) {
          b.create<buckyball::MvoutOp>(loc, mv.getOutput(), mv.getBank(),
                                       mv.getDepth(), mv.getStride());
          mv.getBankOut().replaceAllUsesWith(mv.getBank());
          op.erase();
          continue;
        }

        if (auto mm = dyn_cast<buckyball::BankMulWarp16Op>(op)) {
          b.create<buckyball::MulWarp16Op>(loc, mm.getOp1Bank(), mm.getOp2Bank(),
                                           mm.getWrBank(), mm.getIter(),
                                           mm.getMode());
          mm.getWrBankOut().replaceAllUsesWith(mm.getWrBank());
          op.erase();
          continue;
        }

        if (auto tp = dyn_cast<buckyball::BankTransposeOp>(op)) {
          auto in = getConstI64(tp.getInBank());
          auto out = getConstI64(tp.getOutBank());
          auto cols = getConstI64(tp.getCols());
          if (!in || !out || !cols) {
            op.emitError("bank_transpose expects constant in/out/cols after assignment");
            return signalPassFailure();
          }
          uint64_t rs1 = fieldBits(*in, 0, 9) | fieldBits(*out, 20, 29) |
                         fieldBits(*cols, 30, 63);
          b.create<buckyball::Transpose_IntrOp>(loc, cstI64(b, loc, rs1), tp.getMode());
          tp.getOutBankOut().replaceAllUsesWith(tp.getOutBank());
          op.erase();
          continue;
        }

        if (auto im = dyn_cast<buckyball::BankIm2colOp>(op)) {
          auto in = getConstI64(im.getInBank());
          auto out = getConstI64(im.getOutBank());
          if (!in || !out) {
            op.emitError("bank_im2col expects constant in/out after assignment");
            return signalPassFailure();
          }
          uint64_t rs1 = fieldBits(*in, 0, 9) | fieldBits(*out, 20, 29);
          Value rs2 = im.getKCol();
          rs2 = b.create<arith::OrIOp>(
              loc, rs2,
              b.create<arith::ShLIOp>(loc, im.getKRow(), cstI64(b, loc, 4)));
          rs2 = b.create<arith::OrIOp>(
              loc, rs2,
              b.create<arith::ShLIOp>(loc, im.getInCol(), cstI64(b, loc, 8)));
          rs2 = b.create<arith::OrIOp>(
              loc, rs2,
              b.create<arith::ShLIOp>(loc, im.getInRow(), cstI64(b, loc, 13)));
          rs2 = b.create<arith::OrIOp>(
              loc, rs2,
              b.create<arith::ShLIOp>(loc, im.getStartCol(), cstI64(b, loc, 23)));
          rs2 = b.create<arith::OrIOp>(
              loc, rs2,
              b.create<arith::ShLIOp>(loc, im.getStartRow(), cstI64(b, loc, 28)));
          b.create<buckyball::Im2col_IntrOp>(loc, cstI64(b, loc, rs1), rs2);
          im.getOutBankOut().replaceAllUsesWith(im.getOutBank());
          op.erase();
          continue;
        }

        if (auto q = dyn_cast<buckyball::BankQuantOp>(op)) {
          auto in = getConstI64(q.getInBank());
          auto out = getConstI64(q.getOutBank());
          auto rows = getConstI64(q.getRows());
          if (!in || !out || !rows) {
            op.emitError("bank_quant expects constant in/out/rows after assignment");
            return signalPassFailure();
          }
          uint64_t rs1 = fieldBits(*in, 0, 9) | fieldBits(*out, 20, 29) |
                         fieldBits(*rows, 30, 63);
          Value scaleBits = b.create<arith::BitcastOp>(loc, b.getI32Type(), q.getScale());
          Value scale64 = b.create<arith::ExtUIOp>(loc, b.getI64Type(), scaleBits);
          Value rs2 = b.create<arith::AndIOp>(loc, scale64, cstI64(b, loc, 0xFFFFFFFFULL));
          b.create<buckyball::Quant_IntrOp>(loc, cstI64(b, loc, rs1), rs2);
          q.getOutBankOut().replaceAllUsesWith(q.getOutBank());
          op.erase();
          continue;
        }

        if (auto dq = dyn_cast<buckyball::BankDequantOp>(op)) {
          auto in = getConstI64(dq.getInBank());
          auto out = getConstI64(dq.getOutBank());
          auto rows = getConstI64(dq.getRows());
          if (!in || !out || !rows) {
            op.emitError("bank_dequant expects constant in/out/rows after assignment");
            return signalPassFailure();
          }
          uint64_t rs1 = fieldBits(*in, 0, 9) | fieldBits(*out, 20, 29) |
                         fieldBits(*rows, 30, 63);
          Value scaleBits =
              b.create<arith::BitcastOp>(loc, b.getI32Type(), dq.getScale());
          Value scale64 = b.create<arith::ExtUIOp>(loc, b.getI64Type(), scaleBits);
          Value rs2 = b.create<arith::AndIOp>(loc, scale64, cstI64(b, loc, 0xFFFFFFFFULL));
          b.create<buckyball::Dequant_IntrOp>(loc, cstI64(b, loc, rs1), rs2);
          dq.getOutBankOut().replaceAllUsesWith(dq.getOutBank());
          op.erase();
          continue;
        }
      }
    }

    if (!vm.empty()) {
      func.emitError("assign-physical-banks: leaked virtual bank handles");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
} // namespace buddy
} // namespace mlir
