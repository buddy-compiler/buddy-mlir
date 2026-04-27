//====- ReportBankUsagePass.cpp - Report physical bank usage ---------------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

using namespace mlir;
using namespace buddy;

namespace {

class ReportBankUsagePass
    : public PassWrapper<ReportBankUsagePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportBankUsagePass)
  ReportBankUsagePass() = default;
  ReportBankUsagePass(const ReportBankUsagePass &) {}

  StringRef getArgument() const final { return "report-bank-usage"; }
  StringRef getDescription() const final {
    return "Report physical bank occupancy from bb_mset alloc/release timeline.";
  }

  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Number of physical banks."),
                          llvm::cl::init(16)};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Print per-event timeline."),
                       llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (bankNum <= 0) {
      func.emitError("report-bank-usage: bank_num must be > 0");
      signalPassFailure();
      return;
    }

    llvm::SmallVector<int8_t, 32> used(bankNum, 0);
    llvm::DenseMap<int64_t, int64_t> allocSize;
    int64_t cur = 0;
    int64_t peak = 0;
    int64_t allocCnt = 0;
    int64_t relCnt = 0;
    int64_t evt = 0;

    auto getConstI64 = [&](Value v) -> std::optional<int64_t> {
      auto c = v.getDefiningOp<arith::ConstantOp>();
      if (!c)
        return std::nullopt;
      auto ai = dyn_cast<IntegerAttr>(c.getValue());
      if (!ai)
        return std::nullopt;
      return ai.getInt();
    };

    for (Block &blk : func.getBlocks()) {
      for (Operation &op : blk.getOperations()) {
        auto mset = dyn_cast<buckyball::MsetOp>(op);
        if (!mset)
          continue;
        ++evt;
        auto bid = getConstI64(mset.getBankId());
        if (!bid) {
          func.emitError("report-bank-usage: mset bank id must be constant");
          signalPassFailure();
          return;
        }
        if (*bid < 0 || *bid >= bankNum) {
          func.emitError("report-bank-usage: bank id out of range");
          signalPassFailure();
          return;
        }

        if (mset.getAlloc()) {
          int64_t row = mset.getRow();
          int64_t col = mset.getCol();
          int64_t need = row * col;
          if (row <= 0 || col <= 0 || need <= 0 || *bid + need > bankNum) {
            func.emitError("report-bank-usage: invalid alloc row/col or range");
            signalPassFailure();
            return;
          }
          if (allocSize.count(*bid)) {
            func.emitError("report-bank-usage: double alloc on same base bank");
            signalPassFailure();
            return;
          }
          for (int64_t i = 0; i < need; ++i) {
            if (used[*bid + i]) {
              func.emitError("report-bank-usage: overlapping bank allocation");
              signalPassFailure();
              return;
            }
            used[*bid + i] = 1;
          }
          allocSize[*bid] = need;
          cur += need;
          peak = std::max(peak, cur);
          ++allocCnt;
          if (verbose) {
            llvm::errs() << "[bank-usage] " << func.getName()
                         << " evt=" << evt
                         << " alloc b" << *bid
                         << " row=" << row
                         << " col=" << col
                         << " cur=" << cur << "/" << bankNum << "\n";
          }
        } else {
          auto it = allocSize.find(*bid);
          if (it == allocSize.end()) {
            func.emitError("report-bank-usage: release without prior alloc");
            signalPassFailure();
            return;
          }
          int64_t need = it->second;
          for (int64_t i = 0; i < need; ++i) {
            used[*bid + i] = 0;
          }
          allocSize.erase(it);
          cur -= need;
          ++relCnt;
          if (verbose) {
            llvm::errs() << "[bank-usage] " << func.getName()
                         << " evt=" << evt
                         << " release b" << *bid
                         << " size=" << need
                         << " cur=" << cur << "/" << bankNum << "\n";
          }
        }
      }
    }

    llvm::errs() << "[bank-usage] " << func.getName()
                 << " peak=" << peak << "/" << bankNum
                 << " alloc=" << allocCnt
                 << " release=" << relCnt
                 << " leaked=" << allocSize.size() << "\n";

    if (!allocSize.empty()) {
      func.emitError("report-bank-usage: leaked allocations at function end");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerReportBankUsagePass() { PassRegistration<ReportBankUsagePass>(); }
} // namespace buddy
} // namespace mlir
