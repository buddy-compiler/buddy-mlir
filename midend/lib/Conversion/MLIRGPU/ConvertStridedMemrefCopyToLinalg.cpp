//===- ConvertStridedMemrefCopyToLinalg.cpp -------------------------------===//
//
// Rewrite non-identity-layout memref.copy to linalg.generic so they lower
// through parallel-loops → GPU kernels. Contiguous copies stay as memref.copy
// for convert-memcpy-to-gpu. Identity self-copies are erased.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class ConvertStridedMemrefCopyToLinalgPass
    : public PassWrapper<ConvertStridedMemrefCopyToLinalgPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertStridedMemrefCopyToLinalgPass)
  StringRef getArgument() const final {
    return "convert-strided-memref-copy-to-linalg";
  }
  StringRef getDescription() const final {
    return "Convert strided memref.copy to linalg.generic copy.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<memref::CopyOp> copies;
    funcOp.walk([&](memref::CopyOp op) { copies.push_back(op); });

    for (memref::CopyOp copyOp : copies) {
      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();
      if (src == dst) {
        copyOp.erase();
        continue;
      }

      auto srcType = dyn_cast<MemRefType>(src.getType());
      auto dstType = dyn_cast<MemRefType>(dst.getType());
      if (!srcType || !dstType) {
        copyOp.emitOpError("expected memref operands");
        signalPassFailure();
        return;
      }

      if (srcType.getLayout().isIdentity() && dstType.getLayout().isIdentity())
        continue;

      if (srcType.getRank() != dstType.getRank()) {
        copyOp.emitOpError("rank mismatch between source and target");
        signalPassFailure();
        return;
      }

      OpBuilder builder(copyOp);
      linalg::makeMemRefCopyOp(builder, copyOp.getLoc(), src, dst);
      copyOp.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerConvertStridedMemrefCopyToLinalgPass() {
  PassRegistration<ConvertStridedMemrefCopyToLinalgPass>();
}
} // namespace buddy
} // namespace mlir
