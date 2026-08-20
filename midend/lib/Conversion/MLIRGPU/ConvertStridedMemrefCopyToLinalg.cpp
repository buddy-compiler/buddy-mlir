//===- ConvertStridedMemrefCopyToLinalg.cpp -------------------------------===//
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
// This file implements the pass that converts strided memref.copy operations
// to linalg.generic before GPU lowering.
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

    OpBuilder builder(funcOp.getContext());
    for (memref::CopyOp copyOp : copies) {
      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();

      auto srcType = dyn_cast<MemRefType>(src.getType());
      auto dstType = dyn_cast<MemRefType>(dst.getType());
      if (!srcType || !dstType) {
        copyOp.emitOpError("expected memref operands");
        return signalPassFailure();
      }

      if (srcType.getLayout().isIdentity() && dstType.getLayout().isIdentity())
        continue;

      if (srcType.getRank() != dstType.getRank()) {
        copyOp.emitOpError("rank mismatch between source and target");
        return signalPassFailure();
      }

      builder.setInsertionPoint(copyOp);
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
