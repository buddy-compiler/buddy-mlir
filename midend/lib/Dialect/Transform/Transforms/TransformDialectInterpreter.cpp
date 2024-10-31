//===- TransformDialectInterpreter.cpp ------------------------------------===//
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

#include "Transform/Transforms/TransformDialectInterpreter.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct TransformDialectInterpreterPass
    : public TransformDialectInterpreterBase<TransformDialectInterpreterPass> {

  explicit TransformDialectInterpreterPass(bool erase)
      : TransformDialectInterpreterBase() {
    eraseAfter = erase;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto op : module.getOps<transform::TransformOpInterface>()) {
      RaggedArray<transform::MappedValue> extraMappings;
      if (failed(transform::applyTransforms(
              module, op, extraMappings,
              transform::TransformOptions().enableExpensiveChecks(false)))) {
        return signalPassFailure();
      }
    }
    if (eraseAfter) {
      module.walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
        if (isa<transform::TransformOpInterface>(nestedOp)) {
          nestedOp->erase();
          return WalkResult::skip();
        }
        return WalkResult::advance();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createTransformDialectInterpreter(bool eraseAfter) {
  return std::make_unique<TransformDialectInterpreterPass>(eraseAfter);
}
