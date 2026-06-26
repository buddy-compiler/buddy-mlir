//===- BufferizableOpInterfaceImpl.cpp - Trace bufferization ----*- C++ -*-===//
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

#include "Trace/Transforms/BufferizableOpInterfaceImpl.h"

#include "Trace/TraceDialect.h"
#include "Trace/TraceOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace ::buddy::trace;

namespace {

struct EndOpInterface
    : public BufferizableOpInterface::ExternalModel<EndOpInterface, EndOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &,
                                      const AnalysisState &) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto endOp = cast<EndOp>(op);
    FailureOr<Value> buffer =
        getBuffer(rewriter, endOp.getInput(), options, state);
    if (failed(buffer))
      return failure();

    DictionaryAttr attrs = op->getAttrDictionary();
    auto newOp = replaceOpWithNewBufferizedOp<EndOp>(
        rewriter, op, buffer->getType(), *buffer, endOp.getIdAttr(),
        endOp.getTagAttr());
    newOp->setAttrs(attrs);
    return success();
  }
};

} // namespace

void buddy::trace::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuddyTraceDialect *) {
    EndOp::attachInterface<EndOpInterface>(*ctx);
  });
}
