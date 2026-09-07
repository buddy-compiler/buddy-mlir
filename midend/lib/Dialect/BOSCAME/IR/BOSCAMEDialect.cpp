//====- BOSCAMEDialect.cpp - MLIR BOSCAME dialect implementation ----------===//
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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialect/BOSCAME/BOSCAMEDialect.h"
#include "Dialect/BOSCAME/BOSCAMEOps.h"

using namespace mlir;
using namespace buddy::boscame;

#include "BOSCAME/BOSCAMEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "BOSCAME/BOSCAME.cpp.inc"

namespace {

using bufferization::AnalysisState;
using bufferization::BufferizationOptions;
using bufferization::BufferizationState;
using bufferization::DstBufferizableOpInterfaceExternalModel;

static LogicalResult verifyTensorBufferForm(Operation *op,
                                            ValueRange destinations) {
  const bool tensorForm = llvm::all_of(destinations, [](Value value) {
    return isa<TensorType>(value.getType());
  });
  const bool bufferForm = llvm::all_of(destinations, [](Value value) {
    return isa<BaseMemRefType>(value.getType());
  });
  if (!tensorForm && !bufferForm)
    return op->emitOpError("does not support mixed tensor/buffer destinations");
  if (tensorForm) {
    if (op->getNumResults() != destinations.size())
      return op->emitOpError("tensor form requires one result per destination");
    for (auto [result, destination] : llvm::zip(op->getResults(), destinations))
      if (result.getType() != destination.getType())
        return op->emitOpError("result types must match destination types");
  } else if (op->getNumResults() != 0) {
    return op->emitOpError("buffer form must not produce results");
  }
  return success();
}

static LogicalResult requireStaticShape(Operation *op, Value value,
                                        StringRef name, unsigned rank,
                                        Type elementType) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != rank)
    return op->emitOpError()
           << name << " must be a rank-" << rank << " shaped value";
  if (!shaped.hasStaticShape())
    return op->emitOpError() << name << " must have a static shape";
  if (shaped.getElementType() != elementType)
    return op->emitOpError()
           << name << " must have element type " << elementType;
  return success();
}

static FailureOr<Value> getBufferForOperand(RewriterBase &rewriter, Value value,
                                            const BufferizationOptions &options,
                                            BufferizationState &state) {
  if (isa<BaseMemRefType>(value.getType()))
    return value;
  return bufferization::getBuffer(rewriter, value, options, state);
}

struct QuantizePerGroupOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          QuantizePerGroupOpInterface, QuantizePerGroupOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto quantize = cast<QuantizePerGroupOp>(op);
    return &operand == &quantize.getInputMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto quantize = cast<QuantizePerGroupOp>(op);
    SmallVector<Value> buffers;
    for (Value value : quantize->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(quantize.getLoc(), quantize->getName());
    newState.addOperands(buffers);
    newState.addAttributes(quantize->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[1], buffers[2]});
    return success();
  }
};

struct SiluMulQuantizePerGroupOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          SiluMulQuantizePerGroupOpInterface, SiluMulQuantizePerGroupOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto quantize = cast<SiluMulQuantizePerGroupOp>(op);
    return &operand == &quantize.getSiluMutable() ||
           &operand == &quantize.getUpMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto quantize = cast<SiluMulQuantizePerGroupOp>(op);
    SmallVector<Value> buffers;
    for (Value value : quantize->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(quantize.getLoc(), quantize->getName());
    newState.addOperands(buffers);
    newState.addAttributes(quantize->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[2], buffers[3]});
    return success();
  }
};

struct W8A8LinearOpInterface
    : public DstBufferizableOpInterfaceExternalModel<W8A8LinearOpInterface,
                                                     W8A8LinearOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto linear = cast<W8A8LinearOp>(op);
    return &operand != &linear.getOutputMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto linear = cast<W8A8LinearOp>(op);
    SmallVector<Value> buffers;
    for (Value value : linear->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(linear.getLoc(), linear->getName());
    newState.addOperands(buffers);
    newState.addAttributes(linear->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, buffers[4]);
    return success();
  }
};

struct W8A8LinearPairOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          W8A8LinearPairOpInterface, W8A8LinearPairOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand,
                              const AnalysisState &) const {
    auto pair = cast<W8A8LinearPairOp>(op);
    return &operand != &pair.getOutput0Mutable() &&
           &operand != &pair.getOutput1Mutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto pair = cast<W8A8LinearPairOp>(op);
    SmallVector<Value> buffers;
    for (Value value : pair->getOperands()) {
      FailureOr<Value> buffer =
          getBufferForOperand(rewriter, value, options, state);
      if (failed(buffer))
        return failure();
      buffers.push_back(*buffer);
    }

    OperationState newState(pair.getLoc(), pair->getName());
    newState.addOperands(buffers);
    newState.addAttributes(pair->getAttrs());
    rewriter.create(newState);
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, ValueRange{buffers[6], buffers[7]});
    return success();
  }
};

} // namespace

LogicalResult QuantizePerGroupOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getInput(), "input", 2, f32)) ||
      failed(requireStaticShape(*this, getQuantized(), "quantized", 2, i8)) ||
      failed(requireStaticShape(*this, getScales(), "scales", 2, f32)) ||
      failed(verifyTensorBufferForm(*this,
                                    ValueRange{getQuantized(), getScales()})))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto quantizedType = cast<ShapedType>(getQuantized().getType());
  auto scalesType = cast<ShapedType>(getScales().getType());
  int64_t groupSize = getGroupSize();
  int64_t rows = inputType.getDimSize(0);
  int64_t width = inputType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (quantizedType.getShape() != inputType.getShape())
    return emitOpError("quantized shape must match input shape");
  if (scalesType.getShape() != ArrayRef<int64_t>{rows, width / groupSize})
    return emitOpError("scales shape must be [T, K / group_size]");
  return success();
}

void QuantizePerGroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Tensor form is a deterministic value computation.  Describing it as
  // effect-free lets CSE share activation quantization across linears that
  // consume the same activation (Q/K/V and Gate/Up in Qwen3).  Buffer form
  // must retain its explicit reads and writes so two calls with different
  // destinations are never folded after bufferization.
  if (isa<BaseMemRefType>(getInput().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable());
  if (isa<BaseMemRefType>(getQuantized().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getQuantizedMutable());
  if (isa<BaseMemRefType>(getScales().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getScalesMutable());
}

LogicalResult SiluMulQuantizePerGroupOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getSilu(), "silu", 2, f32)) ||
      failed(requireStaticShape(*this, getUp(), "up", 2, f32)) ||
      failed(requireStaticShape(*this, getQuantized(), "quantized", 2, i8)) ||
      failed(requireStaticShape(*this, getScales(), "scales", 2, f32)) ||
      failed(verifyTensorBufferForm(*this,
                                    ValueRange{getQuantized(), getScales()})))
    return failure();

  auto siluType = cast<ShapedType>(getSilu().getType());
  auto upType = cast<ShapedType>(getUp().getType());
  auto quantizedType = cast<ShapedType>(getQuantized().getType());
  auto scalesType = cast<ShapedType>(getScales().getType());
  // All four shaped values participate in one DPS form; mixed tensor/memref
  // operands cannot be bufferized or reasoned about safely.
  const bool tensorForm = isa<TensorType>(getQuantized().getType());
  const bool inputsMatchForm =
      tensorForm
          ? isa<RankedTensorType>(getSilu().getType()) &&
                isa<RankedTensorType>(getUp().getType())
          : isa<MemRefType>(getSilu().getType()) &&
                isa<MemRefType>(getUp().getType());
  if (!inputsMatchForm)
    return emitOpError(
        "silu/up and destinations must use the same tensor/buffer form");
  int64_t groupSize = getGroupSize();
  int64_t rows = siluType.getDimSize(0);
  int64_t width = siluType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (upType.getShape() != siluType.getShape())
    return emitOpError("up shape must match silu shape");
  if (quantizedType.getShape() != siluType.getShape())
    return emitOpError("quantized shape must match silu shape");
  if (scalesType.getShape() != ArrayRef<int64_t>{rows, width / groupSize})
    return emitOpError("scales shape must be [T, K / group_size]");
  return success();
}

void SiluMulQuantizePerGroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (isa<BaseMemRefType>(getSilu().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getSiluMutable());
  if (isa<BaseMemRefType>(getUp().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getUpMutable());
  if (isa<BaseMemRefType>(getQuantized().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getQuantizedMutable());
  if (isa<BaseMemRefType>(getScales().getType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getScalesMutable());
}

LogicalResult W8A8LinearOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getXq(), "xq", 2, i8)) ||
      failed(requireStaticShape(*this, getXs(), "xs", 2, f32)) ||
      failed(requireStaticShape(*this, getWq(), "wq", 4, i8)) ||
      failed(requireStaticShape(*this, getWs(), "ws", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput(), "output", 2, f32)) ||
      failed(verifyTensorBufferForm(*this, ValueRange{getOutput()})))
    return failure();

  if (getWeightLayout() != "ame_outblk64")
    return emitOpError("weight_layout must be 'ame_outblk64'");

  auto xqType = cast<ShapedType>(getXq().getType());
  auto xsType = cast<ShapedType>(getXs().getType());
  auto wqType = cast<ShapedType>(getWq().getType());
  auto wsType = cast<ShapedType>(getWs().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());
  int64_t groupSize = getGroupSize();
  int64_t tokens = xqType.getDimSize(0);
  int64_t width = xqType.getDimSize(1);
  int64_t outputWidth = outputType.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (width % 64 != 0)
    return emitOpError("K must be divisible by the AME K tile (64)");
  if (groupSize % 64 != 0)
    return emitOpError("group_size must be divisible by the AME K tile (64)");
  if (outputWidth % 64 != 0)
    return emitOpError("D must be divisible by OUTBLK (64)");
  int64_t groups = width / groupSize;
  if (xsType.getShape() != ArrayRef<int64_t>{tokens, groups})
    return emitOpError("xs shape must be [T, K / group_size]");
  if (wqType.getShape() !=
      ArrayRef<int64_t>{outputWidth / 64, groups, 64, groupSize})
    return emitOpError(
        "wq shape must be [D / 64, K / group_size, 64, group_size]");
  if (wsType.getShape() != ArrayRef<int64_t>{groups, outputWidth})
    return emitOpError("ws shape must be [K / group_size, D]");
  if (outputType.getDimSize(0) != tokens)
    return emitOpError("output token dimension must match xq");
  return success();
}

LogicalResult W8A8LinearPairOp::verify() {
  MLIRContext *context = getContext();
  Type f32 = Float32Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  if (failed(requireStaticShape(*this, getXq(), "xq", 2, i8)) ||
      failed(requireStaticShape(*this, getXs(), "xs", 2, f32)) ||
      failed(requireStaticShape(*this, getWq0(), "wq0", 4, i8)) ||
      failed(requireStaticShape(*this, getWs0(), "ws0", 2, f32)) ||
      failed(requireStaticShape(*this, getWq1(), "wq1", 4, i8)) ||
      failed(requireStaticShape(*this, getWs1(), "ws1", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput0(), "output0", 2, f32)) ||
      failed(requireStaticShape(*this, getOutput1(), "output1", 2, f32)) ||
      failed(verifyTensorBufferForm(
          *this, ValueRange{getOutput0(), getOutput1()})))
    return failure();

  if (getWeightLayout() != "ame_outblk64")
    return emitOpError("weight_layout must be 'ame_outblk64'");

  auto xqType = cast<ShapedType>(getXq().getType());
  auto xsType = cast<ShapedType>(getXs().getType());
  auto wq0Type = cast<ShapedType>(getWq0().getType());
  auto ws0Type = cast<ShapedType>(getWs0().getType());
  auto wq1Type = cast<ShapedType>(getWq1().getType());
  auto ws1Type = cast<ShapedType>(getWs1().getType());
  auto output0Type = cast<ShapedType>(getOutput0().getType());
  auto output1Type = cast<ShapedType>(getOutput1().getType());
  int64_t groupSize = getGroupSize();
  int64_t tokens = xqType.getDimSize(0);
  int64_t width = xqType.getDimSize(1);
  int64_t outputWidth = output0Type.getDimSize(1);
  if (groupSize <= 0 || groupSize > 1024)
    return emitOpError("group_size must be in [1, 1024]");
  if (width % groupSize != 0)
    return emitOpError("K must be divisible by group_size");
  if (width % 64 != 0)
    return emitOpError("K must be divisible by the AME K tile (64)");
  if (groupSize % 64 != 0)
    return emitOpError("group_size must be divisible by the AME K tile (64)");
  if (outputWidth % 64 != 0)
    return emitOpError("D must be divisible by OUTBLK (64)");
  if (output1Type.getShape() != output0Type.getShape())
    return emitOpError("paired output shapes must match");
  if (isa<BaseMemRefType>(getOutput0().getType()) &&
      getOperation()->getOperand(6) == getOperation()->getOperand(7))
    return emitOpError("buffer-form paired outputs must not alias");
  int64_t groups = width / groupSize;
  if (xsType.getShape() != ArrayRef<int64_t>{tokens, groups})
    return emitOpError("xs shape must be [T, K / group_size]");
  SmallVector<int64_t> expectedWqShape =
      {outputWidth / 64, groups, 64, groupSize};
  SmallVector<int64_t> expectedWsShape = {groups, outputWidth};
  if (wq0Type.getShape() != ArrayRef<int64_t>(expectedWqShape) ||
      wq1Type.getShape() != ArrayRef<int64_t>(expectedWqShape))
    return emitOpError(
        "both wq shapes must be [D / 64, K / group_size, 64, group_size]");
  if (ws0Type.getShape() != ArrayRef<int64_t>(expectedWsShape) ||
      ws1Type.getShape() != ArrayRef<int64_t>(expectedWsShape))
    return emitOpError("both ws shapes must be [K / group_size, D]");
  if (output0Type.getDimSize(0) != tokens)
    return emitOpError("output token dimension must match xq");
  return success();
}

void BOSCAMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BOSCAME/BOSCAME.cpp.inc"
      >();
  QuantizePerGroupOp::attachInterface<QuantizePerGroupOpInterface>(
      *getContext());
  SiluMulQuantizePerGroupOp::attachInterface<
      SiluMulQuantizePerGroupOpInterface>(*getContext());
  W8A8LinearOp::attachInterface<W8A8LinearOpInterface>(*getContext());
  W8A8LinearPairOp::attachInterface<W8A8LinearPairOpInterface>(*getContext());
}
