//===- RHALOps.cpp - RHAL Dialect Operations ------------------------------===//
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

#include "RHAL/RHALOps.h"
#include "RHAL/RHALDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "RHAL/RHALOps.cpp.inc"

mlir::LogicalResult buddy::rhal::FuncOp::verify() {
  bool hasDispatch = (*this)->hasAttr("dispatch");
  bool hasArgs = (*this)->hasAttr("args");

  if (getBody().empty()) {
    if (!hasDispatch || !hasArgs)
      return emitOpError(
          "legacy form requires 'dispatch' and 'args' attributes");
  } else if (hasDispatch || hasArgs) {
    return emitOpError(
        "body form must not have legacy 'dispatch' or 'args' attributes");
  }

  return mlir::success();
}

mlir::LogicalResult buddy::rhal::CollectiveOp::verify() {
  if (getBuffers().empty())
    return emitOpError("requires at least one buffer");

  mlir::StringAttr kind = getKindAttr();
  mlir::StringAttr reduction = getReductionAttr();
  mlir::IntegerAttr root = getRootAttr();
  mlir::ArrayAttr outputBuffers = getOutputBuffersAttr();
  mlir::DenseI64ArrayAttr recvCounts = getRecvCountsAttr();
  mlir::DenseI64ArrayAttr displacements = getDisplacementsAttr();

  if (kind.getValue() == "all_reduce") {
    if (!reduction || reduction.getValue() != "sum")
      return emitOpError("with kind 'all_reduce' requires reduction = \"sum\"");
    if (outputBuffers)
      return emitOpError(
          "with kind 'all_reduce' does not accept an 'output_buffers' "
          "attribute");
    if (root)
      return emitOpError(
          "with kind 'all_reduce' does not accept a 'root' attribute");
    if (recvCounts)
      return emitOpError(
          "with kind 'all_reduce' does not accept a 'recv_counts' attribute");
    if (displacements)
      return emitOpError("with kind 'all_reduce' does not accept a "
                         "'displacements' attribute");
    return mlir::success();
  }

  if (kind.getValue() == "broadcast") {
    if (!root)
      return emitOpError("with kind 'broadcast' requires a 'root' attribute");
    if (root.getInt() < 0)
      return emitOpError("with kind 'broadcast' requires a non-negative root");
    if (outputBuffers)
      return emitOpError(
          "with kind 'broadcast' does not accept an 'output_buffers' "
          "attribute");
    if (reduction)
      return emitOpError(
          "with kind 'broadcast' does not accept a 'reduction' attribute");
    if (recvCounts)
      return emitOpError(
          "with kind 'broadcast' does not accept a 'recv_counts' attribute");
    if (displacements)
      return emitOpError("with kind 'broadcast' does not accept a "
                         "'displacements' attribute");
    return mlir::success();
  }

  if (kind.getValue() == "all_gatherv") {
    if (getBuffers().size() != 1)
      return emitOpError(
          "with kind 'all_gatherv' requires exactly one input buffer");
    if (!outputBuffers || outputBuffers.size() != 1)
      return emitOpError("with kind 'all_gatherv' requires exactly one "
                         "explicit output buffer");
    if (!recvCounts || recvCounts.empty())
      return emitOpError(
          "with kind 'all_gatherv' requires non-empty 'recv_counts'");
    if (!displacements)
      return emitOpError("with kind 'all_gatherv' requires 'displacements'");
    if (recvCounts.size() != displacements.size())
      return emitOpError("with kind 'all_gatherv' requires 'recv_counts' and "
                         "'displacements' to have equal sizes");
    for (int64_t count : recvCounts.asArrayRef())
      if (count < 0)
        return emitOpError(
            "with kind 'all_gatherv' requires non-negative recv_counts");
    for (int64_t displacement : displacements.asArrayRef())
      if (displacement < 0)
        return emitOpError(
            "with kind 'all_gatherv' requires non-negative displacements");
    if (reduction)
      return emitOpError(
          "with kind 'all_gatherv' does not accept a 'reduction' attribute");
    if (root)
      return emitOpError(
          "with kind 'all_gatherv' does not accept a 'root' attribute");
    return mlir::success();
  }

  if (kind.getValue() == "reduce_scatter") {
    if (getBuffers().size() != 1)
      return emitOpError(
          "with kind 'reduce_scatter' requires exactly one input buffer");
    if (!outputBuffers || outputBuffers.size() != 1)
      return emitOpError("with kind 'reduce_scatter' requires exactly one "
                         "explicit output buffer");
    if (!recvCounts || recvCounts.empty())
      return emitOpError(
          "with kind 'reduce_scatter' requires non-empty 'recv_counts'");
    for (int64_t count : recvCounts.asArrayRef())
      if (count < 0)
        return emitOpError(
            "with kind 'reduce_scatter' requires non-negative recv_counts");
    if (!reduction || reduction.getValue() != "sum")
      return emitOpError(
          "with kind 'reduce_scatter' requires reduction = \"sum\"");
    if (displacements)
      return emitOpError("with kind 'reduce_scatter' does not accept a "
                         "'displacements' attribute");
    if (root)
      return emitOpError(
          "with kind 'reduce_scatter' does not accept a 'root' attribute");
    return mlir::success();
  }

  return emitOpError("has unsupported kind '") << kind.getValue() << "'";
}
