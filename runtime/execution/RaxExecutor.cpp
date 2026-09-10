//===- RaxExecutor.cpp - Execute host RAX functions -----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/RaxExecutor.h"

#include <dlfcn.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace buddy {
namespace runtime {

struct RaxExecutor::HostBufferView {
  void *data = nullptr;
  size_t elementCount = 0;
  size_t bytes = 0;
  rhal::rax::DType dtype = rhal::rax::DType_Invalid;
};

RaxExecutor::RaxExecutor(const ModelManifest &manifest) : manifest_(manifest) {}

RaxExecutor::RaxExecutor(const ModelManifest &manifest,
                         Communicator &communicator)
    : manifest_(manifest), communicator_(&communicator) {}

RaxExecutor::~RaxExecutor() {
  for (const auto &library : libraryHandles_)
    dlclose(library.second);
}

void RaxExecutor::bindBuffer(uint32_t id, void *data) { buffers_[id] = data; }

void RaxExecutor::bindConstant(uint32_t id, void *abiPtr) {
  constants_[id] = abiPtr;
}

RaxHostEntryFn RaxExecutor::resolveEntry(uint32_t codeObjectId) {
  auto cachedEntry = entries_.find(codeObjectId);
  if (cachedEntry != entries_.end())
    return cachedEntry->second;

  const ModelManifest::ResolvedCodeObject *codeObject = nullptr;
  for (const auto &candidate : manifest_.codeObjects) {
    if (candidate.id == codeObjectId) {
      codeObject = &candidate;
      break;
    }
  }
  if (!codeObject)
    throw std::runtime_error("RaxExecutor: unknown code object ID " +
                             std::to_string(codeObjectId));
  if (codeObject->kind != rhal::rax::CodeObjectKind_HostSharedLib)
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " is not a HostSharedLib");
  if (codeObject->path.empty())
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " has no shared-library path");
  if (codeObject->entrySymbol.empty())
    throw std::runtime_error("RaxExecutor: code object " +
                             std::to_string(codeObjectId) +
                             " has no entry symbol");

  void *handle = nullptr;
  auto cachedLibrary = libraryHandles_.find(codeObject->path);
  if (cachedLibrary != libraryHandles_.end()) {
    handle = cachedLibrary->second;
  } else {
    handle = dlopen(codeObject->path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char *error = dlerror();
      throw std::runtime_error("RaxExecutor: dlopen failed for " +
                               codeObject->path + ": " +
                               (error ? error : "unknown error"));
    }
    libraryHandles_.emplace(codeObject->path, handle);
  }

  dlerror();
  void *symbol = dlsym(handle, codeObject->entrySymbol.c_str());
  const char *error = dlerror();
  if (error || !symbol)
    throw std::runtime_error("RaxExecutor: dlsym failed for " +
                             codeObject->entrySymbol + ": " +
                             (error ? error : "symbol is null"));

  auto entry = reinterpret_cast<RaxHostEntryFn>(symbol);
  entries_.emplace(codeObjectId, entry);
  return entry;
}

RaxExecutor::HostBufferView
RaxExecutor::resolveHostBuffer(uint32_t bufferId) const {
  const ModelManifest::RaxBuffer *buffer = nullptr;
  for (const auto &candidate : manifest_.buffers) {
    if (candidate.id == bufferId) {
      buffer = &candidate;
      break;
    }
  }
  if (!buffer)
    throw std::runtime_error("RaxExecutor: unknown buffer ID " +
                             std::to_string(bufferId));
  if (buffer->memorySpace != rhal::rax::MemorySpace_Host &&
      buffer->memorySpace != rhal::rax::MemorySpace_Any)
    throw std::runtime_error("RaxExecutor: collective buffer " +
                             std::to_string(bufferId) + " is not host memory");
  if (buffer->layout != rhal::rax::Layout_RowMajor &&
      buffer->layout != rhal::rax::Layout_Any)
    throw std::runtime_error("RaxExecutor: collective buffer " +
                             std::to_string(bufferId) + " is not row-major");

  size_t elementBytes = 0;
  switch (buffer->dtype) {
  case rhal::rax::DType_I8:
  case rhal::rax::DType_U8:
    elementBytes = 1;
    break;
  case rhal::rax::DType_I16:
  case rhal::rax::DType_U16:
  case rhal::rax::DType_F16:
  case rhal::rax::DType_BF16:
    elementBytes = 2;
    break;
  case rhal::rax::DType_I32:
  case rhal::rax::DType_U32:
  case rhal::rax::DType_F32:
    elementBytes = 4;
    break;
  case rhal::rax::DType_I64:
  case rhal::rax::DType_U64:
  case rhal::rax::DType_F64:
    elementBytes = 8;
    break;
  default:
    throw std::runtime_error("RaxExecutor: unsupported dtype for buffer " +
                             std::to_string(bufferId));
  }

  auto binding = buffers_.find(bufferId);
  if (binding == buffers_.end())
    throw std::runtime_error("RaxExecutor: unbound buffer ID " +
                             std::to_string(bufferId));
  if (!binding->second)
    throw std::runtime_error("RaxExecutor: null data pointer for buffer " +
                             std::to_string(bufferId));

  const size_t rank = buffer->shape.size();
  if (!buffer->strides.empty() && buffer->strides.size() != rank)
    throw std::runtime_error("RaxExecutor: invalid strides for buffer " +
                             std::to_string(bufferId));
  size_t elementCount = 1;
  size_t expectedStride = 1;
  for (size_t reverseIndex = 0; reverseIndex < rank; ++reverseIndex) {
    const size_t dim = rank - reverseIndex - 1;
    const int64_t staticSize = buffer->shape[dim];
    if (staticSize < 0)
      throw std::runtime_error("RaxExecutor: dynamic collective buffer " +
                               std::to_string(bufferId));
    if (!buffer->strides.empty() &&
        buffer->strides[dim] != static_cast<int64_t>(expectedStride))
      throw std::runtime_error("RaxExecutor: non-contiguous buffer " +
                               std::to_string(bufferId));
    const size_t dimension = static_cast<size_t>(staticSize);
    if (dimension != 0 &&
        elementCount > std::numeric_limits<size_t>::max() / dimension)
      throw std::runtime_error("RaxExecutor: element count overflows size_t "
                               "for buffer " +
                               std::to_string(bufferId));
    elementCount *= dimension;
    expectedStride = elementCount;
  }
  if (elementCount > std::numeric_limits<size_t>::max() / elementBytes)
    throw std::runtime_error("RaxExecutor: byte size overflows size_t for "
                             "buffer " +
                             std::to_string(bufferId));
  const size_t bytes = elementCount * elementBytes;
  return {binding->second, elementCount, bytes, buffer->dtype};
}

void RaxExecutor::executeDispatch(const ModelManifest::RaxOperation &op) {
  std::vector<void *> callArgs;
  callArgs.reserve(op.arguments.size());
  for (const auto &argument : op.arguments) {
    if (argument.kind == ModelManifest::RaxDispatchArgument::Kind::Buffer) {
      auto binding = buffers_.find(argument.resourceId);
      if (binding == buffers_.end())
        throw std::runtime_error("RaxExecutor: unbound buffer ID " +
                                 std::to_string(argument.resourceId));
      callArgs.push_back(binding->second);
    } else {
      auto binding = constants_.find(argument.resourceId);
      if (binding == constants_.end())
        throw std::runtime_error("RaxExecutor: unbound constant ID " +
                                 std::to_string(argument.resourceId));
      callArgs.push_back(binding->second);
    }
  }
  resolveEntry(op.codeObjectId)(callArgs.data());
}

void RaxExecutor::executeCollective(const ModelManifest::RaxOperation &op) {
  if (!communicator_)
    throw std::runtime_error("RaxExecutor: Collective requires a communicator");

  if (op.collectiveKind == rhal::rax::CollectiveKind_AllGatherV ||
      op.collectiveKind == rhal::rax::CollectiveKind_ReduceScatter) {
    if (op.collectiveOperands.size() != 1)
      throw std::runtime_error(
          "RaxExecutor: variable Collective requires exactly one operand");

    const auto &operand = op.collectiveOperands.front();
    const HostBufferView input = resolveHostBuffer(operand.inputBufferId);
    const HostBufferView output = resolveHostBuffer(operand.outputBufferId);
    if (input.dtype != rhal::rax::DType_F32 ||
        output.dtype != rhal::rax::DType_F32)
      throw std::runtime_error(
          "RaxExecutor: variable Collective requires F32 buffers");

    const int communicatorSize = communicator_->size();
    const int communicatorRank = communicator_->rank();
    if (operand.recvCounts.size() != static_cast<size_t>(communicatorSize))
      throw std::runtime_error(
          "RaxExecutor: receive counts do not match communicator size");

    if (op.collectiveKind == rhal::rax::CollectiveKind_AllGatherV) {
      if (operand.displacements.size() != static_cast<size_t>(communicatorSize))
        throw std::runtime_error(
            "RaxExecutor: displacements do not match communicator size");

      size_t requiredOutputElements = 0;
      for (size_t i = 0; i < operand.recvCounts.size(); ++i) {
        if (operand.recvCounts[i] < 0 || operand.displacements[i] < 0)
          throw std::runtime_error(
              "RaxExecutor: negative AllGatherV count or displacement");
        const size_t count = static_cast<size_t>(operand.recvCounts[i]);
        const size_t displacement =
            static_cast<size_t>(operand.displacements[i]);
        if (displacement > std::numeric_limits<size_t>::max() - count)
          throw std::runtime_error(
              "RaxExecutor: AllGatherV output extent overflows size_t");
        requiredOutputElements =
            std::max(requiredOutputElements, displacement + count);
      }
      if (input.elementCount !=
          static_cast<size_t>(operand.recvCounts[communicatorRank]))
        throw std::runtime_error(
            "RaxExecutor: AllGatherV input element count mismatch");
      if (output.elementCount < requiredOutputElements)
        throw std::runtime_error(
            "RaxExecutor: AllGatherV output buffer is too small");

      communicator_->allGatherV(input.data, input.elementCount, output.data,
                                operand.recvCounts, operand.displacements,
                                DataType::F32);
      return;
    }

    if (op.reductionKind != rhal::rax::ReductionKind_Sum)
      throw std::runtime_error(
          "RaxExecutor: ReduceScatter requires Sum reduction");
    size_t totalInputElements = 0;
    for (int64_t recvCount : operand.recvCounts) {
      if (recvCount < 0)
        throw std::runtime_error(
            "RaxExecutor: negative ReduceScatter receive count");
      const size_t count = static_cast<size_t>(recvCount);
      if (totalInputElements > std::numeric_limits<size_t>::max() - count)
        throw std::runtime_error(
            "RaxExecutor: ReduceScatter input extent overflows size_t");
      totalInputElements += count;
    }
    if (input.elementCount != totalInputElements)
      throw std::runtime_error(
          "RaxExecutor: ReduceScatter input element count mismatch");
    if (output.elementCount !=
        static_cast<size_t>(operand.recvCounts[communicatorRank]))
      throw std::runtime_error(
          "RaxExecutor: ReduceScatter output element count mismatch");

    communicator_->reduceScatter(input.data, output.data, operand.recvCounts,
                                 DataType::F32, ReductionOp::Sum);
    return;
  }

  if (op.collectiveKind == rhal::rax::CollectiveKind_Broadcast &&
      (op.root < 0 || op.root >= communicator_->size()))
    throw std::runtime_error("RaxExecutor: invalid Broadcast root " +
                             std::to_string(op.root));

  std::vector<HostBufferView> operands;
  operands.reserve(op.collectiveOperands.size());
  for (const auto &operand : op.collectiveOperands) {
    if (operand.inputBufferId != operand.outputBufferId)
      throw std::runtime_error(
          "RaxExecutor: only in-place Collective operands are supported");
    operands.push_back(resolveHostBuffer(operand.inputBufferId));
  }

  for (const auto &operand : operands) {
    if (op.collectiveKind == rhal::rax::CollectiveKind_AllReduce &&
        (operand.dtype != rhal::rax::DType_F32 ||
         op.reductionKind != rhal::rax::ReductionKind_Sum))
      throw std::runtime_error(
          "RaxExecutor: unsupported Collective datatype or reduction");
  }
  if (op.collectiveKind != rhal::rax::CollectiveKind_Broadcast &&
      op.collectiveKind != rhal::rax::CollectiveKind_AllReduce)
    throw std::runtime_error("RaxExecutor: unsupported Collective kind");

  for (const auto &operand : operands) {
    if (op.collectiveKind == rhal::rax::CollectiveKind_Broadcast) {
      communicator_->broadcast(operand.data, operand.bytes, op.root);
      continue;
    }
    communicator_->allReduce(operand.data, operand.data, operand.elementCount,
                             DataType::F32, ReductionOp::Sum);
  }
}

void RaxExecutor::execute(const std::string &functionName) {
  const ModelManifest::RaxFunction *function = nullptr;
  for (const auto &candidate : manifest_.functions) {
    if (candidate.name == functionName) {
      function = &candidate;
      break;
    }
  }
  if (!function)
    throw std::runtime_error("RaxExecutor: unknown function " + functionName);

  for (const auto &op : function->ops) {
    if (op.kind == rhal::rax::OpKind_Dispatch) {
      executeDispatch(op);
      continue;
    }
    if (op.kind == rhal::rax::OpKind_Collective) {
      executeCollective(op);
      continue;
    }
    if (op.kind == rhal::rax::OpKind_Barrier)
      continue;
    throw std::runtime_error("RaxExecutor: unsupported operation kind " +
                             std::to_string(static_cast<int>(op.kind)));
  }
}

} // namespace runtime
} // namespace buddy
