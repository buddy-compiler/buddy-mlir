//===- MpiCommunicator.cpp - MPI collective backend ----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/communication/MpiCommunicator.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

static void checkMpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(std::string("MpiCommunicator: ") + operation +
                             " failed");
}

static std::vector<int> convertMpiValues(const std::vector<int64_t> &values,
                                         const char *description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (int64_t value : values) {
    if (value < 0 || value > std::numeric_limits<int>::max())
      throw std::runtime_error(std::string("MpiCommunicator: ") + description +
                               " exceeds MPI int range");
    result.push_back(static_cast<int>(value));
  }
  return result;
}

MpiCommunicator::MpiCommunicator(MPI_Comm communicator)
    : communicator_(communicator) {}

int MpiCommunicator::rank() const {
  int value = 0;
  checkMpi(MPI_Comm_rank(communicator_, &value), "MPI_Comm_rank");
  return value;
}

int MpiCommunicator::size() const {
  int value = 0;
  checkMpi(MPI_Comm_size(communicator_, &value), "MPI_Comm_size");
  return value;
}

void MpiCommunicator::broadcast(void *buffer, size_t bytes, int root) {
  if (bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "MpiCommunicator: Broadcast count exceeds MPI int");
  checkMpi(
      MPI_Bcast(buffer, static_cast<int>(bytes), MPI_BYTE, root, communicator_),
      "MPI_Bcast");
}

void MpiCommunicator::allReduce(const void *sendBuffer, void *recvBuffer,
                                size_t count, DataType dataType,
                                ReductionOp reduction) {
  if (dataType != DataType::F32 || reduction != ReductionOp::Sum)
    throw std::runtime_error(
        "MpiCommunicator: unsupported AllReduce datatype or reduction");
  if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "MpiCommunicator: AllReduce count exceeds MPI int");
  const void *mpiSendBuffer =
      sendBuffer == recvBuffer ? MPI_IN_PLACE : sendBuffer;
  checkMpi(MPI_Allreduce(mpiSendBuffer, recvBuffer, static_cast<int>(count),
                         MPI_FLOAT, MPI_SUM, communicator_),
           "MPI_Allreduce");
}

void MpiCommunicator::allGatherV(const void *sendBuffer, size_t sendCount,
                                 void *recvBuffer,
                                 const std::vector<int64_t> &recvCounts,
                                 const std::vector<int64_t> &displacements,
                                 DataType dataType) {
  if (dataType != DataType::F32)
    throw std::runtime_error(
        "MpiCommunicator: unsupported AllGatherV datatype");
  if (sendCount > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error(
        "MpiCommunicator: AllGatherV send count exceeds MPI int");
  const std::vector<int> mpiRecvCounts =
      convertMpiValues(recvCounts, "AllGatherV receive count");
  const std::vector<int> mpiDisplacements =
      convertMpiValues(displacements, "AllGatherV displacement");
  checkMpi(MPI_Allgatherv(sendBuffer, static_cast<int>(sendCount), MPI_FLOAT,
                          recvBuffer, mpiRecvCounts.data(),
                          mpiDisplacements.data(), MPI_FLOAT, communicator_),
           "MPI_Allgatherv");
}

void MpiCommunicator::reduceScatter(const void *sendBuffer, void *recvBuffer,
                                    const std::vector<int64_t> &recvCounts,
                                    DataType dataType, ReductionOp reduction) {
  if (dataType != DataType::F32 || reduction != ReductionOp::Sum)
    throw std::runtime_error(
        "MpiCommunicator: unsupported ReduceScatter datatype or reduction");
  const std::vector<int> mpiRecvCounts =
      convertMpiValues(recvCounts, "ReduceScatter receive count");
  checkMpi(MPI_Reduce_scatter(sendBuffer, recvBuffer, mpiRecvCounts.data(),
                              MPI_FLOAT, MPI_SUM, communicator_),
           "MPI_Reduce_scatter");
}

} // namespace runtime
} // namespace buddy
