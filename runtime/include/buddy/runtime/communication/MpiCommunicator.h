//===- MpiCommunicator.h - MPI collective backend --------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_COMMUNICATION_MPICOMMUNICATOR_H
#define BUDDY_RUNTIME_COMMUNICATION_MPICOMMUNICATOR_H

#include "buddy/runtime/communication/Communicator.h"

#include <mpi.h>

namespace buddy {
namespace runtime {

class MpiCommunicator final : public Communicator {
public:
  explicit MpiCommunicator(MPI_Comm communicator);

  int rank() const override;
  int size() const override;
  void broadcast(void *buffer, size_t bytes, int root) override;
  void allReduce(const void *sendBuffer, void *recvBuffer, size_t count,
                 DataType dataType, ReductionOp reduction) override;
  void allGatherV(const void *sendBuffer, size_t sendCount, void *recvBuffer,
                  const std::vector<int64_t> &recvCounts,
                  const std::vector<int64_t> &displacements,
                  DataType dataType) override;
  void reduceScatter(const void *sendBuffer, void *recvBuffer,
                     const std::vector<int64_t> &recvCounts, DataType dataType,
                     ReductionOp reduction) override;

private:
  MPI_Comm communicator_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_COMMUNICATION_MPICOMMUNICATOR_H
