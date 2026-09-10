//===- Communicator.h - Runtime collective interface -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_COMMUNICATION_COMMUNICATOR_H
#define BUDDY_RUNTIME_COMMUNICATION_COMMUNICATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace buddy {
namespace runtime {

enum class DataType { F32 };
enum class ReductionOp { Sum };

class Communicator {
public:
  virtual ~Communicator() = default;

  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual void broadcast(void *buffer, size_t bytes, int root) = 0;
  virtual void allReduce(const void *sendBuffer, void *recvBuffer, size_t count,
                         DataType dataType, ReductionOp reduction) = 0;
  virtual void allGatherV(const void *sendBuffer, size_t sendCount,
                          void *recvBuffer,
                          const std::vector<int64_t> &recvCounts,
                          const std::vector<int64_t> &displacements,
                          DataType dataType) = 0;
  virtual void reduceScatter(const void *sendBuffer, void *recvBuffer,
                             const std::vector<int64_t> &recvCounts,
                             DataType dataType, ReductionOp reduction) = 0;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_COMMUNICATION_COMMUNICATOR_H
