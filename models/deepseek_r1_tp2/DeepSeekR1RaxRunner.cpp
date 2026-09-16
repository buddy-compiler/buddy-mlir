//===- DeepSeekR1RaxRunner.cpp - DeepSeek RAX execution entry ------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DeepSeekR1RaxRunner.h"

#include "buddy/runtime/communication/MpiCommunicator.h"
#include "buddy/runtime/models/DeepSeekR1RaxSession.h"

#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace buddy {
namespace runtime {
namespace {

void checkMpi(int status, const char *operation) {
  if (status != MPI_SUCCESS)
    throw std::runtime_error(std::string("DeepSeekR1RaxRunner: ") + operation +
                             " failed");
}

} // namespace

void runDeepSeekR1Rax(const std::string &raxPath,
                      const DeepSeekR1RaxSessionCallback &callback) {
  if (raxPath.empty())
    throw std::runtime_error("DeepSeekR1RaxRunner requires a RAX artifact");

  int finalized = 0;
  checkMpi(MPI_Finalized(&finalized), "MPI_Finalized");
  if (finalized)
    throw std::runtime_error("DeepSeekR1RaxRunner: MPI is already finalized");

  int initialized = 0;
  checkMpi(MPI_Initialized(&initialized), "MPI_Initialized");
  const bool ownsMpi = initialized == 0;
  if (ownsMpi)
    checkMpi(MPI_Init(nullptr, nullptr), "MPI_Init");

  try {
    MpiCommunicator communicator(MPI_COMM_WORLD);
    const int rank = communicator.rank();
    const int worldSize = communicator.size();
    if (worldSize != 2)
      throw std::runtime_error("DeepSeekR1RaxRunner: MPI world size " +
                               std::to_string(worldSize) + "; expected 2");

    std::cerr << "DeepSeek RAX rank " << rank << '/' << worldSize << ": "
              << raxPath << '\n';

    DeepSeekR1RaxSession session(raxPath, communicator);
    callback(session, rank);
  } catch (...) {
    if (ownsMpi)
      MPI_Finalize();
    throw;
  }

  if (ownsMpi)
    checkMpi(MPI_Finalize(), "MPI_Finalize");
}

} // namespace runtime
} // namespace buddy
