//===- RaxExecutionSession.cpp - Own a RAX execution context --------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/RaxExecutionSession.h"

namespace buddy {
namespace runtime {

RaxExecutionSession::RaxExecutionSession(const std::string &raxPath)
    : manifest_(ModelManifest::loadFromRax(raxPath)), executor_(manifest_) {}

RaxExecutionSession::RaxExecutionSession(const std::string &raxPath,
                                         Communicator &communicator)
    : manifest_(ModelManifest::loadFromRax(raxPath)),
      executor_(manifest_, communicator) {}

void RaxExecutionSession::bindBuffer(uint32_t id, void *ptr) {
  executor_.bindBuffer(id, ptr);
}

void RaxExecutionSession::bindConstant(uint32_t id, void *ptr) {
  executor_.bindConstant(id, ptr);
}

void RaxExecutionSession::execute(const std::string &function) {
  executor_.execute(function);
}

} // namespace runtime
} // namespace buddy
