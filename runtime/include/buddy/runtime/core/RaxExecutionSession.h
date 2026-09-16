//===- RaxExecutionSession.h - Own a RAX execution context ------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_RAXEXECUTIONSESSION_H
#define BUDDY_RUNTIME_CORE_RAXEXECUTIONSESSION_H

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxExecutor.h"

#include <cstdint>
#include <string>

namespace buddy {
namespace runtime {

/// Owns the manifest and low-level executor for one RAX artifact.
///
/// Resource storage is deliberately left to model-specific sessions. The
/// generic session only keeps the resolved artifact metadata alive and
/// forwards bindings and ordered execution to RaxExecutor.
class RaxExecutionSession {
public:
  explicit RaxExecutionSession(const std::string &raxPath);
  RaxExecutionSession(const std::string &raxPath, Communicator &communicator);

  RaxExecutionSession(const RaxExecutionSession &) = delete;
  RaxExecutionSession &operator=(const RaxExecutionSession &) = delete;

  void bindBuffer(uint32_t id, void *ptr);
  void bindConstant(uint32_t id, void *ptr);
  void execute(const std::string &function);

  const ModelManifest &manifest() const { return manifest_; }

private:
  ModelManifest manifest_;
  RaxExecutor executor_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_RAXEXECUTIONSESSION_H
