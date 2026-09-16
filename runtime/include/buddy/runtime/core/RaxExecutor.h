//===- RaxExecutor.h - Execute host RAX functions --------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_RAXEXECUTOR_H
#define BUDDY_RUNTIME_CORE_RAXEXECUTOR_H

#include "buddy/runtime/communication/Communicator.h"
#include "buddy/runtime/core/ModelManifest.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace buddy {
namespace runtime {

using RaxHostEntryFn = void (*)(void **args);

class RaxExecutor {
public:
  explicit RaxExecutor(const ModelManifest &manifest);
  RaxExecutor(const ModelManifest &manifest, Communicator &communicator);
  ~RaxExecutor();

  RaxExecutor(const RaxExecutor &) = delete;
  RaxExecutor &operator=(const RaxExecutor &) = delete;

  void bindBuffer(uint32_t id, void *data);
  void bindConstant(uint32_t id, void *abiPtr);
  void execute(const std::string &functionName);

private:
  struct HostBufferView;

  RaxHostEntryFn resolveEntry(uint32_t codeObjectId);
  HostBufferView resolveHostBuffer(uint32_t bufferId) const;
  void executeDispatch(const ModelManifest::RaxOperation &op);
  void executeCollective(const ModelManifest::RaxOperation &op);

  const ModelManifest &manifest_;
  Communicator *communicator_ = nullptr;
  std::unordered_map<uint32_t, void *> buffers_;
  std::unordered_map<uint32_t, void *> constants_;
  std::unordered_map<std::string, void *> libraryHandles_;
  std::unordered_map<uint32_t, RaxHostEntryFn> entries_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_RAXEXECUTOR_H
