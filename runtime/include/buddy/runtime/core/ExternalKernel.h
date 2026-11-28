//===- ExternalKernel.h - Optional host kernel dispatch -------------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BUDDY_RUNTIME_CORE_EXTERNALKERNEL_H
#define BUDDY_RUNTIME_CORE_EXTERNALKERNEL_H

#include "buddy/runtime/core/ExternalKernelABI.h"

#include <string>

namespace buddy::runtime {

enum class KernelPath { None, External, Fallback };

struct KernelResult {
  int32_t status;
  KernelPath path;
  int32_t externalStatus;
};

/// An optional synchronous host kernel. Resolve once at session creation;
/// keep alive until all calls finish. Missing/incompatible libraries use the
/// supplied Buddy-generated fallback. Diagnostics describe load rejection.
/// Concurrent calls require disjoint outputs/workspaces and a reentrant plugin.
class ExternalKernel {
public:
  explicit ExternalKernel(const std::string &libraryPath);
  ~ExternalKernel();
  ExternalKernel(const ExternalKernel &) = delete;
  ExternalKernel &operator=(const ExternalKernel &) = delete;

  KernelResult dispatch(const BuddyKernelCallV1 &call,
                        BuddyKernelInvokeV1 fallback) const;
  bool matches(const BuddyKernelCallV1 &call) const;
  /// Validate before crossing a compiled graph boundary, even without a plugin.
  static bool isValidCall(const BuddyKernelCallV1 &call);
  static bool hasFeatureRequirements(const std::string &serializedFeatures);
  /// True when every feature token in `serializedFeatures` is declared by the
  /// loaded descriptor. Tokens are separated by commas, semicolons or
  /// whitespace and compared case-insensitively, so RAX metadata ordering and
  /// formatting cannot change the decision. An empty list records no
  /// requirement. The plugin's isCompatibleHost callback remains the
  /// authoritative host-ISA check; this only cross-checks serialized metadata.
  bool satisfiesRequirements(const std::string &serializedFeatures) const;
  /// Borrowed until this handle is destroyed; null if loading was rejected.
  const BuddyKernelDescriptorV1 *descriptorInfo() const { return descriptor; }
  const std::string &diagnostic() const { return message; }

private:
  void *handle = nullptr;
  const BuddyKernelDescriptorV1 *descriptor = nullptr;
  std::string message;
};

} // namespace buddy::runtime
#endif
