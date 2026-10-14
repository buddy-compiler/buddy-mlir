//===- RaxDependenciesHandle.h - Loaded RAX runtime dependencies -*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_RAXDEPENDENCIESHANDLE_H
#define BUDDY_RUNTIME_CORE_RAXDEPENDENCIESHANDLE_H

#include <string>
#include <vector>

namespace buddy {
namespace runtime {

/// Keeps the shared libraries declared by a RAX manifest loaded in manifest
/// order. Dependencies use RTLD_GLOBAL so later libraries can resolve their
/// DT_NEEDED entries against earlier ones.
class RaxDependenciesHandle {
public:
  explicit RaxDependenciesHandle(
      const std::vector<std::string> &dependencyPaths,
      const std::string &errorContext = "RAX");
  ~RaxDependenciesHandle();

  RaxDependenciesHandle(const RaxDependenciesHandle &) = delete;
  RaxDependenciesHandle &operator=(const RaxDependenciesHandle &) = delete;

private:
  void closeHandles();

  std::vector<void *> handles;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_RAXDEPENDENCIESHANDLE_H
