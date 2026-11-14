//===- RaxDependenciesHandle.cpp - Loaded RAX runtime dependencies -------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/core/RaxDependenciesHandle.h"

#include <dlfcn.h>

#include <stdexcept>

namespace buddy {
namespace runtime {

RaxDependenciesHandle::RaxDependenciesHandle(
    const std::vector<std::string> &dependencyPaths,
    const std::string &errorContext) {
  try {
    for (const auto &path : dependencyPaths) {
      dlerror();
      void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!handle) {
        const char *error = dlerror();
        const std::string prefix =
            errorContext.empty() ? std::string() : errorContext + ": ";
        throw std::runtime_error(prefix +
                                 "dlopen RAX dependency failed: " + path +
                                 ": " + (error ? error : "unknown error"));
      }
      handles.push_back(handle);
    }
  } catch (...) {
    closeHandles();
    throw;
  }
}

RaxDependenciesHandle::~RaxDependenciesHandle() { closeHandles(); }

void RaxDependenciesHandle::closeHandles() {
  for (auto it = handles.rbegin(); it != handles.rend(); ++it)
    if (*it)
      dlclose(*it);
  handles.clear();
}

} // namespace runtime
} // namespace buddy
