//===- RaxDependenciesHandleTest.cpp -------------------------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: buddy-rax-dependencies-test 2>&1 | FileCheck %s

#include "buddy/runtime/core/RaxDependenciesHandle.h"

#include <dlfcn.h>

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  const std::string pluginPath = BUDDY_RAX_DEPENDENCY_TEST_PLUGIN;
  {
    buddy::runtime::RaxDependenciesHandle dependencies({pluginPath},
                                                       "dependency-test");
    dlerror();
    void *symbol = dlsym(RTLD_DEFAULT, "buddyRaxDependencyTestValue");
    const char *symbolError = dlerror();
    if (!symbol || symbolError) {
      std::cerr << "failed to resolve test dependency symbol: "
                << (symbolError ? symbolError : "unknown error") << "\n";
      return 1;
    }
    auto value = reinterpret_cast<int (*)()>(symbol);
    if (value() != 42) {
      std::cerr << "test dependency returned the wrong value\n";
      return 1;
    }
  }

  bool rejectedMissingDependency = false;
  try {
    buddy::runtime::RaxDependenciesHandle dependencies(
        {pluginPath, pluginPath + ".missing"}, "dependency-test");
  } catch (const std::runtime_error &error) {
    const std::string message = error.what();
    rejectedMissingDependency =
        message.find("dependency-test: dlopen RAX dependency failed") !=
        std::string::npos;
  }
  if (!rejectedMissingDependency) {
    std::cerr << "missing dependency was not rejected\n";
    return 1;
  }

  // CHECK: RAX dependency handle tests passed
  std::cout << "RAX dependency handle tests passed\n";
  return 0;
}
