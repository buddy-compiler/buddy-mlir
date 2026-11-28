// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "buddy/runtime/core/ModelManifest.h"
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  try {
    auto manifest = buddy::runtime::ModelManifest::loadFromRax(argv[1]);
    for (const auto &code : manifest.codeObjects)
      if (code.optional)
        std::cout << code.path << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
