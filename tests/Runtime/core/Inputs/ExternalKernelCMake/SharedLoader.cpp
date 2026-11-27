// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <dlfcn.h>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::cerr << dlerror() << '\n';
    return 2;
  }
  auto check = reinterpret_cast<int (*)(const char *)>(
      dlsym(handle, "buddy_external_consumer_check"));
  const int status = check ? check(argv[2]) : 3;
  dlclose(handle);
  if (!status)
    std::cout << "PASS CMake shared consumer external=1 fallback=1 invalid=1\n";
  return status;
}
