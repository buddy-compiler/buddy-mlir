//===- rax-inspect.cpp — print RAX module metadata ------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

static std::vector<uint8_t> readAll(const char *path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error(std::string("cannot open: ") + path);
  ifs.seekg(0, std::ios::end);
  size_t n = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(n);
  if (n && !ifs.read(reinterpret_cast<char *>(buf.data()), n))
    throw std::runtime_error("read failed");
  return buf;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: rax-inspect <file.rax>\n";
    return 2;
  }

  try {
    auto buf = readAll(argv[1]);

    // Verify file identifier "RAX0"
    if (buf.size() < 8 ||
        !flatbuffers::BufferHasIdentifier(buf.data(), "RAX0")) {
      std::cerr << "not a valid RAX file (missing identifier RAX0)\n";
      return 1;
    }

    // Verify schema (optional but recommended)
    flatbuffers::Verifier verifier(buf.data(), buf.size());
    if (!rhal::rax::VerifyModuleBuffer(verifier)) {
      std::cerr << "RAX verification failed\n";
      return 1;
    }

    const rhal::rax::Module *m = rhal::rax::GetModule(buf.data());
    std::cout << "magic: " << (m->magic() ? m->magic()->c_str() : "(null)")
              << "\n";
    if (m->version()) {
      std::cout << "version: " << m->version()->major() << "."
                << m->version()->minor() << "." << m->version()->patch()
                << "\n";
    }

    // Module attrs
    if (m->attrs() && m->attrs()->size() > 0) {
      std::cout << "module_attrs:\n";
      for (auto kv : *m->attrs())
        std::cout << "  " << (kv->key() ? kv->key()->c_str() : "?") << " = "
                  << (kv->value() ? kv->value()->c_str() : "") << "\n";
    }

    std::cout << "buffers: " << (m->buffers() ? m->buffers()->size() : 0)
              << "\n";

    // Constants detail
    auto nConst = m->constants() ? m->constants()->size() : 0;
    std::cout << "constants: " << nConst << "\n";
    for (uint32_t i = 0; i < nConst; ++i) {
      auto c = m->constants()->Get(i);
      std::cout << "  [" << c->id() << "] @"
                << (c->name() ? c->name()->c_str() : "?") << "  storage="
                << rhal::rax::EnumNameConstantStorage(c->storage())
                << "  uri=" << (c->uri() ? c->uri()->c_str() : "") << "\n";
    }

    // Code objects detail
    auto nCode = m->code_objects() ? m->code_objects()->size() : 0;
    std::cout << "code_objects: " << nCode << "\n";
    for (uint32_t i = 0; i < nCode; ++i) {
      auto co = m->code_objects()->Get(i);
      std::cout << "  [" << co->id() << "] @"
                << (co->name() ? co->name()->c_str() : "?")
                << "  kind=" << rhal::rax::EnumNameCodeObjectKind(co->kind())
                << "  uri=" << (co->uri() ? co->uri()->c_str() : "") << "\n";
    }

    // Functions
    auto nFunc = m->functions() ? m->functions()->size() : 0;
    std::cout << "functions: " << nFunc << "\n";
    for (uint32_t i = 0; i < nFunc; ++i) {
      auto f = m->functions()->Get(i);
      std::cout << "  @" << (f->name() ? f->name()->c_str() : "?") << "\n";
    }
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
