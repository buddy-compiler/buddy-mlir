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

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

namespace {

constexpr std::array<char, 8> kPayloadIndexMagic = {'R', 'A', 'X', 'I',
                                                    'D', 'X', '0', '1'};
constexpr std::array<char, 8> kPayloadFooterMagic = {'R', 'A', 'X', 'P',
                                                     'A', 'Y', '0', '1'};
constexpr uint64_t kPayloadFooterSize = 64;

struct PayloadFooterInfo {
  uint64_t manifestSize{0};
  uint64_t payloadOffset{0};
  uint64_t payloadSize{0};
  uint64_t indexOffset{0};
  uint64_t indexSize{0};
  uint64_t flags{0};
};

static uint16_t readU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t readU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t readU64LE(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

static std::vector<uint8_t> readRange(const char *path, uint64_t offset,
                                      uint64_t bytes) {
  if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    throw std::runtime_error("range too large");

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error(std::string("cannot open: ") + path);

  ifs.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!ifs)
    throw std::runtime_error("seek failed");

  std::vector<uint8_t> buf(static_cast<size_t>(bytes));
  if (bytes > 0) {
    ifs.read(reinterpret_cast<char *>(buf.data()),
             static_cast<std::streamsize>(bytes));
    if (!ifs)
      throw std::runtime_error("read failed");
  }
  return buf;
}

static bool parsePayloadFooter(const char *path, uint64_t fileSize,
                               PayloadFooterInfo &out) {
  if (fileSize < kPayloadFooterSize)
    return false;

  const auto footer =
      readRange(path, fileSize - kPayloadFooterSize, kPayloadFooterSize);

  if (std::memcmp(footer.data(), kPayloadFooterMagic.data(),
                  kPayloadFooterMagic.size()) != 0)
    return false;

  out.manifestSize = readU64LE(footer.data() + 8);
  out.payloadOffset = readU64LE(footer.data() + 16);
  out.payloadSize = readU64LE(footer.data() + 24);
  out.indexOffset = readU64LE(footer.data() + 32);
  out.indexSize = readU64LE(footer.data() + 40);
  out.flags = readU64LE(footer.data() + 48);

  if (out.manifestSize == 0 || out.manifestSize > fileSize)
    throw std::runtime_error("invalid payload footer: manifest_size");
  if (out.payloadOffset > fileSize ||
      out.payloadSize > fileSize - out.payloadOffset)
    throw std::runtime_error("invalid payload footer: payload range");
  if (out.indexOffset > fileSize || out.indexSize > fileSize - out.indexOffset)
    throw std::runtime_error("invalid payload footer: index range");

  return true;
}

static void printPayloadIndex(const std::vector<uint8_t> &indexBlob) {
  if (indexBlob.size() < 16)
    throw std::runtime_error("payload index too small");

  if (std::memcmp(indexBlob.data(), kPayloadIndexMagic.data(),
                  kPayloadIndexMagic.size()) != 0)
    throw std::runtime_error("payload index magic mismatch");

  const uint32_t version = readU32LE(indexBlob.data() + 8);
  const uint32_t count = readU32LE(indexBlob.data() + 12);

  std::cout << "payload_index_version: " << version << "\n";
  std::cout << "payload_entries: " << count << "\n";

  size_t cursor = 16;
  for (uint32_t i = 0; i < count; ++i) {
    if (cursor + 24 > indexBlob.size())
      throw std::runtime_error("truncated payload entry header");

    const uint16_t kind = readU16LE(indexBlob.data() + cursor);
    const uint32_t nameSize = readU32LE(indexBlob.data() + cursor + 4);
    const uint64_t fileOffset = readU64LE(indexBlob.data() + cursor + 8);
    const uint64_t fileSize = readU64LE(indexBlob.data() + cursor + 16);
    cursor += 24;

    if (nameSize == 0 || cursor + nameSize > indexBlob.size())
      throw std::runtime_error("truncated payload entry name");

    std::string name(reinterpret_cast<const char *>(indexBlob.data() + cursor),
                     nameSize);
    cursor += nameSize;

    std::cout << "  [" << i << "] kind=" << kind << " name=" << name
              << " offset=" << fileOffset << " size=" << fileSize << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: rax-inspect <file.rax>\n";
    return 2;
  }

  try {
    const char *path = argv[1];

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
      throw std::runtime_error(std::string("cannot open: ") + path);
    ifs.seekg(0, std::ios::end);
    const uint64_t fileSize = static_cast<uint64_t>(ifs.tellg());

    PayloadFooterInfo footer;
    const bool hasPayload = parsePayloadFooter(path, fileSize, footer);

    const uint64_t manifestSize = hasPayload ? footer.manifestSize : fileSize;
    const auto manifestBlob = readRange(path, 0, manifestSize);

    // Verify file identifier "RAX0"
    if (manifestBlob.size() < 8 ||
        !flatbuffers::BufferHasIdentifier(manifestBlob.data(), "RAX0")) {
      std::cerr << "not a valid RAX file (missing identifier RAX0)\n";
      return 1;
    }

    flatbuffers::Verifier verifier(manifestBlob.data(), manifestBlob.size());
    if (!rhal::rax::VerifyModuleBuffer(verifier)) {
      std::cerr << "RAX verification failed\n";
      return 1;
    }

    const rhal::rax::Module *m = rhal::rax::GetModule(manifestBlob.data());
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
    const auto nConst = m->constants() ? m->constants()->size() : 0;
    std::cout << "constants: " << nConst << "\n";
    for (uint32_t i = 0; i < nConst; ++i) {
      auto c = m->constants()->Get(i);
      std::cout << "  [" << c->id() << "] @"
                << (c->name() ? c->name()->c_str() : "?") << "  storage="
                << rhal::rax::EnumNameConstantStorage(c->storage())
                << "  uri=" << (c->uri() ? c->uri()->c_str() : "") << "\n";
    }

    // Code objects detail
    const auto nCode = m->code_objects() ? m->code_objects()->size() : 0;
    std::cout << "code_objects: " << nCode << "\n";
    for (uint32_t i = 0; i < nCode; ++i) {
      auto co = m->code_objects()->Get(i);
      std::cout << "  [" << co->id() << "] @"
                << (co->name() ? co->name()->c_str() : "?")
                << "  kind=" << rhal::rax::EnumNameCodeObjectKind(co->kind())
                << "  uri=" << (co->uri() ? co->uri()->c_str() : "") << "\n";
    }

    // Functions
    const auto nFunc = m->functions() ? m->functions()->size() : 0;
    std::cout << "functions: " << nFunc << "\n";
    for (uint32_t i = 0; i < nFunc; ++i) {
      auto f = m->functions()->Get(i);
      std::cout << "  @" << (f->name() ? f->name()->c_str() : "?") << "\n";
    }

    if (hasPayload) {
      std::cout << "payload_footer:\n";
      std::cout << "  manifest_size = " << footer.manifestSize << "\n";
      std::cout << "  payload_offset = " << footer.payloadOffset << "\n";
      std::cout << "  payload_size = " << footer.payloadSize << "\n";
      std::cout << "  index_offset = " << footer.indexOffset << "\n";
      std::cout << "  index_size = " << footer.indexSize << "\n";
      std::cout << "  flags = " << footer.flags << "\n";

      const auto indexBlob =
          readRange(path, footer.indexOffset, footer.indexSize);
      printPayloadIndex(indexBlob);
    }

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
