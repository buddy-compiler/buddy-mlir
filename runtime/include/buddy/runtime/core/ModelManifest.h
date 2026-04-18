//===- ModelManifest.h - Read .rax manifest and resolve paths -------------===//
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
//
// The .rax file produced by:
//   rax-pack model.mlir -o model.rax
//
// encodes:
//   constants[0]              → External weights URI  (e.g. "file:arg0.data")
//   code_objects[0]           → primary HostSharedLib URI (e.g.
//   "file:model.so") code_objects[1..]         → optional dependent
//   HostSharedLib URIs module_attrs["vocab_uri"] → vocab file URI         (e.g.
//   "file:vocab.txt") module_attrs["model_name"]→ model identifier      (e.g.
//   "deepseek_r1_fp32")
//
// URI schemes:
//   file:X     → resolve relative to the .rax directory.
//   payload:X  → extract entry X from the optional payload section appended
//                after the RAX0 FlatBuffer (rax-pack --embed-payload).
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_MODELMANIFEST_H
#define BUDDY_RUNTIME_CORE_MODELMANIFEST_H

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// FlatBuffers + generated RAX schema (provided by buddy_rax INTERFACE target)
#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

namespace buddy {
namespace runtime {

struct ModelManifest {
  // from module_attrs["model_name"] (e.g. "deepseek_r1_fp32")
  std::string modelName;
  // absolute path to the kernel .so  (dlopen)
  std::string soPath;
  // additional HostSharedLib entries from the manifest, resolved to absolute
  // paths (typically runtime deps like libomp/libmlir_c_runner_utils).
  std::vector<std::string> dependentSoPaths;
  // absolute paths to weight files, in manifest order (supports multi-weight
  // variants like w8a16 which have separate f16 + i8 files).
  std::vector<std::string> weightPaths;
  // absolute path to the vocab file   (tokenizer, may be empty)
  std::string vocabPath;

  // Load and resolve from a .rax manifest file.
  // Throws std::runtime_error on any parse / missing-field error.
  static ModelManifest loadFromRax(const std::string &raxPath) {
    namespace fs = std::filesystem;

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

    struct PayloadEntry {
      uint16_t kind{0};
      uint64_t fileOffset{0};
      uint64_t fileSize{0};
    };

    auto hasPrefix = [](const std::string &s, const char *prefix) {
      return s.rfind(prefix, 0) == 0;
    };

    auto readU16LE = [](const uint8_t *p) -> uint16_t {
      return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    };

    auto readU32LE = [](const uint8_t *p) -> uint32_t {
      return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
             (static_cast<uint32_t>(p[2]) << 16) |
             (static_cast<uint32_t>(p[3]) << 24);
    };

    auto readU64LE = [](const uint8_t *p) -> uint64_t {
      uint64_t v = 0;
      for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
      return v;
    };

    auto readRange = [&](const fs::path &path, uint64_t offset,
                         uint64_t bytes) -> std::vector<uint8_t> {
      if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        throw std::runtime_error(
            "ModelManifest: range too large to map into memory");

      std::ifstream ifs(path, std::ios::binary);
      if (!ifs)
        throw std::runtime_error("ModelManifest: cannot open " + path.string());

      ifs.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      if (!ifs)
        throw std::runtime_error("ModelManifest: seek failed in " +
                                 path.string());

      std::vector<uint8_t> out(static_cast<size_t>(bytes));
      if (bytes > 0) {
        ifs.read(reinterpret_cast<char *>(out.data()),
                 static_cast<std::streamsize>(bytes));
        if (!ifs)
          throw std::runtime_error("ModelManifest: read range failed: " +
                                   path.string());
      }
      return out;
    };

    auto parsePayloadFooter = [&](const fs::path &path, uint64_t fileSize,
                                  PayloadFooterInfo &info) -> bool {
      if (fileSize < kPayloadFooterSize)
        return false;

      const auto footerBytes =
          readRange(path, fileSize - kPayloadFooterSize, kPayloadFooterSize);
      if (std::memcmp(footerBytes.data(), kPayloadFooterMagic.data(),
                      kPayloadFooterMagic.size()) != 0)
        return false;

      info.manifestSize = readU64LE(footerBytes.data() + 8);
      info.payloadOffset = readU64LE(footerBytes.data() + 16);
      info.payloadSize = readU64LE(footerBytes.data() + 24);
      info.indexOffset = readU64LE(footerBytes.data() + 32);
      info.indexSize = readU64LE(footerBytes.data() + 40);
      info.flags = readU64LE(footerBytes.data() + 48);

      if (info.manifestSize == 0 || info.manifestSize > fileSize)
        throw std::runtime_error(
            "ModelManifest: invalid payload footer (manifest_size)");

      if (info.payloadOffset > fileSize ||
          info.payloadSize > fileSize - info.payloadOffset)
        throw std::runtime_error(
            "ModelManifest: invalid payload footer (payload range)");

      if (info.indexOffset > fileSize ||
          info.indexSize > fileSize - info.indexOffset)
        throw std::runtime_error(
            "ModelManifest: invalid payload footer (index range)");

      const uint64_t footerOffset = fileSize - kPayloadFooterSize;
      if (info.indexOffset + info.indexSize > footerOffset)
        throw std::runtime_error(
            "ModelManifest: invalid payload footer (index overlaps footer)");

      if (info.payloadOffset != info.manifestSize)
        throw std::runtime_error("ModelManifest: payload layout mismatch "
                                 "(payload_offset != manifest_size)");

      return true;
    };

    auto parsePayloadIndex = [&](const std::vector<uint8_t> &blob,
                                 uint64_t fileSize,
                                 const PayloadFooterInfo &footer)
        -> std::unordered_map<std::string, PayloadEntry> {
      if (blob.size() < 16)
        throw std::runtime_error("ModelManifest: payload index too small");

      if (std::memcmp(blob.data(), kPayloadIndexMagic.data(),
                      kPayloadIndexMagic.size()) != 0)
        throw std::runtime_error("ModelManifest: payload index magic mismatch");

      const uint32_t version = readU32LE(blob.data() + 8);
      if (version != 1)
        throw std::runtime_error(
            "ModelManifest: unsupported payload index version");

      const uint32_t count = readU32LE(blob.data() + 12);
      size_t cursor = 16;

      std::unordered_map<std::string, PayloadEntry> entries;
      entries.reserve(count);

      for (uint32_t i = 0; i < count; ++i) {
        if (cursor + 24 > blob.size())
          throw std::runtime_error(
              "ModelManifest: truncated payload index entry header");

        const uint16_t kind = readU16LE(blob.data() + cursor);
        const uint32_t nameSize = readU32LE(blob.data() + cursor + 4);
        const uint64_t fileOffset = readU64LE(blob.data() + cursor + 8);
        const uint64_t fileBytes = readU64LE(blob.data() + cursor + 16);
        cursor += 24;

        if (nameSize == 0 || cursor + nameSize > blob.size())
          throw std::runtime_error(
              "ModelManifest: truncated payload index entry name");

        std::string name(reinterpret_cast<const char *>(blob.data() + cursor),
                         nameSize);
        cursor += nameSize;

        if (entries.count(name))
          throw std::runtime_error(
              "ModelManifest: duplicate payload entry name: " + name);

        if (fileOffset > fileSize || fileBytes > fileSize - fileOffset)
          throw std::runtime_error(
              "ModelManifest: payload entry out of file bounds: " + name);

        if (fileOffset < footer.payloadOffset ||
            fileOffset + fileBytes > footer.payloadOffset + footer.payloadSize)
          throw std::runtime_error(
              "ModelManifest: payload entry out of payload section: " + name);

        entries.emplace(std::move(name),
                        PayloadEntry{kind, fileOffset, fileBytes});
      }

      return entries;
    };

    // --- Open file and detect optional payload footer ----------------------
    const fs::path raxFs = fs::absolute(fs::path(raxPath));
    std::ifstream ifs(raxFs, std::ios::binary);
    if (!ifs)
      throw std::runtime_error("ModelManifest: cannot open " + raxFs.string());

    ifs.seekg(0, std::ios::end);
    const uint64_t fileSize = static_cast<uint64_t>(ifs.tellg());
    if (fileSize == 0)
      throw std::runtime_error("ModelManifest: empty file: " + raxFs.string());

    PayloadFooterInfo footer;
    const bool hasPayload = parsePayloadFooter(raxFs, fileSize, footer);
    const uint64_t manifestSize = hasPayload ? footer.manifestSize : fileSize;

    // --- Read and verify manifest (RAX0 FlatBuffer) ------------------------
    const std::vector<uint8_t> manifestBytes =
        readRange(raxFs, 0, manifestSize);

    if (manifestBytes.size() < 8 ||
        !flatbuffers::BufferHasIdentifier(manifestBytes.data(), "RAX0"))
      throw std::runtime_error("ModelManifest: not a valid RAX0 file: " +
                               raxFs.string());

    flatbuffers::Verifier ver(manifestBytes.data(), manifestBytes.size());
    if (!rhal::rax::VerifyModuleBuffer(ver))
      throw std::runtime_error("ModelManifest: RAX verification failed: " +
                               raxFs.string());

    const rhal::rax::Module *mod = rhal::rax::GetModule(manifestBytes.data());

    std::unordered_map<std::string, PayloadEntry> payloadEntries;
    if (hasPayload) {
      const auto indexBlob =
          readRange(raxFs, footer.indexOffset, footer.indexSize);
      payloadEntries = parsePayloadIndex(indexBlob, fileSize, footer);
    }

    const fs::path baseDir = raxFs.parent_path();

    // Stable extraction cache directory per (path,size,mtime).
    fs::path extractRoot;
    if (hasPayload) {
      uint64_t mtimeEpoch = 0;
      try {
        mtimeEpoch = static_cast<uint64_t>(
            fs::last_write_time(raxFs).time_since_epoch().count());
      } catch (...) {
        mtimeEpoch = 0;
      }

      const std::string key = raxFs.string() + ":" + std::to_string(fileSize) +
                              ":" + std::to_string(mtimeEpoch);
      const size_t keyHash = std::hash<std::string>{}(key);
      extractRoot = fs::temp_directory_path() / "buddy_rax_payload" /
                    std::to_string(keyHash);
      fs::create_directories(extractRoot);
    }

    auto resolveUri = [&](const flatbuffers::String *uri,
                          const char *field) -> std::string {
      if (!uri || uri->size() == 0)
        throw std::runtime_error(
            std::string("ModelManifest: missing URI for ") + field);

      const std::string s = uri->str();

      if (hasPrefix(s, "file:"))
        return (baseDir / fs::path(s.substr(5))).string();

      if (hasPrefix(s, "payload:")) {
        if (!hasPayload)
          throw std::runtime_error(
              "ModelManifest: payload URI found but payload section missing");

        const std::string name = s.substr(8);
        if (name.empty() || name.find("..") != std::string::npos ||
            name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos) {
          throw std::runtime_error(
              "ModelManifest: invalid payload entry name: " + name);
        }

        const auto it = payloadEntries.find(name);
        if (it == payloadEntries.end())
          throw std::runtime_error("ModelManifest: payload entry not found: " +
                                   name);

        const PayloadEntry &entry = it->second;
        const fs::path outPath = extractRoot / name;

        std::error_code ec;
        const bool reuse = fs::exists(outPath, ec) && !ec &&
                           fs::is_regular_file(outPath, ec) && !ec &&
                           fs::file_size(outPath, ec) == entry.fileSize;

        if (!reuse) {
          fs::create_directories(outPath.parent_path());

          fs::path tmpPath = outPath;
          tmpPath += ".tmp";

          std::ifstream src(raxFs, std::ios::binary);
          if (!src)
            throw std::runtime_error("ModelManifest: cannot reopen " +
                                     raxFs.string());

          src.seekg(static_cast<std::streamoff>(entry.fileOffset),
                    std::ios::beg);
          if (!src)
            throw std::runtime_error(
                "ModelManifest: seek failed when extracting payload");

          std::ofstream dst(tmpPath, std::ios::binary | std::ios::trunc);
          if (!dst)
            throw std::runtime_error(
                "ModelManifest: cannot create extracted payload file: " +
                tmpPath.string());

          static constexpr size_t kChunk = 4 * 1024 * 1024;
          std::vector<char> buf(kChunk);
          uint64_t remain = entry.fileSize;

          while (remain > 0) {
            const size_t chunk = static_cast<size_t>(
                std::min<uint64_t>(remain, static_cast<uint64_t>(buf.size())));
            src.read(buf.data(), static_cast<std::streamsize>(chunk));
            if (static_cast<size_t>(src.gcount()) != chunk) {
              throw std::runtime_error(
                  "ModelManifest: short read while extracting payload entry: " +
                  name);
            }
            dst.write(buf.data(), static_cast<std::streamsize>(chunk));
            if (!dst)
              throw std::runtime_error("ModelManifest: write failed while "
                                       "extracting payload entry: " +
                                       name);
            remain -= static_cast<uint64_t>(chunk);
          }

          dst.close();
          if (!dst)
            throw std::runtime_error(
                "ModelManifest: close failed while extracting payload entry: " +
                name);

          std::error_code renameEc;
          fs::rename(tmpPath, outPath, renameEc);
          if (renameEc) {
            std::error_code rmEc;
            fs::remove(outPath, rmEc);
            renameEc.clear();
            fs::rename(tmpPath, outPath, renameEc);
            if (renameEc)
              throw std::runtime_error(
                  "ModelManifest: rename failed for extracted payload entry: " +
                  name);
          }
        }

        return fs::absolute(outPath).string();
      }

      // passthrough for other URI schemes
      return s;
    };

    ModelManifest out;

    // --- Code objects -> soPath + dependentSoPaths -------------------------
    if (!mod->code_objects() || mod->code_objects()->size() == 0)
      throw std::runtime_error("ModelManifest: no code_objects in " +
                               raxFs.string());
    for (auto co : *mod->code_objects()) {
      if (co->kind() != rhal::rax::CodeObjectKind_HostSharedLib)
        continue;

      const std::string resolved = resolveUri(co->uri(), "code_object.uri");
      if (out.soPath.empty())
        out.soPath = resolved;
      else
        out.dependentSoPaths.push_back(resolved);
    }
    if (out.soPath.empty())
      throw std::runtime_error(
          "ModelManifest: no HostSharedLib code object in " + raxFs.string());

    // --- Constants → weightPaths (all External constants, in order) ------
    if (!mod->constants() || mod->constants()->size() == 0)
      throw std::runtime_error("ModelManifest: no constants in " +
                               raxFs.string());
    for (auto c : *mod->constants()) {
      if (c->storage() == rhal::rax::ConstantStorage_External)
        out.weightPaths.push_back(resolveUri(c->uri(), "constant.uri"));
    }
    if (out.weightPaths.empty())
      throw std::runtime_error(
          "ModelManifest: no External constant (weights) in " + raxFs.string());

    // --- Module attrs: vocab_uri, model_name (optional) ------------------
    if (mod->attrs()) {
      for (auto kv : *mod->attrs()) {
        if (!kv->key())
          continue;
        std::string key = kv->key()->str();
        if (key == "vocab_uri" && kv->value() && kv->value()->size() > 0)
          out.vocabPath = resolveUri(kv->value(), "vocab_uri");
        else if (key == "model_name" && kv->value() && kv->value()->size() > 0)
          out.modelName = kv->value()->str();
      }
    }

    return out;
  }
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MODELMANIFEST_H
