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
//   mandatory code_objects   → primary HostSharedLib URI (e.g.
//   "file:model.so"), then required dependencies. Code objects explicitly
//   marked optional stay in codeObjects for their consumer to select/load.
//   module_attrs["vocab_uri"] → vocab file URI         (e.g.
//   "file:vocab.txt") module_attrs["runner_library"] → runner plugin URI
//   module_attrs["model_library"] → server model plugin URI
//   module_attrs["model_name"]→ model identifier      (e.g.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// FlatBuffers + generated RAX schema (provided by buddy_rax INTERFACE target)
#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

namespace buddy {
namespace runtime {

namespace detail {
/// Directory root for extracted RAX payload blobs (per-user cache).
/// Default: $HOME/.buddy/rax_payload. Override with BUDDY_RAX_PAYLOAD_DIR.
/// If HOME is unset, falls back to temp_directory_path()/buddy_rax_payload.
inline std::filesystem::path buddyRaxPayloadBaseDir() {
  namespace fs = std::filesystem;
  if (const char *env = std::getenv("BUDDY_RAX_PAYLOAD_DIR")) {
    if (env[0] != '\0')
      return fs::path(env);
  }
  if (const char *home = std::getenv("HOME")) {
    if (home[0] != '\0')
      return fs::path(home) / ".buddy" / "rax_payload";
  }
  return fs::temp_directory_path() / "buddy_rax_payload";
}
} // namespace detail

struct ModelManifest {
  struct ResolvedCodeObject {
    uint32_t id = 0;
    std::string name;
    rhal::rax::CodeObjectKind kind = rhal::rax::CodeObjectKind_Unknown;
    std::string backend;
    std::string uri;
    std::string path;
    std::string entrySymbol;
    std::unordered_map<std::string, std::string> attrs;
    // Optional code objects stay discoverable but are never mandatory dlopen
    // dependencies. Their consumer decides whether to load or use a fallback.
    bool optional = false;
  };

  struct ResolvedConstant {
    uint32_t id = 0;
    std::string name;
    rhal::rax::ConstantStorage storage = rhal::rax::ConstantStorage_Inline;
    std::string uri;
    std::string path;
    std::unordered_map<std::string, std::string> attrs;
  };

  struct ResolvedBuffer {
    uint32_t id = 0;
    std::string name;
    rhal::rax::DType dtype = rhal::rax::DType_Invalid;
    rhal::rax::Layout layout = rhal::rax::Layout_Any;
    rhal::rax::MemorySpace space = rhal::rax::MemorySpace_Any;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
  };

  // One DispatchOp: which code object to run, with which args and launch dims.
  struct ResolvedDispatch {
    uint32_t codeObjectId = 0;
    // This convenience view groups tensors then i64 scalars. Consumers must
    // reject programs it cannot faithfully represent instead of dropping args.
    bool supportedArguments = true;
    // Buffer ids in ABI order (tensor arguments).
    std::vector<uint32_t> bufferArgs;
    // Scalar arguments (i64) appended after the buffer args.
    std::vector<int64_t> scalars;
    std::vector<uint32_t> grid;
    std::vector<uint32_t> block;
  };

  struct ResolvedFunction {
    std::string name;
    bool hasUnsupportedOps = false;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    // Dispatch ops (kind == Dispatch) in declaration order; barriers and
    // memory ops are not exposed in this MVP view.
    std::vector<ResolvedDispatch> dispatches;
  };

  // from module_attrs["model_name"] (e.g. "deepseek_r1_fp32")
  std::string modelName;
  // absolute path to the kernel .so  (dlopen)
  std::string soPath;
  // additional mandatory HostSharedLib entries, resolved to absolute
  // paths (typically runtime deps like libomp/libmlir_c_runner_utils).
  std::vector<std::string> dependentSoPaths;
  // absolute paths to weight files, in manifest order (supports multi-weight
  // variants like w8a16 which have separate f16 + i8 files).
  std::vector<std::string> weightPaths;
  // absolute path to the vocab file   (tokenizer, may be empty)
  std::string vocabPath;
  // absolute path to the model runner plugin shared library.
  std::string runnerLibraryPath;
  // absolute path to the server model plugin shared library.
  std::string modelLibraryPath;
  // Raw module attrs plus URI-resolved variants for attrs whose value is a URI.
  std::unordered_map<std::string, std::string> moduleAttrs;
  std::unordered_map<std::string, std::string> resolvedModuleAttrs;
  // All manifest resources, including non-CPU backends such as TTNN
  // flatbuffers.
  std::vector<ResolvedCodeObject> codeObjects;
  std::vector<ResolvedConstant> constants;
  // Descriptive metadata only; storage is allocated and owned by the consumer.
  std::vector<ResolvedBuffer> buffers;
  // Dispatch program: function entry points and their kernel dispatch ops.
  std::vector<ResolvedFunction> functions;

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
    auto hasSuffix = [](const std::string &s, const char *suffix) {
      const std::string suf(suffix);
      return s.size() >= suf.size() &&
             s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
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
      extractRoot = detail::buddyRaxPayloadBaseDir() / std::to_string(keyHash);
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
        bool reuse = fs::exists(outPath, ec) && !ec &&
                     fs::is_regular_file(outPath, ec) && !ec &&
                     fs::file_size(outPath, ec) == entry.fileSize;
        // Size/mtime are only a cache lookup hint. Verify against the package
        // bytes before reusing executable payloads, including same-size edits.
        const bool executablePayload =
            std::strcmp(field, "code_object.uri") == 0 ||
            hasSuffix(std::string(field), "_library");
        if (reuse && executablePayload) {
          std::ifstream cached(outPath, std::ios::binary);
          std::ifstream packed(raxFs, std::ios::binary);
          packed.seekg(static_cast<std::streamoff>(entry.fileOffset));
          std::array<char, 16384> cachedBytes{}, packedBytes{};
          uint64_t remaining = entry.fileSize;
          reuse = bool(cached) && bool(packed);
          while (reuse && remaining) {
            const auto count = static_cast<std::streamsize>(
                std::min<uint64_t>(remaining, cachedBytes.size()));
            cached.read(cachedBytes.data(), count);
            packed.read(packedBytes.data(), count);
            reuse = cached.gcount() == count && packed.gcount() == count &&
                    std::memcmp(cachedBytes.data(), packedBytes.data(),
                                static_cast<size_t>(count)) == 0;
            remaining -= static_cast<uint64_t>(count);
          }
        }

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

    auto attrsToMap = [](auto *attrs) {
      std::unordered_map<std::string, std::string> out;
      if (!attrs)
        return out;
      for (auto kv : *attrs) {
        if (!kv || !kv->key() || !kv->value())
          continue;
        out[kv->key()->str()] = kv->value()->str();
      }
      return out;
    };

    if (mod->buffers()) {
      for (auto buffer : *mod->buffers()) {
        if (!buffer)
          throw std::runtime_error("ModelManifest: null buffer binding");
        ResolvedBuffer rec;
        rec.id = buffer->id();
        if (buffer->name())
          rec.name = buffer->name()->str();
        rec.space = buffer->space();
        if (auto type = buffer->type()) {
          rec.dtype = type->dtype();
          rec.layout = type->layout();
          if (type->shape() && type->shape()->dims())
            rec.shape.assign(type->shape()->dims()->begin(),
                             type->shape()->dims()->end());
          if (type->strides())
            rec.strides.assign(type->strides()->begin(),
                               type->strides()->end());
        }
        out.buffers.push_back(std::move(rec));
      }
    }

    // --- Code objects -> generic list + legacy soPath fields ---------------
    if (!mod->code_objects() || mod->code_objects()->size() == 0)
      throw std::runtime_error("ModelManifest: no code_objects in " +
                               raxFs.string());
    for (auto co : *mod->code_objects()) {
      if (!co)
        throw std::runtime_error("ModelManifest: null code object");

      ResolvedCodeObject rec;
      rec.id = co->id();
      if (co->name())
        rec.name = co->name()->str();
      rec.kind = co->kind();
      if (co->backend())
        rec.backend = co->backend()->str();
      if (co->uri()) {
        rec.uri = co->uri()->str();
        rec.path = resolveUri(co->uri(), "code_object.uri");
      }
      if (co->entry_symbol())
        rec.entrySymbol = co->entry_symbol()->str();
      if (co->attrs()) {
        for (auto kv : *co->attrs()) {
          if (!kv || !kv->key() || !kv->value())
            throw std::runtime_error(
                "ModelManifest: invalid code object metadata");
          if (!rec.attrs.emplace(kv->key()->str(), kv->value()->str()).second)
            throw std::runtime_error(
                "ModelManifest: duplicate code object metadata key");
        }
      }
      auto optional = rec.attrs.find("optional");
      if (optional != rec.attrs.end()) {
        if (optional->second != "true" && optional->second != "false")
          throw std::runtime_error(
              "ModelManifest: optional must be 'true' or 'false'");
        rec.optional = optional->second == "true";
      }
      out.codeObjects.push_back(rec);

      if (co->kind() == rhal::rax::CodeObjectKind_HostSharedLib &&
          !rec.optional) {
        if (out.soPath.empty())
          out.soPath = rec.path;
        else
          out.dependentSoPaths.push_back(rec.path);
      }
    }

    // --- Functions -> dispatch metadata -------------------------------------
    if (mod->functions()) {
      for (auto fn : *mod->functions()) {
        if (!fn)
          throw std::runtime_error("ModelManifest: null function");
        ResolvedFunction rec;
        if (fn->name())
          rec.name = fn->name()->str();
        if (fn->inputs())
          rec.inputs.assign(fn->inputs()->begin(), fn->inputs()->end());
        if (fn->outputs())
          rec.outputs.assign(fn->outputs()->begin(), fn->outputs()->end());
        if (fn->ops()) {
          bool sawTrailingBarrier = false;
          for (auto op : *fn->ops()) {
            // rax-pack appends a plain global barrier to a CPU dispatch. A
            // synchronous host call already completes it; other ops, barrier
            // scopes or program orders must not be silently dropped.
            if (op && op->kind() == rhal::rax::OpKind_Barrier &&
                op->barrier() && !op->dispatch() && !op->memcpy() &&
                !op->fill() && (!op->attrs() || op->attrs()->size() == 0) &&
                (!op->barrier()->attrs() ||
                 op->barrier()->attrs()->size() == 0) &&
                rec.dispatches.size() == 1 && !sawTrailingBarrier) {
              sawTrailingBarrier = true;
              continue;
            }
            if (!op || op->kind() != rhal::rax::OpKind_Dispatch ||
                sawTrailingBarrier || op->barrier() || op->memcpy() ||
                op->fill()) {
              rec.hasUnsupportedOps = true;
              continue;
            }
            const auto *dispatch = op->dispatch();
            if (!dispatch) {
              rec.hasUnsupportedOps = true;
              continue;
            }
            ResolvedDispatch rd;
            rd.codeObjectId = dispatch->code_object_id();
            if (dispatch->args()) {
              bool sawScalar = false;
              for (auto arg : *dispatch->args()) {
                if (!arg) {
                  rd.supportedArguments = false;
                  continue;
                }
                const unsigned alternatives =
                    unsigned(arg->buffer_id() != 0) +
                    unsigned(arg->constant_id() != 0) +
                    unsigned(arg->scalar() != nullptr);
                if (alternatives != 1)
                  rd.supportedArguments = false;
                if (arg->buffer_id()) {
                  if (sawScalar)
                    rd.supportedArguments = false;
                  rd.bufferArgs.push_back(arg->buffer_id());
                } else if (arg->scalar()) {
                  sawScalar = true;
                  if (arg->scalar()->f64() != 0.0 || arg->scalar()->s())
                    rd.supportedArguments = false;
                  rd.scalars.push_back(arg->scalar()->i64());
                } else {
                  rd.supportedArguments = false;
                }
              }
            }
            if (dispatch->dims()) {
              if (dispatch->dims()->grid())
                rd.grid.assign(dispatch->dims()->grid()->begin(),
                               dispatch->dims()->grid()->end());
              if (dispatch->dims()->block())
                rd.block.assign(dispatch->dims()->block()->begin(),
                                dispatch->dims()->block()->end());
            }
            rec.dispatches.push_back(std::move(rd));
          }
        }
        out.functions.push_back(std::move(rec));
      }
    }

    // --- Constants → generic list + legacy weightPaths ---------------------
    if (mod->constants()) {
      for (auto c : *mod->constants()) {
        if (!c)
          continue;
        ResolvedConstant rec;
        rec.id = c->id();
        if (c->name())
          rec.name = c->name()->str();
        rec.storage = c->storage();
        if (c->uri()) {
          rec.uri = c->uri()->str();
          rec.path = resolveUri(c->uri(), "constant.uri");
        }
        rec.attrs = attrsToMap(c->attrs());
        out.constants.push_back(rec);

        if (c->storage() == rhal::rax::ConstantStorage_External)
          out.weightPaths.push_back(rec.path);
      }
    }

    // --- Module attrs: keep all strings; resolve URI-like values -----------
    bool sawModelLibrary = false;
    if (mod->attrs()) {
      for (auto kv : *mod->attrs()) {
        if (!kv->key() || !kv->value())
          continue;
        std::string key = kv->key()->str();
        std::string value = kv->value()->str();
        if (key == "model_library") {
          if (sawModelLibrary)
            throw std::runtime_error(
                "ModelManifest: duplicate model_library attribute");
          sawModelLibrary = true;
        }
        out.moduleAttrs[key] = value;
        if (hasPrefix(value, "file:") || hasPrefix(value, "payload:") ||
            hasSuffix(key, "_uri") || key == "model_library")
          out.resolvedModuleAttrs[key] = resolveUri(kv->value(), key.c_str());
        if (key == "vocab_uri" && kv->value() && kv->value()->size() > 0)
          out.vocabPath = out.resolvedModuleAttrs[key];
        else if (key == "runner_library" && kv->value() &&
                 kv->value()->size() > 0)
          out.runnerLibraryPath = resolveUri(kv->value(), "runner_library");
        else if (key == "model_library" && kv->value() &&
                 kv->value()->size() > 0)
          out.modelLibraryPath = resolveUri(kv->value(), "model_library");
        else if (key == "model_name" && kv->value() && kv->value()->size() > 0)
          out.modelName = value;
      }
    }

    return out;
  }
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_MODELMANIFEST_H
