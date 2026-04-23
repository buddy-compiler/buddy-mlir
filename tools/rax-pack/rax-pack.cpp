//===- rax-pack.cpp - rax-pack: MLIR .rhal dialect → binary .rax ----------===//
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
// Parses an MLIR file containing an `rhal.module` op and serializes it to a
// FlatBuffer .rax file understood by buddy-cli and ModelManifest.
//
// Usage:  rax-pack <input.mlir> -o <output.rax> [--embed-payload]
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "buddy/runtime/rax/RAX.h"
#include "flatbuffers/flatbuffers.h"

#include "RHAL/RHALDialect.h"
#include "RHAL/RHALOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace rhal::rax;
namespace fs = std::filesystem;

//===----------------------------------------------------------------------===//
// Payload container helpers (RAX0 FlatBuffer + appended payload/index/footer)
//===----------------------------------------------------------------------===//

namespace {

constexpr std::array<char, 8> kPayloadIndexMagic = {'R', 'A', 'X', 'I',
                                                    'D', 'X', '0', '1'};
constexpr std::array<char, 8> kPayloadFooterMagic = {'R', 'A', 'X', 'P',
                                                     'A', 'Y', '0', '1'};

enum class PayloadKind : uint16_t {
  Unknown = 0,
  Constant = 1,
  CodeObject = 2,
  Vocab = 3,
};

struct PayloadInput {
  PayloadKind kind{PayloadKind::Unknown};
  std::string entryName;
  fs::path sourcePath;
  uint64_t sizeBytes{0};
  uint64_t fileOffset{0};
};

static bool hasPrefix(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

static std::string sanitizeEntryName(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name) {
    if (std::isalnum(c) || c == '.' || c == '_' || c == '-')
      out.push_back(static_cast<char>(c));
    else
      out.push_back('_');
  }
  if (out.empty())
    out = "payload.bin";
  return out;
}

static std::string
makeUniqueEntryName(const std::string &raw,
                    std::unordered_map<std::string, int> &nameUseCounter) {
  const std::string base = sanitizeEntryName(raw);
  auto it = nameUseCounter.find(base);
  if (it == nameUseCounter.end()) {
    nameUseCounter[base] = 1;
    return base;
  }

  int nextId = it->second;
  std::string candidate;
  do {
    candidate = base + "_" + std::to_string(nextId++);
  } while (nameUseCounter.count(candidate));

  it->second = nextId;
  nameUseCounter[candidate] = 1;
  return candidate;
}

static fs::path resolveFileUri(const std::string &uri,
                               const fs::path &outputPath) {
  if (!hasPrefix(uri, "file:"))
    throw std::runtime_error("URI is not file:*: " + uri);

  fs::path p(uri.substr(5));
  if (p.is_absolute())
    return p;
  return outputPath.parent_path() / p;
}

static void appendBytes(std::vector<uint8_t> &out, const void *data,
                        size_t size) {
  const auto *p = static_cast<const uint8_t *>(data);
  out.insert(out.end(), p, p + size);
}

static void appendU16LE(std::vector<uint8_t> &out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xff));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}

static void appendU32LE(std::vector<uint8_t> &out, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

static void appendU64LE(std::vector<uint8_t> &out, uint64_t v) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

static void copyFileToStream(const fs::path &src, uint64_t expectedSize,
                             std::ofstream &ofs) {
  std::ifstream ifs(src, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("cannot open payload source: " + src.string());

  static constexpr size_t kChunk = 4 * 1024 * 1024;
  std::vector<char> buf(kChunk);

  uint64_t remaining = expectedSize;
  while (remaining > 0) {
    size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, static_cast<uint64_t>(buf.size())));
    ifs.read(buf.data(), static_cast<std::streamsize>(chunk));
    if (static_cast<size_t>(ifs.gcount()) != chunk)
      throw std::runtime_error("short read from payload source: " +
                               src.string());
    ofs.write(buf.data(), static_cast<std::streamsize>(chunk));
    if (!ofs)
      throw std::runtime_error("write failed while writing payload");
    remaining -= static_cast<uint64_t>(chunk);
  }
}

static std::vector<uint8_t>
buildPayloadIndexBlob(const std::vector<PayloadInput> &entries) {
  std::vector<uint8_t> out;
  out.reserve(16 + entries.size() * 64);

  appendBytes(out, kPayloadIndexMagic.data(), kPayloadIndexMagic.size());
  appendU32LE(out, 1); // index version
  appendU32LE(out, static_cast<uint32_t>(entries.size()));

  for (const auto &e : entries) {
    appendU16LE(out, static_cast<uint16_t>(e.kind));
    appendU16LE(out, 0); // reserved
    appendU32LE(out, static_cast<uint32_t>(e.entryName.size()));
    appendU64LE(out, e.fileOffset);
    appendU64LE(out, e.sizeBytes);
    appendBytes(out, e.entryName.data(), e.entryName.size());
  }

  return out;
}

static std::vector<uint8_t>
buildPayloadFooter(uint64_t manifestSize, uint64_t payloadOffset,
                   uint64_t payloadSize, uint64_t indexOffset,
                   uint64_t indexSize, uint64_t flags) {
  std::vector<uint8_t> out;
  out.reserve(64);

  appendBytes(out, kPayloadFooterMagic.data(), kPayloadFooterMagic.size());
  appendU64LE(out, manifestSize);
  appendU64LE(out, payloadOffset);
  appendU64LE(out, payloadSize);
  appendU64LE(out, indexOffset);
  appendU64LE(out, indexSize);
  appendU64LE(out, flags);
  appendU64LE(out, 0); // reserved

  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Type/enum helpers
//===----------------------------------------------------------------------===//

static DType getDType(mlir::Type elemType) {
  if (elemType.isSignlessInteger(8))
    return DType_I8;
  if (elemType.isSignlessInteger(16))
    return DType_I16;
  if (elemType.isSignlessInteger(32))
    return DType_I32;
  if (elemType.isSignlessInteger(64))
    return DType_I64;
  if (elemType.isUnsignedInteger(8))
    return DType_U8;
  if (elemType.isUnsignedInteger(16))
    return DType_U16;
  if (elemType.isUnsignedInteger(32))
    return DType_U32;
  if (elemType.isUnsignedInteger(64))
    return DType_U64;
  if (elemType.isF16())
    return DType_F16;
  if (elemType.isBF16())
    return DType_BF16;
  if (elemType.isF32())
    return DType_F32;
  if (elemType.isF64())
    return DType_F64;
  return DType_Invalid;
}

static MemorySpace parseSpace(llvm::StringRef s) {
  if (s == "host" || s == "Host")
    return MemorySpace_Host;
  if (s == "dram" || s == "DRAM")
    return MemorySpace_DRAM;
  if (s == "sram" || s == "SRAM")
    return MemorySpace_SRAM;
  if (s == "device_local" || s == "DeviceLocal")
    return MemorySpace_DeviceLocal;
  return MemorySpace_Any;
}

static CodeObjectKind parseCodeKind(llvm::StringRef s) {
  if (s == "host_shared_lib" || s == "HostSharedLib")
    return CodeObjectKind_HostSharedLib;
  if (s == "host_elf" || s == "HostELF")
    return CodeObjectKind_HostELF;
  if (s == "device_elf" || s == "DeviceELF")
    return CodeObjectKind_DeviceELF;
  if (s == "raw_bytes" || s == "RawBytes")
    return CodeObjectKind_RawBytes;
  if (s == "bytecode" || s == "Bytecode")
    return CodeObjectKind_Bytecode;
  return CodeObjectKind_Unknown;
}

static void parseVersion(llvm::StringRef v, uint16_t &maj, uint16_t &min,
                         uint16_t &pat) {
  maj = 0;
  min = 1;
  pat = 0;
  auto dot1 = v.find('.');
  if (dot1 == llvm::StringRef::npos)
    return;
  v.substr(0, dot1).getAsInteger(10, maj);
  auto rest = v.substr(dot1 + 1);
  auto dot2 = rest.find('.');
  if (dot2 == llvm::StringRef::npos) {
    rest.getAsInteger(10, min);
    return;
  }
  rest.substr(0, dot2).getAsInteger(10, min);
  rest.substr(dot2 + 1).getAsInteger(10, pat);
}

//===----------------------------------------------------------------------===//
// Shape / type helpers
//===----------------------------------------------------------------------===//

static flatbuffers::Offset<Shape> buildShape(flatbuffers::FlatBufferBuilder &b,
                                             mlir::RankedTensorType tt) {
  std::vector<int64_t> dims(tt.getShape().begin(), tt.getShape().end());
  return CreateShape(b, b.CreateVector(dims));
}

static flatbuffers::Offset<TensorType>
buildTensorType(flatbuffers::FlatBufferBuilder &b, mlir::RankedTensorType tt) {
  auto shape = buildShape(b, tt);
  auto dtype = getDType(tt.getElementType());
  return CreateTensorType(b, dtype, shape, Layout_Any, 0);
}

//===----------------------------------------------------------------------===//
// Output helper
//===----------------------------------------------------------------------===//

static void writeAll(const char *path, const uint8_t *data, size_t n) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs)
    throw std::runtime_error(std::string("cannot write: ") + path);
  ofs.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(n));
  if (!ofs)
    throw std::runtime_error("write failed");
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  const char *inputPath = nullptr;
  const char *outputPath = nullptr;
  bool embedPayload = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--embed-payload") {
      embedPayload = true;
    } else if (arg == "-o" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (!inputPath) {
      inputPath = argv[i];
    } else {
      llvm::errs() << "unexpected argument: " << arg << "\n";
      return 2;
    }
  }

  if (!inputPath || !outputPath) {
    llvm::errs()
        << "usage: rax-pack <input.mlir> -o <output.rax> [--embed-payload]\n";
    return 2;
  }

  const fs::path outputFs = fs::absolute(fs::path(outputPath));

  // ── Set up MLIR context with RHAL dialect ─────────────────────────────────
  mlir::MLIRContext context;
  context.loadDialect<buddy::rhal::RHALDialect>();

  // ── Parse the MLIR source file ────────────────────────────────────────────
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputPath, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse " << inputPath << "\n";
    return 1;
  }

  // ── Find the rhal.module op ───────────────────────────────────────────────
  buddy::rhal::ModuleOp rhalMod;
  module->walk([&](buddy::rhal::ModuleOp op) {
    rhalMod = op;
    return mlir::WalkResult::interrupt();
  });
  if (!rhalMod) {
    llvm::errs() << "error: no rhal.module found in " << inputPath << "\n";
    return 1;
  }

  try {
    flatbuffers::FlatBufferBuilder b(4096);

    // ── Module-level attrs ────────────────────────────────────────────────
    const std::string moduleName = rhalMod.getSymName().str();
    std::string modelNameAttr;
    std::string vocabUriAttr;
    if (auto v = rhalMod.getModelName())
      modelNameAttr = v->str();
    if (auto v = rhalMod.getVocabUri())
      vocabUriAttr = v->str();

    // Version
    uint16_t verMaj = 0, verMin = 1, verPat = 0;
    if (auto v = rhalMod.getVersion())
      parseVersion(*v, verMaj, verMin, verPat);

    // ── Collect children in declaration order ─────────────────────────────
    struct BufRec {
      uint32_t id;
      std::string name;
      mlir::RankedTensorType tt;
      MemorySpace space;
    };
    struct ConstRec {
      uint32_t id;
      std::string name;
      mlir::RankedTensorType tt;
      ConstantStorage storage;
      std::string uri;
    };
    struct CodeRec {
      uint32_t id;
      std::string name;
      CodeObjectKind kind;
      std::string backend;
      std::string uri;
    };
    struct FuncRec {
      std::string name;
      std::vector<std::string> inputs;
      std::vector<std::string> outputs;
      std::string dispatch;
      std::vector<std::string> args;
    };

    std::vector<BufRec> buffers;
    std::vector<ConstRec> constants;
    std::vector<CodeRec> codeobjs;
    std::vector<FuncRec> funcs;

    // Walk the rhal.module body (single block)
    uint32_t nextBufId = 1;
    std::unordered_map<std::string, uint32_t> bufNameToId;

    for (auto &op : rhalMod.getBody().front()) {
      if (auto co = mlir::dyn_cast<buddy::rhal::ConstantOp>(&op)) {
        ConstRec r;
        r.name = co.getSymName().str();
        r.id = static_cast<uint32_t>(co.getId());
        auto tt = mlir::cast<mlir::RankedTensorType>(co.getType());
        r.tt = tt;
        r.storage = (co.getStorage() == "inline") ? ConstantStorage_Inline
                                                  : ConstantStorage_External;
        r.uri = co.getUri().str();
        constants.push_back(std::move(r));

      } else if (auto co = mlir::dyn_cast<buddy::rhal::CodeobjOp>(&op)) {
        CodeRec r;
        r.name = co.getSymName().str();
        r.id = static_cast<uint32_t>(co.getId());
        r.kind = parseCodeKind(co.getKind());
        r.backend = co.getBackend().str();
        r.uri = co.getUri().str();
        codeobjs.push_back(std::move(r));

      } else if (auto bo = mlir::dyn_cast<buddy::rhal::BufferOp>(&op)) {
        BufRec r;
        r.name = bo.getSymName().str();
        r.id = nextBufId++;
        auto tt = mlir::cast<mlir::RankedTensorType>(bo.getType());
        r.tt = tt;
        r.space = parseSpace(bo.getSpace());
        bufNameToId[r.name] = r.id;
        buffers.push_back(std::move(r));

      } else if (auto fo = mlir::dyn_cast<buddy::rhal::FuncOp>(&op)) {
        FuncRec r;
        r.name = fo.getSymName().str();
        r.dispatch = fo.getDispatch().str();
        for (auto a : fo.getInputs())
          r.inputs.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
        for (auto a : fo.getOutputs())
          r.outputs.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
        for (auto a : fo.getArgs())
          r.args.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
        funcs.push_back(std::move(r));
      }
    }

    // Optional: rewrite file:* URIs to payload:* and collect payload inputs.
    std::vector<PayloadInput> payloadInputs;
    std::unordered_map<std::string, std::string> sourceToEntry;
    std::unordered_map<std::string, int> nameUseCounter;

    auto canonicalOrAbsolute = [](const fs::path &p) {
      std::error_code ec;
      fs::path c = fs::weakly_canonical(p, ec);
      if (!ec)
        return c;
      return fs::absolute(p);
    };

    auto registerPayload = [&](PayloadKind kind, std::string &uri,
                               const std::string &owner) {
      if (!embedPayload || !hasPrefix(uri, "file:"))
        return;

      const fs::path src = resolveFileUri(uri, outputFs);
      std::error_code ec;
      if (!fs::exists(src, ec) || !fs::is_regular_file(src, ec)) {
        throw std::runtime_error("missing payload source for " + owner + ": " +
                                 src.string());
      }

      const fs::path normalized = canonicalOrAbsolute(src);
      const std::string sourceKey = normalized.string();

      auto existing = sourceToEntry.find(sourceKey);
      std::string entryName;
      if (existing == sourceToEntry.end()) {
        uint64_t sz = fs::file_size(normalized, ec);
        if (ec) {
          throw std::runtime_error("failed to stat payload source for " +
                                   owner + ": " + normalized.string());
        }

        entryName =
            makeUniqueEntryName(normalized.filename().string(), nameUseCounter);
        payloadInputs.push_back(
            PayloadInput{kind, entryName, normalized, sz, 0});
        sourceToEntry[sourceKey] = entryName;
      } else {
        entryName = existing->second;
      }

      uri = "payload:" + entryName;
    };

    for (auto &r : constants)
      registerPayload(PayloadKind::Constant, r.uri, "rhal.constant @" + r.name);

    for (auto &r : codeobjs)
      registerPayload(PayloadKind::CodeObject, r.uri,
                      "rhal.codeobj @" + r.name);

    if (!vocabUriAttr.empty())
      registerPayload(PayloadKind::Vocab, vocabUriAttr,
                      "module attr vocab_uri");

    // ── Build FlatBuffer ──────────────────────────────────────────────────

    // Module attrs
    std::vector<flatbuffers::Offset<KV>> modAttrs;
    modAttrs.push_back(
        CreateKV(b, b.CreateString("name"), b.CreateString(moduleName)));

    if (!modelNameAttr.empty())
      modAttrs.push_back(CreateKV(b, b.CreateString("model_name"),
                                  b.CreateString(modelNameAttr)));
    if (!vocabUriAttr.empty())
      modAttrs.push_back(CreateKV(b, b.CreateString("vocab_uri"),
                                  b.CreateString(vocabUriAttr)));

    // Buffers
    std::vector<flatbuffers::Offset<BufferBinding>> fbBufs;
    for (auto &r : buffers) {
      auto ttype = buildTensorType(b, r.tt);
      fbBufs.push_back(CreateBufferBinding(b, r.id, b.CreateString(r.name),
                                           ttype, r.space, 0));
    }

    // Constants
    std::vector<flatbuffers::Offset<Constant>> fbConsts;
    for (auto &r : constants) {
      auto ttype = buildTensorType(b, r.tt);
      fbConsts.push_back(CreateConstant(b, r.id, b.CreateString(r.name), ttype,
                                        r.storage,
                                        0, // inline data
                                        b.CreateString(r.uri),
                                        b.CreateString(""), // checksum
                                        0));                // attrs
    }

    // Code objects
    std::vector<flatbuffers::Offset<CodeObject>> fbCodes;
    std::unordered_map<std::string, uint32_t> codeNameToId;
    for (auto &r : codeobjs) {
      codeNameToId[r.name] = r.id;
      fbCodes.push_back(CreateCodeObject(b, r.id, b.CreateString(r.name),
                                         r.kind,
                                         0, // inline data
                                         b.CreateString(r.uri),
                                         b.CreateString(""), // entry
                                         b.CreateString(r.backend), 0));
    }

    // Functions
    std::vector<flatbuffers::Offset<Function>> fbFuncs;
    for (auto &fr : funcs) {
      auto codeIt = codeNameToId.find(fr.dispatch);
      if (codeIt == codeNameToId.end())
        throw std::runtime_error("rhal.func @" + fr.name +
                                 ": unknown dispatch target '" + fr.dispatch +
                                 "'");
      uint32_t codeId = codeIt->second;

      // Build dispatch arg list (buffer name → id)
      std::vector<flatbuffers::Offset<Arg>> dispArgs;
      for (auto &name : fr.args) {
        auto it = bufNameToId.find(name);
        if (it == bufNameToId.end())
          throw std::runtime_error("rhal.func @" + fr.name +
                                   ": unknown buffer '" + name + "' in args");
        dispArgs.push_back(CreateArg(b, it->second, 0, 0, 0));
      }

      // grid/block: (1,1,1) for CPU kernels
      std::vector<uint32_t> g{1, 1, 1}, blk{1, 1, 1};
      auto dims =
          CreateLaunchDims(b, b.CreateVector(g), b.CreateVector(blk), 0);

      auto disp =
          CreateDispatchOp(b, codeId, b.CreateVector(dispArgs), dims, 0);
      auto dispOp = CreateOp(b, OpKind_Dispatch, disp, 0, 0, 0, 0, 0);

      auto barrierOp =
          CreateOp(b, OpKind_Barrier, 0, 0, 0, CreateBarrierOp(b, 0), 0, 0);

      std::vector<flatbuffers::Offset<Op>> ops = {dispOp, barrierOp};

      // input/output buffer id lists
      std::vector<uint32_t> inputIds, outputIds;
      for (auto &name : fr.inputs) {
        auto it = bufNameToId.find(name);
        if (it != bufNameToId.end())
          inputIds.push_back(it->second);
      }
      for (auto &name : fr.outputs) {
        auto it = bufNameToId.find(name);
        if (it != bufNameToId.end())
          outputIds.push_back(it->second);
      }

      fbFuncs.push_back(
          CreateFunction(b, b.CreateString(fr.name), b.CreateVector(inputIds),
                         b.CreateVector(outputIds), 0, b.CreateVector(ops), 0));
    }

    // ── Assemble Module ───────────────────────────────────────────────────
    auto version = CreateVersion(b, verMaj, verMin, verPat);
    auto raxModule =
        CreateModule(b, b.CreateString("RAX"), version, Endianness_Little,
                     b.CreateVector(modAttrs),
                     0, // requires
                     b.CreateVector(fbBufs), b.CreateVector(fbConsts),
                     b.CreateVector(fbCodes), b.CreateVector(fbFuncs));

    b.Finish(raxModule, "RAX0");

    // Legacy behavior: write plain RAX0 FlatBuffer only.
    if (!embedPayload || payloadInputs.empty()) {
      writeAll(outputPath, b.GetBufferPointer(), b.GetSize());
      std::cout << "wrote " << outputPath << " (" << b.GetSize() << " bytes)\n";
      return 0;
    }

    // Payload mode: RAX0 FlatBuffer + payload data + index + footer.
    std::ofstream ofs(outputPath, std::ios::binary);
    if (!ofs)
      throw std::runtime_error(std::string("cannot write: ") + outputPath);

    const uint64_t manifestSize = b.GetSize();
    ofs.write(reinterpret_cast<const char *>(b.GetBufferPointer()),
              static_cast<std::streamsize>(manifestSize));
    if (!ofs)
      throw std::runtime_error("write failed while writing manifest");

    uint64_t cursor = manifestSize;
    const uint64_t payloadOffset = cursor;

    for (auto &entry : payloadInputs) {
      entry.fileOffset = cursor;
      copyFileToStream(entry.sourcePath, entry.sizeBytes, ofs);
      cursor += entry.sizeBytes;
    }

    const uint64_t payloadSize = cursor - payloadOffset;

    const std::vector<uint8_t> indexBlob = buildPayloadIndexBlob(payloadInputs);
    const uint64_t indexOffset = cursor;
    ofs.write(reinterpret_cast<const char *>(indexBlob.data()),
              static_cast<std::streamsize>(indexBlob.size()));
    if (!ofs)
      throw std::runtime_error("write failed while writing payload index");
    cursor += static_cast<uint64_t>(indexBlob.size());

    const std::vector<uint8_t> footer =
        buildPayloadFooter(manifestSize, payloadOffset, payloadSize,
                           indexOffset, static_cast<uint64_t>(indexBlob.size()),
                           1 /* flags: payload-present */);
    ofs.write(reinterpret_cast<const char *>(footer.data()),
              static_cast<std::streamsize>(footer.size()));
    if (!ofs)
      throw std::runtime_error("write failed while writing payload footer");

    ofs.flush();
    if (!ofs)
      throw std::runtime_error("flush failed");

    std::cout << "wrote " << outputPath << " (" << (cursor + footer.size())
              << " bytes, payload entries=" << payloadInputs.size() << ")\n";
    return 0;

  } catch (const std::exception &e) {
    llvm::errs() << "error: " << e.what() << "\n";
    return 1;
  }
}
