//===- Llama31TTRunner.cpp - Tenstorrent Llama 3.1 runner -----------------===//
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

#include "buddy/runtime/models/Llama31TTRunner.h"

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/Sampler.h"

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {
namespace {

namespace fs = std::filesystem;
using TTDataType = ::tt::target::DataType;

extern "C" int Py_IsInitialized(void);

constexpr int kBeginOfText = 128000;
constexpr int kEndOfText = 128001;
constexpr int kStartHeader = 128006;
constexpr int kEndHeader = 128007;
constexpr int kEom = 128008;
constexpr int kEot = 128009;

static std::string getEnvOr(const char *name, const std::string &fallback) {
  if (const char *value = std::getenv(name))
    if (value[0] != '\0')
      return value;
  return fallback;
}

static constexpr const char *kAnsiReset = "\033[0m";
static constexpr const char *kAnsiBlueBold = "\033[34;1m";
static constexpr const char *kAnsiYellowBold = "\033[33;1m";

static std::string colorLabel(const std::string &label,
                              const char *color = kAnsiYellowBold) {
  return std::string(color) + label + kAnsiReset;
}

static std::string formatFixed(double value, int precision = 4) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(precision) << value;
  return os.str();
}

static void printLlamaLog(const std::string &msg, bool suppress = false) {
  if (suppress)
    return;
  std::cout << colorLabel("[Log]", kAnsiBlueBold) << " " << msg << "\n";
}

static std::string lookupAttr(const ModelManifest &manifest,
                              const std::string &key,
                              const std::string &fallback = "") {
  auto resolved = manifest.resolvedModuleAttrs.find(key);
  if (resolved != manifest.resolvedModuleAttrs.end())
    return resolved->second;
  auto raw = manifest.moduleAttrs.find(key);
  if (raw != manifest.moduleAttrs.end())
    return raw->second;
  return fallback;
}

static bool lookupBoolAttr(const ModelManifest &manifest,
                           const std::string &key, bool fallback) {
  const std::string value = lookupAttr(manifest, key);
  if (value.empty())
    return fallback;
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

static int64_t lookupIntAttr(const ModelManifest &manifest,
                             const std::string &key, int64_t fallback) {
  const std::string value = lookupAttr(manifest, key);
  if (value.empty())
    return fallback;
  try {
    return std::stoll(value);
  } catch (...) {
    return fallback;
  }
}

static void raiseMemoryLimitForLlama() {
#if defined(__unix__) || defined(__APPLE__)
  constexpr rlim_t target = static_cast<rlim_t>(95000000) * 1024;
  auto raiseOne = [](int resource) {
    struct rlimit limit{};
    if (getrlimit(resource, &limit) != 0)
      return;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur >= target)
      return;
    rlim_t requested = target;
    if (limit.rlim_max != RLIM_INFINITY && requested > limit.rlim_max)
      requested = limit.rlim_max;
    if (requested <= limit.rlim_cur)
      return;
    limit.rlim_cur = requested;
    setrlimit(resource, &limit);
  };
  raiseOne(RLIMIT_AS);
  raiseOne(RLIMIT_RSS);
#endif
}

static void keepLibPythonVisibleForTTMLIRRuntime() {
  // The current tt-mlir runtime shared object is built together with tt-metal's
  // Python bindings and has an indirect libpython dependency. Reference one
  // stable Python C-API symbol so linkers using --as-needed keep libpython as a
  // direct dependency of buddy-cli. This does not initialize Python or import
  // any Python modules.
  if (std::getenv("BUDDY_LLAMA31_CHECK_PYTHON_ABI"))
    (void)Py_IsInitialized();
}

static std::string findTTNNArtifact(const ModelManifest &manifest,
                                    const std::string &phase) {
  for (const auto &codeObject : manifest.codeObjects) {
    const bool ttBackend = codeObject.backend == "tt" ||
                           codeObject.backend == "ttnn" ||
                           codeObject.backend == "tenstorrent";
    if (!ttBackend)
      continue;
    if (codeObject.name.find(phase) != std::string::npos)
      return codeObject.path;
  }
  throw std::runtime_error("llama31_tt: missing TTNN code object for phase '" +
                           phase + "'");
}

static bool parseArtifactConstantName(const std::string &name,
                                      std::string &phase,
                                      std::string &filename) {
  static const std::string prefix = "artifact_";
  if (name.rfind(prefix, 0) != 0)
    return false;

  std::string rest = name.substr(prefix.size());
  if (rest.rfind("prefill_", 0) == 0) {
    phase = "prefill";
    rest = rest.substr(std::string("prefill_").size());
  } else if (rest.rfind("decode_", 0) == 0) {
    phase = "decode";
    rest = rest.substr(std::string("decode_").size());
  } else {
    return false;
  }

  if (rest == "weights_bin")
    filename = "weights.bin";
  else if (rest == "slot_roles_json")
    filename = "slot_roles.json";
  else if (rest == "shapes_json")
    filename = "shapes.json";
  else if (rest == "dtypes_json")
    filename = "dtypes.json";
  else if (rest == "summary_json")
    filename = "summary.json";
  else if (rest == "inv_freq_npy")
    filename = "inv_freq.npy";
  else
    return false;
  return true;
}

static void linkOrCopyPayloadFile(const fs::path &src, const fs::path &dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  if (ec)
    throw std::runtime_error("llama31_tt: cannot create artifact directory: " +
                             dst.parent_path().string() + ": " + ec.message());

  if (fs::exists(dst, ec)) {
    if (!ec && fs::equivalent(src, dst, ec))
      return;
    ec.clear();
    fs::remove(dst, ec);
    if (ec)
      throw std::runtime_error("llama31_tt: cannot replace artifact file: " +
                               dst.string() + ": " + ec.message());
  }

  fs::create_symlink(src, dst, ec);
  if (!ec)
    return;

  ec.clear();
  fs::create_hard_link(src, dst, ec);
  if (!ec)
    return;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec)
    throw std::runtime_error("llama31_tt: cannot materialize artifact " +
                             src.string() + " -> " + dst.string() + ": " +
                             ec.message());
}

static std::string materializeEmbeddedArtifacts(const ModelManifest &manifest,
                                                const fs::path &raxDir) {
  bool hasArtifactConstants = false;
  fs::path root;

  for (const auto &constant : manifest.constants) {
    std::string phase;
    std::string filename;
    if (!parseArtifactConstantName(constant.name, phase, filename))
      continue;
    if (constant.path.empty())
      continue;

    if (!hasArtifactConstants) {
      const fs::path payloadParent = fs::path(constant.path).parent_path();
      root = payloadParent.empty() ? raxDir / "chat_artifacts"
                                   : payloadParent / "chat_artifacts";
      hasArtifactConstants = true;
    }

    linkOrCopyPayloadFile(fs::path(constant.path), root / phase / filename);
  }

  return hasArtifactConstants ? fs::absolute(root).string() : "";
}

static uint16_t readU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t readU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static std::string readTextFile(const fs::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("llama31_tt: cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

class MappedFile {
public:
  MappedFile() = default;
  explicit MappedFile(const fs::path &path) { open(path); }
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
  MappedFile(MappedFile &&other) noexcept { moveFrom(other); }
  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) {
      close();
      moveFrom(other);
    }
    return *this;
  }
  ~MappedFile() { close(); }

  void open(const fs::path &path) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      throw std::runtime_error("llama31_tt: cannot open " + path.string() +
                               ": " + std::strerror(errno));

    struct stat st{};
    if (::fstat(fd_, &st) != 0)
      throw std::runtime_error("llama31_tt: cannot stat " + path.string());
    if (st.st_size <= 0)
      throw std::runtime_error("llama31_tt: empty file: " + path.string());
    size_ = static_cast<size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("llama31_tt: mmap failed for " + path.string());
    }
  }

  const uint8_t *data() const { return static_cast<const uint8_t *>(data_); }
  size_t size() const { return size_; }
  const fs::path &path() const { return path_; }

private:
  void close() {
    if (data_) {
      ::munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  void moveFrom(MappedFile &other) {
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    path_ = std::move(other.path_);
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  int fd_ = -1;
  void *data_ = nullptr;
  size_t size_ = 0;
  fs::path path_;
};

struct NpyView {
  const uint8_t *data = nullptr;
  size_t bytes = 0;
  std::string descr;
  std::vector<uint32_t> shape;
  bool fortranOrder = false;
};

struct RawTensorView {
  const uint8_t *data = nullptr;
  size_t bytes = 0;
};

static std::string extractPythonDictString(const std::string &header,
                                           const std::string &key) {
  const size_t keyPos = header.find("'" + key + "'");
  if (keyPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing key '" + key +
                             "'");
  const size_t colon = header.find(':', keyPos);
  const size_t quote0 = header.find('\'', colon);
  const size_t quote1 = header.find('\'', quote0 + 1);
  if (colon == std::string::npos || quote0 == std::string::npos ||
      quote1 == std::string::npos)
    throw std::runtime_error("llama31_tt: malformed .npy header key '" + key +
                             "'");
  return header.substr(quote0 + 1, quote1 - quote0 - 1);
}

static std::vector<uint32_t> extractPythonDictShape(const std::string &header) {
  const size_t keyPos = header.find("'shape'");
  if (keyPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing shape");
  const size_t open = header.find('(', keyPos);
  const size_t close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos)
    throw std::runtime_error("llama31_tt: malformed .npy shape");
  std::vector<uint32_t> shape;
  size_t pos = open + 1;
  while (pos < close) {
    while (pos < close && (header[pos] == ' ' || header[pos] == ','))
      ++pos;
    if (pos >= close)
      break;
    size_t end = pos;
    while (end < close && std::isdigit(static_cast<unsigned char>(header[end])))
      ++end;
    if (end == pos)
      break;
    const uint64_t dim = std::stoull(header.substr(pos, end - pos));
    if (dim > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("llama31_tt: .npy dimension too large");
    shape.push_back(static_cast<uint32_t>(dim));
    pos = end;
  }
  return shape;
}

static NpyView parseNpy(const uint8_t *base, size_t size,
                        const std::string &label) {
  static constexpr std::array<uint8_t, 6> magic = {0x93, 'N', 'U',
                                                   'M',  'P', 'Y'};
  if (size < 10 || std::memcmp(base, magic.data(), magic.size()) != 0)
    throw std::runtime_error("llama31_tt: invalid .npy blob: " + label);
  const uint8_t major = base[6];
  uint32_t headerLen = 0;
  size_t headerOffset = 0;
  if (major == 1) {
    headerLen = readU16LE(base + 8);
    headerOffset = 10;
  } else if (major == 2 || major == 3) {
    if (size < 12)
      throw std::runtime_error("llama31_tt: truncated .npy header: " + label);
    headerLen = readU32LE(base + 8);
    headerOffset = 12;
  } else {
    throw std::runtime_error("llama31_tt: unsupported .npy version in " +
                             label);
  }
  if (headerOffset + headerLen > size)
    throw std::runtime_error("llama31_tt: truncated .npy header: " + label);

  const std::string header(reinterpret_cast<const char *>(base + headerOffset),
                           headerLen);
  NpyView view;
  view.descr = extractPythonDictString(header, "descr");
  view.shape = extractPythonDictShape(header);
  const size_t fortranPos = header.find("'fortran_order'");
  if (fortranPos == std::string::npos)
    throw std::runtime_error("llama31_tt: .npy header missing fortran_order");
  const size_t colon = header.find(':', fortranPos);
  view.fortranOrder = header.find("True", colon) != std::string::npos &&
                      header.find("True", colon) < header.find(',', colon);
  if (view.fortranOrder)
    throw std::runtime_error("llama31_tt: Fortran-order .npy is unsupported: " +
                             label);

  const size_t dataOffset = headerOffset + headerLen;
  view.data = base + dataOffset;
  view.bytes = size - dataOffset;
  return view;
}

class NpyFile {
public:
  NpyFile() = default;
  explicit NpyFile(const fs::path &path) : mapped_(path) {
    view_ = parseNpy(mapped_.data(), mapped_.size(), path.string());
  }
  const NpyView &view() const { return view_; }

private:
  MappedFile mapped_;
  NpyView view_;
};

struct RoleEntry {
  uint32_t slot = 0;
  std::string role;
  std::vector<uint32_t> shape;
  std::string dtype;
  uint64_t weightOffset = 0;
  uint64_t weightBytes = 0;
};

static std::string errorToString(llvm::Error error) {
  return llvm::toString(std::move(error));
}

static std::vector<RoleEntry> parseRoles(const fs::path &path) {
  auto parsed = llvm::json::parse(readTextFile(path));
  if (!parsed)
    throw std::runtime_error("llama31_tt: JSON parse failed for " +
                             path.string() + ": " +
                             errorToString(parsed.takeError()));
  const auto *array = parsed->getAsArray();
  if (!array)
    throw std::runtime_error("llama31_tt: slot_roles.json must be an array");

  std::vector<RoleEntry> roles;
  roles.reserve(array->size());
  for (const auto &value : *array) {
    const auto *object = value.getAsObject();
    if (!object)
      throw std::runtime_error("llama31_tt: malformed slot role entry");
    RoleEntry entry;
    auto slot = object->getInteger("slot");
    auto role = object->getString("role");
    auto dtype = object->getString("dtype");
    const auto *shape = object->getArray("shape");
    if (!slot || !role || !dtype || !shape)
      throw std::runtime_error("llama31_tt: incomplete slot role entry");
    entry.slot = static_cast<uint32_t>(*slot);
    entry.role = role->str();
    entry.dtype = dtype->str();
    if (entry.role == "weight") {
      auto offset = object->getInteger("weight_offset");
      auto nbytes = object->getInteger("weight_nbytes");
      if (!offset || !nbytes || *offset < 0 || *nbytes <= 0)
        throw std::runtime_error(
            "llama31_tt: weight slot role missing byte range");
      entry.weightOffset = static_cast<uint64_t>(*offset);
      entry.weightBytes = static_cast<uint64_t>(*nbytes);
    }
    for (const auto &dimValue : *shape) {
      auto dim = dimValue.getAsInteger();
      if (!dim || *dim < 0 ||
          *dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("llama31_tt: invalid tensor shape in " +
                                 path.string());
      entry.shape.push_back(static_cast<uint32_t>(*dim));
    }
    roles.push_back(std::move(entry));
  }
  std::sort(
      roles.begin(), roles.end(),
      [](const RoleEntry &a, const RoleEntry &b) { return a.slot < b.slot; });
  for (size_t i = 0; i < roles.size(); ++i)
    if (roles[i].slot != i)
      throw std::runtime_error("llama31_tt: non-contiguous slot_roles.json");
  return roles;
}

struct PhaseCtx {
  std::string phase;
  fs::path dir;
  std::vector<RoleEntry> roles;
  std::unique_ptr<MappedFile> weights;
  std::unique_ptr<NpyFile> invFreq;

  PhaseCtx(const fs::path &artifactsDir, std::string phaseName)
      : phase(std::move(phaseName)), dir(artifactsDir / phase),
        roles(parseRoles(dir / "slot_roles.json")) {
    const fs::path weightsPath = dir / "weights.bin";
    if (fs::exists(weightsPath))
      weights = std::make_unique<MappedFile>(weightsPath);
    const fs::path invFreqPath = dir / "inv_freq.npy";
    if (fs::exists(invFreqPath))
      invFreq = std::make_unique<NpyFile>(invFreqPath);

    for (const RoleEntry &role : roles)
      if (role.role == "weight") {
        if (!weights)
          throw std::runtime_error("llama31_tt: missing weights.bin for " +
                                   phase);
        if (role.weightOffset > weights->size() ||
            role.weightBytes > weights->size() - role.weightOffset) {
          throw std::runtime_error("llama31_tt: weight byte range out of "
                                   "weights.bin for " +
                                   phase);
        }
      }
  }

  RawTensorView weightForSlot(uint32_t slot) const {
    if (slot >= roles.size() || roles[slot].role != "weight")
      throw std::runtime_error("llama31_tt: slot is not a weight: " +
                               std::to_string(slot));
    if (!weights)
      throw std::runtime_error("llama31_tt: missing weights.bin for " + phase);
    const RoleEntry &role = roles[slot];
    return RawTensorView{weights->data() + role.weightOffset,
                         static_cast<size_t>(role.weightBytes)};
  }
};

static uint64_t volume(const std::vector<uint32_t> &shape) {
  uint64_t result = 1;
  for (uint32_t dim : shape)
    result *= dim;
  return result;
}

static std::vector<uint32_t>
contiguousStride(const std::vector<uint32_t> &shape) {
  std::vector<uint32_t> stride(shape.size(), 1);
  uint64_t running = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    stride[i - 1] = static_cast<uint32_t>(running);
    running *= shape[i - 1];
  }
  return stride;
}

static uint32_t itemSizeFor(TTDataType dtype) {
  switch (dtype) {
  case TTDataType::Float32:
  case TTDataType::Int32:
  case TTDataType::UInt32:
    return 4;
  case TTDataType::Float16:
  case TTDataType::BFloat16:
  case TTDataType::UInt16:
  case TTDataType::Int16:
    return 2;
  case TTDataType::UInt8:
  case TTDataType::Int8:
  case TTDataType::Bool:
    return 1;
  case TTDataType::Float64:
  case TTDataType::Int64:
  case TTDataType::UInt64:
    return 8;
  default:
    throw std::runtime_error("llama31_tt: unsupported scalar tensor dtype");
  }
}

static std::vector<uint8_t> integerBuffer(const std::vector<int64_t> &values,
                                          TTDataType dtype) {
  std::vector<uint8_t> buffer(values.size() * itemSizeFor(dtype));
  auto write = [&](auto tag) {
    using T = decltype(tag);
    T *out = reinterpret_cast<T *>(buffer.data());
    for (size_t i = 0; i < values.size(); ++i)
      out[i] = static_cast<T>(values[i]);
  };
  switch (dtype) {
  case TTDataType::Int64:
    write(int64_t{});
    break;
  case TTDataType::UInt64:
    write(uint64_t{});
    break;
  case TTDataType::Int32:
    write(int32_t{});
    break;
  case TTDataType::UInt32:
    write(uint32_t{});
    break;
  case TTDataType::Int16:
    write(int16_t{});
    break;
  case TTDataType::UInt16:
    write(uint16_t{});
    break;
  case TTDataType::Int8:
    write(int8_t{});
    break;
  case TTDataType::UInt8:
  case TTDataType::Bool:
    write(uint8_t{});
    break;
  default:
    throw std::runtime_error("llama31_tt: expected integer input dtype");
  }
  return buffer;
}

static std::vector<uint8_t> zeroBuffer(const ::tt::runtime::TensorDesc &desc) {
  return std::vector<uint8_t>(volume(desc.shape) * desc.elementSize(), 0);
}

static ::tt::runtime::Tensor
hostTensorFromBytes(const void *data, const ::tt::runtime::TensorDesc &desc) {
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createOwnedHostTensor(
      data, desc.shape, stride, desc.elementSize(), desc.dataType);
}

static ::tt::runtime::Tensor
borrowedHostTensorFromRaw(const RawTensorView &view,
                          const ::tt::runtime::TensorDesc &desc) {
  const uint64_t expectedBytes = volume(desc.shape) * desc.elementSize();
  if (view.bytes != expectedBytes)
    throw std::runtime_error("llama31_tt: raw weight tensor byte size does not "
                             "match TTNN input descriptor");
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createBorrowedHostTensor(
      const_cast<uint8_t *>(view.data), desc.shape, stride, desc.elementSize(),
      desc.dataType);
}

static ::tt::runtime::Tensor
borrowedHostTensorFromNpy(const NpyView &view,
                          const ::tt::runtime::TensorDesc &desc) {
  const uint64_t expectedBytes = volume(desc.shape) * desc.elementSize();
  if (view.bytes < expectedBytes)
    throw std::runtime_error("llama31_tt: .npy tensor is smaller than TTNN "
                             "input descriptor");
  const std::vector<uint32_t> stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createBorrowedHostTensor(
      const_cast<uint8_t *>(view.data), desc.shape, stride, desc.elementSize(),
      desc.dataType);
}

static ::tt::runtime::Tensor toDevice(::tt::runtime::Tensor host,
                                      ::tt::runtime::Device device,
                                      ::tt::runtime::Binary binary,
                                      uint32_t programIndex, uint32_t slot,
                                      bool retain = false) {
  auto layout = ::tt::runtime::getLayout(binary, programIndex, slot);
  auto deviceTensor = ::tt::runtime::toLayout(host, device, layout, retain);
  if (retain)
    ::tt::runtime::setTensorRetain(deviceTensor, true);
  return deviceTensor;
}

static bool startsWith(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

static bool isRemoteModelId(const std::string &value) {
  return !value.empty() && value.find('/') != std::string::npos &&
         !startsWith(value, "/") && !startsWith(value, "./") &&
         !startsWith(value, "../") && !startsWith(value, "file:");
}

static std::vector<uint8_t> decodeBase64(const std::string &text) {
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::array<int, 256> reverse;
  reverse.fill(-1);
  for (int i = 0; i < 64; ++i)
    reverse[static_cast<unsigned char>(table[i])] = i;

  std::vector<uint8_t> out;
  int value = 0;
  int bits = -8;
  for (unsigned char c : text) {
    if (c == '=')
      break;
    if (std::isspace(c))
      continue;
    const int r = reverse[c];
    if (r < 0)
      throw std::runtime_error("llama31_tt: invalid base64 token in tokenizer");
    value = (value << 6) | r;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
      bits -= 8;
    }
  }
  return out;
}

class Llama31Tokenizer {
public:
  explicit Llama31Tokenizer(const fs::path &modelDir) { load(modelDir); }

  std::vector<int> encodeChat(const std::string &userMessage,
                              const std::string &systemPrompt = "") const {
    std::vector<int> ids;
    ids.push_back(kBeginOfText);
    if (!systemPrompt.empty()) {
      appendHeader(ids, "system");
      appendText(ids, systemPrompt);
      ids.push_back(kEot);
    }
    appendHeader(ids, "user");
    appendText(ids, userMessage);
    ids.push_back(kEot);
    appendHeader(ids, "assistant");
    appendText(ids, "");
    return ids;
  }

  std::string decodeToken(int id) const {
    auto special = specialIdToText_.find(id);
    if (special != specialIdToText_.end())
      return special->second;
    auto it = idToBytes_.find(id);
    if (it == idToBytes_.end())
      return "";
    return it->second;
  }

  std::string decodeTokens(const std::vector<int> &ids,
                           bool skipSpecial = false) const {
    std::string out;
    for (int id : ids) {
      if (skipSpecial && specialIdToText_.count(id))
        continue;
      out += decodeToken(id);
    }
    return out;
  }

private:
  void load(const fs::path &modelDir) {
    fs::path tokenizerModel = modelDir / "original" / "tokenizer.model";
    if (!fs::exists(tokenizerModel))
      tokenizerModel = modelDir / "tokenizer.model";
    if (!fs::exists(tokenizerModel))
      throw std::runtime_error(
          "llama31_tt: tokenizer.model not found. Set LLAMA31_MODEL_PATH to a "
          "local Llama-3.1-8B-Instruct directory.");

    std::ifstream input(tokenizerModel);
    if (!input)
      throw std::runtime_error("llama31_tt: cannot read " +
                               tokenizerModel.string());

    std::string line;
    while (std::getline(input, line)) {
      if (line.empty())
        continue;
      const size_t space = line.find(' ');
      if (space == std::string::npos)
        continue;
      std::vector<uint8_t> bytes = decodeBase64(line.substr(0, space));
      const int rank = std::stoi(line.substr(space + 1));
      std::string token(reinterpret_cast<const char *>(bytes.data()),
                        bytes.size());
      tokenToId_[token] = rank;
      idToBytes_[rank] = std::move(token);
    }
    if (tokenToId_.empty())
      throw std::runtime_error("llama31_tt: empty tokenizer ranks");

    specialIdToText_[kBeginOfText] = "<|begin_of_text|>";
    specialIdToText_[kEndOfText] = "<|end_of_text|>";
    specialIdToText_[kStartHeader] = "<|start_header_id|>";
    specialIdToText_[kEndHeader] = "<|end_header_id|>";
    specialIdToText_[kEom] = "<|eom_id|>";
    specialIdToText_[kEot] = "<|eot_id|>";
  }

  void appendHeader(std::vector<int> &ids, const std::string &role) const {
    ids.push_back(kStartHeader);
    appendText(ids, role);
    ids.push_back(kEndHeader);
    appendText(ids, "\n\n");
  }

  static bool isAlpha(unsigned char c) { return std::isalpha(c) != 0; }
  static bool isDigit(unsigned char c) { return std::isdigit(c) != 0; }
  static bool isSpace(unsigned char c) { return std::isspace(c) != 0; }
  static bool isNewline(unsigned char c) { return c == '\n' || c == '\r'; }
  static bool isWord(unsigned char c) { return isAlpha(c) || isDigit(c); }

  static bool matchContraction(std::string_view text, size_t pos, size_t &len) {
    static constexpr std::array<std::string_view, 7> suffixes = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (std::string_view suffix : suffixes) {
      if (pos + suffix.size() > text.size())
        continue;
      bool ok = true;
      for (size_t i = 0; i < suffix.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(text[pos + i]);
        unsigned char b = static_cast<unsigned char>(suffix[i]);
        ok &= std::tolower(a) == std::tolower(b);
      }
      if (ok) {
        len = suffix.size();
        return true;
      }
    }
    return false;
  }

  static std::vector<std::string> splitTiktokenApprox(std::string_view text) {
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
      size_t contractionLen = 0;
      if (matchContraction(text, i, contractionLen)) {
        pieces.emplace_back(text.substr(i, contractionLen));
        i += contractionLen;
        continue;
      }

      unsigned char c = static_cast<unsigned char>(text[i]);
      if (c == ' ' && i + 1 < text.size() &&
          isAlpha(static_cast<unsigned char>(text[i + 1]))) {
        size_t j = i + 2;
        while (j < text.size() && isAlpha(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (isAlpha(c)) {
        size_t j = i + 1;
        while (j < text.size() && isAlpha(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (isDigit(c)) {
        size_t j = i + 1;
        while (j < text.size() && j - i < 3 &&
               isDigit(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }
      if (c == ' ' && i + 1 < text.size()) {
        unsigned char n = static_cast<unsigned char>(text[i + 1]);
        if (!isSpace(n) && !isWord(n)) {
          size_t j = i + 2;
          while (j < text.size()) {
            unsigned char p = static_cast<unsigned char>(text[j]);
            if (isSpace(p) || isWord(p))
              break;
            ++j;
          }
          while (j < text.size() &&
                 isNewline(static_cast<unsigned char>(text[j])))
            ++j;
          pieces.emplace_back(text.substr(i, j - i));
          i = j;
          continue;
        }
      }
      if (!isSpace(c) && !isWord(c)) {
        size_t j = i + 1;
        while (j < text.size()) {
          unsigned char p = static_cast<unsigned char>(text[j]);
          if (isSpace(p) || isWord(p))
            break;
          ++j;
        }
        while (j < text.size() &&
               isNewline(static_cast<unsigned char>(text[j])))
          ++j;
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
        continue;
      }

      size_t j = i + 1;
      while (j < text.size() && isSpace(static_cast<unsigned char>(text[j])) &&
             !isNewline(static_cast<unsigned char>(text[j])))
        ++j;
      if (j < text.size() && isNewline(static_cast<unsigned char>(text[j]))) {
        while (j < text.size() &&
               isNewline(static_cast<unsigned char>(text[j])))
          ++j;
      } else if (j < text.size()) {
        j = i + 1;
      }
      pieces.emplace_back(text.substr(i, j - i));
      i = j;
    }
    return pieces;
  }

  void appendText(std::vector<int> &ids, std::string_view text) const {
    for (const std::string &piece : splitTiktokenApprox(text)) {
      std::vector<std::string> parts;
      parts.reserve(piece.size());
      for (unsigned char c : piece)
        parts.emplace_back(1, static_cast<char>(c));

      while (parts.size() > 1) {
        int bestRank = std::numeric_limits<int>::max();
        size_t bestIndex = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
          std::string merged = parts[i] + parts[i + 1];
          auto it = tokenToId_.find(merged);
          if (it != tokenToId_.end() && it->second < bestRank) {
            bestRank = it->second;
            bestIndex = i;
          }
        }
        if (bestIndex == std::numeric_limits<size_t>::max())
          break;
        std::vector<std::string> next;
        next.reserve(parts.size() - 1);
        for (size_t i = 0; i < parts.size(); ++i) {
          if (i == bestIndex) {
            next.push_back(parts[i] + parts[i + 1]);
            ++i;
          } else {
            next.push_back(std::move(parts[i]));
          }
        }
        parts = std::move(next);
      }

      for (const std::string &part : parts) {
        auto it = tokenToId_.find(part);
        if (it != tokenToId_.end()) {
          ids.push_back(it->second);
          continue;
        }
        for (unsigned char c : part) {
          std::string byte(1, static_cast<char>(c));
          auto byteIt = tokenToId_.find(byte);
          if (byteIt == tokenToId_.end())
            throw std::runtime_error("llama31_tt: tokenizer byte missing");
          ids.push_back(byteIt->second);
        }
      }
    }
  }

  std::unordered_map<std::string, int> tokenToId_;
  std::unordered_map<int, std::string> idToBytes_;
  std::unordered_map<int, std::string> specialIdToText_;
};

static float halfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x03ff;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static float bf16BitsToFloat(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static std::vector<float> loadLogits(const std::vector<std::byte> &bytes,
                                     TTDataType dtype, size_t start,
                                     size_t count) {
  if (count == 0)
    throw std::runtime_error("llama31_tt: empty logits tensor");

  std::vector<float> logits(count);
  switch (dtype) {
  case TTDataType::Float32: {
    const float *data = reinterpret_cast<const float *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = data[start + i];
    break;
  }
  case TTDataType::BFloat16: {
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = bf16BitsToFloat(data[start + i]);
    break;
  }
  case TTDataType::Float16: {
    const uint16_t *data = reinterpret_cast<const uint16_t *>(bytes.data());
    for (size_t i = 0; i < count; ++i)
      logits[i] = halfBitsToFloat(data[start + i]);
    break;
  }
  default:
    throw std::runtime_error("llama31_tt: unsupported logits dtype");
  }
  return logits;
}

static int sampleTensor(const std::vector<std::byte> &bytes, TTDataType dtype,
                        size_t start, size_t count, buddy::Sampler &sampler,
                        const std::vector<int> &recentTokens) {
  switch (dtype) {
  case TTDataType::Int32: {
    const int32_t *data = reinterpret_cast<const int32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::Int64: {
    const int64_t *data = reinterpret_cast<const int64_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  case TTDataType::UInt32: {
    const uint32_t *data = reinterpret_cast<const uint32_t *>(bytes.data());
    return static_cast<int>(data[start]);
  }
  default:
    std::vector<float> logits = loadLogits(bytes, dtype, start, count);
    return sampler.sample(logits.data(), logits.size(), recentTokens);
  }
}

struct ExtractedOutput {
  int token = 0;
  std::unordered_map<std::string, ::tt::runtime::Tensor> kvDevice;
  double logitsToHostSeconds = 0.0;
  double kvRelayoutSeconds = 0.0;
};

static ExtractedOutput extractLogitsAndKV(
    std::vector<::tt::runtime::Tensor> &outputs, int logitsIndex,
    uint32_t tokenPosition, ::tt::runtime::Device device,
    ::tt::runtime::Binary targetBinary, uint32_t targetProgramIndex,
    const std::unordered_map<std::string, uint32_t> &targetKVInputSlots,
    const std::vector<std::string> &kvRoleMap, buddy::Sampler &sampler,
    const std::vector<int> &recentTokens) {
  if (logitsIndex < 0)
    logitsIndex = static_cast<int>(outputs.size()) + logitsIndex;
  if (logitsIndex < 0 || logitsIndex >= static_cast<int>(outputs.size()))
    throw std::runtime_error("llama31_tt: logits output index out of range");

  ExtractedOutput extracted;
  auto t0 = std::chrono::steady_clock::now();
  auto hostShards = ::tt::runtime::toHost(outputs[logitsIndex], true, true);
  if (hostShards.empty())
    throw std::runtime_error("llama31_tt: logits toHost returned no shards");
  auto host = hostShards.front();
  const TTDataType dtype = ::tt::runtime::getTensorDataType(host);
  const auto shape = ::tt::runtime::getTensorShape(host);
  const auto data = ::tt::runtime::getTensorDataBuffer(host);
  auto t1 = std::chrono::steady_clock::now();
  extracted.logitsToHostSeconds =
      std::chrono::duration<double>(t1 - t0).count();

  size_t vocab = 0;
  size_t start = 0;
  if (shape.size() >= 3) {
    vocab = shape.back();
    start = static_cast<size_t>(tokenPosition) * vocab;
  } else if (shape.size() >= 2) {
    vocab = shape.back();
    start = 0;
  } else {
    vocab = data.size() / itemSizeFor(dtype);
    start = 0;
  }
  extracted.token =
      sampleTensor(data, dtype, start, vocab, sampler, recentTokens);

  t0 = std::chrono::steady_clock::now();
  size_t kvIter = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (static_cast<int>(i) == logitsIndex)
      continue;
    if (kvIter >= kvRoleMap.size())
      break;
    const std::string &role = kvRoleMap[kvIter++];
    auto slotIt = targetKVInputSlots.find(role);
    if (slotIt == targetKVInputSlots.end())
      throw std::runtime_error("llama31_tt: missing target KV slot for " +
                               role);
    auto layout = ::tt::runtime::getLayout(targetBinary, targetProgramIndex,
                                           slotIt->second);
    extracted.kvDevice[role] =
        ::tt::runtime::toLayout(outputs[i], device, layout);
  }
  t1 = std::chrono::steady_clock::now();
  extracted.kvRelayoutSeconds = std::chrono::duration<double>(t1 - t0).count();

  for (size_t i = 0; i < outputs.size(); ++i) {
    bool force = static_cast<int>(i) == logitsIndex;
    ::tt::runtime::deallocateTensor(outputs[i], force);
  }
  return extracted;
}

static std::unordered_map<uint32_t, ::tt::runtime::Tensor>
precacheStaticInputs(::tt::runtime::Device device, ::tt::runtime::Binary binary,
                     uint32_t programIndex, const PhaseCtx &ctx,
                     const std::vector<::tt::runtime::TensorDesc> &inputs) {
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> cached;
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    const RoleEntry &role = ctx.roles[slot];
    if (role.role != "weight" && role.role != "inv_freq")
      continue;
    ::tt::runtime::Tensor host;
    if (role.role == "weight") {
      host = borrowedHostTensorFromRaw(ctx.weightForSlot(slot), inputs[slot]);
    } else {
      if (!ctx.invFreq)
        throw std::runtime_error("llama31_tt: missing inv_freq.npy");
      host = borrowedHostTensorFromNpy(ctx.invFreq->view(), inputs[slot]);
    }
    cached[slot] = toDevice(host, device, binary, programIndex, slot, true);
  }
  return cached;
}

static std::vector<::tt::runtime::Tensor> buildPrefillInputs(
    ::tt::runtime::Device device, ::tt::runtime::Binary binary,
    uint32_t programIndex, const PhaseCtx &ctx,
    const std::vector<::tt::runtime::TensorDesc> &programInputs,
    const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cachedInputs,
    const std::vector<int> &paddedTokens) {
  std::vector<::tt::runtime::Tensor> tensors(programInputs.size());
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    auto cached = cachedInputs.find(slot);
    if (cached != cachedInputs.end()) {
      tensors[slot] = cached->second;
      continue;
    }
    const auto &desc = programInputs[slot];
    const RoleEntry &role = ctx.roles[slot];
    std::vector<uint8_t> buffer;
    if (role.role == "input_ids") {
      std::vector<int64_t> values(paddedTokens.begin(), paddedTokens.end());
      buffer = integerBuffer(values, desc.dataType);
    } else if (role.role == "cache_position") {
      std::vector<int64_t> values(volume(desc.shape));
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<int64_t>(i);
      buffer = integerBuffer(values, desc.dataType);
    } else if (startsWith(role.role, "past_K_") ||
               startsWith(role.role, "past_V_")) {
      buffer = zeroBuffer(desc);
    } else {
      throw std::runtime_error("llama31_tt: unexpected prefill role " +
                               role.role);
    }
    auto host = hostTensorFromBytes(buffer.data(), desc);
    tensors[slot] = toDevice(host, device, binary, programIndex, slot);
  }
  return tensors;
}

static std::vector<::tt::runtime::Tensor> buildDecodeInputs(
    ::tt::runtime::Device device, ::tt::runtime::Binary binary,
    uint32_t programIndex, const PhaseCtx &ctx,
    const std::vector<::tt::runtime::TensorDesc> &programInputs,
    const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cachedInputs,
    int token, int cachePos,
    const std::unordered_map<std::string, ::tt::runtime::Tensor> &pastKV) {
  std::vector<::tt::runtime::Tensor> tensors(programInputs.size());
  for (uint32_t slot = 0; slot < ctx.roles.size(); ++slot) {
    auto cached = cachedInputs.find(slot);
    if (cached != cachedInputs.end()) {
      tensors[slot] = cached->second;
      continue;
    }
    const auto &desc = programInputs[slot];
    const RoleEntry &role = ctx.roles[slot];
    if (startsWith(role.role, "past_K_") || startsWith(role.role, "past_V_")) {
      auto kv = pastKV.find(role.role);
      if (kv == pastKV.end())
        throw std::runtime_error("llama31_tt: missing KV tensor " + role.role);
      tensors[slot] = kv->second;
      continue;
    }

    std::vector<uint8_t> buffer;
    if (role.role == "input_ids") {
      buffer = integerBuffer({token}, desc.dataType);
    } else if (role.role == "cache_position") {
      buffer = integerBuffer({cachePos}, desc.dataType);
    } else {
      throw std::runtime_error("llama31_tt: unexpected decode role " +
                               role.role);
    }
    auto host = hostTensorFromBytes(buffer.data(), desc);
    tensors[slot] = toDevice(host, device, binary, programIndex, slot);
  }
  return tensors;
}

static fs::path resolveTokenizerPath(const std::string &tokenizerAttr) {
  std::string value = getEnvOr("LLAMA31_MODEL_PATH", tokenizerAttr);
  if (startsWith(value, "file:"))
    value = value.substr(5);
  if (isRemoteModelId(value))
    throw std::runtime_error(
        "llama31_tt: tokenizer/model path is a remote Hugging Face id ('" +
        value +
        "'). Pure C++ buddy-cli runtime does not download models; set "
        "LLAMA31_MODEL_PATH to a local Llama-3.1-8B-Instruct directory when "
        "running or packaging.");
  return fs::absolute(fs::path(value));
}

} // namespace

void Llama31TTRunner::run(const RunConfig &cfg) {
  raiseMemoryLimitForLlama();
  keepLibPythonVisibleForTTMLIRRuntime();

  if (cfg.raxPath.empty())
    throw std::runtime_error("llama31_tt requires --model <path.rax>");
  const bool suppress = cfg.suppressStats;

  const ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
  const std::string prefillTTNN = findTTNNArtifact(manifest, "prefill");
  const std::string decodeTTNN = findTTNNArtifact(manifest, "decode");

  const fs::path raxDir = fs::absolute(fs::path(cfg.raxPath)).parent_path();
  std::string artifacts = materializeEmbeddedArtifacts(manifest, raxDir);
  if (artifacts.empty())
    artifacts = lookupAttr(manifest, "artifacts_uri",
                           (raxDir / "chat_artifacts").string());

  const int maxCacheLen =
      static_cast<int>(lookupIntAttr(manifest, "max_cache_len", 1024));
  const int eosToken =
      static_cast<int>(lookupIntAttr(manifest, "eos_token_id", kEot));
  const bool ignoreEOS = lookupBoolAttr(manifest, "ignore_eos", false);
  auto isStopToken = [&](int token) {
    if (ignoreEOS)
      return false;
    return token == eosToken || token == kEndOfText || token == kEom ||
           token == kEot;
  };
  const uint32_t programIndex =
      static_cast<uint32_t>(lookupIntAttr(manifest, "program_index", 0));
  const fs::path tokenizerPath = resolveTokenizerPath(lookupAttr(
      manifest, "tokenizer_uri", "meta-llama/Llama-3.1-8B-Instruct"));

  if (!suppress) {
    std::cout << "[buddy-cli] dispatch llama31_tt via native Tenstorrent C++ "
                 "runtime\n";
    std::cout << "[buddy-cli] prefill: " << prefillTTNN << "\n";
    std::cout << "[buddy-cli] decode:  " << decodeTTNN << "\n";
    std::cout << "[buddy-cli] artifacts: " << artifacts << "\n";
  }

  const auto tokenizerStart = std::chrono::steady_clock::now();
  Llama31Tokenizer tokenizer(tokenizerPath);
  const auto tokenizerEnd = std::chrono::steady_clock::now();
  printLlamaLog("tokenizer ready in " +
                    formatFixed(std::chrono::duration<double, std::milli>(
                                    tokenizerEnd - tokenizerStart)
                                    .count()) +
                    "ms",
                suppress);

  std::string prompt = cfg.prompt;
  if (prompt.empty()) {
    std::cout << "Prompt: ";
    std::getline(std::cin, prompt);
    std::cout << "\n";
  }

  auto prefillBinary = ::tt::runtime::Binary::loadFromPath(prefillTTNN.c_str());
  auto decodeBinary = ::tt::runtime::Binary::loadFromPath(decodeTTNN.c_str());
  ::tt::runtime::setCompatibleDeviceRuntime(prefillBinary);

  const auto prefillInputs = prefillBinary.getProgramInputs(programIndex);
  const auto decodeInputs = decodeBinary.getProgramInputs(programIndex);
  PhaseCtx prefillCtx(fs::path(artifacts), "prefill");
  PhaseCtx decodeCtx(fs::path(artifacts), "decode");
  if (prefillInputs.size() != prefillCtx.roles.size())
    throw std::runtime_error("llama31_tt: prefill input count mismatch");
  if (decodeInputs.size() != decodeCtx.roles.size())
    throw std::runtime_error("llama31_tt: decode input count mismatch");

  std::unordered_map<std::string, uint32_t> decodeKVInputSlots;
  std::vector<std::string> decodeKVOutputRoles;
  for (uint32_t slot = 0; slot < decodeCtx.roles.size(); ++slot) {
    const std::string &role = decodeCtx.roles[slot].role;
    if (startsWith(role, "past_K_") || startsWith(role, "past_V_")) {
      decodeKVInputSlots[role] = slot;
      decodeKVOutputRoles.push_back(role);
    }
  }
  if (decodeKVOutputRoles.empty())
    throw std::runtime_error("llama31_tt: decode artifacts have no KV slots");

  ::tt::runtime::MeshDeviceOptions options;
  options.meshShape = std::vector<uint32_t>{1, 1};
  std::optional<::tt::runtime::Device> device;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> prefillCache;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> decodeCache;
  buddy::Sampler sampler(cfg.samplerConfig);

  try {
    device = ::tt::runtime::openMeshDevice(options);
    if (!suppress) {
      std::cout << kAnsiYellowBold
                << "Llama-3.1-8B-Instruct Inference Powered by Buddy Compiler "
                   "(P150A TTIR, native C++)"
                << kAnsiReset << "\n";
    }

    auto cacheStart = std::chrono::steady_clock::now();
    prefillCache = precacheStaticInputs(*device, prefillBinary, programIndex,
                                        prefillCtx, prefillInputs);
    decodeCache = precacheStaticInputs(*device, decodeBinary, programIndex,
                                       decodeCtx, decodeInputs);
    auto cacheEnd = std::chrono::steady_clock::now();
    printLlamaLog(
        "uploaded & retained " + std::to_string(prefillCache.size()) +
            " prefill + " + std::to_string(decodeCache.size()) +
            " decode static tensors in " +
            formatFixed(
                std::chrono::duration<double>(cacheEnd - cacheStart).count()) +
            "s",
        suppress);

    std::vector<int> tokenIds = tokenizer.encodeChat(prompt);
    if (tokenIds.size() >= static_cast<size_t>(maxCacheLen))
      throw std::runtime_error("llama31_tt: prompt is longer than max cache");
    printLlamaLog("prompt tokens = " + std::to_string(tokenIds.size()),
                  suppress);
    std::vector<int> recentTokens = tokenIds;

    std::vector<int> padded(maxCacheLen, eosToken);
    std::copy(tokenIds.begin(), tokenIds.end(), padded.begin());

    auto prefillInputsRT =
        buildPrefillInputs(*device, prefillBinary, programIndex, prefillCtx,
                           prefillInputs, prefillCache, padded);
    auto prefillStart = std::chrono::steady_clock::now();
    auto prefillOutputs = ::tt::runtime::submit(*device, prefillBinary,
                                                programIndex, prefillInputsRT);
    ::tt::runtime::wait(prefillOutputs);
    auto prefillEnd = std::chrono::steady_clock::now();
    const double prefillSeconds =
        std::chrono::duration<double>(prefillEnd - prefillStart).count();

    ExtractedOutput prefillExtracted = extractLogitsAndKV(
        prefillOutputs, -1, static_cast<uint32_t>(tokenIds.size() - 1), *device,
        decodeBinary, programIndex, decodeKVInputSlots, decodeKVOutputRoles,
        sampler, recentTokens);

    std::vector<int> generated;
    generated.push_back(prefillExtracted.token);
    recentTokens.push_back(prefillExtracted.token);
    std::unordered_map<std::string, ::tt::runtime::Tensor> pastKV =
        std::move(prefillExtracted.kvDevice);

    std::string streamedText;
    auto streamGeneratedText = [&]() {
      if (!suppress)
        return;
      std::string current = tokenizer.decodeTokens(generated, true);
      if (current.size() > streamedText.size()) {
        std::cout.write(current.data() + streamedText.size(),
                        current.size() - streamedText.size());
        std::cout.flush();
      }
      streamedText = std::move(current);
    };

    if (!suppress) {
      std::cout << colorLabel("[Iteration 0]")
                << " Token: " << tokenizer.decodeToken(prefillExtracted.token)
                << " | Time: " << formatFixed(prefillSeconds) << "s\n";
    } else {
      streamGeneratedText();
    }

    const int maxNewTokens =
        cfg.maxNewTokens > 0 ? cfg.maxNewTokens : maxCacheLen;
    int cachePos = static_cast<int>(tokenIds.size());
    double decodeWall = 0.0;
    double decodeSubmit = 0.0;
    int decodeCount = 0;
    bool stopped = isStopToken(prefillExtracted.token);
    while (static_cast<int>(generated.size()) < maxNewTokens &&
           cachePos < maxCacheLen && !stopped) {
      const int inputToken = generated.back();
      auto iterStart = std::chrono::steady_clock::now();
      auto decodeInputsRT = buildDecodeInputs(
          *device, decodeBinary, programIndex, decodeCtx, decodeInputs,
          decodeCache, inputToken, cachePos, pastKV);
      auto submitStart = std::chrono::steady_clock::now();
      auto decodeOutputs = ::tt::runtime::submit(*device, decodeBinary,
                                                 programIndex, decodeInputsRT);
      ::tt::runtime::wait(decodeOutputs);
      auto submitEnd = std::chrono::steady_clock::now();

      ExtractedOutput decoded = extractLogitsAndKV(
          decodeOutputs, -1, 0, *device, decodeBinary, programIndex,
          decodeKVInputSlots, decodeKVOutputRoles, sampler, recentTokens);
      for (auto &kv : pastKV) {
        try {
          ::tt::runtime::deallocateTensor(kv.second, true);
        } catch (...) {
        }
      }
      pastKV = std::move(decoded.kvDevice);

      generated.push_back(decoded.token);
      recentTokens.push_back(decoded.token);
      ++cachePos;
      ++decodeCount;
      auto iterEnd = std::chrono::steady_clock::now();
      const double submitSeconds =
          std::chrono::duration<double>(submitEnd - submitStart).count();
      const double iterSeconds =
          std::chrono::duration<double>(iterEnd - iterStart).count();
      decodeSubmit += submitSeconds;
      decodeWall += iterSeconds;

      if (!suppress) {
        std::cout << colorLabel("[Iteration " + std::to_string(decodeCount) +
                                "]")
                  << " Token: " << tokenizer.decodeToken(decoded.token)
                  << " | Time: " << formatFixed(iterSeconds) << "s\n";
      } else {
        streamGeneratedText();
      }
      stopped = isStopToken(decoded.token);
    }

    for (auto &kv : pastKV) {
      try {
        ::tt::runtime::deallocateTensor(kv.second, true);
      } catch (...) {
      }
    }

    if (suppress) {
      std::cout << "\n";
    } else {
      const double total = prefillSeconds + decodeWall;
      std::cout << "\n"
                << colorLabel("[Total time]") << " " << formatFixed(total)
                << "s\n";
      std::cout << colorLabel("[Prefilling]") << " "
                << formatFixed(prefillSeconds > 0.0
                                   ? static_cast<double>(maxCacheLen) /
                                         prefillSeconds
                                   : 0.0)
                << " tokens/s (prefill_time=" << formatFixed(prefillSeconds)
                << "s)\n";
      std::cout << colorLabel("[Decoding wall]") << " "
                << formatFixed(decodeWall > 0.0 ? decodeCount / decodeWall
                                                : 0.0)
                << " tokens/s (" << decodeCount << " tokens in "
                << formatFixed(decodeWall) << "s)\n";
      std::cout << colorLabel("[Decoding device-only]") << " "
                << formatFixed(decodeSubmit > 0.0 ? decodeCount / decodeSubmit
                                                  : 0.0)
                << " tokens/s\n";
      std::cout << colorLabel("[Input]") << " " << prompt << "\n";
      std::cout << colorLabel("[Output]") << " "
                << tokenizer.decodeTokens(generated, true) << "\n";
    }

    ::tt::runtime::closeMeshDevice(*device);
  } catch (...) {
    if (device) {
      try {
        ::tt::runtime::closeMeshDevice(*device);
      } catch (...) {
      }
    }
    throw;
  }
}

} // namespace runtime
} // namespace buddy
