//===- Llama31TTNNBackend.cpp - Native TTNN resident session -------------===//
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

#include "buddy/runtime/models/Llama31TTNNBackend.h"

#ifndef BUDDY_LLAMA31_TT_HAS_TTNN
#define BUDDY_LLAMA31_TT_HAS_TTNN 0
#endif

#if BUDDY_LLAMA31_TT_HAS_TTNN

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {
namespace {

namespace fs = std::filesystem;
using TTDataType = ::tt::target::DataType;

constexpr int kEndOfText = 128001;
constexpr int kEom = 128008;
constexpr int kEot = 128009;

struct Role {
  std::string name;
  std::string dtype;
  std::vector<uint32_t> shape;
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

struct PhaseArtifacts {
  std::string name;
  fs::path root;
  std::vector<Role> roles;
  std::vector<uint8_t> weights;
  std::vector<uint8_t> invFreq;
  size_t invFreqOffset = 0;
};

uint64_t volume(const std::vector<uint32_t> &shape) {
  uint64_t result = 1;
  for (uint32_t dimension : shape) {
    if (dimension == 0 || result > std::numeric_limits<uint64_t>::max() /
                                       static_cast<uint64_t>(dimension))
      throw std::runtime_error("llama31_tt: tensor descriptor has invalid "
                               "zero/overflowing shape");
    result *= dimension;
  }
  return result;
}

std::vector<uint32_t> contiguousStride(const std::vector<uint32_t> &shape) {
  std::vector<uint32_t> stride(shape.size(), 1);
  uint64_t running = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    stride[index - 1] = static_cast<uint32_t>(running);
    running *= shape[index - 1];
  }
  return stride;
}

uint32_t itemSizeFor(TTDataType dtype) {
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
    throw std::runtime_error("llama31_tt: unsupported TTNN scalar dtype");
  }
}

std::vector<uint8_t> readFile(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("llama31_tt: cannot read artifact " +
                             path.string());
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size <= 0)
    throw std::runtime_error("llama31_tt: artifact is empty: " + path.string());
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!input)
    throw std::runtime_error("llama31_tt: failed reading artifact " +
                             path.string());
  return bytes;
}

std::string readText(const fs::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("llama31_tt: cannot read artifact " +
                             path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<Role> readRoles(const fs::path &path) {
  auto parsed = llvm::json::parse(readText(path));
  if (!parsed)
    throw std::runtime_error("llama31_tt: cannot parse " + path.string() +
                             ": " + llvm::toString(parsed.takeError()));
  const auto *array = parsed->getAsArray();
  if (!array)
    throw std::runtime_error("llama31_tt: " + path.string() +
                             " must contain an array");

  std::vector<Role> roles;
  roles.reserve(array->size());
  for (const auto &value : *array) {
    const auto *object = value.getAsObject();
    const auto *shape = object ? object->getArray("shape") : nullptr;
    auto role = object ? object->getString("role") : std::nullopt;
    auto slot = object ? object->getInteger("slot") : std::nullopt;
    if (!object || !shape || !role || !slot || *slot < 0)
      throw std::runtime_error("llama31_tt: malformed slot role in " +
                               path.string());
    Role entry;
    entry.name = role->str();
    if (auto dtype = object->getString("dtype"))
      entry.dtype = dtype->str();
    if (entry.name == "weight") {
      auto offset = object->getInteger("weight_offset");
      auto bytes = object->getInteger("weight_nbytes");
      if (!offset || !bytes || *offset < 0 || *bytes <= 0)
        throw std::runtime_error("llama31_tt: weight role is missing its "
                                 "byte range in " +
                                 path.string());
      entry.offset = static_cast<uint64_t>(*offset);
      entry.bytes = static_cast<uint64_t>(*bytes);
    }
    for (const auto &dimension : *shape) {
      auto number = dimension.getAsInteger();
      if (!number || *number <= 0 ||
          *number > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("llama31_tt: invalid role shape in " +
                                 path.string());
      entry.shape.push_back(static_cast<uint32_t>(*number));
    }
    if (static_cast<size_t>(*slot) != roles.size())
      throw std::runtime_error("llama31_tt: slot_roles.json has non-contiguous "
                               "slots in " +
                               path.string());
    roles.push_back(std::move(entry));
  }
  return roles;
}

uint16_t readU16(const uint8_t *data) {
  return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t readU32(const uint8_t *data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

size_t parseNpyOffset(const std::vector<uint8_t> &bytes, const fs::path &path) {
  if (bytes.size() < 10 || bytes[0] != 0x93 || bytes[1] != 'N' ||
      bytes[2] != 'U' || bytes[3] != 'M' || bytes[4] != 'P' || bytes[5] != 'Y')
    throw std::runtime_error("llama31_tt: invalid inv_freq.npy: " +
                             path.string());
  size_t headerOffset = 10;
  size_t headerBytes = readU16(bytes.data() + 8);
  if (bytes[6] >= 2) {
    if (bytes.size() < 12)
      throw std::runtime_error("llama31_tt: truncated inv_freq.npy: " +
                               path.string());
    headerOffset = 12;
    headerBytes = readU32(bytes.data() + 8);
  }
  if (headerOffset > bytes.size() || headerBytes > bytes.size() - headerOffset)
    throw std::runtime_error("llama31_tt: truncated inv_freq.npy: " +
                             path.string());
  const std::string header(
      reinterpret_cast<const char *>(bytes.data() + headerOffset), headerBytes);
  if (header.find("'fortran_order': False") == std::string::npos &&
      header.find("\"fortran_order\": False") == std::string::npos)
    throw std::runtime_error("llama31_tt: inv_freq.npy must be C-order: " +
                             path.string());
  return headerOffset + headerBytes;
}

PhaseArtifacts loadPhase(const fs::path &root, const std::string &phase) {
  PhaseArtifacts result;
  result.name = phase;
  result.root = root / phase;
  std::error_code error;
  if (!fs::is_directory(result.root, error))
    throw std::runtime_error("llama31_tt: missing " + phase +
                             " artifact directory: " + result.root.string());
  for (const char *file : {"shapes.json", "dtypes.json", "summary.json"})
    (void)readText(result.root / file);
  result.roles = readRoles(result.root / "slot_roles.json");
  const fs::path weights = result.root / "weights.bin";
  if (fs::exists(weights, error))
    result.weights = readFile(weights);
  for (const Role &role : result.roles) {
    if (role.name != "weight")
      continue;
    if (result.weights.empty() || role.offset > result.weights.size() ||
        role.bytes > result.weights.size() - role.offset)
      throw std::runtime_error("llama31_tt: " + phase +
                               " weight byte range exceeds weights.bin");
  }
  const fs::path invFreq = result.root / "inv_freq.npy";
  if (fs::is_regular_file(invFreq, error)) {
    result.invFreq = readFile(invFreq);
    result.invFreqOffset = parseNpyOffset(result.invFreq, invFreq);
    if (result.invFreqOffset == result.invFreq.size())
      throw std::runtime_error("llama31_tt: inv_freq.npy has no data: " +
                               invFreq.string());
  }
  return result;
}

std::vector<float> invFreqValues(const PhaseArtifacts &phase, size_t count) {
  for (const Role &role : phase.roles) {
    if (role.name != "weight" || role.dtype != "float32" ||
        role.shape.size() != 1 || role.shape[0] != count)
      continue;
    if (role.offset > phase.weights.size() ||
        role.bytes < count * sizeof(float) ||
        role.bytes > phase.weights.size() - role.offset)
      throw std::runtime_error("llama31_tt: invalid inv_freq weight range in " +
                               phase.name);
    const auto *values = reinterpret_cast<const float *>(
        phase.weights.data() + static_cast<size_t>(role.offset));
    return std::vector<float>(values, values + count);
  }
  if (phase.invFreq.empty() ||
      phase.invFreq.size() - phase.invFreqOffset < count * sizeof(float))
    throw std::runtime_error("llama31_tt: cannot find inv_freq values for " +
                             phase.name);
  const auto *values = reinterpret_cast<const float *>(phase.invFreq.data() +
                                                       phase.invFreqOffset);
  return std::vector<float>(values, values + count);
}

std::vector<uint8_t> integerBuffer(const std::vector<int64_t> &values,
                                   TTDataType dtype) {
  std::vector<uint8_t> buffer(values.size() * itemSizeFor(dtype));
  auto write = [&](auto tag) {
    using T = decltype(tag);
    auto *out = reinterpret_cast<T *>(buffer.data());
    for (size_t index = 0; index < values.size(); ++index)
      out[index] = static_cast<T>(values[index]);
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
    throw std::runtime_error("llama31_tt: runtime input is not integer dtype");
  }
  return buffer;
}

::tt::runtime::Tensor hostTensor(const std::vector<uint8_t> &bytes,
                                 const ::tt::runtime::TensorDesc &desc) {
  if (bytes.size() != volume(desc.shape) * desc.elementSize())
    throw std::runtime_error("llama31_tt: host tensor byte size does not "
                             "match descriptor");
  const auto stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createOwnedHostTensor(
      bytes.data(), desc.shape, stride, desc.elementSize(), desc.dataType);
}

::tt::runtime::Tensor hostTensorFromRaw(const uint8_t *data, size_t bytes,
                                        const ::tt::runtime::TensorDesc &desc) {
  if (bytes != volume(desc.shape) * desc.elementSize())
    throw std::runtime_error("llama31_tt: static tensor byte size does not "
                             "match descriptor");
  const auto stride =
      desc.stride.empty() ? contiguousStride(desc.shape) : desc.stride;
  return ::tt::runtime::createBorrowedHostTensor(
      const_cast<uint8_t *>(data), desc.shape, stride, desc.elementSize(),
      desc.dataType);
}

::tt::runtime::Tensor toDevice(::tt::runtime::Tensor host,
                               ::tt::runtime::Device device,
                               ::tt::runtime::Binary binary, uint32_t program,
                               uint32_t slot, bool retain = false) {
  auto layout = ::tt::runtime::getLayout(binary, program, slot);
  auto tensor = ::tt::runtime::toLayout(host, device, layout, retain);
  if (retain)
    ::tt::runtime::setTensorRetain(tensor, true);
  return tensor;
}

uint16_t floatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

std::vector<uint8_t> floatBuffer(const std::vector<float> &values,
                                 TTDataType dtype) {
  std::vector<uint8_t> buffer(values.size() * itemSizeFor(dtype));
  if (dtype == TTDataType::Float32) {
    std::memcpy(buffer.data(), values.data(), buffer.size());
    return buffer;
  }
  if (dtype == TTDataType::BFloat16) {
    auto *out = reinterpret_cast<uint16_t *>(buffer.data());
    for (size_t index = 0; index < values.size(); ++index)
      out[index] = floatToBf16(values[index]);
    return buffer;
  }
  throw std::runtime_error("llama31_tt: unsupported floating runtime input "
                           "dtype");
}

float halfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x3ff;
  uint32_t bits = 0;
  if (exponent == 0 && mantissa != 0) {
    exponent = 1;
    while ((mantissa & 0x400) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 0x3ff;
  }
  if (exponent == 0)
    bits = sign;
  else if (exponent == 31)
    bits = sign | 0x7f800000 | (mantissa << 13);
  else
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  float result = 0;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool isTokenDType(TTDataType dtype) {
  return dtype == TTDataType::Int32 || dtype == TTDataType::Int64 ||
         dtype == TTDataType::UInt32;
}

int tokenAt(const std::vector<std::byte> &data, TTDataType dtype,
            size_t index) {
  switch (dtype) {
  case TTDataType::Int32:
    return reinterpret_cast<const int32_t *>(data.data())[index];
  case TTDataType::Int64:
    return static_cast<int>(
        reinterpret_cast<const int64_t *>(data.data())[index]);
  case TTDataType::UInt32:
    return static_cast<int>(
        reinterpret_cast<const uint32_t *>(data.data())[index]);
  default:
    throw std::runtime_error("llama31_tt: output is not token-id dtype");
  }
}

int sampleOutput(::tt::runtime::Tensor &output, int position,
                 buddy::Sampler &sampler, const std::vector<int> &recent) {
  const auto hostShards = ::tt::runtime::toHost(
      output, !isTokenDType(::tt::runtime::getTensorDataType(output)), true);
  if (hostShards.empty())
    throw std::runtime_error("llama31_tt: output toHost returned no shards");
  auto host = hostShards.front();
  const TTDataType dtype = ::tt::runtime::getTensorDataType(host);
  const auto shape = ::tt::runtime::getTensorShape(host);
  const auto data = ::tt::runtime::getTensorDataBuffer(host);
  const size_t elementSize = itemSizeFor(dtype);
  const size_t elements = data.size() / elementSize;
  if (elements == 0)
    throw std::runtime_error("llama31_tt: output tensor is empty");
  if (isTokenDType(dtype)) {
    size_t start = 0;
    if (shape.size() >= 2)
      start = std::min<size_t>(std::max(position, 0), shape.back() - 1);
    else
      start = std::min<size_t>(std::max(position, 0), elements - 1);
    return tokenAt(data, dtype, start);
  }
  size_t start = 0;
  size_t count = elements;
  if (shape.size() >= 3) {
    const size_t sequence = shape[shape.size() - 2];
    count = shape.back();
    start = std::min<size_t>(std::max(position, 0), sequence - 1) * count;
  } else if (shape.size() >= 2) {
    count = shape.back();
    start = 0;
  } else if (isTokenDType(dtype)) {
    start = std::min<size_t>(std::max(position, 0), elements - 1);
    count = 1;
  }
  if (start >= elements || count > elements - start)
    throw std::runtime_error(
        "llama31_tt: output sample range is out of bounds");
  std::vector<float> logits(count);
  if (dtype == TTDataType::Float32) {
    const auto *values = reinterpret_cast<const float *>(data.data());
    std::copy(values + start, values + start + count, logits.begin());
  } else if (dtype == TTDataType::BFloat16) {
    const auto *values = reinterpret_cast<const uint16_t *>(data.data());
    for (size_t index = 0; index < count; ++index)
      logits[index] = bf16ToFloat(values[start + index]);
  } else if (dtype == TTDataType::Float16) {
    const auto *values = reinterpret_cast<const uint16_t *>(data.data());
    for (size_t index = 0; index < count; ++index)
      logits[index] = halfToFloat(values[start + index]);
  } else {
    throw std::runtime_error("llama31_tt: unsupported output dtype");
  }
  return sampler.sample(logits.data(), logits.size(), recent);
}

void release(std::unordered_map<std::string, ::tt::runtime::Tensor> &tensors) {
  for (auto &entry : tensors) {
    try {
      ::tt::runtime::deallocateTensor(entry.second, true);
    } catch (...) {
    }
  }
  tensors.clear();
}

void release(std::unordered_map<uint32_t, ::tt::runtime::Tensor> &tensors) {
  for (auto &entry : tensors) {
    try {
      ::tt::runtime::deallocateTensor(entry.second, true);
    } catch (...) {
    }
  }
  tensors.clear();
}

class Session {
public:
  explicit Session(const Llama31TTExecution::Metadata &metadata)
      : metadata(metadata) {
    try {
      initialize();
    } catch (...) {
      close();
      throw;
    }
  }

  ~Session() { close(); }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    // All request tensors are local to generate(). This method is still an
    // explicit lifecycle hook so a cancelled request cannot leave state in a
    // future implementation that retains KV handles between calls.
    active = false;
  }

  CompletionResult generate(const Llama31TTExecution::BackendRequest &request,
                            const CompletionStreamCallback &callback) {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed || !device)
      throw std::runtime_error("llama31_tt: TTNN session is closed");
    if (active)
      throw std::runtime_error("llama31_tt: TTNN session request is already "
                               "active");
    active = true;
    std::unordered_map<std::string, ::tt::runtime::Tensor> pastKV;
    try {
      CompletionResult result = run(request, callback, pastKV);
      release(pastKV);
      active = false;
      return result;
    } catch (...) {
      release(pastKV);
      active = false;
      throw;
    }
  }

private:
  void initialize() {
    prefillBinary =
        ::tt::runtime::Binary::loadFromPath(metadata.prefillPath.c_str());
    decodeBinary =
        ::tt::runtime::Binary::loadFromPath(metadata.decodePath.c_str());
    ::tt::runtime::setCompatibleDeviceRuntime(*prefillBinary);
    prefillInputs = prefillBinary->getProgramInputs(metadata.programIndex);
    decodeInputs = decodeBinary->getProgramInputs(metadata.programIndex);
    prefill = loadPhase(metadata.artifactsPath, "prefill");
    decode = loadPhase(metadata.artifactsPath, "decode");
    if (prefillInputs.size() != prefill.roles.size())
      throw std::runtime_error("llama31_tt: prefill descriptor/slot count "
                               "mismatch");
    if (decodeInputs.size() != decode.roles.size())
      throw std::runtime_error("llama31_tt: decode descriptor/slot count "
                               "mismatch");

    for (size_t index = 0; index < prefill.roles.size(); ++index)
      validateRole(prefill, prefillInputs[index], index);
    for (size_t index = 0; index < decode.roles.size(); ++index)
      validateRole(decode, decodeInputs[index], index);

    for (uint32_t slot = 0; slot < decode.roles.size(); ++slot) {
      const std::string &role = decode.roles[slot].name;
      if (role.rfind("past_K_", 0) == 0 || role.rfind("past_V_", 0) == 0) {
        decodeKVSlots[role] = slot;
        decodeKVRoles.push_back(role);
      }
    }
    if (decodeKVRoles.empty())
      throw std::runtime_error("llama31_tt: decode artifacts contain no KV "
                               "input roles");
    prefillKVRoles = decodeKVRoles;
    if (metadata.prefillKVOutputOrder == "value_key") {
      if (prefillKVRoles.size() % 2 != 0)
        throw std::runtime_error("llama31_tt: value_key prefill KV order "
                                 "requires K/V pairs");
      for (size_t index = 0; index < prefillKVRoles.size(); index += 2)
        std::swap(prefillKVRoles[index], prefillKVRoles[index + 1]);
    }

    ::tt::runtime::MeshDeviceOptions options;
    options.meshShape = std::vector<uint32_t>{1, 1};
    device = ::tt::runtime::openMeshDevice(options);
    uploadStatic(prefill, prefillInputs, *prefillBinary, prefillStatic);
    if (metadata.disableStaticReuse)
      uploadStatic(decode, decodeInputs, *decodeBinary, decodeStatic);
    else
      uploadStatic(decode, decodeInputs, *decodeBinary, decodeStatic, &prefill,
                   &prefillInputs, &prefillStatic);
  }

  void validateRole(const PhaseArtifacts &phase,
                    const ::tt::runtime::TensorDesc &desc, size_t slot) const {
    if (phase.roles[slot].shape != desc.shape)
      throw std::runtime_error("llama31_tt: " + phase.name + " slot " +
                               std::to_string(slot) + " role '" +
                               phase.roles[slot].name +
                               "' shape differs from TTNN descriptor");
    (void)desc.elementSize();
  }

  void uploadStatic(
      const PhaseArtifacts &phase,
      const std::vector<::tt::runtime::TensorDesc> &descs,
      ::tt::runtime::Binary binary,
      std::unordered_map<uint32_t, ::tt::runtime::Tensor> &out,
      const PhaseArtifacts *sourcePhase = nullptr,
      const std::vector<::tt::runtime::TensorDesc> *sourceDescs = nullptr,
      std::unordered_map<uint32_t, ::tt::runtime::Tensor> *sourceCache =
          nullptr) {
    for (uint32_t slot = 0; slot < phase.roles.size(); ++slot) {
      const Role &role = phase.roles[slot];
      if (role.name != "weight" && role.name != "inv_freq")
        continue;
      if (sourcePhase && sourceDescs && sourceCache) {
        for (uint32_t sourceSlot = 0; sourceSlot < sourcePhase->roles.size();
             ++sourceSlot) {
          const Role &sourceRole = sourcePhase->roles[sourceSlot];
          if (sourceRole.name != role.name || sourceRole.shape != role.shape ||
              sourceRole.dtype != role.dtype ||
              (*sourceDescs)[sourceSlot].shape != descs[slot].shape ||
              (*sourceDescs)[sourceSlot].dataType != descs[slot].dataType)
            continue;
          if (role.name == "weight" &&
              (role.bytes != sourceRole.bytes ||
               std::memcmp(phase.weights.data() + role.offset,
                           sourcePhase->weights.data() + sourceRole.offset,
                           static_cast<size_t>(role.bytes)) != 0))
            continue;
          auto source = sourceCache->find(sourceSlot);
          if (source == sourceCache->end())
            continue;
          auto layout =
              ::tt::runtime::getLayout(binary, metadata.programIndex, slot);
          if (!::tt::runtime::hasLayout(source->second, layout))
            continue;
          ::tt::runtime::setTensorRetain(source->second, true);
          out.emplace(slot, source->second);
          break;
        }
        if (out.find(slot) != out.end())
          continue;
      }
      ::tt::runtime::Tensor host;
      if (role.name == "weight") {
        host = hostTensorFromRaw(phase.weights.data() + role.offset,
                                 static_cast<size_t>(role.bytes), descs[slot]);
      } else {
        if (phase.invFreq.empty())
          throw std::runtime_error("llama31_tt: " + phase.name +
                                   " inv_freq role has no inv_freq.npy");
        host = hostTensorFromRaw(phase.invFreq.data() + phase.invFreqOffset,
                                 phase.invFreq.size() - phase.invFreqOffset,
                                 descs[slot]);
      }
      out.emplace(slot, toDevice(std::move(host), *device, binary,
                                 metadata.programIndex, slot, true));
    }
  }

  std::vector<::tt::runtime::Tensor> buildInputs(
      const PhaseArtifacts &phase,
      const std::vector<::tt::runtime::TensorDesc> &descs,
      ::tt::runtime::Binary binary,
      const std::unordered_map<uint32_t, ::tt::runtime::Tensor> &cache,
      const std::vector<int> &tokens, int cachePosition, bool prefill,
      int padToken,
      const std::unordered_map<std::string, ::tt::runtime::Tensor> &requestKV) {
    std::vector<::tt::runtime::Tensor> result(descs.size());
    for (uint32_t slot = 0; slot < phase.roles.size(); ++slot) {
      if (auto cached = cache.find(slot); cached != cache.end()) {
        result[slot] = cached->second;
        continue;
      }
      const Role &role = phase.roles[slot];
      const auto &desc = descs[slot];
      std::vector<uint8_t> buffer;
      if (role.name == "input_ids") {
        std::vector<int64_t> values(static_cast<size_t>(volume(desc.shape)),
                                    prefill ? padToken : 0);
        if (prefill) {
          if (tokens.size() > values.size())
            throw std::runtime_error("llama31_tt: prompt is larger than "
                                     "prefill input descriptor");
          std::copy(tokens.begin(), tokens.end(), values.begin());
        } else {
          if (tokens.empty())
            throw std::runtime_error("llama31_tt: decode input has no token");
          std::fill(values.begin(), values.end(), tokens.front());
          if (tokens.size() == values.size())
            std::copy(tokens.begin(), tokens.end(), values.begin());
          else if (tokens.size() != 1)
            throw std::runtime_error("llama31_tt: decode input shape does not "
                                     "match batch-1 token");
        }
        buffer = integerBuffer(values, desc.dataType);
      } else if (role.name == "attention_mask") {
        buffer = integerBuffer(
            std::vector<int64_t>(static_cast<size_t>(volume(desc.shape)), 1),
            desc.dataType);
      } else if (role.name == "cache_position" ||
                 role.name == "sdpa_position") {
        std::vector<int64_t> values(static_cast<size_t>(volume(desc.shape)),
                                    prefill ? 0 : cachePosition);
        if (prefill)
          for (size_t index = 0; index < values.size(); ++index)
            values[index] = static_cast<int64_t>(index);
        buffer = integerBuffer(values, desc.dataType);
      } else if (role.name == "sdpa_mask") {
        if (desc.dataType != TTDataType::Float32 &&
            desc.dataType != TTDataType::BFloat16)
          throw std::runtime_error("llama31_tt: decode sdpa_mask slot " +
                                   std::to_string(slot) +
                                   " must be floating point");
        std::vector<float> values(static_cast<size_t>(volume(desc.shape)),
                                  0.0f);
        if (!prefill && desc.shape.size() == 4) {
          const size_t sequence = desc.shape.back();
          const float negInf = -std::numeric_limits<float>::infinity();
          for (size_t index = 0; index < values.size(); ++index)
            if (index % sequence > static_cast<size_t>(cachePosition))
              values[index] = negInf;
        }
        buffer = floatBuffer(values, desc.dataType);
      } else if (role.name == "rope_cos" || role.name == "rope_sin") {
        if (desc.shape.size() != 4 || desc.shape[0] != 1 ||
            desc.shape[1] != 1 || desc.shape[2] != 1 || desc.shape[3] % 2 != 0)
          throw std::runtime_error("llama31_tt: " + phase.name + " slot " +
                                   std::to_string(slot) + " role '" +
                                   role.name + "' has an invalid shape");
        const size_t width = desc.shape[3];
        const std::vector<float> invFreq = invFreqValues(phase, width / 2);
        std::vector<float> values;
        values.reserve(width);
        for (size_t index = 0; index < width; ++index) {
          const float angle = invFreq[index % invFreq.size()] *
                              static_cast<float>(prefill ? 0 : cachePosition);
          values.push_back(role.name == "rope_cos" ? std::cos(angle)
                                                   : std::sin(angle));
        }
        buffer = floatBuffer(values, desc.dataType);
      } else if (role.name.rfind("past_K_", 0) == 0 ||
                 role.name.rfind("past_V_", 0) == 0) {
        if (prefill) {
          buffer = std::vector<uint8_t>(
              static_cast<size_t>(volume(desc.shape)) * desc.elementSize(), 0);
        } else {
          auto found = requestKV.find(role.name);
          if (found == requestKV.end())
            throw std::runtime_error("llama31_tt: missing decode KV tensor '" +
                                     role.name + "'");
          ::tt::runtime::setTensorRetain(found->second, true);
          result[slot] = found->second;
          continue;
        }
      } else {
        throw std::runtime_error("llama31_tt: unsupported " + phase.name +
                                 " input role '" + role.name + "' at slot " +
                                 std::to_string(slot));
      }
      result[slot] = toDevice(hostTensor(buffer, desc), *device, binary,
                              metadata.programIndex, slot);
    }
    return result;
  }

  std::unordered_map<std::string, ::tt::runtime::Tensor>
  extractKV(std::vector<::tt::runtime::Tensor> &outputs,
            const std::vector<std::string> &roles,
            ::tt::runtime::Binary binary) {
    if (outputs.size() < roles.size() + 1)
      throw std::runtime_error("llama31_tt: output count is smaller than KV "
                               "role count");
    std::unordered_map<std::string, ::tt::runtime::Tensor> result;
    for (size_t index = 0; index < roles.size(); ++index) {
      const std::string &role = roles[index];
      auto slot = decodeKVSlots.find(role);
      if (slot == decodeKVSlots.end())
        throw std::runtime_error("llama31_tt: no decode KV slot for '" + role +
                                 "'");
      auto layout =
          ::tt::runtime::getLayout(binary, metadata.programIndex, slot->second);
      auto &output = outputs[index];
      ::tt::runtime::setTensorRetain(output, true);
      result[role] = ::tt::runtime::hasLayout(output, layout)
                         ? output
                         : ::tt::runtime::toLayout(output, *device, layout);
      ::tt::runtime::setTensorRetain(result[role], true);
    }
    return result;
  }

  CompletionResult
  run(const Llama31TTExecution::BackendRequest &request,
      const CompletionStreamCallback &callback,
      std::unordered_map<std::string, ::tt::runtime::Tensor> &pastKV) {
    if (request.promptTokens.empty())
      throw std::invalid_argument("llama31_tt: TTNN prompt is empty");
    if (request.promptTokens.size() >= static_cast<size_t>(request.maxCacheLen))
      throw std::invalid_argument("llama31_tt: TTNN prompt exceeds cache "
                                  "capacity");

    buddy::Sampler sampler(request.sampling.samplerConfig);
    std::vector<int> recent = request.promptTokens;
    const int available = request.maxCacheLen - request.cachePosition;
    const int budget = std::min(
        request.sampling.maxTokens > 0 ? request.sampling.maxTokens : available,
        available);
    CompletionResult result;
    result.model = metadata.modelName;
    result.usage.promptTokens = static_cast<int>(request.promptTokens.size());
    static std::atomic<uint64_t> nextId{1};
    result.id = "llama31-tt-cmpl-" + std::to_string(nextId.fetch_add(1));
    if (budget <= 0) {
      result.finishReason = FinishReason::Length;
      result.usage.totalTokens = result.usage.promptTokens;
      return result;
    }

    const auto prefillStart = std::chrono::steady_clock::now();
    auto inputs =
        buildInputs(prefill, prefillInputs, *prefillBinary, prefillStatic,
                    request.promptTokens, 0, true, request.eosTokenId, pastKV);
    auto outputs = ::tt::runtime::submit(*device, *prefillBinary,
                                         metadata.programIndex, inputs);
    ::tt::runtime::wait(outputs);
    const auto prefillEnd = std::chrono::steady_clock::now();
    result.timings.prefillMs =
        std::chrono::duration<double, std::milli>(prefillEnd - prefillStart)
            .count();
    const int firstToken = sampleOutput(
        outputs.back(), request.cachePosition - 1, sampler, recent);
    pastKV = extractKV(outputs, prefillKVRoles, *decodeBinary);
    try {
      ::tt::runtime::deallocateTensor(outputs.back(), true);
    } catch (...) {
    }
    outputs.clear();

    auto isStop = [&](int token) {
      if (request.ignoreEOS)
        return false;
      if (token == request.eosTokenId || token == kEndOfText || token == kEom ||
          token == kEot)
        return true;
      return std::find(request.sampling.stopTokenIds.begin(),
                       request.sampling.stopTokenIds.end(),
                       token) != request.sampling.stopTokenIds.end();
    };
    auto emit = [&](int token) {
      if (isStop(token))
        return true;
      CompletionChunk chunk;
      chunk.id = result.id;
      chunk.model = metadata.modelName;
      chunk.tokenId = token;
      if (request.decodeToken)
        chunk.delta = request.decodeToken(token);
      if (callback && !callback(chunk))
        return false;
      result.content += chunk.delta;
      ++result.usage.completionTokens;
      return true;
    };

    int lastToken = firstToken;
    if (!emit(firstToken)) {
      result.finishReason = FinishReason::Cancelled;
      result.usage.totalTokens = result.usage.promptTokens;
      return result;
    }
    if (isStop(firstToken)) {
      result.finishReason = FinishReason::Stop;
      result.usage.totalTokens = result.usage.promptTokens;
      return result;
    }
    recent.push_back(firstToken);

    int cachePosition = request.cachePosition;
    const auto decodeStart = std::chrono::steady_clock::now();
    while (result.usage.completionTokens < budget &&
           cachePosition < request.maxCacheLen) {
      auto decodeInputsRT = buildInputs(
          decode, decodeInputs, *decodeBinary, decodeStatic, {lastToken},
          cachePosition, false, request.eosTokenId, pastKV);
      auto decodeOutputs = ::tt::runtime::submit(
          *device, *decodeBinary, metadata.programIndex, decodeInputsRT);
      ::tt::runtime::wait(decodeOutputs);
      const int token = sampleOutput(decodeOutputs.back(), 0, sampler, recent);
      const bool hasDecodeKV = decodeOutputs.size() >= decodeKVRoles.size() + 1;
      if (hasDecodeKV) {
        auto updated = extractKV(decodeOutputs, decodeKVRoles, *decodeBinary);
        release(pastKV);
        pastKV = std::move(updated);
      }
      for (size_t index = 0; index < decodeOutputs.size(); ++index) {
        if (hasDecodeKV && index < decodeKVRoles.size())
          continue;
        try {
          ::tt::runtime::deallocateTensor(decodeOutputs[index], true);
        } catch (...) {
        }
      }
      decodeOutputs.clear();
      lastToken = token;
      recent.push_back(lastToken);
      ++cachePosition;
      if (!emit(token)) {
        result.finishReason = FinishReason::Cancelled;
        break;
      }
      if (isStop(token)) {
        result.finishReason = FinishReason::Stop;
        break;
      }
    }
    if (result.finishReason == FinishReason::None)
      result.finishReason = cachePosition >= request.maxCacheLen ||
                                    result.usage.completionTokens >= budget
                                ? FinishReason::Length
                                : FinishReason::Stop;
    const auto decodeEnd = std::chrono::steady_clock::now();
    result.timings.decodeMs =
        std::chrono::duration<double, std::milli>(decodeEnd - decodeStart)
            .count();
    if (result.timings.decodeMs > 0.0)
      result.timings.tokensPerSecond =
          result.usage.completionTokens * 1000.0 / result.timings.decodeMs;
    result.usage.totalTokens =
        result.usage.promptTokens + result.usage.completionTokens;
    return result;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex);
    if (closed)
      return;
    release(prefillStatic);
    release(decodeStatic);
    if (device) {
      try {
        ::tt::runtime::closeMeshDevice(*device);
      } catch (...) {
      }
      device.reset();
    }
    closed = true;
  }

  Llama31TTExecution::Metadata metadata;
  std::optional<::tt::runtime::Binary> prefillBinary;
  std::optional<::tt::runtime::Binary> decodeBinary;
  PhaseArtifacts prefill;
  PhaseArtifacts decode;
  std::vector<::tt::runtime::TensorDesc> prefillInputs;
  std::vector<::tt::runtime::TensorDesc> decodeInputs;
  std::optional<::tt::runtime::Device> device;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> prefillStatic;
  std::unordered_map<uint32_t, ::tt::runtime::Tensor> decodeStatic;
  std::unordered_map<std::string, uint32_t> decodeKVSlots;
  std::vector<std::string> decodeKVRoles;
  std::vector<std::string> prefillKVRoles;
  std::mutex mutex;
  bool active = false;
  bool closed = false;
};

} // namespace

Llama31TTExecution::BackendHooks
createLlama31TTNNBackend(const Llama31TTExecution::Metadata &metadata) {
  auto session = std::make_shared<Session>(metadata);
  Llama31TTExecution::BackendHooks hooks;
  hooks.generate = [session](const Llama31TTExecution::BackendRequest &request,
                             const CompletionStreamCallback &callback) {
    return session->generate(request, callback);
  };
  hooks.reset = [session] { session->reset(); };
  return hooks;
}

} // namespace runtime
} // namespace buddy

#else

#include <stdexcept>

namespace buddy {
namespace runtime {

Llama31TTExecution::BackendHooks
createLlama31TTNNBackend(const Llama31TTExecution::Metadata &) {
  throw std::runtime_error(
      "llama31_tt: TTNN serving backend was not compiled; configure with "
      "BUDDY_ENABLE_TENSTORRENT=ON");
}

} // namespace runtime
} // namespace buddy

#endif
