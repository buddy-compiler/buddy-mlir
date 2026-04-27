//===- BufferPool.h - Runtime buffer lifecycle management -----------------===//
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
// Source tree: buddy-mlir/runtime/include/buddy/runtime/core/BufferPool.h
// Include as:  #include "buddy/runtime/core/BufferPool.h"
//
// Manages typed, role-annotated buffers for model execution.
// Replaces hand-written MemRefContainer and manual alloc in example mains.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_CORE_BUFFERPOOL_H
#define BUDDY_RUNTIME_CORE_BUFFERPOOL_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace buddy {
namespace runtime {

/// Role of a buffer in the model execution pipeline.
enum class BufferRole {
  Input,     ///< Function call input (ephemeral, host-owned, per-call)
  Output,    ///< Function call output (host-visible, per-call)
  State,     ///< Persistent session state (e.g., KV cache)
  Constant,  ///< Module-scope read-only weights
  Workspace, ///< Internal scratch (runtime-owned, not exposed to host)
};

/// Lifetime of a buffer.
enum class BufferLifetime {
  Call,    ///< Valid for one function invocation
  Session, ///< Valid for the lifetime of a session
  Module,  ///< Valid for the lifetime of the loaded module
};

/// A view into a runtime-managed buffer.
struct BufferView {
  std::string name;
  BufferRole role;
  BufferLifetime lifetime;
  void *data = nullptr;
  uint64_t bytes = 0;
  uint32_t id = 0;
};

/// Typed buffer descriptor used when registering buffers.
struct BufferDesc {
  std::string name;
  BufferRole role;
  BufferLifetime lifetime;
  uint64_t bytes = 0; ///< 0 means externally provided (no allocation)
  uint32_t id = 0;
};

/// Runtime buffer pool.
///
/// Owns allocated buffers and provides lookup by name or id.
/// External buffers (e.g., mmap'd weights) can be registered without
/// transferring ownership.
class BufferPool {
public:
  BufferPool() = default;
  ~BufferPool() = default;

  // Non-copyable, movable
  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;
  BufferPool(BufferPool &&) = default;

  /// Allocate and register a buffer from a descriptor.
  BufferView &allocate(const BufferDesc &desc) {
    if (byName_.count(desc.name))
      throw std::runtime_error("BufferPool: duplicate name: " + desc.name);

    auto storage = std::make_unique<std::vector<uint8_t>>(desc.bytes, 0);
    void *ptr = desc.bytes ? storage->data() : nullptr;

    BufferView bv;
    bv.name = desc.name;
    bv.role = desc.role;
    bv.lifetime = desc.lifetime;
    bv.data = ptr;
    bv.bytes = desc.bytes;
    bv.id = desc.id;

    owned_.push_back(std::move(storage));
    views_.push_back(bv);
    byName_[desc.name] = views_.size() - 1;
    if (desc.id != 0)
      byId_[desc.id] = views_.size() - 1;

    return views_.back();
  }

  /// Register an externally owned buffer (no allocation, no ownership).
  BufferView &registerExternal(const std::string &name, BufferRole role,
                               BufferLifetime lifetime, void *data,
                               uint64_t bytes, uint32_t id = 0) {
    if (byName_.count(name))
      throw std::runtime_error("BufferPool: duplicate name: " + name);

    BufferView bv;
    bv.name = name;
    bv.role = role;
    bv.lifetime = lifetime;
    bv.data = data;
    bv.bytes = bytes;
    bv.id = id;

    // Sentinel: no owned storage for external
    owned_.push_back(nullptr);
    views_.push_back(bv);
    byName_[name] = views_.size() - 1;
    if (id != 0)
      byId_[id] = views_.size() - 1;

    return views_.back();
  }

  BufferView *findByName(const std::string &name) {
    auto it = byName_.find(name);
    if (it == byName_.end())
      return nullptr;
    return &views_[it->second];
  }

  const BufferView *findByName(const std::string &name) const {
    auto it = byName_.find(name);
    if (it == byName_.end())
      return nullptr;
    return &views_[it->second];
  }

  BufferView &getByName(const std::string &name) {
    auto *bv = findByName(name);
    if (!bv)
      throw std::runtime_error("BufferPool: not found: " + name);
    return *bv;
  }

  const std::vector<BufferView> &views() const { return views_; }

  std::vector<BufferView *> byRole(BufferRole role) {
    std::vector<BufferView *> result;
    for (auto &bv : views_)
      if (bv.role == role)
        result.push_back(&bv);
    return result;
  }

private:
  std::vector<BufferView> views_;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> owned_;
  std::unordered_map<std::string, size_t> byName_;
  std::unordered_map<uint32_t, size_t> byId_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_BUFFERPOOL_H
