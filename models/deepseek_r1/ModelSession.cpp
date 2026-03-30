//===- ModelSession.cpp - ModelSession implementation ---------------------===//
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
// Dynamic-loading version: dlopen/dlsym, no static model symbols.
//
// Notes on MemRef default constructor:
//   Container.h declares MemRef(){} as *protected*, so we cannot
//   default-construct array members. We work around this by storing the
//   56 KV MemRef objects in raw char storage (KVCacheABI below) and
//   initialising them via placement new in allocateKVCache().
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/ModelSession.h"
#include "buddy/runtime/llm/KVCacheManager.h"

#include <cstring>
#include <dlfcn.h>
#include <new>
#include <stdexcept>
#include <string>

// std::launder is in <new> (C++17). Use it when accessing a MemRef object
// through a char* pointer obtained by reinterpret_cast.
#include <new> // for std::launder, placement new

namespace buddy {
namespace runtime {

//===----------------------------------------------------------------------===//
// KVCacheABI — contiguous memory block for the MLIR result struct
//
// MLIR-compiled _mlir_ciface_* functions write their outputs into a struct
// whose layout is:
//   [kv0..kv55 : MemRef<float,4> x 56][logits : MemRef<float,3>]
//
// Because MemRef<T,N>() is protected, we store these objects in char arrays
// and construct/destruct them explicitly via placement new.
//
// The pointer arithmetic uses sizeof(MemRef<T,N>) which equals the actual
// object size; the char arrays provide correct alignment via alignas.
//===----------------------------------------------------------------------===//

struct KVCacheABI {
  alignas(MemRef<float, 4>) char kv_[sizeof(MemRef<float, 4>) *
                                     BUDDY_DSR1_KV_LAYERS];
  alignas(MemRef<float, 3>) char logits_[sizeof(MemRef<float, 3>)];

  MemRef<float, 4> &kv(int i) {
    return *std::launder(reinterpret_cast<MemRef<float, 4> *>(
        kv_ + i * sizeof(MemRef<float, 4>)));
  }
  MemRef<float, 3> &logits() {
    return *std::launder(reinterpret_cast<MemRef<float, 3> *>(logits_));
  }
};

//===----------------------------------------------------------------------===//
// Function pointer types (private to this TU)
//===----------------------------------------------------------------------===//

using PrefillFn = void (*)(KVCacheABI *, MemRef<float, 1> *, Text<size_t, 2> *);

using KV4 = MemRef<float, 4> *;
using DecodeFn = void (*)(KVCacheABI *, MemRef<float, 1> *,
                          MemRef<long long, 2> *, MemRef<long long, 1> *, KV4,
                          KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4,
                          KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4,
                          KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4,
                          KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4,
                          KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4, KV4,
                          KV4);

//===----------------------------------------------------------------------===//
// ModelSession::Impl
//===----------------------------------------------------------------------===//

struct ModelSession::Impl {
  KVCacheABI abi;                                 // raw storage for kv + logits
  MemRef<float, 4> *kvPtrs[BUDDY_DSR1_KV_LAYERS]; // host-side pointer array

  bool abiInitialized = false; // placement-new has been called

  void *soHandle = nullptr;
  PrefillFn prefillFn = nullptr;
  DecodeFn decodeFn = nullptr;

  ~Impl() {
    // Explicitly destroy placement-new'd objects before releasing the .so.
    if (abiInitialized) {
      for (int i = 0; i < BUDDY_DSR1_KV_LAYERS; ++i)
        abi.kv(i).~MemRef<float, 4>();
      abi.logits().~MemRef<float, 3>();
    }
    if (soHandle) {
      dlclose(soHandle);
      soHandle = nullptr;
    }
  }

  /// dlopen the model shared library and resolve both entry symbols.
  void loadSo(const std::string &soPath) {
    soHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!soHandle)
      throw std::runtime_error("[BuddyRuntime] dlopen failed: " + soPath +
                               "\n  " + dlerror());

    prefillFn = reinterpret_cast<PrefillFn>(
        dlsym(soHandle, "_mlir_ciface_forward_prefill"));
    if (!prefillFn)
      throw std::runtime_error(
          "[BuddyRuntime] symbol not found: _mlir_ciface_forward_prefill\n  " +
          std::string(dlerror()));

    decodeFn = reinterpret_cast<DecodeFn>(
        dlsym(soHandle, "_mlir_ciface_forward_decode"));
    if (!decodeFn)
      throw std::runtime_error(
          "[BuddyRuntime] symbol not found: _mlir_ciface_forward_decode\n  " +
          std::string(dlerror()));
  }
};

//===----------------------------------------------------------------------===//
// ModelSession lifecycle
//===----------------------------------------------------------------------===//

ModelSession::ModelSession(const Config &cfg) : cfg_(cfg) {
  if (cfg_.modelSoPath.empty())
    throw std::runtime_error(
        "[BuddyRuntime] Config.modelSoPath must not be empty.\n"
        "  Pass --model-so <path/to/deepseek_r1_model.so>");
  allocateKVCache();
  impl_->loadSo(cfg_.modelSoPath);
}

ModelSession::~ModelSession() = default;

std::unique_ptr<ModelSession> ModelSession::create(const Config &cfg) {
  return std::unique_ptr<ModelSession>(new ModelSession(cfg));
}

void ModelSession::allocateKVCache() {
  impl_ = std::make_unique<Impl>();

  const uint64_t elemsPerLayer =
      (uint64_t)cfg_.headNum * cfg_.maxTokenLen * cfg_.hiddenSize;
  const uint64_t bytesPerLayer = elemsPerLayer * sizeof(float);

  // --- KV cache: one MemRef<float,4> per layer, constructed in-place ---
  intptr_t kvShape[4] = {1, cfg_.headNum, cfg_.maxTokenLen, cfg_.hiddenSize};

  for (int i = 0; i < cfg_.kvLayers; ++i) {
    BufferDesc desc;
    desc.name = "kv" + std::to_string(i);
    desc.role = BufferRole::State;
    desc.lifetime = BufferLifetime::Session;
    desc.bytes = bytesPerLayer;
    desc.id = (uint32_t)(100 + i);

    auto &bv = pool_.allocate(desc);

    // Placement-new: construct a MemRef<float,4> view into the BufferPool
    // buffer.
    new (&impl_->abi.kv(i))
        MemRef<float, 4>(reinterpret_cast<float *>(bv.data), kvShape);

    impl_->kvPtrs[i] = &impl_->abi.kv(i);
  }

  // --- Logits: prefill shape {1, maxTokenLen, vocabSize} ---
  const uint64_t logitsBytes =
      (uint64_t)cfg_.maxTokenLen * cfg_.vocabSize * sizeof(float);
  {
    BufferDesc desc;
    desc.name = "logits";
    desc.role = BufferRole::Output;
    desc.lifetime = BufferLifetime::Session;
    desc.bytes = logitsBytes;
    desc.id = 200;

    auto &bv = pool_.allocate(desc);
    intptr_t lshape[3] = {1, cfg_.maxTokenLen, cfg_.vocabSize};
    new (&impl_->abi.logits())
        MemRef<float, 3>(reinterpret_cast<float *>(bv.data), lshape);
  }

  impl_->abiInitialized = true;

  // --- Decode step inputs (small, reused every call) ---
  {
    intptr_t tshape[2] = {1, 1};
    decodeTokenInput_ = std::make_unique<MemRef<long long, 2>>(tshape);
    intptr_t pshape[1] = {1};
    cachePosition_ = std::make_unique<MemRef<long long, 1>>(pshape);
  }
}

//===----------------------------------------------------------------------===//
// Prefill
//===----------------------------------------------------------------------===//

void ModelSession::prefill(MemRef<float, 1> &weights, Text<size_t, 2> &tokens) {
  impl_->prefillFn(&impl_->abi, &weights, &tokens);

  int tokenCount = (int)tokens.getTokenCnt();
  position_ = tokenCount;
}

//===----------------------------------------------------------------------===//
// Decode
//===----------------------------------------------------------------------===//

void ModelSession::decode(MemRef<float, 1> &weights, int tokenId) {
  decodeTokenInput_->getData()[0] = (long long)tokenId;
  cachePosition_->getData()[0] = (long long)position_;

  auto &a = impl_->abi;
  impl_->decodeFn(
      &a, &weights, decodeTokenInput_.get(), cachePosition_.get(), &a.kv(0),
      &a.kv(1), &a.kv(2), &a.kv(3), &a.kv(4), &a.kv(5), &a.kv(6), &a.kv(7),
      &a.kv(8), &a.kv(9), &a.kv(10), &a.kv(11), &a.kv(12), &a.kv(13), &a.kv(14),
      &a.kv(15), &a.kv(16), &a.kv(17), &a.kv(18), &a.kv(19), &a.kv(20),
      &a.kv(21), &a.kv(22), &a.kv(23), &a.kv(24), &a.kv(25), &a.kv(26),
      &a.kv(27), &a.kv(28), &a.kv(29), &a.kv(30), &a.kv(31), &a.kv(32),
      &a.kv(33), &a.kv(34), &a.kv(35), &a.kv(36), &a.kv(37), &a.kv(38),
      &a.kv(39), &a.kv(40), &a.kv(41), &a.kv(42), &a.kv(43), &a.kv(44),
      &a.kv(45), &a.kv(46), &a.kv(47), &a.kv(48), &a.kv(49), &a.kv(50),
      &a.kv(51), &a.kv(52), &a.kv(53), &a.kv(54), &a.kv(55));

  position_ += 1;
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

void ModelSession::resetPosition() { position_ = 0; }

bool ModelSession::handleKVCacheOverflow(int keepTokenNum, float ropeTheta) {
  if (position_ < cfg_.maxTokenLen)
    return false;

  const int currentTokens = std::min(position_, cfg_.maxTokenLen);
  keepTokenNum = std::clamp(keepTokenNum, 0, currentTokens - 1);
  const int discardLen = std::max(1, (currentTokens - keepTokenNum) / 2);

  // Extract raw float* from KV cache MemRefs.
  float *rawPtrs[BUDDY_DSR1_KV_LAYERS];
  for (int i = 0; i < cfg_.kvLayers; ++i)
    rawPtrs[i] = impl_->abi.kv(i).getData();

  // Step 1: Discard tokens (memmove + memset).
  buddy::kvcache::discardKVCache(rawPtrs, cfg_.kvLayers, cfg_.headNum,
                                 cfg_.maxTokenLen, cfg_.hiddenSize,
                                 keepTokenNum, discardLen, currentTokens);

  // Step 2: Adjust RoPE on surviving tail (now relocated to keepTokenNum).
  auto inverseFreqs =
      buddy::kvcache::buildInverseRopeFreqs(ropeTheta, cfg_.hiddenSize);
  buddy::kvcache::adjustKeyCacheRope(
      rawPtrs, cfg_.kvLayers, cfg_.headNum, cfg_.maxTokenLen, cfg_.hiddenSize,
      keepTokenNum, discardLen, currentTokens, inverseFreqs);

  // Step 3: Update position.
  position_ = std::clamp(currentTokens - discardLen, 0, cfg_.maxTokenLen);
  return true;
}

const float *ModelSession::logitsData() const {
  return impl_->abi.logits().getData();
}

std::string ModelSession::loadedSoPath() const { return cfg_.modelSoPath; }

//===----------------------------------------------------------------------===//
// createFromRax: load config from a packed .rax manifest
//===----------------------------------------------------------------------===//

std::unique_ptr<ModelSession>
ModelSession::createFromRax(const std::string &raxPath,
                            ModelManifest &resolvedManifest) {
  resolvedManifest = ModelManifest::loadFromRax(raxPath);

  Config cfg;
  cfg.modelSoPath = resolvedManifest.soPath;
  // Shape constants remain at their compile-time defaults (BUDDY_DSR1_*
  // macros). Callers that need non-default shapes should fill cfg manually
  // instead.

  return create(cfg);
}

} // namespace runtime
} // namespace buddy
