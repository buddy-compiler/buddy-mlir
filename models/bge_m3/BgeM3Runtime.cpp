//===- BgeM3Runtime.cpp - Reusable BGE-M3 inference runtime ---------------===//
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

#include "buddy/runtime/models/BgeM3Runtime.h"
#include "buddy/Core/Container.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/BgeM3Tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace buddy {
namespace runtime {

namespace {
using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<int64_t, 1> *, MemRef<int64_t, 2> *,
                           MemRef<int64_t, 2> *);

size_t parseSizeAttr(const ModelManifest &manifest, const char *key,
                     size_t fallback) {
  auto it = manifest.moduleAttrs.find(key);
  if (it == manifest.moduleAttrs.end() || it->second.empty())
    return fallback;
  try {
    return static_cast<size_t>(std::stoull(it->second));
  } catch (const std::exception &) {
    throw std::runtime_error(std::string("BgeM3Runtime: invalid manifest ") +
                             key + " attribute");
  }
}

std::string findConstantPath(const ModelManifest &manifest,
                             const std::string &name) {
  for (const auto &constant : manifest.constants)
    if (constant.name == name)
      return constant.path;
  return "";
}

void loadWeights(const std::string &path, MemRef<float, 1> &params) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("BgeM3Runtime: failed to open weights: " + path);
  input.read(reinterpret_cast<char *>(params.getData()),
             static_cast<std::streamsize>(sizeof(float) * params.getSize()));
  if (!input)
    throw std::runtime_error("BgeM3Runtime: error reading weights: " + path);
}

void closeHandle(void *&handle) {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
}
} // namespace

struct BgeM3Runtime::Impl {
  std::string name;
  size_t maxSeqLen = 512;
  size_t maxPositionEmbeddings = 8194;
  size_t hiddenSize = 1024;
  BgeM3Tokenizer tokenizer;
  std::vector<void *> dependentHandles;
  void *modelHandle = nullptr;
  ForwardFn forward = nullptr;
  std::unique_ptr<MemRef<float, 1>> params;
  std::unique_ptr<MemRef<int64_t, 2>> inputIds;
  std::unique_ptr<MemRef<int64_t, 2>> attentionMask;
  std::unique_ptr<MemRef<int64_t, 1>> tokenTypeIds;

  ~Impl() {
    closeHandle(modelHandle);
    for (auto it = dependentHandles.rbegin(); it != dependentHandles.rend();
         ++it)
      closeHandle(*it);
  }
};

BgeM3Runtime::~BgeM3Runtime() = default;

std::unique_ptr<BgeM3Runtime>
BgeM3Runtime::load(const std::string &modelSoPath,
                   const std::string &weightsPath, const std::string &vocabPath,
                   size_t maxSeqLen, size_t maxPositionEmbeddings,
                   size_t hiddenSize, const std::string &modelName,
                   const std::vector<std::string> &deps) {
  namespace fs = std::filesystem;
  if (modelSoPath.empty())
    throw std::runtime_error(
        "BgeM3Runtime: model shared library path is empty");
  if (weightsPath.empty())
    throw std::runtime_error("BgeM3Runtime: weights path is empty");
  if (vocabPath.empty())
    throw std::runtime_error("BgeM3Runtime: tokenizer path is empty");
  if (maxSeqLen < 2)
    throw std::runtime_error("BgeM3Runtime: max_seq_len must be >= 2");
  if (maxPositionEmbeddings == 0)
    throw std::runtime_error(
        "BgeM3Runtime: max_position_embeddings must be > 0");
  if (hiddenSize == 0)
    throw std::runtime_error("BgeM3Runtime: embedding dimension must be > 0");
  if (!fs::exists(modelSoPath))
    throw std::runtime_error("BgeM3Runtime: model shared library not found: " +
                             modelSoPath);
  if (!fs::exists(weightsPath))
    throw std::runtime_error("BgeM3Runtime: weights file not found: " +
                             weightsPath);

  auto runtime = std::unique_ptr<BgeM3Runtime>(new BgeM3Runtime());
  runtime->impl = std::make_unique<Impl>();
  Impl &impl = *runtime->impl;
  impl.name = modelName.empty() ? "bge_m3" : modelName;
  impl.maxSeqLen = maxSeqLen;
  impl.maxPositionEmbeddings = maxPositionEmbeddings;
  impl.hiddenSize = hiddenSize;
  impl.tokenizer = BgeM3Tokenizer::loadFromFile(vocabPath);

  try {
    for (const std::string &dep : deps) {
      if (dep.empty())
        continue;
      void *depHandle = dlopen(dep.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!depHandle)
        throw std::runtime_error("BgeM3Runtime: dlopen dependency failed: " +
                                 dep + ": " + dlerror());
      impl.dependentHandles.push_back(depHandle);
    }
    impl.modelHandle = dlopen(modelSoPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!impl.modelHandle)
      throw std::runtime_error("BgeM3Runtime: dlopen model failed: " +
                               modelSoPath + ": " + dlerror());
    dlerror();
    impl.forward = reinterpret_cast<ForwardFn>(
        dlsym(impl.modelHandle, "_mlir_ciface_forward"));
    if (const char *err = dlerror())
      throw std::runtime_error(
          "BgeM3Runtime: missing _mlir_ciface_forward in " + modelSoPath +
          ": " + err);

    const uintmax_t bytes = fs::file_size(weightsPath);
    if (bytes == 0 || bytes % sizeof(float) != 0)
      throw std::runtime_error("BgeM3Runtime: weight file is not f32-aligned");
    impl.params = std::make_unique<MemRef<float, 1>>(
        std::vector<size_t>{static_cast<size_t>(bytes / sizeof(float))});
    loadWeights(weightsPath, *impl.params);
    impl.inputIds = std::make_unique<MemRef<int64_t, 2>>(
        std::vector<size_t>{1, maxSeqLen}, int64_t(0));
    impl.attentionMask = std::make_unique<MemRef<int64_t, 2>>(
        std::vector<size_t>{1, maxSeqLen}, int64_t(0));
    impl.tokenTypeIds = std::make_unique<MemRef<int64_t, 1>>(
        std::vector<size_t>{maxPositionEmbeddings}, int64_t(0));
  } catch (...) {
    runtime.reset();
    throw;
  }
  return runtime;
}

std::unique_ptr<BgeM3Runtime>
BgeM3Runtime::loadFromRax(const std::string &raxPath) {
  if (raxPath.empty())
    throw std::runtime_error("BgeM3Runtime: .rax path is empty");
  ModelManifest manifest = ModelManifest::loadFromRax(raxPath);
  std::string weights = findConstantPath(manifest, "params");
  if (weights.empty() && !manifest.weightPaths.empty())
    weights = manifest.weightPaths.front();
  if (weights.empty())
    throw std::runtime_error("BgeM3Runtime: manifest has no weight file");
  const size_t maxSeqLen = parseSizeAttr(manifest, "max_seq_len", 512);
  const size_t maxPos =
      parseSizeAttr(manifest, "max_position_embeddings", 8194);
  const size_t hidden = parseSizeAttr(manifest, "hidden_size", 1024);
  return load(manifest.soPath, weights, manifest.vocabPath, maxSeqLen, maxPos,
              hidden,
              manifest.modelName.empty() ? "bge_m3" : manifest.modelName,
              manifest.dependentSoPaths);
}

std::unique_ptr<BgeM3Runtime>
BgeM3Runtime::loadLegacy(const std::string &modelSoPath,
                         const std::string &weightsPath,
                         const std::string &vocabPath, size_t maxSeqLen,
                         size_t maxPositionEmbeddings, size_t hiddenSize,
                         const std::string &modelName) {
  return load(modelSoPath, weightsPath, vocabPath, maxSeqLen,
              maxPositionEmbeddings, hiddenSize, modelName);
}

std::vector<float> BgeM3Runtime::embed(const std::string &text,
                                       size_t *tokenCountOut) {
  if (!impl)
    throw std::runtime_error("BgeM3Runtime: runtime is not loaded");
  if (text.empty())
    throw std::runtime_error("BgeM3Runtime: input text is empty");

  std::vector<int64_t> ids, mask;
  impl->tokenizer.encode(text, impl->maxSeqLen, ids, mask);
  if (ids.size() != impl->maxSeqLen || mask.size() != impl->maxSeqLen)
    throw std::runtime_error(
        "BgeM3Runtime: tokenizer returned invalid sequence length");
  if (tokenCountOut)
    *tokenCountOut =
        static_cast<size_t>(std::count(mask.begin(), mask.end(), int64_t(1)));
  std::copy(ids.begin(), ids.end(), impl->inputIds->getData());
  std::copy(mask.begin(), mask.end(), impl->attentionMask->getData());

  MemRef<float, 3> output(
      std::vector<size_t>{1, impl->maxSeqLen, impl->hiddenSize}, false, 0);
  impl->forward(&output, impl->params.get(), impl->tokenTypeIds.get(),
                impl->inputIds.get(), impl->attentionMask.get());
  float *data = output.getData();
  if (!data)
    throw std::runtime_error("BgeM3Runtime: forward returned no output");

  std::vector<float> embedding(impl->hiddenSize);
  float normSquared = 0.0f;
  for (size_t i = 0; i < impl->hiddenSize; ++i) {
    embedding[i] = data[i];
    normSquared += data[i] * data[i];
  }
  if (!(normSquared > 0.0f) || !std::isfinite(normSquared)) {
    free(output.release());
    throw std::runtime_error(
        "BgeM3Runtime: forward returned zero/invalid CLS vector");
  }
  const float invNorm = 1.0f / std::sqrt(normSquared);
  for (float &value : embedding)
    value *= invNorm;
  free(output.release());
  return embedding;
}

const std::string &BgeM3Runtime::modelName() const {
  static const std::string unloaded = "";
  return impl ? impl->name : unloaded;
}

size_t BgeM3Runtime::contextLength() const {
  return impl ? impl->maxSeqLen : 0;
}

size_t BgeM3Runtime::embeddingDimension() const {
  return impl ? impl->hiddenSize : 0;
}

size_t BgeM3Runtime::tokenCount(const std::string &text) const {
  if (!impl)
    throw std::runtime_error("BgeM3Runtime: runtime is not loaded");
  const size_t raw = impl->tokenizer.tokenCount(text);
  return std::min(raw, impl->maxSeqLen - 2) + 2;
}

} // namespace runtime
} // namespace buddy
