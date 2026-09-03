//===- ProteinGLMMaskedLMModel.cpp - ProteinGLM masked-LM model ----------===//
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

#include "buddy/runtime/models/ProteinGLMMaskedLMModel.h"
#include "buddy/Core/Container.h"
#include "buddy/runtime/core/ModelManifest.h"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace buddy {
namespace runtime {
namespace {
using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                           MemRef<int64_t, 2> *);
struct Tokenizer {
  std::vector<std::string> idToToken;
  std::unordered_map<std::string, int64_t> tokenToId;
  int64_t padId = 0, unkId = 35, eosId = 34, maskId = 28;
};
Tokenizer loadTokenizer(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error(
        "ProteinGLM masked-LM: failed to open tokenizer: " + path);
  Tokenizer t;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    int64_t id = static_cast<int64_t>(t.idToToken.size());
    t.tokenToId[line] = id;
    t.idToToken.push_back(line);
  }
  auto idOf = [&](const char *s, int64_t fallback) {
    auto it = t.tokenToId.find(s);
    return it == t.tokenToId.end() ? fallback : it->second;
  };
  t.padId = idOf("<pad>", t.padId);
  t.unkId = idOf("<unk>", t.unkId);
  t.eosId = idOf("<eos>", t.eosId);
  t.maskId = idOf("<mask>", t.maskId);
  return t;
}
std::vector<std::string> pieces(const std::string &s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    if (std::isspace(static_cast<unsigned char>(s[i]))) {
      ++i;
      continue;
    }
    if (s[i] == '<') {
      size_t e = s.find('>', i + 1);
      if (e != std::string::npos) {
        out.push_back(s.substr(i, e - i + 1));
        i = e + 1;
        continue;
      }
    }
    out.emplace_back(1, s[i++]);
  }
  return out;
}
size_t attrSize(const ModelManifest &m, const char *key, size_t fallback) {
  auto it = m.moduleAttrs.find(key);
  if (it == m.moduleAttrs.end() || it->second.empty())
    return fallback;
  return static_cast<size_t>(std::stoull(it->second));
}
std::string constantPath(const ModelManifest &m) {
  for (const auto &c : m.constants)
    if (c.name == "params")
      return c.path;
  return m.weightPaths.empty() ? std::string() : m.weightPaths.front();
}
} // namespace

struct ProteinGLMMaskedLMModel::Impl {
  mutable std::mutex mutex;
  ModelStatus status;
  MaskedLMModelConfig config;
  Tokenizer tokenizer;
  std::string soPath, weightsPath;
  size_t maxSeqLen = 1024, vocabSize = 128, defaultTopK = 5;
  void *soHandle = nullptr;
  ForwardFn forward = nullptr;
  std::unique_ptr<MemRef<float, 1>> params;
  ~Impl() {
    params.reset();
    if (soHandle)
      dlclose(soHandle);
  }
};

ProteinGLMMaskedLMModel::ProteinGLMMaskedLMModel() : impl(new Impl) {}
ProteinGLMMaskedLMModel::~ProteinGLMMaskedLMModel() = default;

void ProteinGLMMaskedLMModel::load(const MaskedLMModelConfig &cfg) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->params.reset();
  if (impl->soHandle) {
    dlclose(impl->soHandle);
    impl->soHandle = nullptr;
    impl->forward = nullptr;
  }
  impl->status = ModelStatus{};
  impl->status.state = ModelLoadState::Loading;
  impl->config = cfg;
  try {
    if (!cfg.raxPath.empty()) {
      auto m = ModelManifest::loadFromRax(cfg.raxPath);
      impl->soPath = m.soPath;
      impl->weightsPath = constantPath(m);
      impl->config.modelName =
          cfg.modelName.empty() ? m.modelName : cfg.modelName;
      if (impl->config.modelName.empty())
        impl->config.modelName = "proteinglm";
      impl->tokenizer =
          loadTokenizer(cfg.vocabPath.empty() ? m.vocabPath : cfg.vocabPath);
      impl->maxSeqLen = cfg.contextLength
                            ? cfg.contextLength
                            : attrSize(m, "max_seq_len", impl->maxSeqLen);
      impl->vocabSize = attrSize(m, "vocab_size", impl->vocabSize);
      impl->defaultTopK =
          cfg.topK ? cfg.topK : attrSize(m, "top_k", impl->defaultTopK);
    } else {
      if (cfg.modelSoPath.empty() || cfg.weightPaths.empty() ||
          cfg.vocabPath.empty())
        throw std::runtime_error(
            "masked-LM legacy mode requires model, weights and vocab");
      impl->soPath = cfg.modelSoPath;
      impl->weightsPath = cfg.weightPaths.front();
      impl->config.modelName =
          cfg.modelName.empty() ? "proteinglm" : cfg.modelName;
      impl->tokenizer = loadTokenizer(cfg.vocabPath);
      impl->defaultTopK = cfg.topK ? cfg.topK : 5;
      if (cfg.contextLength)
        impl->maxSeqLen = cfg.contextLength;
    }
    if (impl->soPath.empty() || impl->weightsPath.empty() ||
        impl->tokenizer.idToToken.empty())
      throw std::runtime_error("masked-LM model resources are incomplete");
    impl->soHandle = dlopen(impl->soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!impl->soHandle)
      throw std::runtime_error(std::string("masked-LM dlopen failed: ") +
                               dlerror());
    dlerror();
    impl->forward = reinterpret_cast<ForwardFn>(
        dlsym(impl->soHandle, "_mlir_ciface_forward"));
    if (const char *e = dlerror())
      throw std::runtime_error(
          std::string("masked-LM kernel symbol missing: ") + e);
    std::ifstream wf(impl->weightsPath, std::ios::binary | std::ios::ate);
    if (!wf)
      throw std::runtime_error("masked-LM weights cannot be opened");
    auto bytes = static_cast<size_t>(wf.tellg());
    if (bytes % sizeof(float))
      throw std::runtime_error("masked-LM weights are not f32-aligned");
    impl->params = std::make_unique<MemRef<float, 1>>(
        std::vector<size_t>{bytes / sizeof(float)});
    wf.seekg(0);
    wf.read(reinterpret_cast<char *>(impl->params->getData()), bytes);
    if (!wf)
      throw std::runtime_error("masked-LM weights read failed");
    impl->status.state = ModelLoadState::Ready;
    impl->status.modelName = impl->config.modelName;
    impl->status.backend = "cpu";
    impl->status.contextLength = static_cast<int>(impl->maxSeqLen);
    impl->status.message.clear();
  } catch (...) {
    impl->status.state = ModelLoadState::Error;
    try {
      throw;
    } catch (const std::exception &e) {
      impl->status.message = e.what();
    }
    throw;
  }
}

ModelStatus ProteinGLMMaskedLMModel::status() const {
  std::lock_guard<std::mutex> lock(impl->mutex);
  return impl->status;
}

MaskedLMResult ProteinGLMMaskedLMModel::predict(const MaskedLMRequest &req) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  if (impl->status.state != ModelLoadState::Ready)
    throw std::runtime_error("masked-LM model is not ready");
  if (req.input.empty())
    throw std::invalid_argument("input must not be empty");
  if (!req.model.empty() && req.model != impl->status.modelName)
    throw std::invalid_argument("model name does not match loaded model");
  size_t topK = req.topK ? req.topK : impl->defaultTopK;
  if (topK == 0)
    throw std::invalid_argument("top_k must be positive");
  topK = std::min(topK, impl->vocabSize);
  std::vector<int64_t> ids(impl->maxSeqLen, impl->tokenizer.padId);
  std::vector<size_t> masks;
  size_t real = 0;
  for (const auto &p : pieces(req.input)) {
    if (real + 1 >= impl->maxSeqLen)
      throw std::invalid_argument("input exceeds context length");
    auto it = impl->tokenizer.tokenToId.find(p);
    ids[real] = it == impl->tokenizer.tokenToId.end() ? impl->tokenizer.unkId
                                                      : it->second;
    if (ids[real] == impl->tokenizer.maskId)
      masks.push_back(real);
    ++real;
  }
  if (real < impl->maxSeqLen)
    ids[real++] = impl->tokenizer.eosId;
  if (masks.empty())
    throw std::invalid_argument("input must contain at least one <mask>");
  MemRef<int64_t, 2> input({1, impl->maxSeqLen}),
      attention({1, impl->maxSeqLen}), position({1, impl->maxSeqLen});
  std::copy(ids.begin(), ids.end(), input.getData());
  for (size_t i = 0; i < impl->maxSeqLen; ++i) {
    attention.getData()[i] = i < real;
    position.getData()[i] = static_cast<int64_t>(i);
  }
  MemRef<float, 3> output({1, impl->maxSeqLen, impl->vocabSize});
  impl->forward(&output, impl->params.get(), &input, &attention, &position);
  MaskedLMResult result;
  result.model = impl->status.modelName;
  result.sequenceLength = real;
  result.promptTokens = real;
  for (size_t pos : masks) {
    std::vector<size_t> order(impl->vocabSize);
    for (size_t i = 0; i < order.size(); ++i)
      order[i] = i;
    const float *row = output.getData() + pos * impl->vocabSize;
    std::partial_sort(order.begin(), order.begin() + topK, order.end(),
                      [&](size_t a, size_t b) { return row[a] > row[b]; });
    MaskedLMPrediction pred;
    pred.position = pos;
    for (size_t i = 0; i < topK; ++i) {
      size_t id = order[i];
      pred.tokens.push_back(MaskedLMToken{static_cast<int64_t>(id),
                                          id < impl->tokenizer.idToToken.size()
                                              ? impl->tokenizer.idToToken[id]
                                              : "?",
                                          row[id]});
    }
    result.predictions.push_back(std::move(pred));
  }
  return result;
}

} // namespace runtime
} // namespace buddy
