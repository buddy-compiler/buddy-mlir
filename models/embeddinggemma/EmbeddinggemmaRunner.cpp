//===- EmbeddinggemmaRunner.cpp - embeddinggemma embedding runner ---------===//
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
// google/embeddinggemma-300m sentence embedding runner.
//
// Loads the compiled `embeddinggemma_model.so` and `arg0.data` weights through
// the `.rax` manifest (or explicit paths), tokenizes the input text with the
// Gemma byte-level BPE tokenizer (EmbeddinggemmaTokenizer.h, a pure-C++
// reimplementation of tokenizer.json), then invokes the AOT forward graph:
//
//   forward(weights: memref<params_size x f32>,
//           input_ids: memref<batch_size x max_seq_len x i64>,
//           attention_mask: memref<batch_size x max_seq_len x i64>)
//     -> (embedding: memref<batch_size x hidden_size x f32>)
//
// The L2-normalized 768-dim embedding (the SentenceTransformer output after
// Gemma3TextModel -> mean pooling -> two dense layers -> L2 normalize) is
// printed together with a short summary.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/EmbeddinggemmaRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/EmbeddinggemmaTokenizer.h"

#include "buddy/Core/Container.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 128;
constexpr size_t kDefaultHiddenSize = 768;

/// Single-result ABI: the wrapper takes one pointer to the output embedding
/// descriptor first, then one pointer per input descriptor.
using ForwardFn = void (*)(MemRef<float, 2> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<int64_t, 2> *);

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

size_t parseSizeAttr(const ModelManifest &manifest, const char *key,
                     size_t fallback) {
  auto it = manifest.moduleAttrs.find(key);
  if (it == manifest.moduleAttrs.end() || it->second.empty())
    return fallback;
  return static_cast<size_t>(std::stoull(it->second));
}

std::string findConstantPath(const ModelManifest &manifest,
                             const std::string &name) {
  for (const auto &constant : manifest.constants) {
    if (constant.name == name)
      return constant.path;
  }
  return "";
}

llvm::sys::fs::mapped_file_region mapWeights(const std::string &path) {
  namespace fs = llvm::sys::fs;
  // A nonblocking open lets us reject a FIFO without waiting for a writer.
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    const std::error_code error(errno, std::generic_category());
    throw std::runtime_error("EmbeddinggemmaRunner: failed to open weights " +
                             path + ": " + error.message());
  }
  llvm::scope_exit closeFile([fd] { close(fd); });

  fs::file_status status;
  if (auto error = fs::status(fd, status))
    throw std::runtime_error("EmbeddinggemmaRunner: failed to stat weights " +
                             path + ": " + error.message());
  if (!fs::is_regular_file(status))
    throw std::runtime_error(
        "EmbeddinggemmaRunner: expected a regular weight file: " + path);
  const uint64_t size = status.getSize();
  if (size == 0 || size % sizeof(float) != 0 ||
      size > std::numeric_limits<size_t>::max())
    throw std::runtime_error(
        "EmbeddinggemmaRunner: invalid weight file size: " + path);

  std::error_code error;
  fs::mapped_file_region mapping(fd, fs::mapped_file_region::readonly,
                                 static_cast<size_t>(size), 0, error);
  if (error)
    throw std::runtime_error("EmbeddinggemmaRunner: failed to map weights " +
                             path + ": " + error.message());
  return mapping;
}

class MappedWeights : public MemRef<float, 1> {
public:
  explicit MappedWeights(const std::string &path) : mapping(mapWeights(path)) {
    allocated = aligned = const_cast<float *>(
        reinterpret_cast<const float *>(mapping.const_data()));
    sizes[0] = mapping.size() / sizeof(float);
    setStrides();
  }

  ~MappedWeights() {
    // The mapping owns the storage. Keep MemRef from freeing it.
    allocated = nullptr;
  }

private:
  llvm::sys::fs::mapped_file_region mapping;
};

} // namespace

void EmbeddinggemmaRunner::run(const RunConfig &cfg) {
  const bool suppress = cfg.suppressStats;

  std::string soPath;
  std::string weightsPath;
  std::string tokenizerPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t hiddenSize = kDefaultHiddenSize;
  size_t batchSize = 1;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error(
          "EmbeddinggemmaRunner: manifest has no weight file");
    tokenizerPath = manifest.vocabPath;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
    batchSize = parseSizeAttr(manifest, "batch_size", batchSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("EmbeddinggemmaRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
  }

  if (tokenizerPath.empty())
    throw std::runtime_error("EmbeddinggemmaRunner: tokenizer path is empty");

  if (batchSize == 0 || maxSeqLen == 0 || hiddenSize == 0)
    throw std::runtime_error("EmbeddinggemmaRunner: invalid tensor dimensions");
  if (cfg.batchSize < 0 ||
      (cfg.batchSize > 0 && static_cast<size_t>(cfg.batchSize) != batchSize))
    throw std::runtime_error(
        "EmbeddinggemmaRunner: batch size differs from compiled model");
  std::vector<std::string> prompts = cfg.prompts;
  if (prompts.empty())
    prompts.assign(batchSize, cfg.prompt);
  if (prompts.size() != batchSize)
    throw std::runtime_error(
        "EmbeddinggemmaRunner: prompt count differs from compiled batch size");

  printLog("Model .so  : " + soPath, suppress);
  printLog("Weights    : " + weightsPath, suppress);
  printLog("Tokenizer  : " + tokenizerPath, suppress);

  EmbeddinggemmaTokenizer tokenizer =
      EmbeddinggemmaTokenizer::loadFromFile(tokenizerPath);
  MemRef<int64_t, 2> inputIds({batchSize, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({batchSize, maxSeqLen});
  size_t validTokens = 0;
  for (size_t row = 0; row < batchSize; ++row) {
    std::vector<int64_t> inputIdVec, attentionMaskVec;
    tokenizer.encode(prompts[row], maxSeqLen, inputIdVec, attentionMaskVec);
    validTokens +=
        std::count(attentionMaskVec.begin(), attentionMaskVec.end(), 1);
    std::copy(inputIdVec.begin(), inputIdVec.end(),
              inputIds.getData() + row * maxSeqLen);
    std::copy(attentionMaskVec.begin(), attentionMaskVec.end(),
              attentionMask.getData() + row * maxSeqLen);
  }

  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("EmbeddinggemmaRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  llvm::scope_exit closeModel([handle] { dlclose(handle); });
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror())
    throw std::runtime_error(
        "EmbeddinggemmaRunner: missing _mlir_ciface_forward in " + soPath +
        ": " + std::string(err));

  MappedWeights paramsContainer(weightsPath);
  MemRef<float, 2> embedding({batchSize, hiddenSize}, false, 0);

  const auto t0 = std::chrono::high_resolution_clock::now();
  forward(&embedding, &paramsContainer, &inputIds, &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();

  const float *data = embedding.getData();
  double norm2 = 0.0;
  for (size_t d = 0; d < hiddenSize; ++d)
    norm2 += static_cast<double>(data[d]) * static_cast<double>(data[d]);

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mEmbeddingGemma Sentence Embedding\033[0m\n";
    std::cerr << "  seq_len : " << maxSeqLen << "\n";
    std::cerr << "  batch_size : " << batchSize << "\n";
    std::cerr << "  valid_tokens : " << validTokens << "\n";
    std::cerr << "  dim     : " << hiddenSize << "\n";
    std::cerr << "  time    : " << seconds << "s\n";
    std::cerr << "  |emb|   : " << std::sqrt(norm2) << " (L2-normalized)\n";
  }

  if (batchSize > 1)
    std::cout << "[";
  for (size_t row = 0; row < batchSize; ++row) {
    if (row)
      std::cout << ", ";
    std::cout << "[";
    for (size_t d = 0; d < hiddenSize; ++d) {
      if (d)
        std::cout << ", ";
      std::cout << data[row * hiddenSize + d];
    }
    std::cout << "]";
  }
  if (batchSize > 1)
    std::cout << "]";
  std::cout << "\n";

  free(embedding.release());
}

} // namespace runtime
} // namespace buddy
