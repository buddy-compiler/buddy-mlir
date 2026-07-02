//===- BgeM3Runner.cpp - BGE-M3 dense embedding runner -------------------===//
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

#include "buddy/runtime/models/BgeM3Runner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 512;
constexpr size_t kDefaultMaxPositionEmbeddings = 8194;
constexpr size_t kDefaultHiddenSize = 1024;

using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<int64_t, 1> *, MemRef<int64_t, 2> *,
                           MemRef<int64_t, 2> *);

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

std::string shellQuote(const std::string &s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

std::string readPipe(const std::string &cmd) {
  std::array<char, 4096> buffer{};
  std::string output;
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    throw std::runtime_error("BgeM3Runner: failed to run tokenizer helper");
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
    output += buffer.data();
  int status = pclose(pipe);
  if (status != 0)
    throw std::runtime_error("BgeM3Runner: tokenizer helper failed");
  return output;
}

std::vector<int64_t> parseI64Line(const std::string &line, size_t expected,
                                  const char *name) {
  std::istringstream iss(line);
  std::vector<int64_t> values;
  int64_t value = 0;
  while (iss >> value)
    values.push_back(value);
  if (values.size() != expected)
    throw std::runtime_error("BgeM3Runner: tokenizer produced " +
                             std::to_string(values.size()) + " " + name +
                             " values, expected " + std::to_string(expected));
  return values;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
tokenizeWithHelper(const std::string &helperPath, const std::string &modelDir,
                   const std::string &prompt, size_t maxSeqLen) {
  namespace fs = std::filesystem;
  fs::path tmp = fs::temp_directory_path() /
                 ("buddy_bge_m3_prompt_" + std::to_string(::getpid()) + ".txt");
  {
    std::ofstream os(tmp, std::ios::binary);
    if (!os)
      throw std::runtime_error("BgeM3Runner: cannot create temp prompt file: " +
                               tmp.string());
    os << prompt;
  }

  const std::string cmd = "python3 " + shellQuote(helperPath) +
                          " --model-dir " + shellQuote(modelDir) +
                          " --input-file " + shellQuote(tmp.string()) +
                          " --max-len " + std::to_string(maxSeqLen);
  std::string output;
  try {
    output = readPipe(cmd);
  } catch (...) {
    std::error_code ec;
    fs::remove(tmp, ec);
    throw;
  }
  std::error_code ec;
  fs::remove(tmp, ec);

  std::istringstream lines(output);
  std::string idsLine;
  std::string maskLine;
  if (!std::getline(lines, idsLine) || !std::getline(lines, maskLine))
    throw std::runtime_error("BgeM3Runner: tokenizer helper output is invalid");
  return {parseI64Line(idsLine, maxSeqLen, "input_id"),
          parseI64Line(maskLine, maxSeqLen, "attention_mask")};
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

void loadWeights(const std::string &weightsPath, MemRef<float, 1> &params) {
  std::ifstream paramFile(weightsPath, std::ios::in | std::ios::binary);
  if (!paramFile)
    throw std::runtime_error("BgeM3Runner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("BgeM3Runner: error reading weights: " +
                             weightsPath);
}

} // namespace

void BgeM3Runner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error("BgeM3Runner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error("BgeM3Runner: only single-prompt inference is "
                             "implemented for BGE-M3");

  std::string soPath;
  std::string weightsPath;
  std::string tokenizerPath;
  std::string tokenizerHelperPath;
  std::string modelDir;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t maxPositionEmbeddings = kDefaultMaxPositionEmbeddings;
  size_t hiddenSize = kDefaultHiddenSize;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    if (weightsPath.empty())
      throw std::runtime_error("BgeM3Runner: manifest has no weight file");
    tokenizerPath = manifest.vocabPath;
    auto tokIt = manifest.resolvedModuleAttrs.find("tokenizer_uri");
    if (tokIt != manifest.resolvedModuleAttrs.end())
      tokenizerPath = tokIt->second;
    auto helperIt = manifest.resolvedModuleAttrs.find("tokenizer_helper_uri");
    if (helperIt != manifest.resolvedModuleAttrs.end())
      tokenizerHelperPath = helperIt->second;
    std::string helperConstant = findConstantPath(manifest, "tokenizer_helper");
    if (!helperConstant.empty())
      tokenizerHelperPath = helperConstant;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    maxPositionEmbeddings = parseSizeAttr(manifest, "max_position_embeddings",
                                          maxPositionEmbeddings);
    hiddenSize = parseSizeAttr(manifest, "hidden_size", hiddenSize);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("BgeM3Runner: legacy mode requires --model-so, "
                               "--weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
    maxSeqLen = cfg.promptLength > 0 ? static_cast<size_t>(cfg.promptLength)
                                     : maxSeqLen;
  }

  if (tokenizerPath.empty())
    throw std::runtime_error("BgeM3Runner: tokenizer path is empty");
  if (tokenizerHelperPath.empty())
    tokenizerHelperPath =
        (fs::path(soPath).parent_path() / "bge_m3_tokenize.py").string();
  modelDir = fs::path(tokenizerPath).parent_path().string();

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so       : " + soPath, suppress);
  printLog("Weights         : " + weightsPath, suppress);
  printLog("Tokenizer dir   : " + modelDir, suppress);
  printLog("Tokenizer helper: " + tokenizerHelperPath, suppress);

  const auto tokenized =
      tokenizeWithHelper(tokenizerHelperPath, modelDir, prompt, maxSeqLen);
  printLog("Tokenization complete", suppress);

  printLog("Loading model shared library", suppress);
  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("BgeM3Runner: dlopen failed: " + soPath + ": " +
                             dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error("BgeM3Runner: missing _mlir_ciface_forward in " +
                             soPath + ": " + std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error("BgeM3Runner: weight file is not f32-aligned");
  }
  printLog("Loading weights", suppress);
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);
  printLog("Weights loaded", suppress);

  MemRef<int64_t, 2> inputIds({1, maxSeqLen});
  MemRef<int64_t, 2> attentionMask({1, maxSeqLen});
  MemRef<int64_t, 1> tokenTypeIds({maxPositionEmbeddings});
  std::copy(tokenized.first.begin(), tokenized.first.end(), inputIds.getData());
  std::copy(tokenized.second.begin(), tokenized.second.end(),
            attentionMask.getData());
  std::fill(tokenTypeIds.getData(),
            tokenTypeIds.getData() + tokenTypeIds.getSize(), 0);

  MemRef<float, 3> resultContainer[1] = {
      MemRef<float, 3>({1, maxSeqLen, hiddenSize}, false, 0),
  };

  const auto t0 = std::chrono::high_resolution_clock::now();
  printLog("Calling _mlir_ciface_forward", suppress);
  forward(resultContainer, &paramsContainer, &tokenTypeIds, &inputIds,
          &attentionMask);
  const auto t1 = std::chrono::high_resolution_clock::now();
  printLog("Forward complete", suppress);

  std::vector<float> embedding(hiddenSize);
  const float *cls = resultContainer[0].getData();
  float norm = 0.0f;
  for (size_t i = 0; i < hiddenSize; ++i) {
    embedding[i] = cls[i];
    norm += embedding[i] * embedding[i];
  }
  norm = std::sqrt(norm);
  if (norm > 0.0f) {
    for (float &v : embedding)
      v /= norm;
  }

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mBGE-M3 Dense Embedding\033[0m\n";
    std::cerr << "  dim: " << hiddenSize << "\n";
    std::cerr << "  time: " << seconds << "s\n";
  }

  std::cout << "[";
  for (size_t i = 0; i < embedding.size(); ++i) {
    if (i)
      std::cout << ", ";
    std::cout << embedding[i];
  }
  std::cout << "]\n";

  free(resultContainer[0].release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
