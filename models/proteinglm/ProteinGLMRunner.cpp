//===- ProteinGLMRunner.cpp - ProteinGLM MLM inference runner ------------===//
//
// Licensed under the Apache License, Version 2.0.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/ProteinGLMRunner.h"
#include "buddy/runtime/core/ModelManifest.h"

#include "buddy/Core/Container.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace buddy {
namespace runtime {

namespace {

constexpr size_t kDefaultMaxSeqLen = 1024;
constexpr size_t kDefaultVocabSize = 128;
constexpr size_t kDefaultTopK = 5;

using ForwardFn = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                           MemRef<int64_t, 2> *, MemRef<int64_t, 2> *,
                           MemRef<int64_t, 2> *);

struct Tokenizer {
  std::vector<std::string> idToToken;
  std::unordered_map<std::string, int64_t> tokenToId;
  int64_t padId = 0;
  int64_t unkId = 35;
  int64_t eosId = 34;
  int64_t maskId = 28;
};

void printLog(const std::string &msg, bool suppress) {
  if (!suppress)
    std::cerr << "\033[34;1m[Log] \033[0m" << msg << "\n";
}

std::string trimCR(std::string s) {
  if (!s.empty() && s.back() == '\r')
    s.pop_back();
  return s;
}

Tokenizer loadTokenizer(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("ProteinGLMRunner: failed to open tokenizer: " +
                             path);

  Tokenizer tok;
  tok.idToToken.clear();
  std::string line;
  while (std::getline(in, line)) {
    line = trimCR(line);
    int64_t id = static_cast<int64_t>(tok.idToToken.size());
    tok.tokenToId[line] = id;
    tok.idToToken.push_back(line);
  }
  auto idOf = [&](const std::string &token, int64_t fallback) {
    auto it = tok.tokenToId.find(token);
    return it == tok.tokenToId.end() ? fallback : it->second;
  };
  tok.padId = idOf("<pad>", tok.padId);
  tok.unkId = idOf("<unk>", tok.unkId);
  tok.eosId = idOf("<eos>", tok.eosId);
  tok.maskId = idOf("<mask>", tok.maskId);
  return tok;
}

std::vector<std::string> splitPrompt(const std::string &prompt) {
  std::vector<std::string> pieces;
  for (size_t i = 0; i < prompt.size();) {
    unsigned char c = static_cast<unsigned char>(prompt[i]);
    if (std::isspace(c)) {
      ++i;
      continue;
    }
    if (prompt[i] == '<') {
      size_t end = prompt.find('>', i + 1);
      if (end != std::string::npos) {
        pieces.push_back(prompt.substr(i, end - i + 1));
        i = end + 1;
        continue;
      }
    }
    pieces.emplace_back(1, prompt[i]);
    ++i;
  }
  return pieces;
}

std::vector<int64_t> encode(const Tokenizer &tok, const std::string &prompt,
                            size_t maxSeqLen,
                            std::vector<size_t> &maskPositions,
                            size_t &realTokens) {
  std::vector<int64_t> ids(maxSeqLen, tok.padId);
  size_t pos = 0;
  for (const std::string &piece : splitPrompt(prompt)) {
    if (pos + 1 >= maxSeqLen)
      break;
    auto it = tok.tokenToId.find(piece);
    ids[pos] = it == tok.tokenToId.end() ? tok.unkId : it->second;
    if (ids[pos] == tok.maskId)
      maskPositions.push_back(pos);
    ++pos;
  }
  if (pos < maxSeqLen)
    ids[pos++] = tok.eosId;
  realTokens = pos;
  return ids;
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
    throw std::runtime_error("ProteinGLMRunner: failed to open weights: " +
                             weightsPath);
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * params.getSize());
  if (!paramFile)
    throw std::runtime_error("ProteinGLMRunner: error reading weights: " +
                             weightsPath);
}

std::vector<size_t> topKIndices(const float *logits, size_t vocabSize,
                                size_t topK) {
  std::vector<size_t> idx(vocabSize);
  for (size_t i = 0; i < vocabSize; ++i)
    idx[i] = i;
  topK = std::min(topK, vocabSize);
  std::partial_sort(idx.begin(), idx.begin() + topK, idx.end(),
                    [&](size_t a, size_t b) { return logits[a] > logits[b]; });
  idx.resize(topK);
  return idx;
}

} // namespace

void ProteinGLMRunner::run(const RunConfig &cfg) {
  namespace fs = std::filesystem;
  const bool suppress = cfg.suppressStats;

  if (cfg.prompt.empty() && cfg.prompts.empty())
    throw std::runtime_error(
        "ProteinGLMRunner: pass --prompt or --prompt-file");
  if (cfg.prompts.size() > 1)
    throw std::runtime_error(
        "ProteinGLMRunner: only single-prompt inference is implemented");

  std::string soPath;
  std::string weightsPath;
  std::string tokenizerPath;
  size_t maxSeqLen = kDefaultMaxSeqLen;
  size_t vocabSize = kDefaultVocabSize;
  size_t topK = cfg.samplerConfig.topK > 0
                    ? static_cast<size_t>(cfg.samplerConfig.topK)
                    : kDefaultTopK;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest = ModelManifest::loadFromRax(cfg.raxPath);
    soPath = manifest.soPath;
    weightsPath = findConstantPath(manifest, "params");
    if (weightsPath.empty() && !manifest.weightPaths.empty())
      weightsPath = manifest.weightPaths.front();
    tokenizerPath = manifest.vocabPath;
    auto tokIt = manifest.resolvedModuleAttrs.find("tokenizer_uri");
    if (tokIt != manifest.resolvedModuleAttrs.end())
      tokenizerPath = tokIt->second;
    maxSeqLen = parseSizeAttr(manifest, "max_seq_len", maxSeqLen);
    vocabSize = parseSizeAttr(manifest, "vocab_size", vocabSize);
    topK = parseSizeAttr(manifest, "top_k", topK);
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty() ||
        cfg.vocabPath.empty())
      throw std::runtime_error("ProteinGLMRunner: legacy mode requires "
                               "--model-so, --weights, and --vocab");
    soPath = cfg.modelSoPath;
    weightsPath = cfg.weightsPath;
    tokenizerPath = cfg.vocabPath;
    if (cfg.promptLength > 0)
      maxSeqLen = static_cast<size_t>(cfg.promptLength);
  }

  if (soPath.empty() || weightsPath.empty() || tokenizerPath.empty())
    throw std::runtime_error("ProteinGLMRunner: missing model resource path");

  const std::string prompt =
      !cfg.prompts.empty() ? cfg.prompts.front() : cfg.prompt;

  printLog("Model .so : " + soPath, suppress);
  printLog("Weights   : " + weightsPath, suppress);
  printLog("Tokenizer : " + tokenizerPath, suppress);

  Tokenizer tokenizer = loadTokenizer(tokenizerPath);
  std::vector<size_t> maskPositions;
  size_t realTokens = 0;
  std::vector<int64_t> inputIds =
      encode(tokenizer, prompt, maxSeqLen, maskPositions, realTokens);
  if (maskPositions.empty() && realTokens > 0)
    maskPositions.push_back(realTokens - 1);

  void *handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("ProteinGLMRunner: dlopen failed: " + soPath +
                             ": " + dlerror());
  dlerror();
  auto forward =
      reinterpret_cast<ForwardFn>(dlsym(handle, "_mlir_ciface_forward"));
  if (const char *err = dlerror()) {
    dlclose(handle);
    throw std::runtime_error(
        "ProteinGLMRunner: missing _mlir_ciface_forward in " + soPath + ": " +
        std::string(err));
  }

  const auto weightBytes = fs::file_size(weightsPath);
  if (weightBytes % sizeof(float) != 0) {
    dlclose(handle);
    throw std::runtime_error(
        "ProteinGLMRunner: weight file is not f32-aligned");
  }
  MemRef<float, 1> paramsContainer({weightBytes / sizeof(float)});
  loadWeights(weightsPath, paramsContainer);

  MemRef<int64_t, 2> inputContainer({1, maxSeqLen});
  MemRef<int64_t, 2> maskContainer({1, maxSeqLen});
  MemRef<int64_t, 2> positionContainer({1, maxSeqLen});
  std::copy(inputIds.begin(), inputIds.end(), inputContainer.getData());
  for (size_t i = 0; i < maxSeqLen; ++i) {
    maskContainer.getData()[i] = i < realTokens ? 1 : 0;
    positionContainer.getData()[i] = static_cast<int64_t>(i);
  }

  MemRef<float, 3> resultContainer[1] = {
      MemRef<float, 3>({1, maxSeqLen, vocabSize}, false, 0),
  };

  const auto t0 = std::chrono::high_resolution_clock::now();
  forward(resultContainer, &paramsContainer, &inputContainer, &maskContainer,
          &positionContainer);
  const auto t1 = std::chrono::high_resolution_clock::now();

  if (!suppress) {
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "\033[33;1mProteinGLM MLM\033[0m\n";
    std::cerr << "  seq_len: " << maxSeqLen << "\n";
    std::cerr << "  vocab: " << vocabSize << "\n";
    std::cerr << "  time: " << seconds << "s\n";
  }

  const float *logits = resultContainer[0].getData();
  for (size_t pos : maskPositions) {
    std::cout << "position " << pos << ":\n";
    const float *row = logits + pos * vocabSize;
    for (size_t id : topKIndices(row, vocabSize, topK)) {
      std::string token =
          id < tokenizer.idToToken.size() ? tokenizer.idToToken[id] : "?";
      std::cout << "  " << token << " (" << id << "): " << row[id] << "\n";
    }
  }

  free(resultContainer[0].release());
  dlclose(handle);
}

} // namespace runtime
} // namespace buddy
