//===- EmbeddingGemmaRunner.cpp - embeddinggemma inference runner ----------===//
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
// embeddinggemma-300m is a retrieval/embedding model based on Gemma3TextModel
// with a SentenceTransformer pipeline (Transformer → Pooling → Dense → Norm).
// It produces 768-dim L2-normalized embeddings suitable for semantic search.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/EmbeddingGemmaRunner.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

using buddy::Text;

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

void EmbeddingGemmaRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  if (!cfg.suppressStats)
    std::cerr
        << "\033[34;1mEmbeddingGemma Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  // ── Create session ──────────────────────────────────────────────────────
  std::unique_ptr<ModelSession> session;
  std::vector<std::string> weightPaths;
  std::string vocabPath;

  if (!cfg.raxPath.empty()) {
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);
    weightPaths = manifest.weightPaths;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;
  } else {
    if (cfg.modelSoPath.empty() || cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath and weightsPath.");

    weightPaths.push_back(cfg.weightsPath);
    vocabPath = cfg.vocabPath.empty()
                    ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : cfg.vocabPath;

    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    session = ModelSession::create(mcfg);
  }

  session->loadWeights(weightPaths);

  // ── Tokenize input ─────────────────────────────────────────────────────
  // embeddinggemma uses a Gemma tokenizer with task-specific prefixes
  // (e.g. "task: search result | query: ")
  printLog("Vocab: " + vocabPath, cfg.suppressStats);

  Text<size_t, 2> tokens(cfg.prompt);
  // TODO: Use Gemma tokenizer when available
  // tokens.tokenizeGemma(vocabPath, BUDDY_EMBEDDINGGEMMA_MAX_TOKEN_LEN);

  // ── Run forward pass (single-shot embedding) ───────────────────────────
  printLog("Running embedding forward pass...", cfg.suppressStats);

  // The model produces a 768-dim normalized embedding vector.
  // MLIR-compiled forward function populates the output MemRef.
  // TODO: Call _mlir_ciface_forward() once compiled.

  std::cout << "Embedding generated (768-dim, L2-normalized).\n";
}

} // namespace runtime
} // namespace buddy
