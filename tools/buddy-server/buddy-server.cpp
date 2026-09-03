//===- buddy-server.cpp - Buddy model HTTP server -------------------------===//
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

#include "EmbeddingModelPluginHandle.h"
#include "JsonCodec.h"
#include "MaskedLMModelPluginHandle.h"
#include "ResidentModelPluginHandle.h"
#include "SimpleHttpServer.h"

#include "buddy/runtime/core/EmbeddingModel.h"
#include "buddy/runtime/core/EmbeddingTypes.h"
#include "buddy/runtime/core/MaskedLMModel.h"
#include "buddy/runtime/core/MaskedLMTypes.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/ResidentModel.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using buddy::runtime::EmbeddingModel;
using buddy::runtime::EmbeddingModelConfig;
using buddy::runtime::MaskedLMModel;
using buddy::runtime::MaskedLMModelConfig;
using buddy::runtime::ModelLoadState;
using buddy::runtime::ModelStatus;
using buddy::runtime::ResidentModel;
using buddy::runtime::ResidentModelConfig;
using buddy::runtime::TokenizeRequest;
using buddy::server::HttpRequest;
using buddy::server::ResponseWriter;
using buddy::server::SimpleHttpServer;

namespace {

enum ServerLoadState { Loading = 0, Ready = 1, Error = 2 };

void usage(const char *prog, std::ostream &os = std::cout) {
  os << "Usage: " << prog << " [options]\n\n"
     << "Model source (one required):\n"
     << "  --model      <path.rax>  Model manifest (recommended)\n"
     << "  --model-so   <path.so>   Model shared library (legacy mode)\n"
     << "  --weights    <path>      Weights file; repeatable in legacy mode\n"
     << "  --vocab      <path>      Vocabulary file (legacy mode)\n"
     << "  --model-type <name>      Model type/name override for status\n"
     << "  --serving-so <path.so>   Resident model plugin shared library\n"
     << "  --embedding-so <path.so> Embedding model plugin shared library\n\n"
     << "  --masked-lm-so <path.so> Masked-LM plugin shared library\n\n"
     << "Server:\n"
     << "  --host       <addr>      Bind address (default 127.0.0.1)\n"
     << "  --port       <port>      Bind port (default 8080)\n\n"
     << "Chat:\n"
     << "  --chat-template <path>   Path to chat template JSON config\n\n"
     << "Other:\n"
     << "  --help / -h\n";
}

bool hasSuffix(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool isReady(const std::atomic<int> &loadState) {
  return loadState.load(std::memory_order_acquire) == Ready;
}

void sendError(ResponseWriter &writer, int status, const std::string &message,
               const std::string &type) {
  writer.sendResponse(buddy::server::jsonResponse(
      status, buddy::server::errorJson(message, type, status)));
}

template <typename Fn> void withJsonErrors(ResponseWriter &writer, Fn fn) {
  try {
    fn();
  } catch (const buddy::server::JsonCodecError &ex) {
    sendError(writer, 400, ex.what(), "bad_request");
  } catch (const std::invalid_argument &ex) {
    sendError(writer, 400, ex.what(), "bad_request");
  } catch (const std::exception &ex) {
    sendError(writer, 500, ex.what(), "internal_error");
  }
}

void handleUnsupported(ResponseWriter &writer, const char *endpoint) {
  sendError(writer, 400,
            std::string(endpoint) + " is not supported by the selected backend",
            "unsupported_endpoint");
}

void handleMaskedLM(MaskedLMModel &model, const std::atomic<int> &loadState,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseMaskedLMRequest(request.body);
    const ModelStatus status = model.status();
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400,
                "model '" + decoded.request.model +
                    "' does not match loaded model '" + status.modelName + "'",
                "model_not_found");
      return;
    }
    auto result = model.predict(decoded.request);
    writer.sendResponse(
        buddy::server::jsonResponse(200, buddy::server::toJson(result)));
  });
}

void handleCompletion(ResidentModel &model, const std::atomic<int> &loadState,
                      const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseCompletionRequest(request.body);
    if (!decoded.stream) {
      auto result = model.complete(decoded.request);
      writer.sendResponse(
          buddy::server::jsonResponse(200, buddy::server::toJson(result)));
      return;
    }

    writer.startSse();
    auto result = model.completeStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toCompletionChunkJson(chunk)))
            return false;
          if (chunk.done)
            return writer.writeSseData("[DONE]");
          return true;
        });
    (void)result;
  });
}

void handleChat(ResidentModel &model, const std::atomic<int> &loadState,
                const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseChatCompletionRequest(request.body);
    if (!decoded.stream) {
      auto result = model.chat(decoded.request);
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toOpenAIChatJson(result)));
      return;
    }

    writer.startSse();
    auto result = model.chatStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toOpenAIChatChunkJson(chunk)))
            return false;
          if (chunk.done)
            return writer.writeSseData("[DONE]");
          return true;
        });
    (void)result;
  });
}

void handleTokenize(ResidentModel &model, const std::atomic<int> &loadState,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    TokenizeRequest tokenRequest =
        buddy::server::parseTokenizeRequest(request.body);
    auto result = model.tokenize(tokenRequest);
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(result, tokenRequest.countOnly)));
  });
}

void handleEmbedding(EmbeddingModel &model, const std::atomic<int> &loadState,
                     const HttpRequest &request, ResponseWriter &writer) {
  if (!isReady(loadState)) {
    sendError(writer, 503, "model is not loaded", "model_not_ready");
    return;
  }

  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseEmbeddingRequest(request.body);
    const ModelStatus status = model.status();
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400,
                "model '" + decoded.request.model +
                    "' does not match loaded model '" + status.modelName + "'",
                "model_not_found");
      return;
    }
    auto result = model.embed(decoded.request);
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toOpenAIEmbeddingJson(result)));
  });
}

} // namespace

int main(int argc, char **argv) {
  ResidentModelConfig modelConfig;
  std::string modelType;
  std::string servingSoPath;
  std::string embeddingSoPath;
  std::string maskedLMSoPath;
  std::string host = "127.0.0.1";
  int port = 8080;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc)
      modelConfig.raxPath = argv[++i];
    else if (arg == "--model-so" && i + 1 < argc)
      modelConfig.modelSoPath = argv[++i];
    else if (arg == "--weights" && i + 1 < argc)
      modelConfig.weightPaths.push_back(argv[++i]);
    else if (arg == "--vocab" && i + 1 < argc)
      modelConfig.vocabPath = argv[++i];
    else if (arg == "--model-type" && i + 1 < argc)
      modelType = argv[++i];
    else if (arg == "--serving-so" && i + 1 < argc)
      servingSoPath = argv[++i];
    else if (arg == "--embedding-so" && i + 1 < argc)
      embeddingSoPath = argv[++i];
    else if (arg == "--masked-lm-so" && i + 1 < argc)
      maskedLMSoPath = argv[++i];
    else if (arg == "--chat-template" && i + 1 < argc)
      modelConfig.chatTemplatePath = argv[++i];
    else if (arg == "--host" && i + 1 < argc)
      host = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      port = std::stoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0], std::cerr);
      return 2;
    }
  }

  if (modelConfig.raxPath.empty() && modelConfig.modelSoPath.empty()) {
    std::cerr << "Provide --model <path.rax> or --model-so <path.so>.\n";
    usage(argv[0], std::cerr);
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() &&
      hasSuffix(modelConfig.modelSoPath, ".rax")) {
    std::cerr << "--model-so expects a shared library. For .rax manifests, use "
                 "--model <path.rax>.\n";
    usage(argv[0], std::cerr);
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() && modelConfig.weightPaths.empty()) {
    std::cerr << "--model-so legacy mode requires at least one --weights "
                 "<path> argument.\n";
    usage(argv[0], std::cerr);
    return 2;
  }
  if ((!servingSoPath.empty() && !embeddingSoPath.empty()) ||
      (!servingSoPath.empty() && !maskedLMSoPath.empty()) ||
      (!embeddingSoPath.empty() && !maskedLMSoPath.empty())) {
    std::cerr
        << "buddy-server: choose at most one of --serving-so, --embedding-so, "
           "--masked-lm-so.\n";
    return 2;
  }

  bool embeddingMode = !embeddingSoPath.empty();
  bool maskedLMMode = !maskedLMSoPath.empty();
  if (!modelConfig.raxPath.empty()) {
    try {
      auto manifest =
          buddy::runtime::ModelManifest::loadFromRax(modelConfig.raxPath);
      if (servingSoPath.empty() && embeddingSoPath.empty() &&
          maskedLMSoPath.empty()) {
        const bool hasServing = !manifest.servingLibraryPath.empty();
        const bool hasEmbedding = !manifest.embeddingLibraryPath.empty();
        const bool hasMasked = !manifest.maskedLMLibraryPath.empty();
        const int backends = static_cast<int>(hasServing) +
                             static_cast<int>(hasEmbedding) +
                             static_cast<int>(hasMasked);
        if (backends != 1) {
          std::cerr << "buddy-server: manifest must provide exactly one of "
                       "serving_library, embedding_library, masked_lm_library; "
                       "pass an explicit plugin option.\n";
          return 1;
        }
        if (hasEmbedding) {
          embeddingSoPath = manifest.embeddingLibraryPath;
          embeddingMode = true;
        } else if (hasMasked) {
          maskedLMSoPath = manifest.maskedLMLibraryPath;
          maskedLMMode = true;
        } else {
          servingSoPath = manifest.servingLibraryPath;
        }
      }
    } catch (const std::exception &ex) {
      std::cerr << "buddy-server: failed to read model manifest: " << ex.what()
                << "\n";
      return 1;
    }
  }

  std::unique_ptr<buddy::server::ResidentModelPluginHandle> residentPlugin;
  std::unique_ptr<buddy::server::EmbeddingModelPluginHandle> embeddingPlugin;
  std::unique_ptr<buddy::server::MaskedLMModelPluginHandle> maskedLMPlugin;
  buddy::server::ResidentModelPluginHandle::ModelPtr residentModel(
      nullptr, [](ResidentModel *m) { delete m; });
  buddy::server::EmbeddingModelPluginHandle::ModelPtr embeddingModel(
      nullptr, [](EmbeddingModel *m) { delete m; });
  buddy::server::MaskedLMModelPluginHandle::ModelPtr maskedLMModel(
      nullptr, [](MaskedLMModel *m) { delete m; });

  if (maskedLMMode) {
    try {
      maskedLMPlugin =
          std::make_unique<buddy::server::MaskedLMModelPluginHandle>(
              maskedLMSoPath);
      if (modelType.empty()) {
        const std::string pluginType = maskedLMPlugin->modelType();
        modelType = pluginType.empty() ? "masked_lm" : pluginType;
      }
      maskedLMModel = maskedLMPlugin->createModel();
    } catch (const std::exception &ex) {
      std::cerr << ex.what() << "\n";
      return 1;
    }
  } else if (embeddingMode) {
    if (embeddingSoPath.empty()) {
      std::cerr << "buddy-server: no embedding plugin specified.\n";
      return 1;
    }
    try {
      embeddingPlugin =
          std::make_unique<buddy::server::EmbeddingModelPluginHandle>(
              embeddingSoPath);
      if (modelType.empty()) {
        const std::string pluginType = embeddingPlugin->modelType();
        modelType = pluginType.empty() ? "embedding" : pluginType;
      }
      embeddingModel = embeddingPlugin->createModel();
    } catch (const std::exception &ex) {
      std::cerr << ex.what() << "\n";
      return 1;
    }
  } else {
    if (servingSoPath.empty()) {
      std::cerr << "buddy-server: no resident serving plugin specified.\n";
      if (!modelConfig.raxPath.empty())
        std::cerr
            << "The .rax manifest has no serving_library; pass --serving-so "
               "<path.so> or rebuild the .rax with serving_library.\n";
      else
        std::cerr
            << "Pass --serving-so <path.so> for legacy --model-so mode.\n";
      return 1;
    }
    try {
      residentPlugin =
          std::make_unique<buddy::server::ResidentModelPluginHandle>(
              servingSoPath);
      if (modelType.empty()) {
        const std::string pluginType = residentPlugin->modelType();
        modelType = pluginType.empty() ? "plugin" : pluginType;
      }
      residentModel = residentPlugin->createModel();
    } catch (const std::exception &ex) {
      std::cerr << ex.what() << "\n";
      return 1;
    }
  }

  EmbeddingModelConfig embeddingConfig;
  embeddingConfig.raxPath = modelConfig.raxPath;
  embeddingConfig.modelSoPath = modelConfig.modelSoPath;
  embeddingConfig.weightPaths = modelConfig.weightPaths;
  embeddingConfig.vocabPath = modelConfig.vocabPath;
  embeddingConfig.modelName = modelConfig.modelName;
  MaskedLMModelConfig maskedConfig;
  maskedConfig.raxPath = modelConfig.raxPath;
  maskedConfig.modelSoPath = modelConfig.modelSoPath;
  maskedConfig.weightPaths = modelConfig.weightPaths;
  maskedConfig.vocabPath = modelConfig.vocabPath;
  maskedConfig.modelName = modelConfig.modelName;

  std::atomic<int> loadState{Loading};
  std::mutex loadErrorMutex;
  std::string loadError;

  SimpleHttpServer server;
  server.get("/health", [&](const HttpRequest &, ResponseWriter &writer) {
    const int state = loadState.load(std::memory_order_acquire);
    if (state == Ready) {
      ModelStatus status = maskedLMMode
                               ? maskedLMModel->status()
                               : (embeddingMode ? embeddingModel->status()
                                                : residentModel->status());
      writer.sendResponse(
          buddy::server::jsonResponse(200, buddy::server::toJson(status)));
      return;
    }

    ModelStatus status;
    status.modelName =
        modelConfig.modelName.empty() ? modelType : modelConfig.modelName;
    status.backend = maskedLMMode ? "cpu" : (embeddingMode ? "cpu" : "");
    if (state == Error) {
      status.state = ModelLoadState::Error;
      std::lock_guard<std::mutex> lock(loadErrorMutex);
      status.message = loadError;
    } else {
      status.state = ModelLoadState::Loading;
      status.message = "model is loading";
    }
    writer.sendResponse(
        buddy::server::jsonResponse(200, buddy::server::toJson(status)));
  });

  if (maskedLMMode) {
    server.post("/v1/masked-lm",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleMaskedLM(*maskedLMModel, loadState, request, writer);
                });
    server.post("/masked-lm",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleMaskedLM(*maskedLMModel, loadState, request, writer);
                });
    server.post("/completion",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/completion");
                });
    server.post("/v1/chat/completions",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/v1/chat/completions");
                });
    server.post("/tokenize", [&](const HttpRequest &, ResponseWriter &writer) {
      handleUnsupported(writer, "/tokenize");
    });
    server.post("/v1/embeddings",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/v1/embeddings");
                });
    server.post("/embeddings",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/embeddings");
                });
  } else if (embeddingMode) {
    server.post("/v1/embeddings",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleEmbedding(*embeddingModel, loadState, request, writer);
                });
    server.post("/embeddings",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleEmbedding(*embeddingModel, loadState, request, writer);
                });
    server.post("/completion",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/completion");
                });
    server.post("/v1/chat/completions",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/v1/chat/completions");
                });
    server.post("/tokenize", [&](const HttpRequest &, ResponseWriter &writer) {
      handleUnsupported(writer, "/tokenize");
    });
  } else {
    server.post("/completion",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleCompletion(*residentModel, loadState, request, writer);
                });
    server.post("/v1/chat/completions",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleChat(*residentModel, loadState, request, writer);
                });
    server.post("/tokenize",
                [&](const HttpRequest &request, ResponseWriter &writer) {
                  handleTokenize(*residentModel, loadState, request, writer);
                });
    server.post("/v1/embeddings",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/v1/embeddings");
                });
    server.post("/embeddings",
                [&](const HttpRequest &, ResponseWriter &writer) {
                  handleUnsupported(writer, "/embeddings");
                });
  }

  std::thread loadThread;
  if (maskedLMMode) {
    loadThread = std::thread([&maskedLMModel, &maskedConfig, &loadState,
                              &loadError, &loadErrorMutex] {
      try {
        std::cerr << "[buddy-server] loading masked-LM model...\n";
        maskedLMModel->load(maskedConfig);
        loadState.store(Ready, std::memory_order_release);
        std::cerr << "[buddy-server] masked-LM model loaded\n";
      } catch (const std::exception &ex) {
        {
          std::lock_guard<std::mutex> lock(loadErrorMutex);
          loadError = ex.what();
        }
        loadState.store(Error, std::memory_order_release);
        std::cerr << "[buddy-server] failed to load masked-LM model: "
                  << ex.what() << "\n";
      }
    });
  } else if (embeddingMode) {
    loadThread = std::thread([&embeddingModel, &embeddingConfig, &loadState,
                              &loadError, &loadErrorMutex] {
      try {
        std::cerr << "[buddy-server] loading embedding model...\n";
        embeddingModel->load(embeddingConfig);
        loadState.store(Ready, std::memory_order_release);
        std::cerr << "[buddy-server] embedding model loaded\n";
      } catch (const std::exception &ex) {
        {
          std::lock_guard<std::mutex> lock(loadErrorMutex);
          loadError = ex.what();
        }
        loadState.store(Error, std::memory_order_release);
        std::cerr << "[buddy-server] failed to load embedding model: "
                  << ex.what() << "\n";
      }
    });
  } else {
    loadThread = std::thread([&residentModel, &modelConfig, &loadState,
                              &loadError, &loadErrorMutex] {
      try {
        std::cerr << "[buddy-server] loading model...\n";
        residentModel->load(modelConfig);
        loadState.store(Ready, std::memory_order_release);
        std::cerr << "[buddy-server] model loaded\n";
      } catch (const std::exception &ex) {
        {
          std::lock_guard<std::mutex> lock(loadErrorMutex);
          loadError = ex.what();
        }
        loadState.store(Error, std::memory_order_release);
        std::cerr << "[buddy-server] failed to load model: " << ex.what()
                  << "\n";
      }
    });
  }

  try {
    server.listen(host, port);
  } catch (const std::exception &ex) {
    std::cerr << "[buddy-server] " << ex.what() << "\n";
    if (loadThread.joinable())
      loadThread.join();
    return 1;
  }
  if (loadThread.joinable())
    loadThread.join();
  return 0;
}
