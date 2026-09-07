//===- buddy-server.cpp - Buddy model HTTP server -----------------------===//
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

#include "AudioTranscriptionModelPluginHandle.h"
#include "BackendSelection.h"
#include "EmbeddingModelPluginHandle.h"
#include "JsonCodec.h"
#include "MaskedLMModelPluginHandle.h"
#include "ResidentModelPluginHandle.h"
#include "SimpleHttpServer.h"

#include "buddy/runtime/core/AudioTranscriptionModel.h"
#include "buddy/runtime/core/AudioTranscriptionTypes.h"
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
#include <stdexcept>
#include <string>
#include <thread>

using buddy::runtime::AudioTranscriptionModel;
using buddy::runtime::AudioTranscriptionModelConfig;
using buddy::runtime::EmbeddingModel;
using buddy::runtime::EmbeddingModelConfig;
using buddy::runtime::MaskedLMModel;
using buddy::runtime::MaskedLMModelConfig;
using buddy::runtime::ModelLoadState;
using buddy::runtime::ModelStatus;
using buddy::runtime::ResidentModel;
using buddy::runtime::ResidentModelConfig;
using buddy::server::BackendKind;
using buddy::server::HttpRequest;
using buddy::server::ResponseWriter;
using buddy::server::SimpleHttpServer;

namespace {

enum ServerLoadState { Loading = 0, Ready = 1, Error = 2 };

void usage(const char *program, std::ostream &stream = std::cout) {
  stream << "Usage: " << program << " [options]\n\n"
         << "Model source (one required):\n"
         << "  --model <path.rax>       Model manifest (recommended)\n"
         << "  --model-so <path.so>     Model library (legacy mode)\n"
         << "  --weights <path>         Weights; repeatable in legacy mode\n"
         << "  --vocab <path>           Vocabulary (legacy mode)\n"
         << "  --model-type <name>      Model name override\n\n"
         << "Backend plugin (at most one explicit override):\n"
         << "  --serving-so <path.so>       Resident completion plugin\n"
         << "  --embedding-so <path.so>     Embedding plugin\n"
         << "  --masked-lm-so <path.so>     Masked-LM plugin\n"
         << "  --transcription-so <path.so> Audio transcription plugin\n\n"
         << "Server:\n"
         << "  --host <addr>            Bind address (default 127.0.0.1)\n"
         << "  --port <port>            Bind port (default 8080)\n"
         << "  --chat-template <path>   Chat template JSON\n"
         << "  --help / -h\n";
}

bool hasSuffix(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool isReady(const std::atomic<int> &state) {
  return state.load(std::memory_order_acquire) == Ready;
}

void sendError(ResponseWriter &writer, int status, const std::string &message,
               const std::string &type) {
  writer.sendResponse(buddy::server::jsonResponse(
      status, buddy::server::errorJson(message, type, status)));
}

template <typename Function>
void withJsonErrors(ResponseWriter &writer, Function function) {
  try {
    function();
  } catch (const buddy::server::JsonCodecError &error) {
    sendError(writer, 400, error.what(), "bad_request");
  } catch (const std::invalid_argument &error) {
    sendError(writer, 400, error.what(), "bad_request");
  } catch (const std::exception &error) {
    sendError(writer, 500, error.what(), "internal_error");
  }
}

bool requireReady(const std::atomic<int> &state, ResponseWriter &writer) {
  if (isReady(state))
    return true;
  sendError(writer, 503, "model is not loaded", "model_not_ready");
  return false;
}

void unsupported(ResponseWriter &writer, const char *endpoint) {
  sendError(writer, 400,
            std::string(endpoint) + " is not supported by the selected backend",
            "unsupported_endpoint");
}

EmbeddingModelConfig embeddingConfig(const ResidentModelConfig &config) {
  return {config.raxPath, config.modelSoPath, config.weightPaths,
          config.vocabPath, config.modelName};
}

MaskedLMModelConfig maskedLMConfig(const ResidentModelConfig &config) {
  MaskedLMModelConfig value;
  value.raxPath = config.raxPath;
  value.modelSoPath = config.modelSoPath;
  value.weightPaths = config.weightPaths;
  value.vocabPath = config.vocabPath;
  value.modelName = config.modelName;
  return value;
}

AudioTranscriptionModelConfig
transcriptionConfig(const ResidentModelConfig &config) {
  return {config.raxPath, config.modelSoPath, config.weightPaths,
          config.vocabPath, config.modelName};
}

void handleCompletion(ResidentModel &model, const std::atomic<int> &state,
                      const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseCompletionRequest(request.body);
    if (!decoded.stream) {
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toJson(model.complete(decoded.request))));
      return;
    }
    writer.startSse();
    (void)model.completeStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toCompletionChunkJson(chunk)))
            return false;
          return !chunk.done || writer.writeSseData("[DONE]");
        });
  });
}

void handleChat(ResidentModel &model, const std::atomic<int> &state,
                const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseChatCompletionRequest(request.body);
    if (!decoded.stream) {
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toOpenAIChatJson(model.chat(decoded.request))));
      return;
    }
    writer.startSse();
    (void)model.chatStream(
        decoded.request, [&](const buddy::runtime::CompletionChunk &chunk) {
          if (!writer.writeSseData(buddy::server::toOpenAIChatChunkJson(chunk)))
            return false;
          return !chunk.done || writer.writeSseData("[DONE]");
        });
  });
}

void handleTokenize(ResidentModel &model, const std::atomic<int> &state,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseTokenizeRequest(request.body);
    writer.sendResponse(buddy::server::jsonResponse(
        200,
        buddy::server::toJson(model.tokenize(decoded), decoded.countOnly)));
  });
}

void handleEmbedding(EmbeddingModel &model, const std::atomic<int> &state,
                     const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseEmbeddingRequest(request.body);
    const ModelStatus status = model.status();
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    writer.sendResponse(buddy::server::jsonResponse(
        200,
        buddy::server::toOpenAIEmbeddingJson(model.embed(decoded.request))));
  });
}

void handleMaskedLM(MaskedLMModel &model, const std::atomic<int> &state,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseMaskedLMRequest(request.body);
    const ModelStatus status = model.status();
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(model.predict(decoded.request))));
  });
}

void handleTranscription(AudioTranscriptionModel &model,
                         const std::atomic<int> &state,
                         const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto contentType = request.headers.find("content-type");
    if (contentType != request.headers.end() &&
        contentType->second.rfind("multipart/", 0) == 0)
      throw buddy::server::JsonCodecError(
          "multipart audio uploads are not supported");
    auto decoded = buddy::server::parseAudioTranscriptionRequest(request.body);
    const ModelStatus status = model.status();
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toOpenAIAudioTranscriptionJson(
                 model.transcribe(decoded.request))));
  });
}

} // namespace

int main(int argc, char **argv) {
  ResidentModelConfig modelConfig;
  buddy::server::BackendPluginPaths explicitPlugins;
  std::string modelType;
  std::string host = "127.0.0.1";
  int port = 8080;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc)
      modelConfig.raxPath = argv[++index];
    else if (argument == "--model-so" && index + 1 < argc)
      modelConfig.modelSoPath = argv[++index];
    else if (argument == "--weights" && index + 1 < argc)
      modelConfig.weightPaths.push_back(argv[++index]);
    else if (argument == "--vocab" && index + 1 < argc)
      modelConfig.vocabPath = argv[++index];
    else if (argument == "--model-type" && index + 1 < argc) {
      modelType = argv[++index];
      modelConfig.modelName = modelType;
    } else if (argument == "--serving-so" && index + 1 < argc)
      explicitPlugins.resident = argv[++index];
    else if (argument == "--embedding-so" && index + 1 < argc)
      explicitPlugins.embedding = argv[++index];
    else if (argument == "--masked-lm-so" && index + 1 < argc)
      explicitPlugins.maskedLM = argv[++index];
    else if (argument == "--transcription-so" && index + 1 < argc)
      explicitPlugins.transcription = argv[++index];
    else if (argument == "--chat-template" && index + 1 < argc)
      modelConfig.chatTemplatePath = argv[++index];
    else if (argument == "--host" && index + 1 < argc)
      host = argv[++index];
    else if (argument == "--port" && index + 1 < argc)
      port = std::stoi(argv[++index]);
    else if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argument << "\n";
      usage(argv[0], std::cerr);
      return 2;
    }
  }

  if (modelConfig.raxPath.empty() && modelConfig.modelSoPath.empty()) {
    std::cerr << "Provide --model <path.rax> or --model-so <path.so>.\n";
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() &&
      hasSuffix(modelConfig.modelSoPath, ".rax")) {
    std::cerr << "Use --model, not --model-so, for .rax packages.\n";
    return 2;
  }
  if (!modelConfig.modelSoPath.empty() && modelConfig.weightPaths.empty()) {
    std::cerr << "Legacy --model-so mode requires --weights.\n";
    return 2;
  }

  buddy::server::BackendPluginPaths manifestPlugins;
  if (!modelConfig.raxPath.empty()) {
    try {
      const auto manifest =
          buddy::runtime::ModelManifest::loadFromRax(modelConfig.raxPath);
      manifestPlugins = buddy::server::pluginPathsFromManifest(manifest);
      if (modelConfig.modelName.empty())
        modelConfig.modelName = manifest.modelName;
    } catch (const std::exception &error) {
      std::cerr << "buddy-server: failed to read model manifest: "
                << error.what() << "\n";
      return 1;
    }
  }

  buddy::server::BackendSelection selection;
  try {
    selection = buddy::server::selectBackend(explicitPlugins, manifestPlugins);
  } catch (const std::invalid_argument &error) {
    std::cerr << "buddy-server: " << error.what() << "\n";
    return 2;
  }

  std::unique_ptr<buddy::server::ResidentModelPluginHandle> residentPlugin;
  std::unique_ptr<buddy::server::EmbeddingModelPluginHandle> embeddingPlugin;
  std::unique_ptr<buddy::server::MaskedLMModelPluginHandle> maskedLMPlugin;
  std::unique_ptr<buddy::server::AudioTranscriptionModelPluginHandle>
      transcriptionPlugin;
  buddy::server::ResidentModelPluginHandle::ModelPtr residentModel(
      nullptr, [](ResidentModel *) {});
  buddy::server::EmbeddingModelPluginHandle::ModelPtr embeddingModel(
      nullptr, [](EmbeddingModel *) {});
  buddy::server::MaskedLMModelPluginHandle::ModelPtr maskedLMModel(
      nullptr, [](MaskedLMModel *) {});
  buddy::server::AudioTranscriptionModelPluginHandle::ModelPtr
      transcriptionModel(nullptr, [](AudioTranscriptionModel *) {});

  try {
    switch (selection.kind) {
    case BackendKind::Resident:
      residentPlugin =
          std::make_unique<buddy::server::ResidentModelPluginHandle>(
              selection.pluginPath);
      modelType = modelType.empty() ? residentPlugin->modelType() : modelType;
      residentModel = residentPlugin->createModel();
      break;
    case BackendKind::Embedding:
      embeddingPlugin =
          std::make_unique<buddy::server::EmbeddingModelPluginHandle>(
              selection.pluginPath);
      modelType = modelType.empty() ? embeddingPlugin->modelType() : modelType;
      embeddingModel = embeddingPlugin->createModel();
      break;
    case BackendKind::MaskedLM:
      maskedLMPlugin =
          std::make_unique<buddy::server::MaskedLMModelPluginHandle>(
              selection.pluginPath);
      modelType = modelType.empty() ? maskedLMPlugin->modelType() : modelType;
      maskedLMModel = maskedLMPlugin->createModel();
      break;
    case BackendKind::Transcription:
      transcriptionPlugin =
          std::make_unique<buddy::server::AudioTranscriptionModelPluginHandle>(
              selection.pluginPath);
      modelType =
          modelType.empty() ? transcriptionPlugin->modelType() : modelType;
      transcriptionModel = transcriptionPlugin->createModel();
      break;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }

  const EmbeddingModelConfig embeddingCfg = embeddingConfig(modelConfig);
  const MaskedLMModelConfig maskedCfg = maskedLMConfig(modelConfig);
  const AudioTranscriptionModelConfig transcriptionCfg =
      transcriptionConfig(modelConfig);

  std::atomic<int> loadState{Loading};
  std::mutex loadErrorMutex;
  std::string loadError;
  SimpleHttpServer server;

  server.get("/health", [&](const HttpRequest &, ResponseWriter &writer) {
    ModelStatus status;
    const int state = loadState.load(std::memory_order_acquire);
    if (state == Ready) {
      switch (selection.kind) {
      case BackendKind::Resident:
        status = residentModel->status();
        break;
      case BackendKind::Embedding:
        status = embeddingModel->status();
        break;
      case BackendKind::MaskedLM:
        status = maskedLMModel->status();
        break;
      case BackendKind::Transcription:
        status = transcriptionModel->status();
        break;
      }
    } else {
      status.state =
          state == Error ? ModelLoadState::Error : ModelLoadState::Loading;
      status.modelName =
          modelConfig.modelName.empty() ? modelType : modelConfig.modelName;
      status.backend = selection.kind == BackendKind::Resident ? "" : "cpu";
      if (state == Error) {
        std::lock_guard<std::mutex> lock(loadErrorMutex);
        status.message = loadError;
      } else {
        status.message = "model is loading";
      }
    }
    writer.sendResponse(
        buddy::server::jsonResponse(200, buddy::server::toJson(status)));
  });

  server.post("/completion",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.kind != BackendKind::Resident)
                  return unsupported(writer, "/completion");
                handleCompletion(*residentModel, loadState, request, writer);
              });
  server.post("/v1/chat/completions",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.kind != BackendKind::Resident)
                  return unsupported(writer, "/v1/chat/completions");
                handleChat(*residentModel, loadState, request, writer);
              });
  server.post("/tokenize",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.kind != BackendKind::Resident)
                  return unsupported(writer, "/tokenize");
                handleTokenize(*residentModel, loadState, request, writer);
              });

  auto embeddings = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.kind != BackendKind::Embedding)
      return unsupported(writer, "/v1/embeddings");
    handleEmbedding(*embeddingModel, loadState, request, writer);
  };
  server.post("/v1/embeddings", embeddings);
  server.post("/embeddings", embeddings);

  auto maskedLM = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.kind != BackendKind::MaskedLM)
      return unsupported(writer, "/v1/masked-lm");
    handleMaskedLM(*maskedLMModel, loadState, request, writer);
  };
  server.post("/v1/masked-lm", maskedLM);
  server.post("/masked-lm", maskedLM);

  auto transcription = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.kind != BackendKind::Transcription)
      return unsupported(writer, "/v1/audio/transcriptions");
    handleTranscription(*transcriptionModel, loadState, request, writer);
  };
  server.post("/v1/audio/transcriptions", transcription);
  server.post("/audio/transcriptions", transcription);

  std::thread loadThread([&] {
    try {
      std::cerr << "[buddy-server] loading "
                << buddy::server::backendKindName(selection.kind)
                << " model...\n";
      switch (selection.kind) {
      case BackendKind::Resident:
        residentModel->load(modelConfig);
        break;
      case BackendKind::Embedding:
        embeddingModel->load(embeddingCfg);
        break;
      case BackendKind::MaskedLM:
        maskedLMModel->load(maskedCfg);
        break;
      case BackendKind::Transcription:
        transcriptionModel->load(transcriptionCfg);
        break;
      }
      loadState.store(Ready, std::memory_order_release);
      std::cerr << "[buddy-server] model loaded\n";
    } catch (const std::exception &error) {
      {
        std::lock_guard<std::mutex> lock(loadErrorMutex);
        loadError = error.what();
      }
      loadState.store(Error, std::memory_order_release);
      std::cerr << "[buddy-server] failed to load model: " << error.what()
                << "\n";
    }
  });

  try {
    server.listen(host, port);
  } catch (const std::exception &error) {
    std::cerr << "[buddy-server] " << error.what() << "\n";
    if (loadThread.joinable())
      loadThread.join();
    return 1;
  }
  if (loadThread.joinable())
    loadThread.join();
  return 0;
}
