//===- BuddyServerApp.cpp - Buddy model HTTP server application ---------===//
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

#include "BuddyServerApp.h"

#include "AudioTranscriptionModelPluginHandle.h"
#include "EmbeddingModelPluginHandle.h"
#include "JsonCodec.h"
#include "MaskedLMModelPluginHandle.h"
#include "ModelPluginDiscovery.h"
#include "ResidentModelPluginHandle.h"
#include "SimpleHttpServer.h"

#include "buddy/runtime/core/AudioTranscriptionModel.h"
#include "buddy/runtime/core/AudioTranscriptionTypes.h"
#include "buddy/runtime/core/EmbeddingModel.h"
#include "buddy/runtime/core/EmbeddingTypes.h"
#include "buddy/runtime/core/MaskedLMModel.h"
#include "buddy/runtime/core/MaskedLMTypes.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"
#include "buddy/runtime/core/ResidentModel.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using buddy::runtime::AudioTranscriptionModel;
using buddy::runtime::AudioTranscriptionModelConfig;
using buddy::runtime::EmbeddingModel;
using buddy::runtime::EmbeddingModelConfig;
using buddy::runtime::MaskedLMModel;
using buddy::runtime::MaskedLMModelConfig;
using buddy::runtime::ModelLoadState;
using buddy::runtime::ModelServiceKind;
using buddy::runtime::ModelStatus;
using buddy::runtime::ResidentModel;
using buddy::runtime::ResidentModelConfig;
using buddy::server::HttpRequest;
using buddy::server::ResponseWriter;
using buddy::server::SimpleHttpServer;

namespace {

enum ServerLoadState { Loading = 0, Ready = 1, Error = 2 };

bool isReady(const std::atomic<int> &state) {
  return state.load(std::memory_order_acquire) == Ready;
}

void sendError(ResponseWriter &writer, int status, const std::string &message,
               const std::string &type) {
  const auto body = buddy::server::errorJson(message, type, status);
  if (writer.isSse())
    writer.writeSseData(body);
  else
    writer.sendResponse(buddy::server::jsonResponse(status, body));
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

bool requireResidentReady(ResidentModel &model, const std::atomic<int> &state,
                          ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return false;
  if (model.status().state == ModelLoadState::Ready)
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

struct ResponseIdentity {
  std::string id;
  std::string model;
  int64_t created;
};

ResponseIdentity responseIdentity(const ResidentModel &model, bool chat) {
  static std::atomic<uint64_t> nextId{0};
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  return {std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(micros) +
              "-" + std::to_string(nextId++),
          model.status().modelName,
          std::chrono::duration_cast<std::chrono::seconds>(now).count()};
}

// Start SSE only when the backend has completed preflight and reports its first
// event. A preflight failure can still return a normal HTTP error response.
template <typename Generate>
void streamGeneration(ResponseWriter &writer, const ResponseIdentity &identity,
                      bool chat, bool includeUsage, Generate generate) {
  bool connected = true;
  bool terminalReceived = false;
  bool eventReceived = false;
  buddy::runtime::CompletionChunk terminal;
  auto start = [&] {
    if (writer.hasStarted())
      return connected;
    connected = writer.startSse();
    if (connected && chat)
      connected = writer.writeSseData(buddy::server::toOpenAIChatRoleChunkJson(
          identity.id, identity.model, identity.created));
    return connected;
  };
  auto result = generate([&](const buddy::runtime::CompletionChunk &event) {
    if (!connected)
      return false;
    if (terminalReceived)
      return true;
    eventReceived = true;
    if (!start())
      return false;
    auto chunk = event;
    chunk.id = identity.id;
    chunk.model = identity.model;
    if (chunk.done) {
      terminalReceived = true;
      terminal = std::move(chunk);
      return true;
    }
    connected = writer.writeSseData(
        chat ? buddy::server::toOpenAIChatChunkJson(chunk, identity.created)
             : buddy::server::toCompletionChunkJson(chunk));
    return connected;
  });
  if (!connected ||
      result.finishReason == buddy::runtime::FinishReason::Cancelled)
    return;
  if (!start())
    return;
  result.id = identity.id;
  result.model = identity.model;
  if (!eventReceived)
    terminal.delta = result.content;
  terminal.id = identity.id;
  terminal.model = identity.model;
  terminal.done = true;
  terminal.finishReason = result.finishReason;
  terminal.usage = result.usage;
  terminal.timings = result.timings;
  if (!chat) {
    writer.writeSseData(buddy::server::toCompletionChunkJson(terminal));
    return;
  }
  // A backend may attach the last text to its terminal event. Preserve it in a
  // content chunk before sending the independent finish and usage events.
  if (!terminal.delta.empty()) {
    auto content = terminal;
    content.done = false;
    content.finishReason = buddy::runtime::FinishReason::None;
    if (!writer.writeSseData(
            buddy::server::toOpenAIChatChunkJson(content, identity.created)))
      return;
  }
  terminal.delta.clear();
  if (!writer.writeSseData(
          buddy::server::toOpenAIChatChunkJson(terminal, identity.created)))
    return;
  if (includeUsage &&
      !writer.writeSseData(
          buddy::server::toOpenAIChatUsageChunkJson(result, identity.created)))
    return;
  writer.writeSseData("[DONE]");
}

void handleCompletion(ResidentModel &model, const std::atomic<int> &state,
                      const HttpRequest &request, ResponseWriter &writer) {
  if (!requireResidentReady(model, state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseCompletionRequest(request.body);
    buddy::server::validateCapabilities(decoded, model.capabilities());
    const auto identity = responseIdentity(model, false);
    if (!decoded.stream) {
      auto result = model.complete(decoded.request);
      result.id = identity.id;
      result.model = identity.model;
      writer.sendResponse(
          buddy::server::jsonResponse(200, buddy::server::toJson(result)));
      return;
    }
    streamGeneration(writer, identity, false, false, [&](const auto &callback) {
      return model.completeStream(decoded.request, callback);
    });
  });
}

void handleChat(ResidentModel &model, const std::atomic<int> &state,
                const HttpRequest &request, ResponseWriter &writer) {
  if (!requireResidentReady(model, state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseChatCompletionRequest(request.body);
    buddy::server::validateCapabilities(decoded, model.capabilities());
    const auto identity = responseIdentity(model, true);
    if (!decoded.stream) {
      auto result = model.chat(decoded.request);
      result.id = identity.id;
      result.model = identity.model;
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toOpenAIChatJson(result, identity.created)));
      return;
    }
    streamGeneration(writer, identity, true, decoded.includeUsage,
                     [&](const auto &callback) {
                       return model.chatStream(decoded.request, callback);
                     });
  });
}

void handleTokenize(ResidentModel &model, const std::atomic<int> &state,
                    const HttpRequest &request, ResponseWriter &writer) {
  if (!requireResidentReady(model, state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseTokenizeRequest(request.body);
    buddy::server::validateCapabilities(decoded, model.capabilities());
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(model.tokenize(decoded), decoded.countOnly,
                                   decoded.withPieces)));
  });
}

void handleDetokenize(ResidentModel &model, const std::atomic<int> &state,
                      const HttpRequest &request, ResponseWriter &writer) {
  if (!requireResidentReady(model, state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseDetokenizeRequest(request.body);
    buddy::server::validateCapabilities(decoded, model.capabilities());
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(model.detokenize(decoded))));
  });
}

void handleEmbedding(EmbeddingModel &model, const std::atomic<int> &state,
                     const HttpRequest &request, ResponseWriter &writer) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    auto decoded = buddy::server::parseEmbeddingRequest(request.body);
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

buddy::server::BuddyServerApp::BuddyServerApp(BuddyServerOptions options)
    : options(std::move(options)) {}

int buddy::server::BuddyServerApp::run() {
  ResidentModelConfig modelConfig;
  modelConfig.raxPath = options.modelPath;
  modelConfig.chatTemplatePath = options.chatTemplatePath;
  std::string modelType = options.modelType;
  if (!modelType.empty())
    modelConfig.modelName = modelType;
  std::unique_ptr<buddy::runtime::RaxDependenciesHandle> raxDependencies;
  buddy::server::ModelPluginDiscoveryResult selection;
  try {
    const auto manifest =
        buddy::runtime::ModelManifest::loadFromRax(modelConfig.raxPath);
    if (manifest.modelLibraryPath.empty())
      throw std::runtime_error("model manifest is missing model_library");
    if (modelConfig.modelName.empty())
      modelConfig.modelName = manifest.modelName;
    raxDependencies = std::make_unique<buddy::runtime::RaxDependenciesHandle>(
        manifest.dependentSoPaths, "");
    selection = buddy::server::discoverModelPlugin(manifest.modelLibraryPath);
  } catch (const std::exception &error) {
    std::cerr << "buddy-server: " << error.what() << "\n";
    return 1;
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
    switch (selection.serviceKind) {
    case ModelServiceKind::Resident:
      residentPlugin =
          std::make_unique<buddy::server::ResidentModelPluginHandle>(
              selection.pluginPath);
      if (!modelConfig.modelName.empty() &&
          !residentPlugin->modelType().empty() &&
          modelConfig.modelName != residentPlugin->modelType() &&
          modelConfig.modelName.rfind(residentPlugin->modelType() + "_", 0) !=
              0) {
        std::cerr << "buddy-server: resident plugin type '"
                  << residentPlugin->modelType()
                  << "' does not match manifest model '"
                  << modelConfig.modelName << "'\n";
        return 2;
      }
      modelType = modelType.empty() ? residentPlugin->modelType() : modelType;
      residentModel = residentPlugin->createModel();
      break;
    case ModelServiceKind::Embedding:
      embeddingPlugin =
          std::make_unique<buddy::server::EmbeddingModelPluginHandle>(
              selection.pluginPath);
      modelType = modelType.empty() ? embeddingPlugin->modelType() : modelType;
      embeddingModel = embeddingPlugin->createModel();
      break;
    case ModelServiceKind::MaskedLM:
      maskedLMPlugin =
          std::make_unique<buddy::server::MaskedLMModelPluginHandle>(
              selection.pluginPath);
      modelType = modelType.empty() ? maskedLMPlugin->modelType() : modelType;
      maskedLMModel = maskedLMPlugin->createModel();
      break;
    case ModelServiceKind::Transcription:
      transcriptionPlugin =
          std::make_unique<buddy::server::AudioTranscriptionModelPluginHandle>(
              selection.pluginPath);
      modelType =
          modelType.empty() ? transcriptionPlugin->modelType() : modelType;
      transcriptionModel = transcriptionPlugin->createModel();
      break;
    case ModelServiceKind::Invalid:
      throw std::runtime_error("buddy-server: invalid model service kind");
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

  auto currentStatus = [&] {
    ModelStatus status;
    const int state = loadState.load(std::memory_order_acquire);
    if (state == Ready) {
      switch (selection.serviceKind) {
      case ModelServiceKind::Resident:
        status = residentModel->status();
        break;
      case ModelServiceKind::Embedding:
        status = embeddingModel->status();
        break;
      case ModelServiceKind::MaskedLM:
        status = maskedLMModel->status();
        break;
      case ModelServiceKind::Transcription:
        status = transcriptionModel->status();
        break;
      case ModelServiceKind::Invalid:
        throw std::runtime_error("invalid model service kind");
      }
    } else {
      status.state =
          state == Error ? ModelLoadState::Error : ModelLoadState::Loading;
      status.modelName =
          modelConfig.modelName.empty() ? modelType : modelConfig.modelName;
      status.backend =
          selection.serviceKind == ModelServiceKind::Resident ? "" : "cpu";
      if (state == Error) {
        std::lock_guard<std::mutex> lock(loadErrorMutex);
        status.message = loadError;
      } else {
        status.message = "model is loading";
      }
    }
    return status;
  };
  server.get("/health", [&](const HttpRequest &, ResponseWriter &writer) {
    const auto status = currentStatus();
    writer.sendResponse(buddy::server::jsonResponse(
        status.state == ModelLoadState::Ready ? 200 : 503,
        buddy::server::toJson(status)));
  });
  server.get("/v1/models", [&](const HttpRequest &, ResponseWriter &writer) {
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toModelListJson(currentStatus())));
  });

  server.post("/completion",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.serviceKind != ModelServiceKind::Resident)
                  return unsupported(writer, "/completion");
                handleCompletion(*residentModel, loadState, request, writer);
              });
  server.post("/v1/chat/completions",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.serviceKind != ModelServiceKind::Resident)
                  return unsupported(writer, "/v1/chat/completions");
                handleChat(*residentModel, loadState, request, writer);
              });
  server.post("/tokenize",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.serviceKind != ModelServiceKind::Resident)
                  return unsupported(writer, "/tokenize");
                handleTokenize(*residentModel, loadState, request, writer);
              });

  server.post("/detokenize",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                if (selection.serviceKind != ModelServiceKind::Resident)
                  return unsupported(writer, "/detokenize");
                handleDetokenize(*residentModel, loadState, request, writer);
              });

  auto embeddings = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.serviceKind != ModelServiceKind::Embedding)
      return unsupported(writer, "/v1/embeddings");
    handleEmbedding(*embeddingModel, loadState, request, writer);
  };
  server.post("/v1/embeddings", embeddings);
  server.post("/embeddings", embeddings);

  auto maskedLM = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.serviceKind != ModelServiceKind::MaskedLM)
      return unsupported(writer, "/v1/masked-lm");
    handleMaskedLM(*maskedLMModel, loadState, request, writer);
  };
  server.post("/v1/masked-lm", maskedLM);
  server.post("/masked-lm", maskedLM);

  auto transcription = [&](const HttpRequest &request, ResponseWriter &writer) {
    if (selection.serviceKind != ModelServiceKind::Transcription)
      return unsupported(writer, "/v1/audio/transcriptions");
    handleTranscription(*transcriptionModel, loadState, request, writer);
  };
  server.post("/v1/audio/transcriptions", transcription);
  server.post("/audio/transcriptions", transcription);

  std::thread loadThread([&] {
    try {
      std::cerr << "[buddy-server] loading "
                << buddy::server::modelServiceKindName(selection.serviceKind)
                << " model...\n";
      switch (selection.serviceKind) {
      case ModelServiceKind::Resident:
        residentModel->load(modelConfig);
        break;
      case ModelServiceKind::Embedding:
        embeddingModel->load(embeddingCfg);
        break;
      case ModelServiceKind::MaskedLM:
        maskedLMModel->load(maskedCfg);
        break;
      case ModelServiceKind::Transcription:
        transcriptionModel->load(transcriptionCfg);
        break;
      case ModelServiceKind::Invalid:
        throw std::runtime_error("invalid model service kind");
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
    server.listen(options.host, options.port);
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
