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

#include "JsonCodec.h"
#include "ServerModelPluginHandle.h"
#include "ServerModelRequest.h"
#include "SimpleHttpServer.h"

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/core/RaxDependenciesHandle.h"
#include "buddy/runtime/core/ServerModelABI.h"
#include "buddy/runtime/core/ServingTypes.h"

#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using buddy::runtime::AudioTranscriptionResult;
using buddy::runtime::CompletionChunk;
using buddy::runtime::CompletionResult;
using buddy::runtime::EmbeddingResult;
using buddy::runtime::FinishReason;
using buddy::runtime::MaskedLMResult;
using buddy::runtime::ModelLoadState;
using buddy::runtime::ModelStatus;
using buddy::runtime::TokenizeResult;
using buddy::server::HttpRequest;
using buddy::server::ResponseWriter;
using buddy::server::ServerModelError;
using buddy::server::ServerModelPluginHandle;
using buddy::server::ServerModelRequest;
using buddy::server::ServerModelStatus;
using buddy::server::SimpleHttpServer;

namespace {

enum ServerLoadState { Loading = 0, Ready = 1, Error = 2 };

bool isReady(const std::atomic<int> &state) {
  return state.load(std::memory_order_acquire) == Ready;
}

std::string copyView(BuddyServerStringViewV1 value) {
  if (!value.data || value.size == 0)
    return {};
  return {value.data, static_cast<size_t>(value.size)};
}

ModelStatus statusFromPlugin(const ServerModelStatus &source) {
  ModelStatus result;
  result.state = static_cast<ModelLoadState>(source.state);
  result.modelName = source.modelName;
  result.backend = source.backend;
  result.message = source.message;
  result.contextLength = source.contextLength;
  return result;
}

void sendError(ResponseWriter &writer, int status, const std::string &message,
               const std::string &type) {
  const std::string body = buddy::server::errorJson(message, type, status);
  if (writer.isSse()) {
    (void)writer.writeSseEvent("error", body);
    return;
  }
  if (!writer.hasStarted())
    (void)writer.sendResponse(buddy::server::jsonResponse(status, body));
}

struct HttpErrorMapping {
  int status;
  const char *type;
};

HttpErrorMapping mapServerModelError(BuddyServerStatusCodeV1 status) {
  switch (status) {
  case BUDDY_SERVER_STATUS_INVALID_ARGUMENT:
    return {400, "bad_request"};
  case BUDDY_SERVER_STATUS_NOT_SUPPORTED:
    return {400, "unsupported_endpoint"};
  case BUDDY_SERVER_STATUS_MODEL_NOT_READY:
    return {503, "model_not_ready"};
  case BUDDY_SERVER_STATUS_RESOURCE_EXHAUSTED:
    return {429, "resource_exhausted"};
  case BUDDY_SERVER_STATUS_CANCELLED:
    return {499, "cancelled"};
  case BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED:
    return {504, "deadline_exceeded"};
  case BUDDY_SERVER_STATUS_MODEL_ERROR:
    return {500, "model_error"};
  case BUDDY_SERVER_STATUS_INTERNAL_ERROR:
  case BUDDY_SERVER_STATUS_OK:
  default:
    return {500, "internal_error"};
  }
}

template <typename Function>
void withJsonErrors(ResponseWriter &writer, Function function) {
  try {
    function();
  } catch (const buddy::server::JsonCodecError &error) {
    sendError(writer, 400, error.what(), "bad_request");
  } catch (const std::invalid_argument &error) {
    sendError(writer, 400, error.what(), "bad_request");
  } catch (const ServerModelError &error) {
    const HttpErrorMapping mapping = mapServerModelError(error.status());
    sendError(writer, mapping.status, error.what(), mapping.type);
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

uint64_t operationBit(uint32_t operation) { return uint64_t{1} << operation; }

bool supports(const ServerModelPluginHandle &model, uint32_t operation) {
  return (model.operationMask() & operationBit(operation)) != 0;
}

bool requireStreamingCapability(const ServerModelPluginHandle &model,
                                bool streaming, ResponseWriter &writer) {
  if (!streaming ||
      (model.capabilityMask() & BUDDY_SERVER_CAPABILITY_STREAMING) != 0)
    return true;
  sendError(writer, 400, "streaming is not supported by the selected backend",
            "unsupported_endpoint");
  return false;
}

void unsupported(ResponseWriter &writer, const char *endpoint) {
  sendError(writer, 400,
            std::string(endpoint) + " is not supported by the selected backend",
            "unsupported_endpoint");
}

struct LoadArgsStorage {
  LoadArgsStorage(const buddy::runtime::ModelManifest &manifest,
                  const std::string &raxPath, const std::string &modelName,
                  const std::string &chatTemplatePath)
      : manifestPath(raxPath), modelName(modelName) {
    weightViews.reserve(manifest.weightPaths.size());
    for (const auto &path : manifest.weightPaths)
      weightViews.push_back({path.data(), static_cast<uint64_t>(path.size())});

    attributeValues.reserve(manifest.moduleAttrs.size());
    for (const auto &attribute : manifest.moduleAttrs)
      attributeValues.emplace_back(attribute.first, attribute.second);
    attributeViews.reserve(attributeValues.size());
    for (const auto &attribute : attributeValues) {
      attributeViews.push_back(
          {{attribute.first.data(),
            static_cast<uint64_t>(attribute.first.size())},
           {attribute.second.data(),
            static_cast<uint64_t>(attribute.second.size())}});
    }

    args.structSize = sizeof(args);
    args.raxPath = view(manifestPath);
    args.modelName = view(modelName);
    args.modelSoPath = view(manifest.soPath);
    args.weightPaths = weightViews.data();
    args.weightPathCount = weightViews.size();
    args.vocabPath = view(manifest.vocabPath);
    args.chatTemplatePath = view(chatTemplatePath);
    args.attributes = attributeViews.data();
    args.attributeCount = attributeViews.size();
  }

  const BuddyServerLoadArgsV1 &get() const { return args; }

private:
  static BuddyServerStringViewV1 view(const std::string &value) {
    return {value.data(), static_cast<uint64_t>(value.size())};
  }

  std::string manifestPath;
  const std::string &modelName;
  BuddyServerLoadArgsV1 args{};
  std::vector<BuddyServerStringViewV1> weightViews;
  std::vector<std::pair<std::string, std::string>> attributeValues;
  std::vector<BuddyServerStringPairV1> attributeViews;
};

struct CompletionEmitContext {
  ResponseWriter *writer = nullptr;
  bool chat = false;
  CompletionResult result;
  bool gotResult = false;
};

CompletionChunk
completionChunkFromAbi(const BuddyServerCompletionChunkV1 &source) {
  CompletionChunk result;
  result.id = copyView(source.id);
  result.model = copyView(source.model);
  result.delta = copyView(source.delta);
  result.tokenId = source.tokenId;
  result.done = source.done != 0;
  result.finishReason = static_cast<FinishReason>(source.finishReason);
  result.usage = {source.usage.promptTokens, source.usage.completionTokens,
                  source.usage.totalTokens};
  result.timings = {source.timings.prefillMs, source.timings.decodeMs,
                    source.timings.tokensPerSecond};
  return result;
}

CompletionResult
completionResultFromAbi(const BuddyServerCompletionResultV1 &source) {
  CompletionResult result;
  result.id = copyView(source.id);
  result.model = copyView(source.model);
  result.content = copyView(source.content);
  result.finishReason = static_cast<FinishReason>(source.finishReason);
  result.usage = {source.usage.promptTokens, source.usage.completionTokens,
                  source.usage.totalTokens};
  result.timings = {source.timings.prefillMs, source.timings.decodeMs,
                    source.timings.tokensPerSecond};
  return result;
}

BuddyServerEmitResultV1 emitCompletion(void *userData,
                                       const BuddyServerEventV1 *event) {
  auto &context = *static_cast<CompletionEmitContext *>(userData);
  if (!event)
    return BUDDY_SERVER_EMIT_ERROR;
  if (event->kind == BUDDY_SERVER_EVENT_FINAL) {
    context.result = completionResultFromAbi(event->completion);
    context.gotResult = true;
    if (context.writer && !context.writer->writeSseData("[DONE]"))
      return BUDDY_SERVER_EMIT_CANCEL;
    return BUDDY_SERVER_EMIT_CONTINUE;
  }
  if (event->kind != BUDDY_SERVER_EVENT_CHUNK || !context.writer)
    return BUDDY_SERVER_EMIT_ERROR;
  const CompletionChunk chunk = completionChunkFromAbi(event->completionChunk);
  const std::string json = context.chat
                               ? buddy::server::toOpenAIChatChunkJson(chunk)
                               : buddy::server::toCompletionChunkJson(chunk);
  if (!context.writer->writeSseData(json))
    return BUDDY_SERVER_EMIT_CANCEL;
  return BUDDY_SERVER_EMIT_CONTINUE;
}

struct TokenizeEmitContext {
  TokenizeResult result;
  bool gotResult = false;
};

BuddyServerEmitResultV1 emitTokenize(void *userData,
                                     const BuddyServerEventV1 *event) {
  auto &context = *static_cast<TokenizeEmitContext *>(userData);
  if (!event || event->kind != BUDDY_SERVER_EVENT_FINAL)
    return BUDDY_SERVER_EMIT_ERROR;
  context.result.tokens.clear();
  if (event->tokenize.tokens && event->tokenize.tokenCount)
    context.result.tokens.assign(event->tokenize.tokens,
                                 event->tokenize.tokens +
                                     event->tokenize.tokenCount);
  context.result.count = static_cast<size_t>(event->tokenize.count);
  context.gotResult = true;
  return BUDDY_SERVER_EMIT_CONTINUE;
}

struct EmbeddingEmitContext {
  EmbeddingResult result;
  bool gotResult = false;
};

BuddyServerEmitResultV1 emitEmbedding(void *userData,
                                      const BuddyServerEventV1 *event) {
  auto &context = *static_cast<EmbeddingEmitContext *>(userData);
  if (!event || event->kind != BUDDY_SERVER_EVENT_FINAL)
    return BUDDY_SERVER_EMIT_ERROR;
  context.result.model = copyView(event->embedding.model);
  if (event->embedding.embedding && event->embedding.embeddingCount)
    context.result.embedding.assign(event->embedding.embedding,
                                    event->embedding.embedding +
                                        event->embedding.embeddingCount);
  context.result.promptTokens =
      static_cast<size_t>(event->embedding.promptTokens);
  context.result.totalTokens =
      static_cast<size_t>(event->embedding.totalTokens);
  context.gotResult = true;
  return BUDDY_SERVER_EMIT_CONTINUE;
}

struct MaskedLMEmitContext {
  MaskedLMResult result;
  bool gotResult = false;
};

BuddyServerEmitResultV1 emitMaskedLM(void *userData,
                                     const BuddyServerEventV1 *event) {
  auto &context = *static_cast<MaskedLMEmitContext *>(userData);
  if (!event || event->kind != BUDDY_SERVER_EVENT_FINAL)
    return BUDDY_SERVER_EMIT_ERROR;
  context.result.model = copyView(event->maskedLM.model);
  context.result.sequenceLength =
      static_cast<size_t>(event->maskedLM.sequenceLength);
  context.result.promptTokens =
      static_cast<size_t>(event->maskedLM.promptTokens);
  context.result.predictions.clear();
  for (uint64_t i = 0; i < event->maskedLM.predictionCount; ++i) {
    const auto &source = event->maskedLM.predictions[i];
    buddy::runtime::MaskedLMPrediction prediction;
    prediction.position = static_cast<size_t>(source.position);
    for (uint64_t j = 0; j < source.tokenCount; ++j) {
      const auto &token = source.tokens[j];
      prediction.tokens.push_back(
          {token.tokenId, copyView(token.token), token.score});
    }
    context.result.predictions.push_back(std::move(prediction));
  }
  context.gotResult = true;
  return BUDDY_SERVER_EMIT_CONTINUE;
}

struct TranscriptionEmitContext {
  AudioTranscriptionResult result;
  bool gotResult = false;
};

BuddyServerEmitResultV1 emitTranscription(void *userData,
                                          const BuddyServerEventV1 *event) {
  auto &context = *static_cast<TranscriptionEmitContext *>(userData);
  if (!event || event->kind != BUDDY_SERVER_EVENT_FINAL)
    return BUDDY_SERVER_EMIT_ERROR;
  context.result.model = copyView(event->transcription.model);
  context.result.text = copyView(event->transcription.text);
  context.result.generatedTokens =
      static_cast<size_t>(event->transcription.generatedTokens);
  context.result.timings = {event->transcription.timings.preprocessMs,
                            event->transcription.timings.inferenceMs,
                            event->transcription.timings.totalMs};
  context.gotResult = true;
  return BUDDY_SERVER_EMIT_CONTINUE;
}

BuddyServerCallV1 makeCall(ServerModelRequest &request, uint64_t requestId) {
  request.setRequestId(requestId);
  BuddyServerCallV1 call{};
  call.structSize = sizeof(call);
  call.requestId = requestId;
  call.request = &request.request();
  return call;
}

void handleCompletion(ServerModelPluginHandle &model,
                      const std::atomic<int> &state, const HttpRequest &request,
                      ResponseWriter &writer,
                      std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_COMPLETION)) {
      unsupported(writer, "/completion");
      return;
    }
    const auto decoded = buddy::server::parseCompletionRequest(request.body);
    if (!requireStreamingCapability(model, decoded.stream, writer))
      return;
    ServerModelRequest serverRequest;
    serverRequest.setCompletion(decoded.request, decoded.stream);
    CompletionEmitContext context{decoded.stream ? &writer : nullptr, false};
    if (decoded.stream)
      writer.startSse();
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitCompletion, &context);
    if (!decoded.stream) {
      if (!context.gotResult)
        throw std::runtime_error("server model returned no completion result");
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toJson(context.result)));
    }
  });
}

void handleChat(ServerModelPluginHandle &model, const std::atomic<int> &state,
                const HttpRequest &request, ResponseWriter &writer,
                std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_CHAT)) {
      unsupported(writer, "/v1/chat/completions");
      return;
    }
    const auto decoded =
        buddy::server::parseChatCompletionRequest(request.body);
    if (!requireStreamingCapability(model, decoded.stream, writer))
      return;
    ServerModelRequest serverRequest;
    serverRequest.setChat(decoded.request, decoded.stream);
    CompletionEmitContext context{decoded.stream ? &writer : nullptr, true};
    if (decoded.stream)
      writer.startSse();
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitCompletion, &context);
    if (!decoded.stream) {
      if (!context.gotResult)
        throw std::runtime_error("server model returned no chat result");
      writer.sendResponse(buddy::server::jsonResponse(
          200, buddy::server::toOpenAIChatJson(context.result)));
    }
  });
}

void handleTokenize(ServerModelPluginHandle &model,
                    const std::atomic<int> &state, const HttpRequest &request,
                    ResponseWriter &writer, std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_TOKENIZE)) {
      unsupported(writer, "/tokenize");
      return;
    }
    const auto decoded = buddy::server::parseTokenizeRequest(request.body);
    ServerModelRequest serverRequest;
    serverRequest.setTokenize(decoded);
    TokenizeEmitContext context;
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitTokenize, &context);
    if (!context.gotResult)
      throw std::runtime_error("server model returned no tokenize result");
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(context.result, decoded.countOnly)));
  });
}

void handleEmbedding(ServerModelPluginHandle &model,
                     const std::atomic<int> &state, const HttpRequest &request,
                     ResponseWriter &writer,
                     std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_EMBEDDING)) {
      unsupported(writer, "/v1/embeddings");
      return;
    }
    const auto decoded = buddy::server::parseEmbeddingRequest(request.body);
    const ModelStatus status = statusFromPlugin(model.status());
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    ServerModelRequest serverRequest;
    serverRequest.setEmbedding(decoded.request);
    EmbeddingEmitContext context;
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitEmbedding, &context);
    if (!context.gotResult)
      throw std::runtime_error("server model returned no embedding result");
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toOpenAIEmbeddingJson(context.result)));
  });
}

void handleMaskedLM(ServerModelPluginHandle &model,
                    const std::atomic<int> &state, const HttpRequest &request,
                    ResponseWriter &writer, std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_MASKED_LM)) {
      unsupported(writer, "/v1/masked-lm");
      return;
    }
    const auto decoded = buddy::server::parseMaskedLMRequest(request.body);
    const ModelStatus status = statusFromPlugin(model.status());
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    ServerModelRequest serverRequest;
    serverRequest.setMaskedLM(decoded.request);
    MaskedLMEmitContext context;
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitMaskedLM, &context);
    if (!context.gotResult)
      throw std::runtime_error("server model returned no masked-LM result");
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toJson(context.result)));
  });
}

void handleTranscription(ServerModelPluginHandle &model,
                         const std::atomic<int> &state,
                         const HttpRequest &request, ResponseWriter &writer,
                         std::atomic<uint64_t> &requestIds) {
  if (!requireReady(state, writer))
    return;
  withJsonErrors(writer, [&] {
    if (!supports(model, BUDDY_SERVER_OPERATION_TRANSCRIPTION)) {
      unsupported(writer, "/v1/audio/transcriptions");
      return;
    }
    const auto contentType = request.headers.find("content-type");
    if (contentType != request.headers.end() &&
        contentType->second.rfind("multipart/", 0) == 0)
      throw buddy::server::JsonCodecError(
          "multipart audio uploads are not supported");
    const auto decoded =
        buddy::server::parseAudioTranscriptionRequest(request.body);
    const ModelStatus status = statusFromPlugin(model.status());
    if (!decoded.request.model.empty() &&
        decoded.request.model != status.modelName) {
      sendError(writer, 400, "requested model does not match loaded model",
                "model_not_found");
      return;
    }
    ServerModelRequest serverRequest;
    serverRequest.setTranscription(decoded.request);
    TranscriptionEmitContext context;
    BuddyServerCallV1 call = makeCall(serverRequest, requestIds.fetch_add(1));
    model.invoke(call, &emitTranscription, &context);
    if (!context.gotResult)
      throw std::runtime_error("server model returned no transcription result");
    writer.sendResponse(buddy::server::jsonResponse(
        200, buddy::server::toOpenAIAudioTranscriptionJson(context.result)));
  });
}

} // namespace

buddy::server::BuddyServerApp::BuddyServerApp(BuddyServerOptions options)
    : options(std::move(options)) {}

int buddy::server::BuddyServerApp::run() {
  buddy::runtime::ModelManifest manifest;
  std::string modelName;
  std::unique_ptr<buddy::runtime::RaxDependenciesHandle> raxDependencies;
  std::unique_ptr<ServerModelPluginHandle> model;

  try {
    manifest = buddy::runtime::ModelManifest::loadFromRax(options.modelPath);
    if (manifest.modelLibraryPath.empty())
      throw std::runtime_error("model manifest is missing model_library");
    modelName =
        options.modelType.empty() ? manifest.modelName : options.modelType;
    raxDependencies = std::make_unique<buddy::runtime::RaxDependenciesHandle>(
        manifest.dependentSoPaths, "buddy-server");
    model =
        std::make_unique<ServerModelPluginHandle>(manifest.modelLibraryPath);
    if (modelName.empty())
      modelName = model->modelType();
    if (modelName.empty())
      modelName = "unknown";
    if (!model->modelType().empty() && modelName != model->modelType() &&
        modelName.rfind(model->modelType() + "_", 0) != 0) {
      throw std::runtime_error("server plugin type '" + model->modelType() +
                               "' does not match manifest model '" + modelName +
                               "'");
    }
    model->create(modelName);
  } catch (const std::exception &error) {
    std::cerr << "buddy-server: " << error.what() << "\n";
    return 1;
  }

  LoadArgsStorage loadArgs(manifest, options.modelPath, modelName,
                           options.chatTemplatePath);
  std::atomic<int> loadState{Loading};
  std::mutex loadErrorMutex;
  std::string loadError;
  std::atomic<uint64_t> requestIds{1};
  SimpleHttpServer server;

  server.get("/health", [&](const HttpRequest &, ResponseWriter &writer) {
    ModelStatus status;
    const int state = loadState.load(std::memory_order_acquire);
    if (state == Ready) {
      try {
        status = statusFromPlugin(model->status());
      } catch (const std::exception &error) {
        status.state = ModelLoadState::Error;
        status.modelName = modelName;
        status.message = error.what();
      }
    } else {
      status.state =
          state == Error ? ModelLoadState::Error : ModelLoadState::Loading;
      status.modelName = modelName;
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

  server.post(
      "/completion", [&](const HttpRequest &request, ResponseWriter &writer) {
        handleCompletion(*model, loadState, request, writer, requestIds);
      });
  server.post("/v1/chat/completions",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                handleChat(*model, loadState, request, writer, requestIds);
              });
  server.post("/tokenize",
              [&](const HttpRequest &request, ResponseWriter &writer) {
                handleTokenize(*model, loadState, request, writer, requestIds);
              });

  auto embeddings = [&](const HttpRequest &request, ResponseWriter &writer) {
    handleEmbedding(*model, loadState, request, writer, requestIds);
  };
  server.post("/v1/embeddings", embeddings);
  server.post("/embeddings", embeddings);

  auto maskedLM = [&](const HttpRequest &request, ResponseWriter &writer) {
    handleMaskedLM(*model, loadState, request, writer, requestIds);
  };
  server.post("/v1/masked-lm", maskedLM);
  server.post("/masked-lm", maskedLM);

  auto transcription = [&](const HttpRequest &request, ResponseWriter &writer) {
    handleTranscription(*model, loadState, request, writer, requestIds);
  };
  server.post("/v1/audio/transcriptions", transcription);
  server.post("/audio/transcriptions", transcription);

  std::thread loadThread([&] {
    try {
      std::cerr << "[buddy-server] loading "
                << buddy::server::serverServiceKindName(model->serviceKind())
                << " model...\n";
      model->load(loadArgs.get());
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
