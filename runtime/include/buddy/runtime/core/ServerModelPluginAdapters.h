//===- ServerModelPluginAdapters.h - Typed model to server ABI adapters
//----===//
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

#ifndef BUDDY_RUNTIME_CORE_SERVERMODELPLUGINADAPTERS_H
#define BUDDY_RUNTIME_CORE_SERVERMODELPLUGINADAPTERS_H

#include "buddy/runtime/core/AudioTranscriptionModel.h"
#include "buddy/runtime/core/EmbeddingModel.h"
#include "buddy/runtime/core/MaskedLMModel.h"
#include "buddy/runtime/core/ModelError.h"
#include "buddy/runtime/core/ResidentModel.h"
#include "buddy/runtime/core/ServerModelABIConversion.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace buddy {
namespace runtime {
namespace server_abi {

inline void clearError(BuddyServerErrorV1 *error) {
  if (!error)
    return;
  error->code = BUDDY_SERVER_STATUS_OK;
  error->message = nullptr;
  error->messageSize = 0;
}

inline BuddyServerStatusCodeV1 setError(BuddyServerErrorV1 *error,
                                        BuddyServerStatusCodeV1 code,
                                        const std::string &message) {
  if (error) {
    thread_local std::string storage;
    storage = message;
    error->code = static_cast<uint32_t>(code);
    error->message = storage.c_str();
    error->messageSize = static_cast<uint64_t>(storage.size());
  }
  return code;
}

inline BuddyServerStatusCodeV1 statusCode(ModelErrorCode code) {
  switch (code) {
  case ModelErrorCode::InvalidArgument:
    return BUDDY_SERVER_STATUS_INVALID_ARGUMENT;
  case ModelErrorCode::NotSupported:
    return BUDDY_SERVER_STATUS_NOT_SUPPORTED;
  case ModelErrorCode::ModelNotReady:
    return BUDDY_SERVER_STATUS_MODEL_NOT_READY;
  case ModelErrorCode::ResourceExhausted:
    return BUDDY_SERVER_STATUS_RESOURCE_EXHAUSTED;
  case ModelErrorCode::Cancelled:
    return BUDDY_SERVER_STATUS_CANCELLED;
  case ModelErrorCode::DeadlineExceeded:
    return BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED;
  case ModelErrorCode::ModelError:
    return BUDDY_SERVER_STATUS_MODEL_ERROR;
  case ModelErrorCode::InternalError:
    return BUDDY_SERVER_STATUS_INTERNAL_ERROR;
  }
  return BUDDY_SERVER_STATUS_INTERNAL_ERROR;
}

// This helper must be called from a catch block so it can rethrow and classify
// the active exception without relying on unstable message text.
inline BuddyServerStatusCodeV1
setExceptionError(BuddyServerErrorV1 *error,
                  BuddyServerStatusCodeV1 fallbackCode,
                  const char *fallbackMessage) {
  try {
    throw;
  } catch (const ModelException &ex) {
    return setError(error, statusCode(ex.code()), ex.what());
  } catch (const std::invalid_argument &ex) {
    return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT, ex.what());
  } catch (const std::bad_alloc &ex) {
    return setError(error, BUDDY_SERVER_STATUS_RESOURCE_EXHAUSTED, ex.what());
  } catch (const std::exception &ex) {
    return setError(error, fallbackCode, ex.what());
  } catch (...) {
    return setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR, fallbackMessage);
  }
}

inline std::vector<std::string> paths(const BuddyServerStringViewV1 *values,
                                      uint64_t count) {
  if (!values && count != 0)
    throw std::invalid_argument("server ABI path array is null");
  std::vector<std::string> result;
  result.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i)
    result.push_back(copy(values[i]));
  return result;
}

inline ResidentModelConfig residentConfig(const BuddyServerLoadArgsV1 &args) {
  ResidentModelConfig result;
  result.raxPath = copy(args.raxPath);
  result.modelName = copy(args.modelName);
  result.modelSoPath = copy(args.modelSoPath);
  result.weightPaths = paths(args.weightPaths, args.weightPathCount);
  result.vocabPath = copy(args.vocabPath);
  result.chatTemplatePath = copy(args.chatTemplatePath);
  return result;
}

inline EmbeddingModelConfig embeddingConfig(const BuddyServerLoadArgsV1 &args) {
  EmbeddingModelConfig result;
  result.raxPath = copy(args.raxPath);
  result.modelName = copy(args.modelName);
  result.modelSoPath = copy(args.modelSoPath);
  result.weightPaths = paths(args.weightPaths, args.weightPathCount);
  result.vocabPath = copy(args.vocabPath);
  return result;
}

inline MaskedLMModelConfig maskedLMConfig(const BuddyServerLoadArgsV1 &args) {
  MaskedLMModelConfig result;
  result.raxPath = copy(args.raxPath);
  result.modelName = copy(args.modelName);
  result.modelSoPath = copy(args.modelSoPath);
  result.weightPaths = paths(args.weightPaths, args.weightPathCount);
  result.vocabPath = copy(args.vocabPath);
  return result;
}

inline AudioTranscriptionModelConfig
transcriptionConfig(const BuddyServerLoadArgsV1 &args) {
  AudioTranscriptionModelConfig result;
  result.raxPath = copy(args.raxPath);
  result.modelName = copy(args.modelName);
  result.modelSoPath = copy(args.modelSoPath);
  result.weightPaths = paths(args.weightPaths, args.weightPathCount);
  result.vocabPath = copy(args.vocabPath);
  return result;
}

inline void fillStatus(const ModelStatus &source, BuddyServerStatusV1 *target) {
  if (!target)
    return;
  target->structSize = sizeof(*target);
  target->state = static_cast<uint32_t>(source.state);
  target->modelName = view(source.modelName);
  target->backend = view(source.backend);
  target->message = view(source.message);
  target->contextLength = source.contextLength;
}

inline uint32_t finishReason(FinishReason value) {
  return static_cast<uint32_t>(value);
}

inline BuddyServerEventV1 completionChunkEvent(const CompletionChunk &source) {
  BuddyServerEventV1 event{};
  event.structSize = sizeof(event);
  event.kind = BUDDY_SERVER_EVENT_CHUNK;
  event.operation = BUDDY_SERVER_OPERATION_COMPLETION;
  event.completionChunk.id = view(source.id);
  event.completionChunk.model = view(source.model);
  event.completionChunk.delta = view(source.delta);
  event.completionChunk.tokenId = source.tokenId;
  event.completionChunk.done = source.done ? 1u : 0u;
  event.completionChunk.finishReason = finishReason(source.finishReason);
  event.completionChunk.usage = usage(source.usage);
  event.completionChunk.timings = timings(source.timings);
  return event;
}

inline BuddyServerEventV1 completionResultEvent(const CompletionResult &source,
                                                uint32_t operation) {
  BuddyServerEventV1 event{};
  event.structSize = sizeof(event);
  event.kind = BUDDY_SERVER_EVENT_FINAL;
  event.operation = operation;
  event.completion.id = view(source.id);
  event.completion.model = view(source.model);
  event.completion.content = view(source.content);
  event.completion.finishReason = finishReason(source.finishReason);
  event.completion.usage = usage(source.usage);
  event.completion.timings = timings(source.timings);
  return event;
}

inline BuddyServerEventV1 tokenizeResultEvent(const TokenizeResult &source) {
  static_assert(sizeof(int) == sizeof(int32_t),
                "server ABI token type requires 32-bit int");
  BuddyServerEventV1 event{};
  event.structSize = sizeof(event);
  event.kind = BUDDY_SERVER_EVENT_FINAL;
  event.operation = BUDDY_SERVER_OPERATION_TOKENIZE;
  event.tokenize.tokens =
      reinterpret_cast<const int32_t *>(source.tokens.data());
  event.tokenize.tokenCount = static_cast<uint64_t>(source.tokens.size());
  event.tokenize.count = static_cast<uint64_t>(source.count);
  return event;
}

inline BuddyServerEventV1 embeddingResultEvent(const EmbeddingResult &source) {
  BuddyServerEventV1 event{};
  event.structSize = sizeof(event);
  event.kind = BUDDY_SERVER_EVENT_FINAL;
  event.operation = BUDDY_SERVER_OPERATION_EMBEDDING;
  event.embedding.model = view(source.model);
  event.embedding.embedding = source.embedding.data();
  event.embedding.embeddingCount =
      static_cast<uint64_t>(source.embedding.size());
  event.embedding.promptTokens = static_cast<uint64_t>(source.promptTokens);
  event.embedding.totalTokens = static_cast<uint64_t>(source.totalTokens);
  return event;
}

inline BuddyServerEventV1
transcriptionResultEvent(const AudioTranscriptionResult &source) {
  BuddyServerEventV1 event{};
  event.structSize = sizeof(event);
  event.kind = BUDDY_SERVER_EVENT_FINAL;
  event.operation = BUDDY_SERVER_OPERATION_TRANSCRIPTION;
  event.transcription.model = view(source.model);
  event.transcription.text = view(source.text);
  event.transcription.generatedTokens =
      static_cast<uint64_t>(source.generatedTokens);
  event.transcription.timings = timings(source.timings);
  return event;
}

inline bool cancelled(const BuddyServerCallV1 &call) {
  return call.isCancelled && call.isCancelled(call.cancellationUserData);
}

template <typename Factory> class ResidentServerModelAdapter {
  struct Object {
    std::unique_ptr<ResidentModel> model;
    ModelStatus status;
  };

  static Object *object(void *value) { return static_cast<Object *>(value); }

  static void *create(const BuddyServerCreateArgsV1 *,
                      BuddyServerErrorV1 *error) {
    clearError(error);
    try {
      auto result = std::make_unique<Object>();
      result->model = Factory::create();
      if (!result->model) {
        setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                 "resident model factory returned null");
        return nullptr;
      }
      return result.release();
    } catch (...) {
      setExceptionError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                        "resident model factory failed");
      return nullptr;
    }
  }

  static BuddyServerStatusCodeV1 load(void *value,
                                      const BuddyServerLoadArgsV1 *args,
                                      BuddyServerErrorV1 *error) {
    clearError(error);
    if (!args)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "resident load arguments are null");
    try {
      object(value)->model->load(residentConfig(*args));
      object(value)->status = object(value)->model->status();
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "resident model load failed");
    }
  }

  static BuddyServerStatusCodeV1 getStatus(void *value,
                                           BuddyServerStatusV1 *status,
                                           BuddyServerErrorV1 *error) {
    clearError(error);
    if (!status)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "resident status output is null");
    try {
      object(value)->status = object(value)->model->status();
      fillStatus(object(value)->status, status);
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "resident status failed");
    }
  }

  static BuddyServerStatusCodeV1 invoke(void *value,
                                        const BuddyServerCallV1 *call,
                                        BuddyServerEmitFnV1 emit, void *data,
                                        BuddyServerErrorV1 *error) {
    clearError(error);
    if (!call || !call->request || !emit)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "resident invoke arguments are incomplete");
    if (cancelled(*call))
      return BUDDY_SERVER_STATUS_CANCELLED;
    try {
      const BuddyServerRequestV1 &request = *call->request;
      switch (request.operation) {
      case BUDDY_SERVER_OPERATION_COMPLETION: {
        const CompletionRequest decoded = completion(request.completion);
        if (request.completion.stream) {
          bool wasCancelled = false;
          const CompletionResult result = object(value)->model->completeStream(
              decoded, [&](const CompletionChunk &chunk) {
                if (wasCancelled || cancelled(*call)) {
                  wasCancelled = true;
                  return false;
                }
                BuddyServerEventV1 event = completionChunkEvent(chunk);
                if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE) {
                  wasCancelled = true;
                  return false;
                }
                return true;
              });
          if (wasCancelled)
            return BUDDY_SERVER_STATUS_CANCELLED;
          BuddyServerEventV1 event =
              completionResultEvent(result, BUDDY_SERVER_OPERATION_COMPLETION);
          if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
            return BUDDY_SERVER_STATUS_CANCELLED;
        } else {
          const CompletionResult result =
              object(value)->model->complete(decoded);
          BuddyServerEventV1 event =
              completionResultEvent(result, BUDDY_SERVER_OPERATION_COMPLETION);
          if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
            return BUDDY_SERVER_STATUS_CANCELLED;
        }
        return BUDDY_SERVER_STATUS_OK;
      }
      case BUDDY_SERVER_OPERATION_CHAT: {
        const ChatCompletionRequest decoded = chat(request.chat);
        if (request.chat.stream) {
          bool wasCancelled = false;
          const CompletionResult result = object(value)->model->chatStream(
              decoded, [&](const CompletionChunk &chunk) {
                if (wasCancelled || cancelled(*call)) {
                  wasCancelled = true;
                  return false;
                }
                BuddyServerEventV1 event = completionChunkEvent(chunk);
                event.operation = BUDDY_SERVER_OPERATION_CHAT;
                if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE) {
                  wasCancelled = true;
                  return false;
                }
                return true;
              });
          if (wasCancelled)
            return BUDDY_SERVER_STATUS_CANCELLED;
          BuddyServerEventV1 event =
              completionResultEvent(result, BUDDY_SERVER_OPERATION_CHAT);
          if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
            return BUDDY_SERVER_STATUS_CANCELLED;
        } else {
          const CompletionResult result = object(value)->model->chat(decoded);
          BuddyServerEventV1 event =
              completionResultEvent(result, BUDDY_SERVER_OPERATION_CHAT);
          if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
            return BUDDY_SERVER_STATUS_CANCELLED;
        }
        return BUDDY_SERVER_STATUS_OK;
      }
      case BUDDY_SERVER_OPERATION_TOKENIZE: {
        const TokenizeResult result =
            object(value)->model->tokenize(tokenize(request.tokenize));
        BuddyServerEventV1 event = tokenizeResultEvent(result);
        if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
          return BUDDY_SERVER_STATUS_CANCELLED;
        return BUDDY_SERVER_STATUS_OK;
      }
      default:
        return setError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED,
                        "operation is not supported by resident model");
      }
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "resident model invocation failed");
    }
  }

  static void destroy(void *value) { delete object(value); }

public:
  static const BuddyServerModelOpsV1 *ops() {
    static const BuddyServerModelOpsV1 value = {
        sizeof(BuddyServerModelOpsV1),
        BUDDY_SERVER_MODEL_ABI_VERSION_V1,
        &create,
        &load,
        &getStatus,
        &invoke,
        &destroy};
    return &value;
  }
};

template <typename Factory> class EmbeddingServerModelAdapter {
  struct Object {
    std::unique_ptr<EmbeddingModel> model;
    ModelStatus status;
  };
  static Object *object(void *value) { return static_cast<Object *>(value); }
  static void *create(const BuddyServerCreateArgsV1 *,
                      BuddyServerErrorV1 *error) {
    clearError(error);
    try {
      auto result = std::make_unique<Object>();
      result->model = Factory::create();
      if (!result->model) {
        setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                 "embedding model factory returned null");
        return nullptr;
      }
      return result.release();
    } catch (...) {
      setExceptionError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                        "embedding model factory failed");
      return nullptr;
    }
  }
  static BuddyServerStatusCodeV1 load(void *value,
                                      const BuddyServerLoadArgsV1 *args,
                                      BuddyServerErrorV1 *error) {
    clearError(error);
    if (!args)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "embedding load arguments are null");
    try {
      object(value)->model->load(embeddingConfig(*args));
      object(value)->status = object(value)->model->status();
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "embedding model load failed");
    }
  }
  static BuddyServerStatusCodeV1 getStatus(void *value,
                                           BuddyServerStatusV1 *status,
                                           BuddyServerErrorV1 *error) {
    clearError(error);
    if (!status)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "embedding status output is null");
    try {
      object(value)->status = object(value)->model->status();
      fillStatus(object(value)->status, status);
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "embedding status failed");
    }
  }
  static BuddyServerStatusCodeV1 invoke(void *value,
                                        const BuddyServerCallV1 *call,
                                        BuddyServerEmitFnV1 emit, void *data,
                                        BuddyServerErrorV1 *error) {
    clearError(error);
    if (!call || !call->request || !emit)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "embedding invoke arguments are incomplete");
    if (cancelled(*call))
      return BUDDY_SERVER_STATUS_CANCELLED;
    try {
      if (call->request->operation != BUDDY_SERVER_OPERATION_EMBEDDING)
        return setError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED,
                        "operation is not supported by embedding model");
      const EmbeddingResult result =
          object(value)->model->embed(embedding(call->request->embedding));
      BuddyServerEventV1 event = embeddingResultEvent(result);
      if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
        return BUDDY_SERVER_STATUS_CANCELLED;
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "embedding model invocation failed");
    }
  }
  static void destroy(void *value) { delete object(value); }

public:
  static const BuddyServerModelOpsV1 *ops() {
    static const BuddyServerModelOpsV1 value = {
        sizeof(BuddyServerModelOpsV1),
        BUDDY_SERVER_MODEL_ABI_VERSION_V1,
        &create,
        &load,
        &getStatus,
        &invoke,
        &destroy};
    return &value;
  }
};

template <typename Factory> class MaskedLMServerModelAdapter {
  struct Object {
    std::unique_ptr<MaskedLMModel> model;
    ModelStatus status;
  };
  static Object *object(void *value) { return static_cast<Object *>(value); }
  static void *create(const BuddyServerCreateArgsV1 *,
                      BuddyServerErrorV1 *error) {
    clearError(error);
    try {
      auto result = std::make_unique<Object>();
      result->model = Factory::create();
      if (!result->model) {
        setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                 "masked-LM model factory returned null");
        return nullptr;
      }
      return result.release();
    } catch (...) {
      setExceptionError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                        "masked-LM model factory failed");
      return nullptr;
    }
  }
  static BuddyServerStatusCodeV1 load(void *value,
                                      const BuddyServerLoadArgsV1 *args,
                                      BuddyServerErrorV1 *error) {
    clearError(error);
    if (!args)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "masked-LM load arguments are null");
    try {
      object(value)->model->load(maskedLMConfig(*args));
      object(value)->status = object(value)->model->status();
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "masked-LM model load failed");
    }
  }
  static BuddyServerStatusCodeV1 getStatus(void *value,
                                           BuddyServerStatusV1 *status,
                                           BuddyServerErrorV1 *error) {
    clearError(error);
    if (!status)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "masked-LM status output is null");
    try {
      object(value)->status = object(value)->model->status();
      fillStatus(object(value)->status, status);
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "masked-LM status failed");
    }
  }
  static BuddyServerStatusCodeV1 invoke(void *value,
                                        const BuddyServerCallV1 *call,
                                        BuddyServerEmitFnV1 emit, void *data,
                                        BuddyServerErrorV1 *error) {
    clearError(error);
    if (!call || !call->request || !emit)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "masked-LM invoke arguments are incomplete");
    if (cancelled(*call))
      return BUDDY_SERVER_STATUS_CANCELLED;
    try {
      if (call->request->operation != BUDDY_SERVER_OPERATION_MASKED_LM)
        return setError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED,
                        "operation is not supported by masked-LM model");
      const MaskedLMResult result =
          object(value)->model->predict(maskedLM(call->request->maskedLM));
      std::vector<std::vector<BuddyServerMaskedLMTokenV1>> tokenStorage;
      std::vector<BuddyServerMaskedLMPredictionV1> predictions;
      tokenStorage.reserve(result.predictions.size());
      predictions.reserve(result.predictions.size());
      for (const auto &prediction : result.predictions) {
        tokenStorage.emplace_back();
        auto &tokens = tokenStorage.back();
        tokens.reserve(prediction.tokens.size());
        for (const auto &token : prediction.tokens)
          tokens.push_back({token.tokenId, view(token.token), token.score});
        predictions.push_back({static_cast<uint64_t>(prediction.position),
                               tokens.data(),
                               static_cast<uint64_t>(tokens.size())});
      }
      BuddyServerEventV1 event{};
      event.structSize = sizeof(event);
      event.kind = BUDDY_SERVER_EVENT_FINAL;
      event.operation = BUDDY_SERVER_OPERATION_MASKED_LM;
      event.maskedLM.model = view(result.model);
      event.maskedLM.sequenceLength =
          static_cast<uint64_t>(result.sequenceLength);
      event.maskedLM.predictions = predictions.data();
      event.maskedLM.predictionCount =
          static_cast<uint64_t>(predictions.size());
      event.maskedLM.promptTokens = static_cast<uint64_t>(result.promptTokens);
      if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
        return BUDDY_SERVER_STATUS_CANCELLED;
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "masked-LM model invocation failed");
    }
  }
  static void destroy(void *value) { delete object(value); }

public:
  static const BuddyServerModelOpsV1 *ops() {
    static const BuddyServerModelOpsV1 value = {
        sizeof(BuddyServerModelOpsV1),
        BUDDY_SERVER_MODEL_ABI_VERSION_V1,
        &create,
        &load,
        &getStatus,
        &invoke,
        &destroy};
    return &value;
  }
};

template <typename Factory> class TranscriptionServerModelAdapter {
  struct Object {
    std::unique_ptr<AudioTranscriptionModel> model;
    ModelStatus status;
  };
  static Object *object(void *value) { return static_cast<Object *>(value); }
  static void *create(const BuddyServerCreateArgsV1 *,
                      BuddyServerErrorV1 *error) {
    clearError(error);
    try {
      auto result = std::make_unique<Object>();
      result->model = Factory::create();
      if (!result->model) {
        setError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                 "transcription model factory returned null");
        return nullptr;
      }
      return result.release();
    } catch (...) {
      setExceptionError(error, BUDDY_SERVER_STATUS_INTERNAL_ERROR,
                        "transcription model factory failed");
      return nullptr;
    }
  }
  static BuddyServerStatusCodeV1 load(void *value,
                                      const BuddyServerLoadArgsV1 *args,
                                      BuddyServerErrorV1 *error) {
    clearError(error);
    if (!args)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "transcription load arguments are null");
    try {
      object(value)->model->load(transcriptionConfig(*args));
      object(value)->status = object(value)->model->status();
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "transcription model load failed");
    }
  }
  static BuddyServerStatusCodeV1 getStatus(void *value,
                                           BuddyServerStatusV1 *status,
                                           BuddyServerErrorV1 *error) {
    clearError(error);
    if (!status)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "transcription status output is null");
    try {
      object(value)->status = object(value)->model->status();
      fillStatus(object(value)->status, status);
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "transcription status failed");
    }
  }
  static BuddyServerStatusCodeV1 invoke(void *value,
                                        const BuddyServerCallV1 *call,
                                        BuddyServerEmitFnV1 emit, void *data,
                                        BuddyServerErrorV1 *error) {
    clearError(error);
    if (!call || !call->request || !emit)
      return setError(error, BUDDY_SERVER_STATUS_INVALID_ARGUMENT,
                      "transcription invoke arguments are incomplete");
    if (cancelled(*call))
      return BUDDY_SERVER_STATUS_CANCELLED;
    try {
      if (call->request->operation != BUDDY_SERVER_OPERATION_TRANSCRIPTION)
        return setError(error, BUDDY_SERVER_STATUS_NOT_SUPPORTED,
                        "operation is not supported by transcription model");
      const AudioTranscriptionResult result = object(value)->model->transcribe(
          transcription(call->request->transcription));
      BuddyServerEventV1 event = transcriptionResultEvent(result);
      if (emit(data, &event) != BUDDY_SERVER_EMIT_CONTINUE)
        return BUDDY_SERVER_STATUS_CANCELLED;
      return BUDDY_SERVER_STATUS_OK;
    } catch (...) {
      return setExceptionError(error, BUDDY_SERVER_STATUS_MODEL_ERROR,
                               "transcription model invocation failed");
    }
  }
  static void destroy(void *value) { delete object(value); }

public:
  static const BuddyServerModelOpsV1 *ops() {
    static const BuddyServerModelOpsV1 value = {
        sizeof(BuddyServerModelOpsV1),
        BUDDY_SERVER_MODEL_ABI_VERSION_V1,
        &create,
        &load,
        &getStatus,
        &invoke,
        &destroy};
    return &value;
  }
};

} // namespace server_abi
} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_CORE_SERVERMODELPLUGINADAPTERS_H
