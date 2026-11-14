//===- ServerModelABI.h - Buddy server model C ABI -------------*- C -*-===//
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

#ifndef BUDDY_RUNTIME_CORE_SERVERMODELABI_H
#define BUDDY_RUNTIME_CORE_SERVERMODELABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BUDDY_SERVER_MODEL_ABI_VERSION_V1 1u
#define BUDDY_SERVER_MODEL_DESCRIPTOR_SYMBOL_V1                                \
  "buddy_server_model_descriptor_v1"

typedef struct BuddyServerStringViewV1 {
  const char *data;
  uint64_t size;
} BuddyServerStringViewV1;

typedef struct BuddyServerBytesViewV1 {
  const uint8_t *data;
  uint64_t size;
} BuddyServerBytesViewV1;

typedef struct BuddyServerStringPairV1 {
  BuddyServerStringViewV1 key;
  BuddyServerStringViewV1 value;
} BuddyServerStringPairV1;

typedef enum BuddyServerServiceKindV1 {
  BUDDY_SERVER_SERVICE_INVALID = 0,
  BUDDY_SERVER_SERVICE_RESIDENT = 1,
  BUDDY_SERVER_SERVICE_EMBEDDING = 2,
  BUDDY_SERVER_SERVICE_MASKED_LM = 3,
  BUDDY_SERVER_SERVICE_TRANSCRIPTION = 4
} BuddyServerServiceKindV1;

typedef enum BuddyServerOperationV1 {
  BUDDY_SERVER_OPERATION_COMPLETION = 1,
  BUDDY_SERVER_OPERATION_CHAT = 2,
  BUDDY_SERVER_OPERATION_TOKENIZE = 3,
  BUDDY_SERVER_OPERATION_EMBEDDING = 4,
  BUDDY_SERVER_OPERATION_MASKED_LM = 5,
  BUDDY_SERVER_OPERATION_TRANSCRIPTION = 6
} BuddyServerOperationV1;

typedef enum BuddyServerModalityV1 {
  BUDDY_SERVER_MODALITY_TEXT = 1ull << 0,
  BUDDY_SERVER_MODALITY_IMAGE = 1ull << 1,
  BUDDY_SERVER_MODALITY_AUDIO = 1ull << 2,
  BUDDY_SERVER_MODALITY_VIDEO = 1ull << 3
} BuddyServerModalityV1;

typedef enum BuddyServerCapabilityV1 {
  BUDDY_SERVER_CAPABILITY_STREAMING = 1ull << 0,
  BUDDY_SERVER_CAPABILITY_BATCH = 1ull << 1,
  BUDDY_SERVER_CAPABILITY_CANCEL = 1ull << 2,
  BUDDY_SERVER_CAPABILITY_THREAD_SAFE = 1ull << 3
} BuddyServerCapabilityV1;

typedef enum BuddyServerResultKindV1 {
  BUDDY_SERVER_RESULT_UNSPECIFIED = 0,
  BUDDY_SERVER_RESULT_TEXT = 1,
  BUDDY_SERVER_RESULT_EMBEDDING_VECTOR = 2,
  BUDDY_SERVER_RESULT_TOKEN_SCORES = 3,
  BUDDY_SERVER_RESULT_AUDIO = 4
} BuddyServerResultKindV1;

typedef enum BuddyServerModelLoadStateV1 {
  BUDDY_SERVER_MODEL_UNLOADED = 0,
  BUDDY_SERVER_MODEL_LOADING = 1,
  BUDDY_SERVER_MODEL_READY = 2,
  BUDDY_SERVER_MODEL_ERROR = 3
} BuddyServerModelLoadStateV1;

typedef enum BuddyServerStatusCodeV1 {
  BUDDY_SERVER_STATUS_OK = 0,
  BUDDY_SERVER_STATUS_INVALID_ARGUMENT = 1,
  BUDDY_SERVER_STATUS_NOT_SUPPORTED = 2,
  BUDDY_SERVER_STATUS_MODEL_NOT_READY = 3,
  BUDDY_SERVER_STATUS_RESOURCE_EXHAUSTED = 4,
  BUDDY_SERVER_STATUS_CANCELLED = 5,
  BUDDY_SERVER_STATUS_DEADLINE_EXCEEDED = 6,
  BUDDY_SERVER_STATUS_MODEL_ERROR = 7,
  BUDDY_SERVER_STATUS_INTERNAL_ERROR = 8
} BuddyServerStatusCodeV1;

typedef enum BuddyServerEventKindV1 {
  BUDDY_SERVER_EVENT_CHUNK = 1,
  BUDDY_SERVER_EVENT_FINAL = 2,
  BUDDY_SERVER_EVENT_ERROR = 3,
  BUDDY_SERVER_EVENT_METRICS = 4
} BuddyServerEventKindV1;

typedef struct BuddyServerErrorV1 {
  uint32_t code;
  const char *message;
  uint64_t messageSize;
} BuddyServerErrorV1;

typedef struct BuddyServerSamplingV1 {
  int32_t maxTokens;
  float temperature;
  int32_t topK;
  float topP;
  float minP;
  float repeatPenalty;
  int32_t repeatLastN;
  uint64_t seed;
  const int64_t *stopTokenIds;
  uint64_t stopTokenIdCount;
  const BuddyServerStringViewV1 *stop;
  uint64_t stopCount;
} BuddyServerSamplingV1;

typedef struct BuddyServerImageInputV1 {
  BuddyServerStringViewV1 uri;
  BuddyServerStringViewV1 mimeType;
  BuddyServerBytesViewV1 bytes;
} BuddyServerImageInputV1;

typedef struct BuddyServerAudioInputV1 {
  BuddyServerStringViewV1 uri;
  BuddyServerStringViewV1 mimeType;
  BuddyServerBytesViewV1 bytes;
} BuddyServerAudioInputV1;

typedef struct BuddyServerCompletionRequestV1 {
  BuddyServerStringViewV1 prompt;
  BuddyServerSamplingV1 sampling;
  const BuddyServerImageInputV1 *images;
  uint64_t imageCount;
  uint32_t stream;
} BuddyServerCompletionRequestV1;

typedef struct BuddyServerChatMessageV1 {
  BuddyServerStringViewV1 role;
  BuddyServerStringViewV1 content;
  const BuddyServerImageInputV1 *images;
  uint64_t imageCount;
} BuddyServerChatMessageV1;

typedef struct BuddyServerChatRequestV1 {
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 input;
  const BuddyServerChatMessageV1 *messages;
  uint64_t messageCount;
  BuddyServerSamplingV1 sampling;
  const BuddyServerImageInputV1 *images;
  uint64_t imageCount;
  uint32_t stream;
} BuddyServerChatRequestV1;

typedef struct BuddyServerTokenizeRequestV1 {
  BuddyServerStringViewV1 content;
  uint32_t addSpecial;
  uint32_t countOnly;
} BuddyServerTokenizeRequestV1;

typedef struct BuddyServerEmbeddingRequestV1 {
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 input;
} BuddyServerEmbeddingRequestV1;

typedef struct BuddyServerMaskedLMRequestV1 {
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 input;
  uint64_t topK;
} BuddyServerMaskedLMRequestV1;

typedef struct BuddyServerTranscriptionRequestV1 {
  BuddyServerStringViewV1 model;
  BuddyServerAudioInputV1 audio;
  int32_t maxTokens;
} BuddyServerTranscriptionRequestV1;

typedef struct BuddyServerRequestV1 {
  uint32_t structSize;
  uint32_t operation;
  uint64_t requestId;
  union {
    BuddyServerCompletionRequestV1 completion;
    BuddyServerChatRequestV1 chat;
    BuddyServerTokenizeRequestV1 tokenize;
    BuddyServerEmbeddingRequestV1 embedding;
    BuddyServerMaskedLMRequestV1 maskedLM;
    BuddyServerTranscriptionRequestV1 transcription;
  };
} BuddyServerRequestV1;

typedef uint32_t (*BuddyServerCancellationFnV1)(void *userData);

typedef struct BuddyServerCallV1 {
  uint32_t structSize;
  uint64_t requestId;
  uint64_t deadlineUnixNs;
  BuddyServerCancellationFnV1 isCancelled;
  void *cancellationUserData;
  const BuddyServerRequestV1 *request;
} BuddyServerCallV1;

typedef struct BuddyServerCreateArgsV1 {
  uint32_t structSize;
  BuddyServerStringViewV1 modelName;
} BuddyServerCreateArgsV1;

typedef struct BuddyServerLoadArgsV1 {
  uint32_t structSize;
  BuddyServerStringViewV1 raxPath;
  BuddyServerStringViewV1 modelName;
  BuddyServerStringViewV1 modelSoPath;
  const BuddyServerStringViewV1 *weightPaths;
  uint64_t weightPathCount;
  BuddyServerStringViewV1 vocabPath;
  BuddyServerStringViewV1 chatTemplatePath;
  const BuddyServerStringPairV1 *attributes;
  uint64_t attributeCount;
} BuddyServerLoadArgsV1;

typedef struct BuddyServerStatusV1 {
  uint32_t structSize;
  uint32_t state;
  BuddyServerStringViewV1 modelName;
  BuddyServerStringViewV1 backend;
  BuddyServerStringViewV1 message;
  int32_t contextLength;
} BuddyServerStatusV1;

typedef struct BuddyServerUsageV1 {
  int32_t promptTokens;
  int32_t completionTokens;
  int32_t totalTokens;
} BuddyServerUsageV1;

typedef struct BuddyServerTimingsV1 {
  double prefillMs;
  double decodeMs;
  double tokensPerSecond;
  double preprocessMs;
  double inferenceMs;
  double totalMs;
} BuddyServerTimingsV1;

typedef struct BuddyServerCompletionChunkV1 {
  BuddyServerStringViewV1 id;
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 delta;
  int32_t tokenId;
  uint32_t done;
  uint32_t finishReason;
  BuddyServerUsageV1 usage;
  BuddyServerTimingsV1 timings;
} BuddyServerCompletionChunkV1;

typedef struct BuddyServerCompletionResultV1 {
  BuddyServerStringViewV1 id;
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 content;
  uint32_t finishReason;
  BuddyServerUsageV1 usage;
  BuddyServerTimingsV1 timings;
} BuddyServerCompletionResultV1;

typedef struct BuddyServerTokenizeResultV1 {
  const int32_t *tokens;
  uint64_t tokenCount;
  uint64_t count;
} BuddyServerTokenizeResultV1;

typedef struct BuddyServerEmbeddingResultV1 {
  BuddyServerStringViewV1 model;
  const float *embedding;
  uint64_t embeddingCount;
  uint64_t promptTokens;
  uint64_t totalTokens;
} BuddyServerEmbeddingResultV1;

typedef struct BuddyServerMaskedLMTokenV1 {
  int64_t tokenId;
  BuddyServerStringViewV1 token;
  float score;
} BuddyServerMaskedLMTokenV1;

typedef struct BuddyServerMaskedLMPredictionV1 {
  uint64_t position;
  const BuddyServerMaskedLMTokenV1 *tokens;
  uint64_t tokenCount;
} BuddyServerMaskedLMPredictionV1;

typedef struct BuddyServerMaskedLMResultV1 {
  BuddyServerStringViewV1 model;
  uint64_t sequenceLength;
  const BuddyServerMaskedLMPredictionV1 *predictions;
  uint64_t predictionCount;
  uint64_t promptTokens;
} BuddyServerMaskedLMResultV1;

typedef struct BuddyServerTranscriptionResultV1 {
  BuddyServerStringViewV1 model;
  BuddyServerStringViewV1 text;
  uint64_t generatedTokens;
  BuddyServerTimingsV1 timings;
} BuddyServerTranscriptionResultV1;

typedef struct BuddyServerEventV1 {
  uint32_t structSize;
  uint32_t kind;
  uint32_t operation;
  union {
    BuddyServerCompletionChunkV1 completionChunk;
    BuddyServerCompletionResultV1 completion;
    BuddyServerTokenizeResultV1 tokenize;
    BuddyServerEmbeddingResultV1 embedding;
    BuddyServerMaskedLMResultV1 maskedLM;
    BuddyServerTranscriptionResultV1 transcription;
  };
} BuddyServerEventV1;

typedef enum BuddyServerEmitResultV1 {
  BUDDY_SERVER_EMIT_CONTINUE = 0,
  BUDDY_SERVER_EMIT_CANCEL = 1,
  BUDDY_SERVER_EMIT_ERROR = 2
} BuddyServerEmitResultV1;

typedef BuddyServerEmitResultV1 (*BuddyServerEmitFnV1)(
    void *userData, const BuddyServerEventV1 *event);

typedef struct BuddyServerModelOpsV1 BuddyServerModelOpsV1;

typedef void *(*BuddyServerCreateFnV1)(const BuddyServerCreateArgsV1 *,
                                       BuddyServerErrorV1 *);
typedef BuddyServerStatusCodeV1 (*BuddyServerLoadFnV1)(
    void *, const BuddyServerLoadArgsV1 *, BuddyServerErrorV1 *);
typedef BuddyServerStatusCodeV1 (*BuddyServerGetStatusFnV1)(
    void *, BuddyServerStatusV1 *, BuddyServerErrorV1 *);
typedef BuddyServerStatusCodeV1 (*BuddyServerInvokeFnV1)(
    void *, const BuddyServerCallV1 *, BuddyServerEmitFnV1, void *,
    BuddyServerErrorV1 *);
typedef void (*BuddyServerDestroyFnV1)(void *);

struct BuddyServerModelOpsV1 {
  uint32_t structSize;
  uint32_t abiVersion;
  BuddyServerCreateFnV1 create;
  BuddyServerLoadFnV1 load;
  BuddyServerGetStatusFnV1 getStatus;
  BuddyServerInvokeFnV1 invoke;
  BuddyServerDestroyFnV1 destroy;
};

typedef struct BuddyServerModelDescriptorV1 {
  uint32_t structSize;
  uint32_t abiVersion;
  uint32_t serviceKind;
  uint64_t operationMask;
  uint64_t capabilityMask;
  uint64_t inputModalityMask;
  uint32_t resultKind;
  uint32_t maxConcurrency;
  const char *modelType;
  const BuddyServerModelOpsV1 *ops;
  uint32_t reserved;
} BuddyServerModelDescriptorV1;

typedef const BuddyServerModelDescriptorV1 *(*BuddyServerModelDescriptorFnV1)(
    void);

#ifdef __cplusplus
} // extern "C"

static_assert(sizeof(BuddyServerStringViewV1) == 16,
              "Buddy server ABI string view layout changed");
static_assert(sizeof(BuddyServerModelDescriptorV1) == 72,
              "Buddy server ABI descriptor layout changed");
#endif

#endif // BUDDY_RUNTIME_CORE_SERVERMODELABI_H
