#ifndef BUDDY_QWEN3_LAYERWISE_SCHEDULER_H
#define BUDDY_QWEN3_LAYERWISE_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} BuddyQwen3MemRef1;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} BuddyQwen3MemRef2;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[3];
  int64_t strides[3];
} BuddyQwen3MemRef3;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} BuddyQwen3MemRef4;

typedef struct {
  /*
   * The imported MLIR functions return (value, key, hidden) in this order.
   * Prefill K/V cover the full prompt. Decode imports may return either the
   * current [1, kv_heads, 1, head_dim] update or the full updated static
   * cache; the scheduler commits only the cache_write_mask position.
   */
  BuddyQwen3MemRef4 value;
  BuddyQwen3MemRef4 key;
  BuddyQwen3MemRef3 hidden;
} BuddyQwen3LayerResult;

typedef struct {
  const uint8_t *params_i8;
  const float *params_f32;
} BuddyQwen3PackedParams;

typedef struct {
  int32_t prefill_len;
  int32_t max_cache_len;
  int32_t num_layers;
  int32_t hidden_size;
  int32_t kv_heads;
  int32_t head_dim;
  int32_t vocab_size;
} BuddyQwen3ModelSpec;

typedef struct {
  int (*embedding_prefill)(const BuddyQwen3PackedParams *params,
                           BuddyQwen3MemRef3 *result, const int64_t *input_ids);
  int (*embedding_decode)(const BuddyQwen3PackedParams *params,
                          BuddyQwen3MemRef3 *result, const int64_t *input_id);
  int (*decoder_layer_prefill)(const BuddyQwen3PackedParams *params,
                               int32_t layer, BuddyQwen3LayerResult *result,
                               const float *hidden, const float *rope_cos,
                               const float *rope_sin,
                               const float *attention_mask);
  int (*decoder_layer_decode)(const BuddyQwen3PackedParams *params,
                              int32_t layer, BuddyQwen3LayerResult *result,
                              const float *hidden, const float *rope_cos,
                              const float *rope_sin,
                              const float *attention_mask,
                              const float *cache_write_mask,
                              const float *old_key, const float *old_value);
  int (*final_head)(const BuddyQwen3PackedParams *params,
                    BuddyQwen3MemRef3 *result, const float *last_hidden);
} BuddyQwen3LayerwiseOps;

typedef void (*BuddyQwen3ArenaResetFn)(void *opaque);
typedef void (*BuddyQwen3ProgressFn)(void *opaque, const char *stage,
                                     int32_t layer);
typedef void (*BuddyQwen3CheckpointFn)(void *opaque, const char *stage,
                                       int32_t layer, const float *values,
                                       size_t count);
typedef uint64_t (*BuddyQwen3ReadCyclesFn)(void *opaque);
typedef void (*BuddyQwen3TimingFn)(void *opaque, const char *stage,
                                   int32_t layer, uint64_t cycles);

typedef struct {
  BuddyQwen3ReadCyclesFn read_cycles;
  BuddyQwen3TimingFn record;
  void *opaque;
} BuddyQwen3Profiler;

typedef struct {
  BuddyQwen3ModelSpec spec;
  BuddyQwen3PackedParams params;
  const BuddyQwen3LayerwiseOps *ops;

  float *hidden_ping;
  float *hidden_pong;
  size_t hidden_capacity;
  float *key_cache;
  float *value_cache;
  size_t cache_capacity;
  float *logits;
  size_t logits_capacity;

  BuddyQwen3ArenaResetFn reset_arena;
  void *arena_opaque;
  BuddyQwen3ProgressFn progress;
  void *progress_opaque;
  BuddyQwen3CheckpointFn checkpoint;
  void *checkpoint_opaque;
  /* Optional cycle profiler. NULL keeps the scheduler instrumentation-free. */
  const BuddyQwen3Profiler *profiler;
} BuddyQwen3LayerwiseContext;

typedef struct {
  const int64_t *input_ids;
  const float *rope_cos;
  const float *rope_sin;
  const float *attention_mask;
  /* Number of real tokens in the left-padded static prefill tensor. */
  int32_t valid_tokens;
} BuddyQwen3PrefillInputs;

typedef struct {
  const int64_t *input_id;
  const float *rope_cos;
  const float *rope_sin;
  const float *attention_mask;
  const float *cache_write_mask;
} BuddyQwen3DecodeInputs;

typedef struct {
  int64_t initial_token;
  const float *rope_cos;
  const float *rope_sin;
  size_t rope_elements;
  float *attention_mask_workspace;
  float *cache_write_mask_workspace;
  size_t mask_capacity;
  int64_t *output_ids;
  size_t output_capacity;
  int32_t decode_steps;
  /* Number of real tokens in the left-padded prefill tensor. */
  int32_t prompt_tokens;
  /* A negative value disables early stopping. */
  int32_t eos_token_id;
} BuddyQwen3GreedyDecodeInputs;

enum {
  BUDDY_QWEN3_OK = 0,
  BUDDY_QWEN3_ERR_ARGUMENT = -1,
  BUDDY_QWEN3_ERR_CAPACITY = -2,
  BUDDY_QWEN3_ERR_KERNEL = -3,
  BUDDY_QWEN3_ERR_RESULT = -4,
};

int buddy_qwen3_layerwise_prefill(BuddyQwen3LayerwiseContext *context,
                                  const BuddyQwen3PrefillInputs *inputs);

int buddy_qwen3_layerwise_decode(BuddyQwen3LayerwiseContext *context,
                                 const BuddyQwen3DecodeInputs *inputs);

int buddy_qwen3_layerwise_generate_greedy(
    BuddyQwen3LayerwiseContext *context,
    const BuddyQwen3GreedyDecodeInputs *inputs, int32_t *generated_count);

extern const BuddyQwen3ModelSpec buddy_qwen3_generated_spec;
void buddy_qwen3_initialize_generated_ops(BuddyQwen3LayerwiseOps *ops);

#ifdef __cplusplus
}
#endif

#endif
