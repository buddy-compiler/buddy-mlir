#include "layerwise_scheduler.h"

#include <stddef.h>
#include <stdint.h>

static void reset_arena(BuddyQwen3LayerwiseContext *context) {
  if (context->reset_arena)
    context->reset_arena(context->arena_opaque);
}

static void progress(BuddyQwen3LayerwiseContext *context, const char *stage,
                     int32_t layer) {
  if (context->progress)
    context->progress(context->progress_opaque, stage, layer);
}

static void checkpoint(BuddyQwen3LayerwiseContext *context, const char *stage,
                       int32_t layer, const float *values, size_t count) {
  if (context->checkpoint)
    context->checkpoint(context->checkpoint_opaque, stage, layer, values,
                        count);
}

static uint64_t timing_begin(const BuddyQwen3LayerwiseContext *context) {
  if (!context->profiler || !context->profiler->read_cycles ||
      !context->profiler->record)
    return 0;
  return context->profiler->read_cycles(context->profiler->opaque);
}

static void timing_end(const BuddyQwen3LayerwiseContext *context,
                       const char *stage, int32_t layer, uint64_t start) {
  uint64_t end;
  if (!context->profiler || !context->profiler->read_cycles ||
      !context->profiler->record)
    return;
  end = context->profiler->read_cycles(context->profiler->opaque);
  context->profiler->record(context->profiler->opaque, stage, layer,
                            end - start);
}

static int checked_elements(size_t a, size_t b, size_t *result) {
  if (a != 0 && b > SIZE_MAX / a)
    return 0;
  *result = a * b;
  return 1;
}

static int32_t greedy_argmax(const float *values, int32_t count) {
  int32_t best = 0;
  for (int32_t index = 1; index < count; ++index)
    if (values[index] > values[best])
      best = index;
  return best;
}

static int validate_context(const BuddyQwen3LayerwiseContext *context) {
  size_t hidden_elements;
  size_t cache_elements;
  size_t value;

  if (!context || !context->ops || !context->params.params_i8 ||
      !context->params.params_f32 || !context->hidden_ping ||
      !context->hidden_pong || !context->key_cache || !context->value_cache ||
      !context->logits)
    return BUDDY_QWEN3_ERR_ARGUMENT;
  if (!context->ops->embedding_prefill || !context->ops->embedding_decode ||
      !context->ops->decoder_layer_prefill ||
      !context->ops->decoder_layer_decode || !context->ops->final_head)
    return BUDDY_QWEN3_ERR_ARGUMENT;
  if (context->spec.prefill_len <= 0 || context->spec.max_cache_len <= 0 ||
      context->spec.prefill_len > context->spec.max_cache_len ||
      context->spec.num_layers <= 0 || context->spec.hidden_size <= 0 ||
      context->spec.kv_heads <= 0 || context->spec.head_dim <= 0 ||
      context->spec.vocab_size <= 0)
    return BUDDY_QWEN3_ERR_ARGUMENT;

  if (!checked_elements((size_t)context->spec.prefill_len,
                        (size_t)context->spec.hidden_size, &hidden_elements))
    return BUDDY_QWEN3_ERR_CAPACITY;
  if (context->hidden_capacity < hidden_elements ||
      context->logits_capacity < (size_t)context->spec.vocab_size)
    return BUDDY_QWEN3_ERR_CAPACITY;

  if (!checked_elements((size_t)context->spec.num_layers,
                        (size_t)context->spec.kv_heads, &value) ||
      !checked_elements(value, (size_t)context->spec.max_cache_len, &value) ||
      !checked_elements(value, (size_t)context->spec.head_dim, &cache_elements))
    return BUDDY_QWEN3_ERR_CAPACITY;
  if (context->cache_capacity < cache_elements)
    return BUDDY_QWEN3_ERR_CAPACITY;
  return BUDDY_QWEN3_OK;
}

static int copy_rank3_contiguous(float *destination, size_t elements,
                                 const BuddyQwen3MemRef3 *source, int64_t size0,
                                 int64_t size1, int64_t size2) {
  size_t index = 0;
  const float *data;
  if (!source || !source->aligned || source->sizes[0] != size0 ||
      source->sizes[1] != size1 || source->sizes[2] != size2)
    return BUDDY_QWEN3_ERR_RESULT;
  data = (const float *)source->aligned;
  for (int64_t i = 0; i < size0; ++i) {
    for (int64_t j = 0; j < size1; ++j) {
      for (int64_t k = 0; k < size2; ++k) {
        int64_t source_index = source->offset + i * source->strides[0] +
                               j * source->strides[1] + k * source->strides[2];
        if (source_index < 0 || index >= elements)
          return BUDDY_QWEN3_ERR_RESULT;
        destination[index++] = data[source_index];
      }
    }
  }
  return index == elements ? BUDDY_QWEN3_OK : BUDDY_QWEN3_ERR_RESULT;
}

static int copy_prefill_cache(float *destination,
                              const BuddyQwen3MemRef4 *source,
                              const BuddyQwen3ModelSpec *spec, int32_t layer) {
  const float *data;
  size_t layer_stride = (size_t)spec->kv_heads * (size_t)spec->max_cache_len *
                        (size_t)spec->head_dim;
  if (!source || !source->aligned || source->sizes[0] != 1 ||
      source->sizes[1] != spec->kv_heads ||
      source->sizes[2] != spec->prefill_len ||
      source->sizes[3] != spec->head_dim)
    return BUDDY_QWEN3_ERR_RESULT;
  data = (const float *)source->aligned;
  destination += (size_t)layer * layer_stride;
  for (int64_t head = 0; head < spec->kv_heads; ++head) {
    for (int64_t position = 0; position < spec->prefill_len; ++position) {
      for (int64_t dim = 0; dim < spec->head_dim; ++dim) {
        int64_t source_index = source->offset + head * source->strides[1] +
                               position * source->strides[2] +
                               dim * source->strides[3];
        size_t destination_index =
            ((size_t)head * (size_t)spec->max_cache_len + (size_t)position) *
                (size_t)spec->head_dim +
            (size_t)dim;
        if (source_index < 0)
          return BUDDY_QWEN3_ERR_RESULT;
        destination[destination_index] = data[source_index];
      }
    }
  }
  return BUDDY_QWEN3_OK;
}

static int find_cache_write_position(const float *cache_write_mask,
                                     int32_t max_cache_len,
                                     int32_t *position) {
  int32_t found = -1;
  if (!cache_write_mask || !position || max_cache_len <= 0)
    return BUDDY_QWEN3_ERR_ARGUMENT;
  for (int32_t index = 0; index < max_cache_len; ++index) {
    if (cache_write_mask[index] == 0.0f)
      continue;
    if (cache_write_mask[index] != 1.0f || found >= 0)
      return BUDDY_QWEN3_ERR_ARGUMENT;
    found = index;
  }
  if (found < 0)
    return BUDDY_QWEN3_ERR_ARGUMENT;
  *position = found;
  return BUDDY_QWEN3_OK;
}

static int copy_decode_cache_update(float *destination,
                                    const BuddyQwen3MemRef4 *source,
                                    const BuddyQwen3ModelSpec *spec,
                                    int32_t layer, int32_t position) {
  const float *data;
  int64_t source_position;
  size_t layer_stride = (size_t)spec->kv_heads * (size_t)spec->max_cache_len *
                        (size_t)spec->head_dim;
  if (!source || !source->aligned || source->sizes[0] != 1 ||
      source->sizes[1] != spec->kv_heads ||
      (source->sizes[2] != 1 &&
       source->sizes[2] != spec->max_cache_len) ||
      source->sizes[3] != spec->head_dim ||
      position < 0 || position >= spec->max_cache_len)
    return BUDDY_QWEN3_ERR_RESULT;
  data = (const float *)source->aligned;
  source_position = source->sizes[2] == 1 ? 0 : position;
  destination += (size_t)layer * layer_stride;
  for (int64_t head = 0; head < spec->kv_heads; ++head) {
    for (int64_t dim = 0; dim < spec->head_dim; ++dim) {
      int64_t source_index = source->offset + head * source->strides[1] +
                             source_position * source->strides[2] +
                             dim * source->strides[3];
      size_t destination_index =
          ((size_t)head * (size_t)spec->max_cache_len + (size_t)position) *
              (size_t)spec->head_dim +
          (size_t)dim;
      if (source_index < 0)
        return BUDDY_QWEN3_ERR_RESULT;
      destination[destination_index] = data[source_index];
    }
  }
  return BUDDY_QWEN3_OK;
}

int buddy_qwen3_layerwise_prefill(BuddyQwen3LayerwiseContext *context,
                                  const BuddyQwen3PrefillInputs *inputs) {
  BuddyQwen3MemRef3 embedding_result;
  BuddyQwen3LayerResult layer_result;
  BuddyQwen3MemRef3 logits_result;
  float *current;
  float *next;
  size_t hidden_elements;
  int status = validate_context(context);
  if (status != BUDDY_QWEN3_OK || !inputs || !inputs->input_ids ||
      !inputs->rope_cos || !inputs->rope_sin || !inputs->attention_mask ||
      inputs->valid_tokens <= 0 ||
      inputs->valid_tokens > context->spec.prefill_len)
    return status != BUDDY_QWEN3_OK ? status : BUDDY_QWEN3_ERR_ARGUMENT;

  hidden_elements =
      (size_t)context->spec.prefill_len * (size_t)context->spec.hidden_size;
  progress(context, "embedding_prefill", -1);
  reset_arena(context);
  status = context->ops->embedding_prefill(&context->params, &embedding_result,
                                           inputs->input_ids);
  if (status != BUDDY_QWEN3_OK)
    return BUDDY_QWEN3_ERR_KERNEL;
  status = copy_rank3_contiguous(
      context->hidden_ping, hidden_elements, &embedding_result, 1,
      context->spec.prefill_len, context->spec.hidden_size);
  if (status == BUDDY_QWEN3_OK)
    checkpoint(context, "prefill_embedding", -1, context->hidden_ping,
               hidden_elements);
  reset_arena(context);
  if (status != BUDDY_QWEN3_OK)
    return status;

  current = context->hidden_ping;
  next = context->hidden_pong;
  for (int32_t layer = 0; layer < context->spec.num_layers; ++layer) {
    status = context->ops->decoder_layer_prefill(
        &context->params, layer, &layer_result, current, inputs->rope_cos,
        inputs->rope_sin, inputs->attention_mask);
    if (status != BUDDY_QWEN3_OK)
      return BUDDY_QWEN3_ERR_KERNEL;
    progress(context, "decoder_layer_prefill", layer);
    status = copy_prefill_cache(context->key_cache, &layer_result.key,
                                &context->spec, layer);
    if (status == BUDDY_QWEN3_OK)
      status = copy_prefill_cache(context->value_cache, &layer_result.value,
                                  &context->spec, layer);
    if (status == BUDDY_QWEN3_OK)
      status = copy_rank3_contiguous(
          next, hidden_elements, &layer_result.hidden, 1,
          context->spec.prefill_len, context->spec.hidden_size);
    if (status == BUDDY_QWEN3_OK) {
      size_t layer_stride =
          (size_t)context->spec.kv_heads *
          (size_t)context->spec.max_cache_len *
          (size_t)context->spec.head_dim;
      checkpoint(context, "prefill_key", layer,
                 context->key_cache + (size_t)layer * layer_stride,
                 layer_stride);
      checkpoint(context, "prefill_value", layer,
                 context->value_cache + (size_t)layer * layer_stride,
                 layer_stride);
      checkpoint(context, "prefill_hidden", layer, next, hidden_elements);
    }
    reset_arena(context);
    if (status != BUDDY_QWEN3_OK)
      return status;
    {
      float *swap = current;
      current = next;
      next = swap;
    }
  }

  progress(context, "final_head", -1);
  status = context->ops->final_head(
      &context->params, &logits_result,
      current + ((size_t)context->spec.prefill_len - 1u) *
                    (size_t)context->spec.hidden_size);
  if (status != BUDDY_QWEN3_OK)
    return BUDDY_QWEN3_ERR_KERNEL;
  status =
      copy_rank3_contiguous(context->logits, (size_t)context->spec.vocab_size,
                            &logits_result, 1, 1, context->spec.vocab_size);
  if (status == BUDDY_QWEN3_OK)
    checkpoint(context, "prefill_logits", -1, context->logits,
               (size_t)context->spec.vocab_size);
  reset_arena(context);
  if (status == BUDDY_QWEN3_OK)
    progress(context, "prefill_done", -1);
  return status;
}

int buddy_qwen3_layerwise_decode(BuddyQwen3LayerwiseContext *context,
                                 const BuddyQwen3DecodeInputs *inputs) {
  BuddyQwen3MemRef3 embedding_result;
  BuddyQwen3LayerResult layer_result;
  BuddyQwen3MemRef3 logits_result;
  float *current;
  float *next;
  size_t layer_stride;
  int32_t cache_write_position;
  uint64_t cycle_start;
  int status = validate_context(context);
  if (status != BUDDY_QWEN3_OK || !inputs || !inputs->input_id ||
      !inputs->rope_cos || !inputs->rope_sin || !inputs->attention_mask ||
      !inputs->cache_write_mask)
    return status != BUDDY_QWEN3_OK ? status : BUDDY_QWEN3_ERR_ARGUMENT;
  status = find_cache_write_position(inputs->cache_write_mask,
                                     context->spec.max_cache_len,
                                     &cache_write_position);
  if (status != BUDDY_QWEN3_OK)
    return status;

  progress(context, "embedding_decode", -1);
  reset_arena(context);
  cycle_start = timing_begin(context);
  status = context->ops->embedding_decode(&context->params, &embedding_result,
                                          inputs->input_id);
  timing_end(context, "decode_embedding_kernel", -1, cycle_start);
  if (status != BUDDY_QWEN3_OK)
    return BUDDY_QWEN3_ERR_KERNEL;
  cycle_start = timing_begin(context);
  status = copy_rank3_contiguous(
      context->hidden_ping, (size_t)context->spec.hidden_size,
      &embedding_result, 1, 1, context->spec.hidden_size);
  timing_end(context, "decode_embedding_copy", -1, cycle_start);
  if (status == BUDDY_QWEN3_OK)
    checkpoint(context, "decode_embedding", -1, context->hidden_ping,
               (size_t)context->spec.hidden_size);
  reset_arena(context);
  if (status != BUDDY_QWEN3_OK)
    return status;

  layer_stride = (size_t)context->spec.kv_heads *
                 (size_t)context->spec.max_cache_len *
                 (size_t)context->spec.head_dim;
  current = context->hidden_ping;
  next = context->hidden_pong;
  for (int32_t layer = 0; layer < context->spec.num_layers; ++layer) {
    float *old_key = context->key_cache + (size_t)layer * layer_stride;
    float *old_value = context->value_cache + (size_t)layer * layer_stride;
    cycle_start = timing_begin(context);
    status = context->ops->decoder_layer_decode(
        &context->params, layer, &layer_result, current, inputs->rope_cos,
        inputs->rope_sin, inputs->attention_mask, inputs->cache_write_mask,
        old_key, old_value);
    timing_end(context, "decode_layer_kernel", layer, cycle_start);
    if (status != BUDDY_QWEN3_OK)
      return BUDDY_QWEN3_ERR_KERNEL;
    progress(context, "decoder_layer_decode", layer);
    cycle_start = timing_begin(context);
    status = copy_decode_cache_update(
        context->key_cache, &layer_result.key, &context->spec, layer,
        cache_write_position);
    if (status == BUDDY_QWEN3_OK)
      status = copy_decode_cache_update(
          context->value_cache, &layer_result.value, &context->spec, layer,
          cache_write_position);
    timing_end(context, "decode_cache_commit", layer, cycle_start);
    cycle_start = timing_begin(context);
    if (status == BUDDY_QWEN3_OK)
      status = copy_rank3_contiguous(next, (size_t)context->spec.hidden_size,
                                     &layer_result.hidden, 1, 1,
                                     context->spec.hidden_size);
    timing_end(context, "decode_hidden_copy", layer, cycle_start);
    if (status == BUDDY_QWEN3_OK) {
      checkpoint(context, "decode_key", layer, old_key, layer_stride);
      checkpoint(context, "decode_value", layer, old_value, layer_stride);
      checkpoint(context, "decode_hidden", layer, next,
                 (size_t)context->spec.hidden_size);
    }
    reset_arena(context);
    if (status != BUDDY_QWEN3_OK)
      return status;
    {
      float *swap = current;
      current = next;
      next = swap;
    }
  }

  progress(context, "final_head", -1);
  cycle_start = timing_begin(context);
  status = context->ops->final_head(&context->params, &logits_result, current);
  timing_end(context, "decode_final_head_kernel", -1, cycle_start);
  if (status != BUDDY_QWEN3_OK)
    return BUDDY_QWEN3_ERR_KERNEL;
  cycle_start = timing_begin(context);
  status =
      copy_rank3_contiguous(context->logits, (size_t)context->spec.vocab_size,
                            &logits_result, 1, 1, context->spec.vocab_size);
  timing_end(context, "decode_logits_copy", -1, cycle_start);
  if (status == BUDDY_QWEN3_OK)
    checkpoint(context, "decode_logits", -1, context->logits,
               (size_t)context->spec.vocab_size);
  reset_arena(context);
  if (status == BUDDY_QWEN3_OK)
    progress(context, "decode_done", -1);
  return status;
}

int buddy_qwen3_layerwise_generate_greedy(
    BuddyQwen3LayerwiseContext *context,
    const BuddyQwen3GreedyDecodeInputs *inputs, int32_t *generated_count) {
  BuddyQwen3DecodeInputs step_inputs;
  int64_t current_token;
  size_t required_rope_elements;
  int status = validate_context(context);

  if (generated_count)
    *generated_count = 0;
  if (status != BUDDY_QWEN3_OK)
    return status;
  if (!inputs || !generated_count || !inputs->rope_cos || !inputs->rope_sin ||
      !inputs->attention_mask_workspace ||
      !inputs->cache_write_mask_workspace || !inputs->output_ids ||
      inputs->decode_steps <= 0 || inputs->initial_token < 0 ||
      inputs->initial_token >= context->spec.vocab_size ||
      inputs->prompt_tokens <= 0 ||
      inputs->prompt_tokens > context->spec.prefill_len ||
      (inputs->eos_token_id >= context->spec.vocab_size))
    return BUDDY_QWEN3_ERR_ARGUMENT;
  if (inputs->decode_steps >
      context->spec.max_cache_len - context->spec.prefill_len)
    return BUDDY_QWEN3_ERR_CAPACITY;
  if (!checked_elements((size_t)inputs->decode_steps,
                        (size_t)context->spec.head_dim,
                        &required_rope_elements))
    return BUDDY_QWEN3_ERR_CAPACITY;
  if (inputs->rope_elements < required_rope_elements ||
      inputs->mask_capacity < (size_t)context->spec.max_cache_len ||
      inputs->output_capacity < (size_t)inputs->decode_steps)
    return BUDDY_QWEN3_ERR_CAPACITY;

  current_token = inputs->initial_token;
  step_inputs.input_id = &current_token;
  step_inputs.attention_mask = inputs->attention_mask_workspace;
  step_inputs.cache_write_mask = inputs->cache_write_mask_workspace;
  for (int32_t step = 0; step < inputs->decode_steps; ++step) {
    int32_t position = context->spec.prefill_len + step;
    int32_t masked_prefix =
        context->spec.prefill_len - inputs->prompt_tokens;
    size_t rope_offset = (size_t)step * (size_t)context->spec.head_dim;
    int32_t next_token;

    for (int32_t index = 0; index < context->spec.max_cache_len; ++index) {
      inputs->attention_mask_workspace[index] =
          index >= masked_prefix && index <= position
              ? 0.0f
              : -3.402823466e+38F;
      inputs->cache_write_mask_workspace[index] =
          index == position ? 1.0f : 0.0f;
    }
    step_inputs.rope_cos = inputs->rope_cos + rope_offset;
    step_inputs.rope_sin = inputs->rope_sin + rope_offset;
    progress(context, "greedy_decode_step", step);
    status = buddy_qwen3_layerwise_decode(context, &step_inputs);
    if (status != BUDDY_QWEN3_OK)
      return status;
    next_token = greedy_argmax(context->logits, context->spec.vocab_size);
    inputs->output_ids[step] = (int64_t)next_token;
    *generated_count = step + 1;
    current_token = next_token;
    if (inputs->eos_token_id >= 0 && next_token == inputs->eos_token_id)
      break;
  }
  return BUDDY_QWEN3_OK;
}
