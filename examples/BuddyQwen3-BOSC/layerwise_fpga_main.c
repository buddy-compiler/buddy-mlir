#include "bare_runtime.h"
#include "layerwise_scheduler.h"
#include "uart.h"

#include <stddef.h>
#include <stdint.h>

#ifndef QWEN3_PREFILL_LEN
#define QWEN3_PREFILL_LEN 22
#endif

#ifndef QWEN3_MAX_CACHE_LEN
#define QWEN3_MAX_CACHE_LEN 32
#endif

#ifndef QWEN3_ENABLE_CHECKPOINTS
#define QWEN3_ENABLE_CHECKPOINTS 0
#endif

#ifndef QWEN3_DECODE_STEPS
#define QWEN3_DECODE_STEPS 1
#endif

#ifndef QWEN3_EOS_TOKEN_ID
#define QWEN3_EOS_TOKEN_ID 151645
#endif

#define QWEN3_LAYERS 28
#define QWEN3_HIDDEN 1024
#define QWEN3_KV_HEADS 8
#define QWEN3_HEAD_DIM 128
#define QWEN3_VOCAB 151936
#define QWEN3_HIDDEN_ELEMENTS (QWEN3_PREFILL_LEN * QWEN3_HIDDEN)
#define QWEN3_CACHE_ELEMENTS                                                   \
  (QWEN3_LAYERS * QWEN3_KV_HEADS * QWEN3_MAX_CACHE_LEN * QWEN3_HEAD_DIM)

#if QWEN3_DECODE_STEPS <= 0
#error "QWEN3_DECODE_STEPS must be positive"
#endif
#if QWEN3_DECODE_STEPS > QWEN3_MAX_CACHE_LEN - QWEN3_PREFILL_LEN
#error "QWEN3_DECODE_STEPS exceeds the remaining KV-cache capacity"
#endif

extern uint8_t _buddy_qwen3_params_i8[];
extern float _buddy_qwen3_params_f32[];
extern int64_t _buddy_qwen3_prefill_ids[];
extern int32_t _buddy_qwen3_prompt_length[];
extern float _buddy_qwen3_prefill_rope_cos[];
extern float _buddy_qwen3_prefill_rope_sin[];
extern float _buddy_qwen3_prefill_mask[];
extern float _buddy_qwen3_decode_rope_cos[];
extern float _buddy_qwen3_decode_rope_sin[];

static float hidden_ping[QWEN3_HIDDEN_ELEMENTS] __attribute__((aligned(64)));
static float hidden_pong[QWEN3_HIDDEN_ELEMENTS] __attribute__((aligned(64)));
static float key_cache[QWEN3_CACHE_ELEMENTS] __attribute__((aligned(64)));
static float value_cache[QWEN3_CACHE_ELEMENTS] __attribute__((aligned(64)));
static float logits[QWEN3_VOCAB] __attribute__((aligned(64)));
static float decode_attention_mask[QWEN3_MAX_CACHE_LEN]
    __attribute__((aligned(64)));
static float decode_cache_write_mask[QWEN3_MAX_CACHE_LEN]
    __attribute__((aligned(64)));
static int64_t generated_tokens[QWEN3_DECODE_STEPS]
    __attribute__((aligned(64)));

static __attribute__((noinline)) void *runtime_address(void *pointer) {
  __asm__ volatile("" : "+r"(pointer));
  return pointer;
}

static inline uint64_t read_cycle_counter(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

static void print_timing(const char *stage, uint64_t cycles) {
  print_uart("[timing] ");
  print_uart(stage);
  print_uart(" cycles=0x");
  print_uart_addr(cycles);
  print_uart("\r\n");
}

static void clear_floats(float *data, size_t count) {
  for (size_t index = 0; index < count; ++index)
    data[index] = 0.0f;
}

static void scheduler_reset(void *opaque) {
  (void)opaque;
  bare_runtime_reset_heap();
}

static void scheduler_progress(void *opaque, const char *stage, int32_t layer) {
  (void)opaque;
  print_uart("scheduler: ");
  print_uart(stage);
  if (layer >= 0) {
    print_uart(" layer=");
    print_uart_int((uint32_t)layer);
  }
  print_uart("\r\n");
}

#if QWEN3_ENABLE_CHECKPOINTS
static uint32_t float_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } converted;
  converted.value = value;
  return converted.bits;
}

static uint64_t double_bits(double value) {
  union {
    double value;
    uint64_t bits;
  } converted;
  converted.value = value;
  return converted.bits;
}

static void scheduler_checkpoint(void *opaque, const char *stage,
                                 int32_t layer, const float *values,
                                 size_t count) {
  enum { SAMPLE_COUNT = 8 };
  double sum = 0.0;
  double absolute_sum = 0.0;
  float minimum;
  float maximum;
  uint32_t nonfinite = 0;
  (void)opaque;

  if (!values || count == 0)
    return;
  minimum = values[0];
  maximum = values[0];
  for (size_t index = 0; index < count; ++index) {
    float value = values[index];
    uint32_t bits = float_bits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u)
      ++nonfinite;
    if (value < minimum)
      minimum = value;
    if (value > maximum)
      maximum = value;
    sum += (double)value;
    absolute_sum += value < 0.0f ? -(double)value : (double)value;
  }

  print_uart("CKPT stage=");
  print_uart(stage);
  print_uart(" layer=");
  print_uart_int((uint32_t)layer);
  print_uart(" count=");
  print_uart_int((uint32_t)count);
  print_uart(" nonfinite=");
  print_uart_int(nonfinite);
  print_uart(" min=");
  print_uart_int(float_bits(minimum));
  print_uart(" max=");
  print_uart_int(float_bits(maximum));
  print_uart(" sum=");
  print_uart_addr(double_bits(sum));
  print_uart(" abs=");
  print_uart_addr(double_bits(absolute_sum));
  print_uart(" samples=");
  for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
    size_t index = sample * (count - 1u) / (SAMPLE_COUNT - 1u);
    if (sample != 0)
      write_serial(',');
    print_uart_int(float_bits(values[index]));
  }
  print_uart("\r\n");
}
#endif

static int32_t argmax(const float *values, int32_t count) {
  int32_t best = 0;
  for (int32_t index = 1; index < count; ++index)
    if (values[index] > values[best])
      best = index;
  return best;
}

static void print_status(const char *stage, int status) {
  print_uart(stage);
  print_uart(status == BUDDY_QWEN3_OK ? ": PASS\r\n" : ": FAIL code=");
  if (status != BUDDY_QWEN3_OK) {
    print_uart_int((uint32_t)(-status));
    print_uart("\r\n");
  }
}

int main(void) {
  BuddyQwen3LayerwiseContext context;
  BuddyQwen3LayerwiseOps generated_ops;
  BuddyQwen3PrefillInputs prefill;
  BuddyQwen3GreedyDecodeInputs generation;
  int32_t generated_count;
  int32_t prompt_tokens;
  int status;
  int32_t token;
  uint64_t cycle_start;

  print_uart("\r\n========================================\r\n");
  print_uart("  Buddy Qwen3-0.6B Layer-wise W8A8\r\n");
  print_uart("  Variable Prompt Prefill + Greedy Decode\r\n");
  print_uart("========================================\r\n");

  if (buddy_qwen3_generated_spec.prefill_len != QWEN3_PREFILL_LEN ||
      buddy_qwen3_generated_spec.max_cache_len != QWEN3_MAX_CACHE_LEN ||
      buddy_qwen3_generated_spec.num_layers != QWEN3_LAYERS ||
      buddy_qwen3_generated_spec.hidden_size != QWEN3_HIDDEN ||
      buddy_qwen3_generated_spec.kv_heads != QWEN3_KV_HEADS ||
      buddy_qwen3_generated_spec.head_dim != QWEN3_HEAD_DIM ||
      buddy_qwen3_generated_spec.vocab_size != QWEN3_VOCAB) {
    print_uart("scheduler config: FAIL generated ABI mismatch\r\n");
    return 1;
  }

  print_uart("runtime params_i8=");
  print_uart_addr((uint64_t)(uintptr_t)runtime_address(
      _buddy_qwen3_params_i8));
  print_uart(" params_f32=");
  print_uart_addr((uint64_t)(uintptr_t)runtime_address(
      _buddy_qwen3_params_f32));
  print_uart("\r\nruntime prompt=");
  print_uart_int((uint32_t)_buddy_qwen3_prefill_ids[0]);
  print_uart(" ... ");
  print_uart_int((uint32_t)_buddy_qwen3_prefill_ids[QWEN3_PREFILL_LEN - 1]);
  print_uart("\r\n");
  prompt_tokens = *(const int32_t *)runtime_address(
      _buddy_qwen3_prompt_length);
  if (prompt_tokens <= 0 || prompt_tokens > QWEN3_PREFILL_LEN) {
    print_uart("runtime prompt length: FAIL\r\n");
    return 1;
  }
  print_uart("runtime prompt tokens=");
  print_uart_int((uint32_t)prompt_tokens);
  print_uart(" padded prefill=");
  print_uart_int(QWEN3_PREFILL_LEN);
  print_uart("\r\n");

  clear_floats(key_cache, QWEN3_CACHE_ELEMENTS);
  clear_floats(value_cache, QWEN3_CACHE_ELEMENTS);

  context.spec = buddy_qwen3_generated_spec;
  context.params.params_i8 =
      (const uint8_t *)runtime_address(_buddy_qwen3_params_i8);
  context.params.params_f32 =
      (const float *)runtime_address(_buddy_qwen3_params_f32);
  buddy_qwen3_initialize_generated_ops(&generated_ops);
  context.ops = &generated_ops;
  context.hidden_ping = hidden_ping;
  context.hidden_pong = hidden_pong;
  context.hidden_capacity = QWEN3_HIDDEN_ELEMENTS;
  context.key_cache = key_cache;
  context.value_cache = value_cache;
  context.cache_capacity = QWEN3_CACHE_ELEMENTS;
  context.logits = logits;
  context.logits_capacity = QWEN3_VOCAB;
  context.reset_arena = scheduler_reset;
  context.arena_opaque = NULL;
  context.progress = scheduler_progress;
  context.progress_opaque = NULL;
#if QWEN3_ENABLE_CHECKPOINTS
  context.checkpoint = scheduler_checkpoint;
#else
  context.checkpoint = NULL;
#endif
  context.checkpoint_opaque = NULL;
  context.profiler = NULL;

  prefill.input_ids =
      (const int64_t *)runtime_address(_buddy_qwen3_prefill_ids);
  prefill.rope_cos =
      (const float *)runtime_address(_buddy_qwen3_prefill_rope_cos);
  prefill.rope_sin =
      (const float *)runtime_address(_buddy_qwen3_prefill_rope_sin);
  prefill.attention_mask =
      (const float *)runtime_address(_buddy_qwen3_prefill_mask);
  prefill.valid_tokens = prompt_tokens;

  bare_trace_reset();
  cycle_start = read_cycle_counter();
  status = buddy_qwen3_layerwise_prefill(&context, &prefill);
  print_timing("prefill", read_cycle_counter() - cycle_start);
  bare_trace_print();
  print_status("verify layerwise prefill", status);
  if (status != BUDDY_QWEN3_OK)
    return 1;
  token = argmax(logits, QWEN3_VOCAB);
  print_uart("prefill argmax token=");
  print_uart_int((uint32_t)token);
  print_uart("\r\n");

  generation.initial_token = token;
  generation.rope_cos =
      (const float *)runtime_address(_buddy_qwen3_decode_rope_cos);
  generation.rope_sin =
      (const float *)runtime_address(_buddy_qwen3_decode_rope_sin);
  generation.rope_elements =
      (size_t)QWEN3_DECODE_STEPS * (size_t)QWEN3_HEAD_DIM;
  generation.attention_mask_workspace = decode_attention_mask;
  generation.cache_write_mask_workspace = decode_cache_write_mask;
  generation.mask_capacity = QWEN3_MAX_CACHE_LEN;
  generation.output_ids = generated_tokens;
  generation.output_capacity = QWEN3_DECODE_STEPS;
  generation.decode_steps = QWEN3_DECODE_STEPS;
  generation.prompt_tokens = prompt_tokens;
  generation.eos_token_id = QWEN3_EOS_TOKEN_ID;

  bare_trace_reset();
  cycle_start = read_cycle_counter();
  status = buddy_qwen3_layerwise_generate_greedy(
      &context, &generation, &generated_count);
  print_timing("decode", read_cycle_counter() - cycle_start);
  bare_trace_print();
  print_status("verify layerwise decode", status);
  if (status != BUDDY_QWEN3_OK)
    return 1;
  print_uart("generated token[0]=");
  print_uart_int((uint32_t)token);
  print_uart("\r\n");
  for (int32_t step = 0; step < generated_count; ++step) {
    print_uart("generated token[");
    print_uart_int((uint32_t)(step + 1));
    print_uart("]=");
    print_uart_int((uint32_t)generated_tokens[step]);
    print_uart(" position=");
    print_uart_int((uint32_t)(QWEN3_PREFILL_LEN + step));
    print_uart("\r\n");
  }
  token = (int32_t)generated_tokens[generated_count - 1];
  print_uart("decode argmax token=");
  print_uart_int((uint32_t)token);
  print_uart("\r\n=== Buddy Qwen3 Layer-wise Done ===\r\n");
  return 0;
}
