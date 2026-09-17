#include "support.h"
#include "nr_runtime.h"
extern void qwen_intermediate_begin(unsigned, unsigned);
extern int qwen_intermediate_end(void);
#include "tokenizer_resource.h"
#define LAYERS 1
#define CAPACITY 512
#define KV_ELEMENTS 524288
#define VOCAB 151936
typedef struct { MemRef1 position; MemRef4 key, value; } CacheResult;
typedef struct { CacheResult cache[LAYERS]; MemRef3 logits; } GraphResults;
_Static_assert(sizeof(CacheResult) == 216, "cache result ABI");
_Static_assert(sizeof(GraphResults) == LAYERS * 216 + 72, "graph result ABI");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "weight_arena_raw:\n"
        ".skip 171981568\n"
        "weight_arena_end:\n"
        ".previous\n");
extern unsigned char weight_arena_raw[] __asm__("weight_arena_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "k_cache_raw:\n"
        ".skip 2097152\n"
        "k_cache_end:\n"
        ".previous\n");
extern unsigned char k_cache_raw[] __asm__("k_cache_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "v_cache_raw:\n"
        ".skip 2097152\n"
        "v_cache_end:\n"
        ".previous\n");
extern unsigned char v_cache_raw[] __asm__("v_cache_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "input_ids_raw:\n"
        ".skip 128\n"
        "input_ids_end:\n"
        ".previous\n");
extern unsigned char input_ids_raw[] __asm__("input_ids_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "cache_position_raw:\n"
        ".skip 64\n"
        "cache_position_end:\n"
        ".previous\n");
extern unsigned char cache_position_raw[] __asm__("cache_position_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "tokenizer_blob_raw:\n"
        ".skip 5222976\n"
        "tokenizer_blob_end:\n"
        ".previous\n");
extern unsigned char tokenizer_blob_raw[] __asm__("tokenizer_blob_raw");
static float *const k_cache_f = (float *)k_cache_raw;
static float *const v_cache_f = (float *)v_cache_raw;
static int64_t *const input_ids = (int64_t *)input_ids_raw;
static int64_t *const cache_position = (int64_t *)cache_position_raw;
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "ws_prefill_raw:\n"
        ".skip 4856384\n"
        "ws_prefill_end:\n"
        ".previous\n");
extern unsigned char ws_prefill_raw[] __asm__("ws_prefill_raw");
extern void _mlir_ciface_forward_prefill(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "ws_decode_raw:\n"
        ".skip 1444544\n"
        "ws_decode_end:\n"
        ".previous\n");
extern unsigned char ws_decode_raw[] __asm__("ws_decode_raw");
extern void _mlir_ciface_forward_decode(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
#define REF_LENGTH 24
#define REF_LOGITS_ROWS 9
extern const float model_reference_raw[];
static int compare_reference(const GraphResults *, unsigned, const float *, int64_t);

static void zero_bytes(void *p, size_t n) { memset(p, 0, n); }
/* Returned buffers may alias inputs or belong to the graph's scoped heap.
 * Preserve every layer before releasing that scope. Honor offsets/strides. */
static int retain_cache(float *dst, const MemRef4 *m) {
  if (!m->aligned || m->sizes[0] != 1 || m->sizes[1] != 8 ||
      m->sizes[2] != CAPACITY || m->sizes[3] != 128) return -1;
  const float *src = (const float *)m->aligned + m->offset;
  if (m->strides[3] == 1 && m->strides[2] == 128 &&
      m->strides[1] == CAPACITY * 128) {
    if (src != dst) nr_copy_bytes(dst, src, KV_ELEMENTS * sizeof(float));
  } else {
    for (unsigned h = 0; h < 8; ++h)
      for (unsigned t = 0; t < CAPACITY; ++t)
        for (unsigned d = 0; d < 128; ++d)
          dst[(h * CAPACITY + t) * 128 + d] =
              src[h * m->strides[1] + t * m->strides[2] + d * m->strides[3]];
  }
  return 0;
}
static void float_bits(float f) {
  union { float f; uint32_t u; } bits = {f}; nr_hex32(bits.u);
}
static uint64_t selection_cycles, cache_retention_cycles;
static int collect(GraphResults *r, unsigned position, unsigned *token,
                   float *score, int trace) {
  uint64_t selection_begin = nr_cycles();
  if (!r->logits.aligned || r->logits.sizes[0] != 1 ||
      r->logits.sizes[1] < 1 || r->logits.sizes[2] != VOCAB) return -1;
  const float *scores = (const float *)r->logits.aligned + r->logits.offset +
                        (r->logits.sizes[1]-1) * r->logits.strides[1];
  *token = 0; *score = scores[0];
  for (unsigned i = 0; i < VOCAB; ++i) {
    float value = scores[i * r->logits.strides[2]];
    if (!(value <= 3.402823466e38f && value >= -3.402823466e38f)) return -1;
    if (value > *score) { *score = value; *token = i; }
  }
  uint64_t retention_begin = nr_cycles();
  selection_cycles = retention_begin - selection_begin;
  for (unsigned l = 0; l < LAYERS; ++l) {
    if (retain_cache(k_cache_f + l * KV_ELEMENTS, &r->cache[l].key) ||
        retain_cache(v_cache_f + l * KV_ELEMENTS, &r->cache[l].value)) return -1;
  }
  cache_retention_cycles = nr_cycles() - retention_begin;
#ifdef REF_LENGTH
  int numeric_status = compare_reference(r, position, scores, r->logits.strides[2]);
#else
  int numeric_status = 0;
#endif
  if (trace) {
    nr_puts("[model] selected position="); nr_hex32(position);
    static const unsigned ids[] = {49000,374,264,3146,7407,304,4787,5159,11};
    for (unsigned i = 0; i < sizeof(ids)/sizeof(ids[0]); ++i) {
      nr_puts(" "); nr_hex32(ids[i]); nr_puts(":");
      float_bits(scores[ids[i] * r->logits.strides[2]]);
    }
    nr_puts("\r\n");
    for (unsigned l = 0; l < LAYERS; ++l) {
      nr_puts("[model] kv position="); nr_hex32(position);
      nr_puts(" layer="); nr_hex32(l);
      nr_puts(" k="); float_bits(k_cache_f[l * KV_ELEMENTS + position * 128]);
      nr_puts(" v="); float_bits(v_cache_f[l * KV_ELEMENTS + position * 128]);
      nr_puts("\r\n");
    }
  }
  return numeric_status;
}
static void reset_cache(void) {
  zero_bytes(k_cache_f, LAYERS * KV_ELEMENTS * sizeof(float));
  zero_bytes(v_cache_f, LAYERS * KV_ELEMENTS * sizeof(float));
}

static const float max_atol = 1.000000000e-03f;
static const float mean_atol = 1.000000000e-04f;

static int report_error(const char *name, unsigned position, float max, float sum,
                        unsigned count) {
  float mean = sum / (float)count;
  nr_puts("[compare] "); nr_puts(name); nr_puts(" position="); nr_hex32(position);
  nr_puts(" count="); nr_hex32(count);
  nr_puts(" max_abs_bits="); float_bits(max);
  nr_puts(" mean_abs_bits="); float_bits(mean);
  int ok = max <= max_atol && mean <= mean_atol;
  nr_puts(ok ? " PASS\r\n" : " FAIL\r\n");
  return ok ? 0 : -1;
}
static int compare_reference(const GraphResults *r, unsigned position,
                             const float *scores, int64_t stride) {
  (void)r;

  unsigned row = position - 15;

  if (row >= REF_LOGITS_ROWS) return -1;
  float max = 0, sum = 0;
  for (unsigned i = 0; i < VOCAB; ++i) {
    float diff = scores[i * stride] - model_reference_raw[row * VOCAB + i];
    if (diff < 0) diff = -diff;
    if (!(diff <= 3.402823466e38f)) return -1;
    if (diff > max) max = diff; sum += diff;
  }
  int status = report_error("logits", position, max, sum, VOCAB);
  for (unsigned which = 0; which < 2; ++which) {
    const float *gold = model_reference_raw + REF_LOGITS_ROWS * VOCAB +
                        which * LAYERS * 8 * REF_LENGTH * 128;
    const float *actual = which ? v_cache_f : k_cache_f;
    max = 0; sum = 0;
    for (unsigned l = 0; l < LAYERS; ++l)
      for (unsigned h = 0; h < 8; ++h)
        for (unsigned t = 0; t <= position; ++t)
          for (unsigned d = 0; d < 128; ++d) {
            float diff = actual[(l*8+h)*CAPACITY*128+t*128+d] -
                         gold[(l*8+h)*REF_LENGTH*128+t*128+d];
            if (diff < 0) diff = -diff;
            if (!(diff <= 3.402823466e38f)) return -1;
            if (diff > max) max = diff; sum += diff;
          }
    status |= report_error(which ? "value_cache" : "key_cache", position, max, sum,
                           LAYERS * 8 * (position+1) * 128);
  }
  return status;
}

static int run_prefill(unsigned position, unsigned *token,
                      float *score, int trace) {
  if (position + 16 > CAPACITY) return -1;
  uint64_t preparation_begin = nr_cycles();
  *cache_position = position;
  uintptr_t mark = nr_heap_mark();
  GraphResults result;
  MemRef2 ids = make_2(input_ids, 1, 16);
  MemRef1 pos = make_1(cache_position, 1);
  MemRef4 k[LAYERS], v[LAYERS];
  for (unsigned l = 0; l < LAYERS; ++l) {
    k[l] = make_4(k_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);
    v[l] = make_4(v_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);
  }
  MemRef1 p0 = make_1(weight_arena_raw + 0, 1024);
  MemRef1 p1 = make_1(weight_arena_raw + 4096, 128);
  MemRef1 p2 = make_1(weight_arena_raw + 4608, 128);
  MemRef1 p3 = make_1(weight_arena_raw + 5120, 1024);
  MemRef1 p4 = make_1(weight_arena_raw + 9216, 1024);
  MemRef1 p5 = make_1(weight_arena_raw + 13312, 64);
  MemRef2 p6 = make_2(weight_arena_raw + 13568, 151936, 1024);
  MemRef1 p7 = make_1(weight_arena_raw + 155596032, 151936);
  MemRef2 p8 = make_2(weight_arena_raw + 156203776, 1024, 3072);
  MemRef1 p9 = make_1(weight_arena_raw + 159349504, 1024);
  MemRef2 p10 = make_2(weight_arena_raw + 159353600, 3072, 1024);
  MemRef1 p11 = make_1(weight_arena_raw + 162499328, 3072);
  MemRef2 p12 = make_2(weight_arena_raw + 162511616, 3072, 1024);
  MemRef1 p13 = make_1(weight_arena_raw + 165657344, 3072);
  MemRef2 p14 = make_2(weight_arena_raw + 165669632, 1024, 2048);
  MemRef1 p15 = make_1(weight_arena_raw + 167766784, 1024);
  MemRef2 p16 = make_2(weight_arena_raw + 167770880, 1024, 1024);
  MemRef1 p17 = make_1(weight_arena_raw + 168819456, 1024);
  MemRef2 p18 = make_2(weight_arena_raw + 168823552, 1024, 1024);
  MemRef1 p19 = make_1(weight_arena_raw + 169872128, 1024);
  MemRef2 p20 = make_2(weight_arena_raw + 169876224, 2048, 1024);
  MemRef1 p21 = make_1(weight_arena_raw + 171973376, 2048);
  MemRef4 w0 = make_4(ws_prefill_raw + 0, 1, 16, 16, 512);
  MemRef4 w1 = make_4(ws_prefill_raw + 524288, 1, 16, 16, 512);
  MemRef1 w2 = make_1(ws_prefill_raw + 1048576, 256);
  MemRef1 w3 = make_1(ws_prefill_raw + 1049600, 256);
  MemRef4 w4 = make_4(ws_prefill_raw + 1050624, 1, 16, 16, 512);
  MemRef4 w5 = make_4(ws_prefill_raw + 1574912, 1, 16, 16, 128);
  MemRef1 w6 = make_1(ws_prefill_raw + 1705984, 16);
  for (unsigned j = 0; j < 16; ++j) ((int32_t *)(ws_prefill_raw + 1705984))[j] = position + j;
  MemRef4 w7 = make_4(ws_prefill_raw + 1706048, 1, 8, 16, 128);
  MemRef4 w8 = make_4(ws_prefill_raw + 1771584, 1, 8, 16, 128);
  MemRef3 w9 = make_3(ws_prefill_raw + 1837120, 1, 16, 1024);
  MemRef2 w10 = make_2(ws_prefill_raw + 1902656, 16, 1024);
  MemRef1 w11 = make_1(ws_prefill_raw + 1919040, 16);
  MemRef2 w12 = make_2(ws_prefill_raw + 1919104, 16, 2048);
  zero_bytes(ws_prefill_raw + 1919104, 131072);
  MemRef2 w13 = make_2(ws_prefill_raw + 2050176, 16, 2048);
  MemRef2 w14 = make_2(ws_prefill_raw + 2181248, 16, 1024);
  MemRef1 w15 = make_1(ws_prefill_raw + 2197632, 16);
  MemRef2 w16 = make_2(ws_prefill_raw + 2197696, 16, 1024);
  zero_bytes(ws_prefill_raw + 2197696, 65536);
  MemRef2 w17 = make_2(ws_prefill_raw + 2263232, 16, 1024);
  MemRef2 w18 = make_2(ws_prefill_raw + 2328768, 16, 1024);
  MemRef1 w19 = make_1(ws_prefill_raw + 2345152, 16);
  MemRef2 w20 = make_2(ws_prefill_raw + 2345216, 16, 1024);
  zero_bytes(ws_prefill_raw + 2345216, 65536);
  MemRef2 w21 = make_2(ws_prefill_raw + 2410752, 16, 1024);
  MemRef2 w22 = make_2(ws_prefill_raw + 2476288, 16, 2048);
  MemRef1 w23 = make_1(ws_prefill_raw + 2509056, 16);
  MemRef2 w24 = make_2(ws_prefill_raw + 2509120, 16, 1024);
  zero_bytes(ws_prefill_raw + 2509120, 65536);
  MemRef2 w25 = make_2(ws_prefill_raw + 2574656, 16, 1024);
  MemRef2 w26 = make_2(ws_prefill_raw + 2640192, 16, 1024);
  MemRef1 w27 = make_1(ws_prefill_raw + 2656576, 16);
  MemRef2 w28 = make_2(ws_prefill_raw + 2656640, 16, 3072);
  zero_bytes(ws_prefill_raw + 2656640, 196608);
  MemRef2 w29 = make_2(ws_prefill_raw + 2853248, 16, 3072);
  MemRef2 w30 = make_2(ws_prefill_raw + 3049856, 16, 1024);
  MemRef1 w31 = make_1(ws_prefill_raw + 3066240, 16);
  MemRef2 w32 = make_2(ws_prefill_raw + 3066304, 16, 3072);
  zero_bytes(ws_prefill_raw + 3066304, 196608);
  MemRef2 w33 = make_2(ws_prefill_raw + 3262912, 16, 3072);
  MemRef2 w34 = make_2(ws_prefill_raw + 3459520, 16, 3072);
  MemRef1 w35 = make_1(ws_prefill_raw + 3508672, 16);
  MemRef2 w36 = make_2(ws_prefill_raw + 3508736, 16, 1024);
  zero_bytes(ws_prefill_raw + 3508736, 65536);
  MemRef2 w37 = make_2(ws_prefill_raw + 3574272, 16, 1024);
  MemRef2 w38 = make_2(ws_prefill_raw + 3639808, 1, 1024);
  MemRef1 w39 = make_1(ws_prefill_raw + 3640832, 1);
  MemRef2 w40 = make_2(ws_prefill_raw + 3640896, 1, 151936);
  zero_bytes(ws_prefill_raw + 3640896, 607744);
  MemRef2 w41 = make_2(ws_prefill_raw + 4248640, 1, 151936);
  uint64_t preparation_cycles = nr_cycles() - preparation_begin;
  if (trace) { nr_puts("[model] prefill begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_intermediate_begin(position, 16);
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_prefill(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &ids, &pos, &k[0], &v[0], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41);
  ame_fence();
  uint64_t compute = nr_cycles() - begin;
  int status = collect(&result, position + 16 - 1, token, score, trace);
  uintptr_t peak = nr_heap_mark();
  nr_heap_reset(mark);
  if (trace) { nr_puts("[model] prefill position="); nr_hex32(position);
    nr_puts(" token="); nr_hex32(*token);
    nr_puts(" logit_bits="); float_bits(*score);
    nr_puts(" compute_cycles="); nr_hex64(compute);
    nr_puts(" preparation_cycles="); nr_hex64(preparation_cycles);
    nr_puts(" selection_cycles="); nr_hex64(selection_cycles);
    nr_puts(" cache_retention_cycles="); nr_hex64(cache_retention_cycles);
    nr_puts(" model_cycles=");
    nr_hex64(preparation_cycles + compute + selection_cycles + cache_retention_cycles);
    nr_puts(" total_with_uart_cycles="); nr_hex64(nr_cycles() - begin);
    nr_puts(" scratch_bytes="); nr_hex64(peak - mark);
    nr_puts("\r\n"); }
  status |= qwen_intermediate_end();
  return status;
}
static int run_decode(unsigned position, unsigned *token,
                      float *score, int trace) {
  if (position + 1 > CAPACITY) return -1;
  uint64_t preparation_begin = nr_cycles();
  *cache_position = position;
  uintptr_t mark = nr_heap_mark();
  GraphResults result;
  MemRef2 ids = make_2(input_ids, 1, 1);
  MemRef1 pos = make_1(cache_position, 1);
  MemRef4 k[LAYERS], v[LAYERS];
  for (unsigned l = 0; l < LAYERS; ++l) {
    k[l] = make_4(k_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);
    v[l] = make_4(v_cache_f + l * KV_ELEMENTS, 1, 8, CAPACITY, 128);
  }
  MemRef1 p0 = make_1(weight_arena_raw + 0, 1024);
  MemRef1 p1 = make_1(weight_arena_raw + 4096, 128);
  MemRef1 p2 = make_1(weight_arena_raw + 4608, 128);
  MemRef1 p3 = make_1(weight_arena_raw + 5120, 1024);
  MemRef1 p4 = make_1(weight_arena_raw + 9216, 1024);
  MemRef1 p5 = make_1(weight_arena_raw + 13312, 64);
  MemRef2 p6 = make_2(weight_arena_raw + 13568, 151936, 1024);
  MemRef1 p7 = make_1(weight_arena_raw + 155596032, 151936);
  MemRef2 p8 = make_2(weight_arena_raw + 156203776, 1024, 3072);
  MemRef1 p9 = make_1(weight_arena_raw + 159349504, 1024);
  MemRef2 p10 = make_2(weight_arena_raw + 159353600, 3072, 1024);
  MemRef1 p11 = make_1(weight_arena_raw + 162499328, 3072);
  MemRef2 p12 = make_2(weight_arena_raw + 162511616, 3072, 1024);
  MemRef1 p13 = make_1(weight_arena_raw + 165657344, 3072);
  MemRef2 p14 = make_2(weight_arena_raw + 165669632, 1024, 2048);
  MemRef1 p15 = make_1(weight_arena_raw + 167766784, 1024);
  MemRef2 p16 = make_2(weight_arena_raw + 167770880, 1024, 1024);
  MemRef1 p17 = make_1(weight_arena_raw + 168819456, 1024);
  MemRef2 p18 = make_2(weight_arena_raw + 168823552, 1024, 1024);
  MemRef1 p19 = make_1(weight_arena_raw + 169872128, 1024);
  MemRef2 p20 = make_2(weight_arena_raw + 169876224, 2048, 1024);
  MemRef1 p21 = make_1(weight_arena_raw + 171973376, 2048);
  MemRef4 w0 = make_4(ws_decode_raw + 0, 1, 16, 1, 512);
  MemRef4 w1 = make_4(ws_decode_raw + 32768, 1, 16, 1, 512);
  MemRef1 w2 = make_1(ws_decode_raw + 65536, 16);
  MemRef1 w3 = make_1(ws_decode_raw + 65600, 16);
  MemRef4 w4 = make_4(ws_decode_raw + 65664, 1, 16, 1, 512);
  MemRef4 w5 = make_4(ws_decode_raw + 98432, 1, 16, 1, 128);
  MemRef1 w6 = make_1(ws_decode_raw + 106624, 1);
  for (unsigned j = 0; j < 1; ++j) ((int32_t *)(ws_decode_raw + 106624))[j] = position + j;
  MemRef4 w7 = make_4(ws_decode_raw + 106688, 1, 8, 1, 128);
  MemRef4 w8 = make_4(ws_decode_raw + 110784, 1, 8, 1, 128);
  MemRef3 w9 = make_3(ws_decode_raw + 114880, 1, 1, 1024);
  MemRef2 w10 = make_2(ws_decode_raw + 118976, 1, 1024);
  MemRef1 w11 = make_1(ws_decode_raw + 120000, 1);
  MemRef2 w12 = make_2(ws_decode_raw + 120064, 1, 2048);
  zero_bytes(ws_decode_raw + 120064, 8192);
  MemRef2 w13 = make_2(ws_decode_raw + 128256, 1, 2048);
  MemRef2 w14 = make_2(ws_decode_raw + 136448, 1, 1024);
  MemRef1 w15 = make_1(ws_decode_raw + 137472, 1);
  MemRef2 w16 = make_2(ws_decode_raw + 137536, 1, 1024);
  zero_bytes(ws_decode_raw + 137536, 4096);
  MemRef2 w17 = make_2(ws_decode_raw + 141632, 1, 1024);
  MemRef2 w18 = make_2(ws_decode_raw + 145728, 1, 1024);
  MemRef1 w19 = make_1(ws_decode_raw + 146752, 1);
  MemRef2 w20 = make_2(ws_decode_raw + 146816, 1, 1024);
  zero_bytes(ws_decode_raw + 146816, 4096);
  MemRef2 w21 = make_2(ws_decode_raw + 150912, 1, 1024);
  MemRef2 w22 = make_2(ws_decode_raw + 155008, 1, 2048);
  MemRef1 w23 = make_1(ws_decode_raw + 157056, 1);
  MemRef2 w24 = make_2(ws_decode_raw + 157120, 1, 1024);
  zero_bytes(ws_decode_raw + 157120, 4096);
  MemRef2 w25 = make_2(ws_decode_raw + 161216, 1, 1024);
  MemRef2 w26 = make_2(ws_decode_raw + 165312, 1, 1024);
  MemRef1 w27 = make_1(ws_decode_raw + 166336, 1);
  MemRef2 w28 = make_2(ws_decode_raw + 166400, 1, 3072);
  zero_bytes(ws_decode_raw + 166400, 12288);
  MemRef2 w29 = make_2(ws_decode_raw + 178688, 1, 3072);
  MemRef2 w30 = make_2(ws_decode_raw + 190976, 1, 1024);
  MemRef1 w31 = make_1(ws_decode_raw + 192000, 1);
  MemRef2 w32 = make_2(ws_decode_raw + 192064, 1, 3072);
  zero_bytes(ws_decode_raw + 192064, 12288);
  MemRef2 w33 = make_2(ws_decode_raw + 204352, 1, 3072);
  MemRef2 w34 = make_2(ws_decode_raw + 216640, 1, 3072);
  MemRef1 w35 = make_1(ws_decode_raw + 219712, 1);
  MemRef2 w36 = make_2(ws_decode_raw + 219776, 1, 1024);
  zero_bytes(ws_decode_raw + 219776, 4096);
  MemRef2 w37 = make_2(ws_decode_raw + 223872, 1, 1024);
  MemRef2 w38 = make_2(ws_decode_raw + 227968, 1, 1024);
  MemRef1 w39 = make_1(ws_decode_raw + 228992, 1);
  MemRef2 w40 = make_2(ws_decode_raw + 229056, 1, 151936);
  zero_bytes(ws_decode_raw + 229056, 607744);
  MemRef2 w41 = make_2(ws_decode_raw + 836800, 1, 151936);
  uint64_t preparation_cycles = nr_cycles() - preparation_begin;
  if (trace) { nr_puts("[model] decode begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_intermediate_begin(position, 1);
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_decode(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &ids, &pos, &k[0], &v[0], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41);
  ame_fence();
  uint64_t compute = nr_cycles() - begin;
  int status = collect(&result, position + 1 - 1, token, score, trace);
  uintptr_t peak = nr_heap_mark();
  nr_heap_reset(mark);
  if (trace) { nr_puts("[model] decode position="); nr_hex32(position);
    nr_puts(" token="); nr_hex32(*token);
    nr_puts(" logit_bits="); float_bits(*score);
    nr_puts(" compute_cycles="); nr_hex64(compute);
    nr_puts(" preparation_cycles="); nr_hex64(preparation_cycles);
    nr_puts(" selection_cycles="); nr_hex64(selection_cycles);
    nr_puts(" cache_retention_cycles="); nr_hex64(cache_retention_cycles);
    nr_puts(" model_cycles=");
    nr_hex64(preparation_cycles + compute + selection_cycles + cache_retention_cycles);
    nr_puts(" total_with_uart_cycles="); nr_hex64(nr_cycles() - begin);
    nr_puts(" scratch_bytes="); nr_hex64(peak - mark);
    nr_puts("\r\n"); }
  status |= qwen_intermediate_end();
  return status;
}
static const uint8_t fixed_user_text[] = {87,104,97,116,32,105,115,32,70,114,97,110,99,101,63};
static const uint32_t expected_prompt_ids[] = {151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271};
#define TEXT_OUTPUT_CAPACITY 65536

typedef struct { uint8_t data[TEXT_OUTPUT_CAPACITY]; size_t length; int overflow; } TextOutput;
static TextOutput text_output;
static void append_text(void *context, const uint8_t *bytes, size_t count) {
  TextOutput *out = (TextOutput *)context;
  if (count > sizeof(out->data) - out->length) { out->overflow = 1; return; }
  memcpy(out->data + out->length, bytes, count);
  out->length += count;
}
static void text_hex(const uint8_t *bytes, size_t count) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < count; ++i) {
    char pair[3] = {digits[bytes[i] >> 4], digits[bytes[i] & 15], 0};
    nr_puts(pair);
  }
}
static int record_text_token(const QwenTokenizerResource *resource,
                             QwenUtf8Decoder *utf8, unsigned step, unsigned token) {
  size_t previous = text_output.length;
  if (qwen_decode_token(resource, utf8, token, 1, append_text, &text_output) ||
      text_output.overflow) {
    nr_puts("[text] decode or output capacity: FAIL\r\n"); return 1;
  }
  nr_puts("[text] prediction="); nr_hex32(step);
  nr_puts(" token="); nr_hex32(token);
  nr_puts(" eos="); nr_hex32(token == 151643 || token == 151645);
  nr_puts(" bytes="); nr_hex32(text_output.length - previous);
  nr_puts(" hex="); text_hex(text_output.data + previous, text_output.length - previous);
  nr_puts("\r\n");
  return 0;
}
int launch(void) {
  QwenTokenizerResource resource;

  if (qwen_tokenizer_open(&resource, tokenizer_blob_raw, 5222928)) {
    nr_puts("[text] tokenizer resource: FAIL\r\n"); return 1; }

  static uint8_t prompt[8192];
  static uint32_t encoded_ids[2048];
  size_t bytes = 0, count = 0;
  uint64_t encode_begin = nr_cycles();
  if (qwen_chat_single_turn(prompt, sizeof(prompt), &bytes, 0, 0, 0,
                            fixed_user_text, sizeof(fixed_user_text), 0) ||
      qwen_encode(&resource, prompt, bytes, encoded_ids, 2048, &count)) {
    nr_puts("[text] template or tokenizer: FAIL\r\n"); return 1;
  }
  uint64_t encode_cycles = nr_cycles() - encode_begin;
  nr_puts("[text] prompt count="); nr_hex32(count); nr_puts(" ids=");
  for (size_t i = 0; i < count; ++i) {
    if (i) nr_puts(" "); nr_hex32(encoded_ids[i]);
  }
  nr_puts(" encode_cycles="); nr_hex64(encode_cycles); nr_puts("\r\n");

  if (count != 16) {
    nr_puts("[text] prompt length differs from compiled prefill: FAIL\r\n"); return 1; }

  for (size_t i = 0; i < count; ++i) {
    if (encoded_ids[i] != expected_prompt_ids[i]) {
      nr_puts("[text] prompt IDs differ from expected reference: FAIL\r\n"); return 1;
    }
  }
  nr_puts("verify fixed prompt tokenizer: PASS\r\n");
  /* The actual encoder output feeds the graph. The expected array is read only
   * above for validation; it never supplies model operands. */
  for (size_t i = 0; i < count; ++i) input_ids[i] = encoded_ids[i];
  reset_cache();
  text_output.length = 0; text_output.overflow = 0;
  QwenUtf8Decoder utf8 = {{0,0,0,0},0,0};
  unsigned token = 0; float score = 0;
  nr_puts("[text] mode=fixed-validation; EOS is recorded, never early-stops\r\n");
  if (run_prefill(0, &token, &score, 1)) return 1;
  if (record_text_token(&resource, &utf8, 0, token)) return 1;

  for (unsigned step = 0; step < 8; ++step) {
    input_ids[0] = token;
    if (run_decode(16 + step, &token, &score, 1)) return 1;
    if (record_text_token(&resource, &utf8, step + 1, token)) return 1;
  }

  size_t previous = text_output.length;
  qwen_decode_finish(&utf8, append_text, &text_output);
  if (text_output.overflow) { nr_puts("[text] output capacity: FAIL\r\n"); return 1; }
  nr_puts("[text] finish hex=");
  text_hex(text_output.data + previous, text_output.length - previous);
  nr_puts("\r\n[text] output bytes="); nr_hex32(text_output.length);
  nr_puts(" hex="); text_hex(text_output.data, text_output.length);
  nr_puts("\r\n[text] BEGIN\r\n");
  /* A single length-delimited UART write keeps model traces outside the text.
   * UTF-8 was incrementally decoded on board after each actual prediction. */
  nr_write(text_output.data, text_output.length);
  nr_puts("\r\n[text] END\r\nverify fixed text validation: PASS\r\n");
  return 0;
}

