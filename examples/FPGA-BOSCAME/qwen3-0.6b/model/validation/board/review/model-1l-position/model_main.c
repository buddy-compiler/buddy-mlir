#include "support.h"
#include "nr_runtime.h"
extern void qwen_profile_reset(void);
extern void qwen_profile_report(unsigned position);
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
static float *const k_cache_f = (float *)k_cache_raw;
static float *const v_cache_f = (float *)v_cache_raw;
static int64_t *const input_ids = (int64_t *)input_ids_raw;
static int64_t *const cache_position = (int64_t *)cache_position_raw;
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "ws_prefill_raw:\n"
        ".skip 9050688\n"
        "ws_prefill_end:\n"
        ".previous\n");
extern unsigned char ws_prefill_raw[] __asm__("ws_prefill_raw");
extern void _mlir_ciface_forward_prefill(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "ws_decode_raw:\n"
        ".skip 5638848\n"
        "ws_decode_end:\n"
        ".previous\n");
extern unsigned char ws_decode_raw[] __asm__("ws_decode_raw");
extern void _mlir_ciface_forward_decode(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
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
static int collect(GraphResults *r, unsigned position, unsigned *token,
                   float *score, int trace) {
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
  for (unsigned l = 0; l < LAYERS; ++l) {
    if (retain_cache(k_cache_f + l * KV_ELEMENTS, &r->cache[l].key) ||
        retain_cache(v_cache_f + l * KV_ELEMENTS, &r->cache[l].value)) return -1;
  }
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
  MemRef4 w0 = make_4(ws_prefill_raw + 0, 1, 16, 128, 512);
  MemRef4 w1 = make_4(ws_prefill_raw + 4194304, 1, 16, 16, 512);
  MemRef4 w2 = make_4(ws_prefill_raw + 4718592, 1, 16, 16, 512);
  MemRef1 w3 = make_1(ws_prefill_raw + 5242880, 256);
  MemRef1 w4 = make_1(ws_prefill_raw + 5243904, 256);
  MemRef4 w5 = make_4(ws_prefill_raw + 5244928, 1, 16, 16, 512);
  MemRef4 w6 = make_4(ws_prefill_raw + 5769216, 1, 16, 16, 128);
  MemRef1 w7 = make_1(ws_prefill_raw + 5900288, 16);
  for (unsigned j = 0; j < 16; ++j) ((int32_t *)(ws_prefill_raw + 5900288))[j] = position + j;
  MemRef4 w8 = make_4(ws_prefill_raw + 5900352, 1, 8, 16, 128);
  MemRef4 w9 = make_4(ws_prefill_raw + 5965888, 1, 8, 16, 128);
  MemRef3 w10 = make_3(ws_prefill_raw + 6031424, 1, 16, 1024);
  MemRef2 w11 = make_2(ws_prefill_raw + 6096960, 16, 1024);
  MemRef1 w12 = make_1(ws_prefill_raw + 6113344, 16);
  MemRef2 w13 = make_2(ws_prefill_raw + 6113408, 16, 2048);
  zero_bytes(ws_prefill_raw + 6113408, 131072);
  MemRef2 w14 = make_2(ws_prefill_raw + 6244480, 16, 2048);
  MemRef2 w15 = make_2(ws_prefill_raw + 6375552, 16, 1024);
  MemRef1 w16 = make_1(ws_prefill_raw + 6391936, 16);
  MemRef2 w17 = make_2(ws_prefill_raw + 6392000, 16, 1024);
  zero_bytes(ws_prefill_raw + 6392000, 65536);
  MemRef2 w18 = make_2(ws_prefill_raw + 6457536, 16, 1024);
  MemRef2 w19 = make_2(ws_prefill_raw + 6523072, 16, 1024);
  MemRef1 w20 = make_1(ws_prefill_raw + 6539456, 16);
  MemRef2 w21 = make_2(ws_prefill_raw + 6539520, 16, 1024);
  zero_bytes(ws_prefill_raw + 6539520, 65536);
  MemRef2 w22 = make_2(ws_prefill_raw + 6605056, 16, 1024);
  MemRef2 w23 = make_2(ws_prefill_raw + 6670592, 16, 2048);
  MemRef1 w24 = make_1(ws_prefill_raw + 6703360, 16);
  MemRef2 w25 = make_2(ws_prefill_raw + 6703424, 16, 1024);
  zero_bytes(ws_prefill_raw + 6703424, 65536);
  MemRef2 w26 = make_2(ws_prefill_raw + 6768960, 16, 1024);
  MemRef2 w27 = make_2(ws_prefill_raw + 6834496, 16, 1024);
  MemRef1 w28 = make_1(ws_prefill_raw + 6850880, 16);
  MemRef2 w29 = make_2(ws_prefill_raw + 6850944, 16, 3072);
  zero_bytes(ws_prefill_raw + 6850944, 196608);
  MemRef2 w30 = make_2(ws_prefill_raw + 7047552, 16, 3072);
  MemRef2 w31 = make_2(ws_prefill_raw + 7244160, 16, 1024);
  MemRef1 w32 = make_1(ws_prefill_raw + 7260544, 16);
  MemRef2 w33 = make_2(ws_prefill_raw + 7260608, 16, 3072);
  zero_bytes(ws_prefill_raw + 7260608, 196608);
  MemRef2 w34 = make_2(ws_prefill_raw + 7457216, 16, 3072);
  MemRef2 w35 = make_2(ws_prefill_raw + 7653824, 16, 3072);
  MemRef1 w36 = make_1(ws_prefill_raw + 7702976, 16);
  MemRef2 w37 = make_2(ws_prefill_raw + 7703040, 16, 1024);
  zero_bytes(ws_prefill_raw + 7703040, 65536);
  MemRef2 w38 = make_2(ws_prefill_raw + 7768576, 16, 1024);
  MemRef2 w39 = make_2(ws_prefill_raw + 7834112, 1, 1024);
  MemRef1 w40 = make_1(ws_prefill_raw + 7835136, 1);
  MemRef2 w41 = make_2(ws_prefill_raw + 7835200, 1, 151936);
  zero_bytes(ws_prefill_raw + 7835200, 607744);
  MemRef2 w42 = make_2(ws_prefill_raw + 8442944, 1, 151936);
  if (trace) { nr_puts("[model] prefill begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_profile_reset();
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_prefill(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &ids, &pos, &k[0], &v[0], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41, &w42);
  ame_fence();
  uint64_t compute = nr_cycles() - begin;
  int status = collect(&result, position + 16 - 1, token, score, trace);
  uintptr_t peak = nr_heap_mark();
  nr_heap_reset(mark);
  if (trace) { nr_puts("[model] prefill position="); nr_hex32(position);
    nr_puts(" token="); nr_hex32(*token);
    nr_puts(" logit_bits="); float_bits(*score);
    nr_puts(" compute_cycles="); nr_hex64(compute);
    nr_puts(" total_with_uart_cycles="); nr_hex64(nr_cycles() - begin);
    nr_puts(" scratch_bytes="); nr_hex64(peak - mark);
    nr_puts("\r\n"); }
  if (trace) qwen_profile_report(position + 16 - 1);
  return status;
}
static int run_decode(unsigned position, unsigned *token,
                      float *score, int trace) {
  if (position + 1 > CAPACITY) return -1;
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
  MemRef4 w0 = make_4(ws_decode_raw + 0, 1, 16, 128, 512);
  MemRef4 w1 = make_4(ws_decode_raw + 4194304, 1, 16, 1, 512);
  MemRef4 w2 = make_4(ws_decode_raw + 4227072, 1, 16, 1, 512);
  MemRef1 w3 = make_1(ws_decode_raw + 4259840, 16);
  MemRef1 w4 = make_1(ws_decode_raw + 4259904, 16);
  MemRef4 w5 = make_4(ws_decode_raw + 4259968, 1, 16, 1, 512);
  MemRef4 w6 = make_4(ws_decode_raw + 4292736, 1, 16, 1, 128);
  MemRef1 w7 = make_1(ws_decode_raw + 4300928, 1);
  for (unsigned j = 0; j < 1; ++j) ((int32_t *)(ws_decode_raw + 4300928))[j] = position + j;
  MemRef4 w8 = make_4(ws_decode_raw + 4300992, 1, 8, 1, 128);
  MemRef4 w9 = make_4(ws_decode_raw + 4305088, 1, 8, 1, 128);
  MemRef3 w10 = make_3(ws_decode_raw + 4309184, 1, 1, 1024);
  MemRef2 w11 = make_2(ws_decode_raw + 4313280, 1, 1024);
  MemRef1 w12 = make_1(ws_decode_raw + 4314304, 1);
  MemRef2 w13 = make_2(ws_decode_raw + 4314368, 1, 2048);
  zero_bytes(ws_decode_raw + 4314368, 8192);
  MemRef2 w14 = make_2(ws_decode_raw + 4322560, 1, 2048);
  MemRef2 w15 = make_2(ws_decode_raw + 4330752, 1, 1024);
  MemRef1 w16 = make_1(ws_decode_raw + 4331776, 1);
  MemRef2 w17 = make_2(ws_decode_raw + 4331840, 1, 1024);
  zero_bytes(ws_decode_raw + 4331840, 4096);
  MemRef2 w18 = make_2(ws_decode_raw + 4335936, 1, 1024);
  MemRef2 w19 = make_2(ws_decode_raw + 4340032, 1, 1024);
  MemRef1 w20 = make_1(ws_decode_raw + 4341056, 1);
  MemRef2 w21 = make_2(ws_decode_raw + 4341120, 1, 1024);
  zero_bytes(ws_decode_raw + 4341120, 4096);
  MemRef2 w22 = make_2(ws_decode_raw + 4345216, 1, 1024);
  MemRef2 w23 = make_2(ws_decode_raw + 4349312, 1, 2048);
  MemRef1 w24 = make_1(ws_decode_raw + 4351360, 1);
  MemRef2 w25 = make_2(ws_decode_raw + 4351424, 1, 1024);
  zero_bytes(ws_decode_raw + 4351424, 4096);
  MemRef2 w26 = make_2(ws_decode_raw + 4355520, 1, 1024);
  MemRef2 w27 = make_2(ws_decode_raw + 4359616, 1, 1024);
  MemRef1 w28 = make_1(ws_decode_raw + 4360640, 1);
  MemRef2 w29 = make_2(ws_decode_raw + 4360704, 1, 3072);
  zero_bytes(ws_decode_raw + 4360704, 12288);
  MemRef2 w30 = make_2(ws_decode_raw + 4372992, 1, 3072);
  MemRef2 w31 = make_2(ws_decode_raw + 4385280, 1, 1024);
  MemRef1 w32 = make_1(ws_decode_raw + 4386304, 1);
  MemRef2 w33 = make_2(ws_decode_raw + 4386368, 1, 3072);
  zero_bytes(ws_decode_raw + 4386368, 12288);
  MemRef2 w34 = make_2(ws_decode_raw + 4398656, 1, 3072);
  MemRef2 w35 = make_2(ws_decode_raw + 4410944, 1, 3072);
  MemRef1 w36 = make_1(ws_decode_raw + 4414016, 1);
  MemRef2 w37 = make_2(ws_decode_raw + 4414080, 1, 1024);
  zero_bytes(ws_decode_raw + 4414080, 4096);
  MemRef2 w38 = make_2(ws_decode_raw + 4418176, 1, 1024);
  MemRef2 w39 = make_2(ws_decode_raw + 4422272, 1, 1024);
  MemRef1 w40 = make_1(ws_decode_raw + 4423296, 1);
  MemRef2 w41 = make_2(ws_decode_raw + 4423360, 1, 151936);
  zero_bytes(ws_decode_raw + 4423360, 607744);
  MemRef2 w42 = make_2(ws_decode_raw + 5031104, 1, 151936);
  if (trace) { nr_puts("[model] decode begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_profile_reset();
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_decode(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &ids, &pos, &k[0], &v[0], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41, &w42);
  ame_fence();
  uint64_t compute = nr_cycles() - begin;
  int status = collect(&result, position + 1 - 1, token, score, trace);
  uintptr_t peak = nr_heap_mark();
  nr_heap_reset(mark);
  if (trace) { nr_puts("[model] decode position="); nr_hex32(position);
    nr_puts(" token="); nr_hex32(*token);
    nr_puts(" logit_bits="); float_bits(*score);
    nr_puts(" compute_cycles="); nr_hex64(compute);
    nr_puts(" total_with_uart_cycles="); nr_hex64(nr_cycles() - begin);
    nr_puts(" scratch_bytes="); nr_hex64(peak - mark);
    nr_puts("\r\n"); }
  if (trace) qwen_profile_report(position + 1 - 1);
  return status;
}
int launch(void) {
  reset_cache();
  static const int64_t prompt[] = {151644, 872, 198, 3838, 374, 9625, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271};
  for (unsigned i = 0; i < sizeof(prompt)/sizeof(prompt[0]); ++i) input_ids[i] = prompt[i];
  unsigned token = 0; float score = 0;
  if (run_prefill(0, &token, &score, 1)) return 1;
  for (unsigned step = 0; step < 8; ++step) {
    input_ids[0] = token;
    if (run_decode(16 + step, &token, &score, 1)) return 1;
  }
  return 0;
}
