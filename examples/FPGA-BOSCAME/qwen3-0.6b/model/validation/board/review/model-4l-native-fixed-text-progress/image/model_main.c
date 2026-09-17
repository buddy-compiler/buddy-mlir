#include "support.h"
#include "nr_runtime.h"
extern void qwen_profile_reset(void);
extern void qwen_profile_report(unsigned position);
#include "tokenizer_resource.h"
#define LAYERS 4
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
        ".skip 219342592\n"
        "weight_arena_end:\n"
        ".previous\n");
extern unsigned char weight_arena_raw[] __asm__("weight_arena_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "k_cache_raw:\n"
        ".skip 8388608\n"
        "k_cache_end:\n"
        ".previous\n");
extern unsigned char k_cache_raw[] __asm__("k_cache_raw");
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "v_cache_raw:\n"
        ".skip 8388608\n"
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
        ".skip 15579008\n"
        "ws_prefill_end:\n"
        ".previous\n");
extern unsigned char ws_prefill_raw[] __asm__("ws_prefill_raw");
extern void _mlir_ciface_forward_prefill(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
__asm__(".section .workspace,\"aw\",@nobits\n"
        ".balign 64\n"
        "ws_decode_raw:\n"
        ".skip 2115968\n"
        "ws_decode_end:\n"
        ".previous\n");
extern unsigned char ws_decode_raw[] __asm__("ws_decode_raw");
extern void _mlir_ciface_forward_decode(GraphResults *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef1 *, MemRef1 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef4 *, MemRef3 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *, MemRef2 *, MemRef1 *, MemRef2 *, MemRef2 *);
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
  MemRef1 p5 = make_1(weight_arena_raw + 13312, 128);
  MemRef1 p6 = make_1(weight_arena_raw + 13824, 128);
  MemRef1 p7 = make_1(weight_arena_raw + 14336, 1024);
  MemRef1 p8 = make_1(weight_arena_raw + 18432, 1024);
  MemRef1 p9 = make_1(weight_arena_raw + 22528, 128);
  MemRef1 p10 = make_1(weight_arena_raw + 23040, 128);
  MemRef1 p11 = make_1(weight_arena_raw + 23552, 1024);
  MemRef1 p12 = make_1(weight_arena_raw + 27648, 1024);
  MemRef1 p13 = make_1(weight_arena_raw + 31744, 128);
  MemRef1 p14 = make_1(weight_arena_raw + 32256, 128);
  MemRef1 p15 = make_1(weight_arena_raw + 32768, 1024);
  MemRef1 p16 = make_1(weight_arena_raw + 36864, 1024);
  MemRef1 p17 = make_1(weight_arena_raw + 40960, 64);
  MemRef2 p18 = make_2(weight_arena_raw + 41216, 151936, 1024);
  MemRef1 p19 = make_1(weight_arena_raw + 155623680, 151936);
  MemRef2 p20 = make_2(weight_arena_raw + 156231424, 1024, 3072);
  MemRef1 p21 = make_1(weight_arena_raw + 159377152, 1024);
  MemRef2 p22 = make_2(weight_arena_raw + 159381248, 3072, 1024);
  MemRef1 p23 = make_1(weight_arena_raw + 162526976, 3072);
  MemRef2 p24 = make_2(weight_arena_raw + 162539264, 3072, 1024);
  MemRef1 p25 = make_1(weight_arena_raw + 165684992, 3072);
  MemRef2 p26 = make_2(weight_arena_raw + 165697280, 1024, 2048);
  MemRef1 p27 = make_1(weight_arena_raw + 167794432, 1024);
  MemRef2 p28 = make_2(weight_arena_raw + 167798528, 1024, 1024);
  MemRef1 p29 = make_1(weight_arena_raw + 168847104, 1024);
  MemRef2 p30 = make_2(weight_arena_raw + 168851200, 1024, 1024);
  MemRef1 p31 = make_1(weight_arena_raw + 169899776, 1024);
  MemRef2 p32 = make_2(weight_arena_raw + 169903872, 2048, 1024);
  MemRef1 p33 = make_1(weight_arena_raw + 172001024, 2048);
  MemRef2 p34 = make_2(weight_arena_raw + 172009216, 1024, 3072);
  MemRef1 p35 = make_1(weight_arena_raw + 175154944, 1024);
  MemRef2 p36 = make_2(weight_arena_raw + 175159040, 3072, 1024);
  MemRef1 p37 = make_1(weight_arena_raw + 178304768, 3072);
  MemRef2 p38 = make_2(weight_arena_raw + 178317056, 3072, 1024);
  MemRef1 p39 = make_1(weight_arena_raw + 181462784, 3072);
  MemRef2 p40 = make_2(weight_arena_raw + 181475072, 1024, 2048);
  MemRef1 p41 = make_1(weight_arena_raw + 183572224, 1024);
  MemRef2 p42 = make_2(weight_arena_raw + 183576320, 1024, 1024);
  MemRef1 p43 = make_1(weight_arena_raw + 184624896, 1024);
  MemRef2 p44 = make_2(weight_arena_raw + 184628992, 1024, 1024);
  MemRef1 p45 = make_1(weight_arena_raw + 185677568, 1024);
  MemRef2 p46 = make_2(weight_arena_raw + 185681664, 2048, 1024);
  MemRef1 p47 = make_1(weight_arena_raw + 187778816, 2048);
  MemRef2 p48 = make_2(weight_arena_raw + 187787008, 1024, 3072);
  MemRef1 p49 = make_1(weight_arena_raw + 190932736, 1024);
  MemRef2 p50 = make_2(weight_arena_raw + 190936832, 3072, 1024);
  MemRef1 p51 = make_1(weight_arena_raw + 194082560, 3072);
  MemRef2 p52 = make_2(weight_arena_raw + 194094848, 3072, 1024);
  MemRef1 p53 = make_1(weight_arena_raw + 197240576, 3072);
  MemRef2 p54 = make_2(weight_arena_raw + 197252864, 1024, 2048);
  MemRef1 p55 = make_1(weight_arena_raw + 199350016, 1024);
  MemRef2 p56 = make_2(weight_arena_raw + 199354112, 1024, 1024);
  MemRef1 p57 = make_1(weight_arena_raw + 200402688, 1024);
  MemRef2 p58 = make_2(weight_arena_raw + 200406784, 1024, 1024);
  MemRef1 p59 = make_1(weight_arena_raw + 201455360, 1024);
  MemRef2 p60 = make_2(weight_arena_raw + 201459456, 2048, 1024);
  MemRef1 p61 = make_1(weight_arena_raw + 203556608, 2048);
  MemRef2 p62 = make_2(weight_arena_raw + 203564800, 1024, 3072);
  MemRef1 p63 = make_1(weight_arena_raw + 206710528, 1024);
  MemRef2 p64 = make_2(weight_arena_raw + 206714624, 3072, 1024);
  MemRef1 p65 = make_1(weight_arena_raw + 209860352, 3072);
  MemRef2 p66 = make_2(weight_arena_raw + 209872640, 3072, 1024);
  MemRef1 p67 = make_1(weight_arena_raw + 213018368, 3072);
  MemRef2 p68 = make_2(weight_arena_raw + 213030656, 1024, 2048);
  MemRef1 p69 = make_1(weight_arena_raw + 215127808, 1024);
  MemRef2 p70 = make_2(weight_arena_raw + 215131904, 1024, 1024);
  MemRef1 p71 = make_1(weight_arena_raw + 216180480, 1024);
  MemRef2 p72 = make_2(weight_arena_raw + 216184576, 1024, 1024);
  MemRef1 p73 = make_1(weight_arena_raw + 217233152, 1024);
  MemRef2 p74 = make_2(weight_arena_raw + 217237248, 2048, 1024);
  MemRef1 p75 = make_1(weight_arena_raw + 219334400, 2048);
  MemRef4 w0 = make_4(ws_prefill_raw + 0, 1, 16, 16, 512);
  MemRef4 w1 = make_4(ws_prefill_raw + 524288, 1, 16, 16, 512);
  MemRef1 w2 = make_1(ws_prefill_raw + 1048576, 256);
  MemRef1 w3 = make_1(ws_prefill_raw + 1049600, 256);
  MemRef4 w4 = make_4(ws_prefill_raw + 1050624, 1, 16, 16, 512);
  MemRef4 w5 = make_4(ws_prefill_raw + 1574912, 1, 16, 16, 128);
  MemRef1 w6 = make_1(ws_prefill_raw + 1705984, 16);
  for (unsigned j = 0; j < 16; ++j) ((int32_t *)(ws_prefill_raw + 1705984))[j] = position + j;
  MemRef4 w7 = make_4(ws_prefill_raw + 1706048, 1, 16, 16, 512);
  MemRef4 w8 = make_4(ws_prefill_raw + 2230336, 1, 16, 16, 512);
  MemRef1 w9 = make_1(ws_prefill_raw + 2754624, 256);
  MemRef1 w10 = make_1(ws_prefill_raw + 2755648, 256);
  MemRef4 w11 = make_4(ws_prefill_raw + 2756672, 1, 16, 16, 512);
  MemRef4 w12 = make_4(ws_prefill_raw + 3280960, 1, 16, 16, 128);
  MemRef4 w13 = make_4(ws_prefill_raw + 3412032, 1, 16, 16, 512);
  MemRef4 w14 = make_4(ws_prefill_raw + 3936320, 1, 16, 16, 512);
  MemRef1 w15 = make_1(ws_prefill_raw + 4460608, 256);
  MemRef1 w16 = make_1(ws_prefill_raw + 4461632, 256);
  MemRef4 w17 = make_4(ws_prefill_raw + 4462656, 1, 16, 16, 512);
  MemRef4 w18 = make_4(ws_prefill_raw + 4986944, 1, 16, 16, 128);
  MemRef4 w19 = make_4(ws_prefill_raw + 5118016, 1, 16, 16, 512);
  MemRef4 w20 = make_4(ws_prefill_raw + 5642304, 1, 16, 16, 512);
  MemRef1 w21 = make_1(ws_prefill_raw + 6166592, 256);
  MemRef1 w22 = make_1(ws_prefill_raw + 6167616, 256);
  MemRef4 w23 = make_4(ws_prefill_raw + 6168640, 1, 16, 16, 512);
  MemRef4 w24 = make_4(ws_prefill_raw + 6692928, 1, 16, 16, 128);
  MemRef4 w25 = make_4(ws_prefill_raw + 6824000, 1, 8, 16, 128);
  MemRef4 w26 = make_4(ws_prefill_raw + 6889536, 1, 8, 16, 128);
  MemRef4 w27 = make_4(ws_prefill_raw + 6955072, 1, 8, 16, 128);
  MemRef4 w28 = make_4(ws_prefill_raw + 7020608, 1, 8, 16, 128);
  MemRef4 w29 = make_4(ws_prefill_raw + 7086144, 1, 8, 16, 128);
  MemRef4 w30 = make_4(ws_prefill_raw + 7151680, 1, 8, 16, 128);
  MemRef4 w31 = make_4(ws_prefill_raw + 7217216, 1, 8, 16, 128);
  MemRef4 w32 = make_4(ws_prefill_raw + 7282752, 1, 8, 16, 128);
  MemRef3 w33 = make_3(ws_prefill_raw + 7348288, 1, 16, 1024);
  MemRef2 w34 = make_2(ws_prefill_raw + 7413824, 16, 1024);
  MemRef1 w35 = make_1(ws_prefill_raw + 7430208, 16);
  MemRef2 w36 = make_2(ws_prefill_raw + 7430272, 16, 2048);
  zero_bytes(ws_prefill_raw + 7430272, 131072);
  MemRef2 w37 = make_2(ws_prefill_raw + 7561344, 16, 2048);
  MemRef2 w38 = make_2(ws_prefill_raw + 7692416, 16, 1024);
  MemRef1 w39 = make_1(ws_prefill_raw + 7708800, 16);
  MemRef2 w40 = make_2(ws_prefill_raw + 7708864, 16, 1024);
  zero_bytes(ws_prefill_raw + 7708864, 65536);
  MemRef2 w41 = make_2(ws_prefill_raw + 7774400, 16, 1024);
  MemRef2 w42 = make_2(ws_prefill_raw + 7839936, 16, 1024);
  MemRef1 w43 = make_1(ws_prefill_raw + 7856320, 16);
  MemRef2 w44 = make_2(ws_prefill_raw + 7856384, 16, 1024);
  zero_bytes(ws_prefill_raw + 7856384, 65536);
  MemRef2 w45 = make_2(ws_prefill_raw + 7921920, 16, 1024);
  MemRef2 w46 = make_2(ws_prefill_raw + 7987456, 16, 2048);
  MemRef1 w47 = make_1(ws_prefill_raw + 8020224, 16);
  MemRef2 w48 = make_2(ws_prefill_raw + 8020288, 16, 1024);
  zero_bytes(ws_prefill_raw + 8020288, 65536);
  MemRef2 w49 = make_2(ws_prefill_raw + 8085824, 16, 1024);
  MemRef2 w50 = make_2(ws_prefill_raw + 8151360, 16, 1024);
  MemRef1 w51 = make_1(ws_prefill_raw + 8167744, 16);
  MemRef2 w52 = make_2(ws_prefill_raw + 8167808, 16, 3072);
  zero_bytes(ws_prefill_raw + 8167808, 196608);
  MemRef2 w53 = make_2(ws_prefill_raw + 8364416, 16, 3072);
  MemRef2 w54 = make_2(ws_prefill_raw + 8561024, 16, 1024);
  MemRef1 w55 = make_1(ws_prefill_raw + 8577408, 16);
  MemRef2 w56 = make_2(ws_prefill_raw + 8577472, 16, 3072);
  zero_bytes(ws_prefill_raw + 8577472, 196608);
  MemRef2 w57 = make_2(ws_prefill_raw + 8774080, 16, 3072);
  MemRef2 w58 = make_2(ws_prefill_raw + 8970688, 16, 3072);
  MemRef1 w59 = make_1(ws_prefill_raw + 9019840, 16);
  MemRef2 w60 = make_2(ws_prefill_raw + 9019904, 16, 1024);
  zero_bytes(ws_prefill_raw + 9019904, 65536);
  MemRef2 w61 = make_2(ws_prefill_raw + 9085440, 16, 1024);
  MemRef2 w62 = make_2(ws_prefill_raw + 9150976, 16, 1024);
  MemRef1 w63 = make_1(ws_prefill_raw + 9167360, 16);
  MemRef2 w64 = make_2(ws_prefill_raw + 9167424, 16, 2048);
  zero_bytes(ws_prefill_raw + 9167424, 131072);
  MemRef2 w65 = make_2(ws_prefill_raw + 9298496, 16, 2048);
  MemRef2 w66 = make_2(ws_prefill_raw + 9429568, 16, 1024);
  MemRef1 w67 = make_1(ws_prefill_raw + 9445952, 16);
  MemRef2 w68 = make_2(ws_prefill_raw + 9446016, 16, 1024);
  zero_bytes(ws_prefill_raw + 9446016, 65536);
  MemRef2 w69 = make_2(ws_prefill_raw + 9511552, 16, 1024);
  MemRef2 w70 = make_2(ws_prefill_raw + 9577088, 16, 1024);
  MemRef1 w71 = make_1(ws_prefill_raw + 9593472, 16);
  MemRef2 w72 = make_2(ws_prefill_raw + 9593536, 16, 1024);
  zero_bytes(ws_prefill_raw + 9593536, 65536);
  MemRef2 w73 = make_2(ws_prefill_raw + 9659072, 16, 1024);
  MemRef2 w74 = make_2(ws_prefill_raw + 9724608, 16, 2048);
  MemRef1 w75 = make_1(ws_prefill_raw + 9757376, 16);
  MemRef2 w76 = make_2(ws_prefill_raw + 9757440, 16, 1024);
  zero_bytes(ws_prefill_raw + 9757440, 65536);
  MemRef2 w77 = make_2(ws_prefill_raw + 9822976, 16, 1024);
  MemRef2 w78 = make_2(ws_prefill_raw + 9888512, 16, 1024);
  MemRef1 w79 = make_1(ws_prefill_raw + 9904896, 16);
  MemRef2 w80 = make_2(ws_prefill_raw + 9904960, 16, 3072);
  zero_bytes(ws_prefill_raw + 9904960, 196608);
  MemRef2 w81 = make_2(ws_prefill_raw + 10101568, 16, 3072);
  MemRef2 w82 = make_2(ws_prefill_raw + 10298176, 16, 1024);
  MemRef1 w83 = make_1(ws_prefill_raw + 10314560, 16);
  MemRef2 w84 = make_2(ws_prefill_raw + 10314624, 16, 3072);
  zero_bytes(ws_prefill_raw + 10314624, 196608);
  MemRef2 w85 = make_2(ws_prefill_raw + 10511232, 16, 3072);
  MemRef2 w86 = make_2(ws_prefill_raw + 10707840, 16, 3072);
  MemRef1 w87 = make_1(ws_prefill_raw + 10756992, 16);
  MemRef2 w88 = make_2(ws_prefill_raw + 10757056, 16, 1024);
  zero_bytes(ws_prefill_raw + 10757056, 65536);
  MemRef2 w89 = make_2(ws_prefill_raw + 10822592, 16, 1024);
  MemRef2 w90 = make_2(ws_prefill_raw + 10888128, 16, 1024);
  MemRef1 w91 = make_1(ws_prefill_raw + 10904512, 16);
  MemRef2 w92 = make_2(ws_prefill_raw + 10904576, 16, 2048);
  zero_bytes(ws_prefill_raw + 10904576, 131072);
  MemRef2 w93 = make_2(ws_prefill_raw + 11035648, 16, 2048);
  MemRef2 w94 = make_2(ws_prefill_raw + 11166720, 16, 1024);
  MemRef1 w95 = make_1(ws_prefill_raw + 11183104, 16);
  MemRef2 w96 = make_2(ws_prefill_raw + 11183168, 16, 1024);
  zero_bytes(ws_prefill_raw + 11183168, 65536);
  MemRef2 w97 = make_2(ws_prefill_raw + 11248704, 16, 1024);
  MemRef2 w98 = make_2(ws_prefill_raw + 11314240, 16, 1024);
  MemRef1 w99 = make_1(ws_prefill_raw + 11330624, 16);
  MemRef2 w100 = make_2(ws_prefill_raw + 11330688, 16, 1024);
  zero_bytes(ws_prefill_raw + 11330688, 65536);
  MemRef2 w101 = make_2(ws_prefill_raw + 11396224, 16, 1024);
  MemRef2 w102 = make_2(ws_prefill_raw + 11461760, 16, 2048);
  MemRef1 w103 = make_1(ws_prefill_raw + 11494528, 16);
  MemRef2 w104 = make_2(ws_prefill_raw + 11494592, 16, 1024);
  zero_bytes(ws_prefill_raw + 11494592, 65536);
  MemRef2 w105 = make_2(ws_prefill_raw + 11560128, 16, 1024);
  MemRef2 w106 = make_2(ws_prefill_raw + 11625664, 16, 1024);
  MemRef1 w107 = make_1(ws_prefill_raw + 11642048, 16);
  MemRef2 w108 = make_2(ws_prefill_raw + 11642112, 16, 3072);
  zero_bytes(ws_prefill_raw + 11642112, 196608);
  MemRef2 w109 = make_2(ws_prefill_raw + 11838720, 16, 3072);
  MemRef2 w110 = make_2(ws_prefill_raw + 12035328, 16, 1024);
  MemRef1 w111 = make_1(ws_prefill_raw + 12051712, 16);
  MemRef2 w112 = make_2(ws_prefill_raw + 12051776, 16, 3072);
  zero_bytes(ws_prefill_raw + 12051776, 196608);
  MemRef2 w113 = make_2(ws_prefill_raw + 12248384, 16, 3072);
  MemRef2 w114 = make_2(ws_prefill_raw + 12444992, 16, 3072);
  MemRef1 w115 = make_1(ws_prefill_raw + 12494144, 16);
  MemRef2 w116 = make_2(ws_prefill_raw + 12494208, 16, 1024);
  zero_bytes(ws_prefill_raw + 12494208, 65536);
  MemRef2 w117 = make_2(ws_prefill_raw + 12559744, 16, 1024);
  MemRef2 w118 = make_2(ws_prefill_raw + 12625280, 16, 1024);
  MemRef1 w119 = make_1(ws_prefill_raw + 12641664, 16);
  MemRef2 w120 = make_2(ws_prefill_raw + 12641728, 16, 2048);
  zero_bytes(ws_prefill_raw + 12641728, 131072);
  MemRef2 w121 = make_2(ws_prefill_raw + 12772800, 16, 2048);
  MemRef2 w122 = make_2(ws_prefill_raw + 12903872, 16, 1024);
  MemRef1 w123 = make_1(ws_prefill_raw + 12920256, 16);
  MemRef2 w124 = make_2(ws_prefill_raw + 12920320, 16, 1024);
  zero_bytes(ws_prefill_raw + 12920320, 65536);
  MemRef2 w125 = make_2(ws_prefill_raw + 12985856, 16, 1024);
  MemRef2 w126 = make_2(ws_prefill_raw + 13051392, 16, 1024);
  MemRef1 w127 = make_1(ws_prefill_raw + 13067776, 16);
  MemRef2 w128 = make_2(ws_prefill_raw + 13067840, 16, 1024);
  zero_bytes(ws_prefill_raw + 13067840, 65536);
  MemRef2 w129 = make_2(ws_prefill_raw + 13133376, 16, 1024);
  MemRef2 w130 = make_2(ws_prefill_raw + 13198912, 16, 2048);
  MemRef1 w131 = make_1(ws_prefill_raw + 13231680, 16);
  MemRef2 w132 = make_2(ws_prefill_raw + 13231744, 16, 1024);
  zero_bytes(ws_prefill_raw + 13231744, 65536);
  MemRef2 w133 = make_2(ws_prefill_raw + 13297280, 16, 1024);
  MemRef2 w134 = make_2(ws_prefill_raw + 13362816, 16, 1024);
  MemRef1 w135 = make_1(ws_prefill_raw + 13379200, 16);
  MemRef2 w136 = make_2(ws_prefill_raw + 13379264, 16, 3072);
  zero_bytes(ws_prefill_raw + 13379264, 196608);
  MemRef2 w137 = make_2(ws_prefill_raw + 13575872, 16, 3072);
  MemRef2 w138 = make_2(ws_prefill_raw + 13772480, 16, 1024);
  MemRef1 w139 = make_1(ws_prefill_raw + 13788864, 16);
  MemRef2 w140 = make_2(ws_prefill_raw + 13788928, 16, 3072);
  zero_bytes(ws_prefill_raw + 13788928, 196608);
  MemRef2 w141 = make_2(ws_prefill_raw + 13985536, 16, 3072);
  MemRef2 w142 = make_2(ws_prefill_raw + 14182144, 16, 3072);
  MemRef1 w143 = make_1(ws_prefill_raw + 14231296, 16);
  MemRef2 w144 = make_2(ws_prefill_raw + 14231360, 16, 1024);
  zero_bytes(ws_prefill_raw + 14231360, 65536);
  MemRef2 w145 = make_2(ws_prefill_raw + 14296896, 16, 1024);
  MemRef2 w146 = make_2(ws_prefill_raw + 14362432, 1, 1024);
  MemRef1 w147 = make_1(ws_prefill_raw + 14363456, 1);
  MemRef2 w148 = make_2(ws_prefill_raw + 14363520, 1, 151936);
  zero_bytes(ws_prefill_raw + 14363520, 607744);
  MemRef2 w149 = make_2(ws_prefill_raw + 14971264, 1, 151936);
  uint64_t preparation_cycles = nr_cycles() - preparation_begin;
  if (trace) { nr_puts("[model] prefill begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_profile_reset();
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_prefill(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23, &p24, &p25, &p26, &p27, &p28, &p29, &p30, &p31, &p32, &p33, &p34, &p35, &p36, &p37, &p38, &p39, &p40, &p41, &p42, &p43, &p44, &p45, &p46, &p47, &p48, &p49, &p50, &p51, &p52, &p53, &p54, &p55, &p56, &p57, &p58, &p59, &p60, &p61, &p62, &p63, &p64, &p65, &p66, &p67, &p68, &p69, &p70, &p71, &p72, &p73, &p74, &p75, &ids, &pos, &k[0], &v[0], &pos, &k[1], &v[1], &pos, &k[2], &v[2], &pos, &k[3], &v[3], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41, &w42, &w43, &w44, &w45, &w46, &w47, &w48, &w49, &w50, &w51, &w52, &w53, &w54, &w55, &w56, &w57, &w58, &w59, &w60, &w61, &w62, &w63, &w64, &w65, &w66, &w67, &w68, &w69, &w70, &w71, &w72, &w73, &w74, &w75, &w76, &w77, &w78, &w79, &w80, &w81, &w82, &w83, &w84, &w85, &w86, &w87, &w88, &w89, &w90, &w91, &w92, &w93, &w94, &w95, &w96, &w97, &w98, &w99, &w100, &w101, &w102, &w103, &w104, &w105, &w106, &w107, &w108, &w109, &w110, &w111, &w112, &w113, &w114, &w115, &w116, &w117, &w118, &w119, &w120, &w121, &w122, &w123, &w124, &w125, &w126, &w127, &w128, &w129, &w130, &w131, &w132, &w133, &w134, &w135, &w136, &w137, &w138, &w139, &w140, &w141, &w142, &w143, &w144, &w145, &w146, &w147, &w148, &w149);
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
  if (trace) qwen_profile_report(position + 16 - 1);
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
  MemRef1 p5 = make_1(weight_arena_raw + 13312, 128);
  MemRef1 p6 = make_1(weight_arena_raw + 13824, 128);
  MemRef1 p7 = make_1(weight_arena_raw + 14336, 1024);
  MemRef1 p8 = make_1(weight_arena_raw + 18432, 1024);
  MemRef1 p9 = make_1(weight_arena_raw + 22528, 128);
  MemRef1 p10 = make_1(weight_arena_raw + 23040, 128);
  MemRef1 p11 = make_1(weight_arena_raw + 23552, 1024);
  MemRef1 p12 = make_1(weight_arena_raw + 27648, 1024);
  MemRef1 p13 = make_1(weight_arena_raw + 31744, 128);
  MemRef1 p14 = make_1(weight_arena_raw + 32256, 128);
  MemRef1 p15 = make_1(weight_arena_raw + 32768, 1024);
  MemRef1 p16 = make_1(weight_arena_raw + 36864, 1024);
  MemRef1 p17 = make_1(weight_arena_raw + 40960, 64);
  MemRef2 p18 = make_2(weight_arena_raw + 41216, 151936, 1024);
  MemRef1 p19 = make_1(weight_arena_raw + 155623680, 151936);
  MemRef2 p20 = make_2(weight_arena_raw + 156231424, 1024, 3072);
  MemRef1 p21 = make_1(weight_arena_raw + 159377152, 1024);
  MemRef2 p22 = make_2(weight_arena_raw + 159381248, 3072, 1024);
  MemRef1 p23 = make_1(weight_arena_raw + 162526976, 3072);
  MemRef2 p24 = make_2(weight_arena_raw + 162539264, 3072, 1024);
  MemRef1 p25 = make_1(weight_arena_raw + 165684992, 3072);
  MemRef2 p26 = make_2(weight_arena_raw + 165697280, 1024, 2048);
  MemRef1 p27 = make_1(weight_arena_raw + 167794432, 1024);
  MemRef2 p28 = make_2(weight_arena_raw + 167798528, 1024, 1024);
  MemRef1 p29 = make_1(weight_arena_raw + 168847104, 1024);
  MemRef2 p30 = make_2(weight_arena_raw + 168851200, 1024, 1024);
  MemRef1 p31 = make_1(weight_arena_raw + 169899776, 1024);
  MemRef2 p32 = make_2(weight_arena_raw + 169903872, 2048, 1024);
  MemRef1 p33 = make_1(weight_arena_raw + 172001024, 2048);
  MemRef2 p34 = make_2(weight_arena_raw + 172009216, 1024, 3072);
  MemRef1 p35 = make_1(weight_arena_raw + 175154944, 1024);
  MemRef2 p36 = make_2(weight_arena_raw + 175159040, 3072, 1024);
  MemRef1 p37 = make_1(weight_arena_raw + 178304768, 3072);
  MemRef2 p38 = make_2(weight_arena_raw + 178317056, 3072, 1024);
  MemRef1 p39 = make_1(weight_arena_raw + 181462784, 3072);
  MemRef2 p40 = make_2(weight_arena_raw + 181475072, 1024, 2048);
  MemRef1 p41 = make_1(weight_arena_raw + 183572224, 1024);
  MemRef2 p42 = make_2(weight_arena_raw + 183576320, 1024, 1024);
  MemRef1 p43 = make_1(weight_arena_raw + 184624896, 1024);
  MemRef2 p44 = make_2(weight_arena_raw + 184628992, 1024, 1024);
  MemRef1 p45 = make_1(weight_arena_raw + 185677568, 1024);
  MemRef2 p46 = make_2(weight_arena_raw + 185681664, 2048, 1024);
  MemRef1 p47 = make_1(weight_arena_raw + 187778816, 2048);
  MemRef2 p48 = make_2(weight_arena_raw + 187787008, 1024, 3072);
  MemRef1 p49 = make_1(weight_arena_raw + 190932736, 1024);
  MemRef2 p50 = make_2(weight_arena_raw + 190936832, 3072, 1024);
  MemRef1 p51 = make_1(weight_arena_raw + 194082560, 3072);
  MemRef2 p52 = make_2(weight_arena_raw + 194094848, 3072, 1024);
  MemRef1 p53 = make_1(weight_arena_raw + 197240576, 3072);
  MemRef2 p54 = make_2(weight_arena_raw + 197252864, 1024, 2048);
  MemRef1 p55 = make_1(weight_arena_raw + 199350016, 1024);
  MemRef2 p56 = make_2(weight_arena_raw + 199354112, 1024, 1024);
  MemRef1 p57 = make_1(weight_arena_raw + 200402688, 1024);
  MemRef2 p58 = make_2(weight_arena_raw + 200406784, 1024, 1024);
  MemRef1 p59 = make_1(weight_arena_raw + 201455360, 1024);
  MemRef2 p60 = make_2(weight_arena_raw + 201459456, 2048, 1024);
  MemRef1 p61 = make_1(weight_arena_raw + 203556608, 2048);
  MemRef2 p62 = make_2(weight_arena_raw + 203564800, 1024, 3072);
  MemRef1 p63 = make_1(weight_arena_raw + 206710528, 1024);
  MemRef2 p64 = make_2(weight_arena_raw + 206714624, 3072, 1024);
  MemRef1 p65 = make_1(weight_arena_raw + 209860352, 3072);
  MemRef2 p66 = make_2(weight_arena_raw + 209872640, 3072, 1024);
  MemRef1 p67 = make_1(weight_arena_raw + 213018368, 3072);
  MemRef2 p68 = make_2(weight_arena_raw + 213030656, 1024, 2048);
  MemRef1 p69 = make_1(weight_arena_raw + 215127808, 1024);
  MemRef2 p70 = make_2(weight_arena_raw + 215131904, 1024, 1024);
  MemRef1 p71 = make_1(weight_arena_raw + 216180480, 1024);
  MemRef2 p72 = make_2(weight_arena_raw + 216184576, 1024, 1024);
  MemRef1 p73 = make_1(weight_arena_raw + 217233152, 1024);
  MemRef2 p74 = make_2(weight_arena_raw + 217237248, 2048, 1024);
  MemRef1 p75 = make_1(weight_arena_raw + 219334400, 2048);
  MemRef4 w0 = make_4(ws_decode_raw + 0, 1, 16, 1, 512);
  MemRef4 w1 = make_4(ws_decode_raw + 32768, 1, 16, 1, 512);
  MemRef1 w2 = make_1(ws_decode_raw + 65536, 16);
  MemRef1 w3 = make_1(ws_decode_raw + 65600, 16);
  MemRef4 w4 = make_4(ws_decode_raw + 65664, 1, 16, 1, 512);
  MemRef4 w5 = make_4(ws_decode_raw + 98432, 1, 16, 1, 128);
  MemRef1 w6 = make_1(ws_decode_raw + 106624, 1);
  for (unsigned j = 0; j < 1; ++j) ((int32_t *)(ws_decode_raw + 106624))[j] = position + j;
  MemRef4 w7 = make_4(ws_decode_raw + 106688, 1, 16, 1, 512);
  MemRef4 w8 = make_4(ws_decode_raw + 139456, 1, 16, 1, 512);
  MemRef1 w9 = make_1(ws_decode_raw + 172224, 16);
  MemRef1 w10 = make_1(ws_decode_raw + 172288, 16);
  MemRef4 w11 = make_4(ws_decode_raw + 172352, 1, 16, 1, 512);
  MemRef4 w12 = make_4(ws_decode_raw + 205120, 1, 16, 1, 128);
  MemRef4 w13 = make_4(ws_decode_raw + 213312, 1, 16, 1, 512);
  MemRef4 w14 = make_4(ws_decode_raw + 246080, 1, 16, 1, 512);
  MemRef1 w15 = make_1(ws_decode_raw + 278848, 16);
  MemRef1 w16 = make_1(ws_decode_raw + 278912, 16);
  MemRef4 w17 = make_4(ws_decode_raw + 278976, 1, 16, 1, 512);
  MemRef4 w18 = make_4(ws_decode_raw + 311744, 1, 16, 1, 128);
  MemRef4 w19 = make_4(ws_decode_raw + 319936, 1, 16, 1, 512);
  MemRef4 w20 = make_4(ws_decode_raw + 352704, 1, 16, 1, 512);
  MemRef1 w21 = make_1(ws_decode_raw + 385472, 16);
  MemRef1 w22 = make_1(ws_decode_raw + 385536, 16);
  MemRef4 w23 = make_4(ws_decode_raw + 385600, 1, 16, 1, 512);
  MemRef4 w24 = make_4(ws_decode_raw + 418368, 1, 16, 1, 128);
  MemRef4 w25 = make_4(ws_decode_raw + 426560, 1, 8, 1, 128);
  MemRef4 w26 = make_4(ws_decode_raw + 430656, 1, 8, 1, 128);
  MemRef4 w27 = make_4(ws_decode_raw + 434752, 1, 8, 1, 128);
  MemRef4 w28 = make_4(ws_decode_raw + 438848, 1, 8, 1, 128);
  MemRef4 w29 = make_4(ws_decode_raw + 442944, 1, 8, 1, 128);
  MemRef4 w30 = make_4(ws_decode_raw + 447040, 1, 8, 1, 128);
  MemRef4 w31 = make_4(ws_decode_raw + 451136, 1, 8, 1, 128);
  MemRef4 w32 = make_4(ws_decode_raw + 455232, 1, 8, 1, 128);
  MemRef3 w33 = make_3(ws_decode_raw + 459328, 1, 1, 1024);
  MemRef2 w34 = make_2(ws_decode_raw + 463424, 1, 1024);
  MemRef1 w35 = make_1(ws_decode_raw + 464448, 1);
  MemRef2 w36 = make_2(ws_decode_raw + 464512, 1, 2048);
  zero_bytes(ws_decode_raw + 464512, 8192);
  MemRef2 w37 = make_2(ws_decode_raw + 472704, 1, 2048);
  MemRef2 w38 = make_2(ws_decode_raw + 480896, 1, 1024);
  MemRef1 w39 = make_1(ws_decode_raw + 481920, 1);
  MemRef2 w40 = make_2(ws_decode_raw + 481984, 1, 1024);
  zero_bytes(ws_decode_raw + 481984, 4096);
  MemRef2 w41 = make_2(ws_decode_raw + 486080, 1, 1024);
  MemRef2 w42 = make_2(ws_decode_raw + 490176, 1, 1024);
  MemRef1 w43 = make_1(ws_decode_raw + 491200, 1);
  MemRef2 w44 = make_2(ws_decode_raw + 491264, 1, 1024);
  zero_bytes(ws_decode_raw + 491264, 4096);
  MemRef2 w45 = make_2(ws_decode_raw + 495360, 1, 1024);
  MemRef2 w46 = make_2(ws_decode_raw + 499456, 1, 2048);
  MemRef1 w47 = make_1(ws_decode_raw + 501504, 1);
  MemRef2 w48 = make_2(ws_decode_raw + 501568, 1, 1024);
  zero_bytes(ws_decode_raw + 501568, 4096);
  MemRef2 w49 = make_2(ws_decode_raw + 505664, 1, 1024);
  MemRef2 w50 = make_2(ws_decode_raw + 509760, 1, 1024);
  MemRef1 w51 = make_1(ws_decode_raw + 510784, 1);
  MemRef2 w52 = make_2(ws_decode_raw + 510848, 1, 3072);
  zero_bytes(ws_decode_raw + 510848, 12288);
  MemRef2 w53 = make_2(ws_decode_raw + 523136, 1, 3072);
  MemRef2 w54 = make_2(ws_decode_raw + 535424, 1, 1024);
  MemRef1 w55 = make_1(ws_decode_raw + 536448, 1);
  MemRef2 w56 = make_2(ws_decode_raw + 536512, 1, 3072);
  zero_bytes(ws_decode_raw + 536512, 12288);
  MemRef2 w57 = make_2(ws_decode_raw + 548800, 1, 3072);
  MemRef2 w58 = make_2(ws_decode_raw + 561088, 1, 3072);
  MemRef1 w59 = make_1(ws_decode_raw + 564160, 1);
  MemRef2 w60 = make_2(ws_decode_raw + 564224, 1, 1024);
  zero_bytes(ws_decode_raw + 564224, 4096);
  MemRef2 w61 = make_2(ws_decode_raw + 568320, 1, 1024);
  MemRef2 w62 = make_2(ws_decode_raw + 572416, 1, 1024);
  MemRef1 w63 = make_1(ws_decode_raw + 573440, 1);
  MemRef2 w64 = make_2(ws_decode_raw + 573504, 1, 2048);
  zero_bytes(ws_decode_raw + 573504, 8192);
  MemRef2 w65 = make_2(ws_decode_raw + 581696, 1, 2048);
  MemRef2 w66 = make_2(ws_decode_raw + 589888, 1, 1024);
  MemRef1 w67 = make_1(ws_decode_raw + 590912, 1);
  MemRef2 w68 = make_2(ws_decode_raw + 590976, 1, 1024);
  zero_bytes(ws_decode_raw + 590976, 4096);
  MemRef2 w69 = make_2(ws_decode_raw + 595072, 1, 1024);
  MemRef2 w70 = make_2(ws_decode_raw + 599168, 1, 1024);
  MemRef1 w71 = make_1(ws_decode_raw + 600192, 1);
  MemRef2 w72 = make_2(ws_decode_raw + 600256, 1, 1024);
  zero_bytes(ws_decode_raw + 600256, 4096);
  MemRef2 w73 = make_2(ws_decode_raw + 604352, 1, 1024);
  MemRef2 w74 = make_2(ws_decode_raw + 608448, 1, 2048);
  MemRef1 w75 = make_1(ws_decode_raw + 610496, 1);
  MemRef2 w76 = make_2(ws_decode_raw + 610560, 1, 1024);
  zero_bytes(ws_decode_raw + 610560, 4096);
  MemRef2 w77 = make_2(ws_decode_raw + 614656, 1, 1024);
  MemRef2 w78 = make_2(ws_decode_raw + 618752, 1, 1024);
  MemRef1 w79 = make_1(ws_decode_raw + 619776, 1);
  MemRef2 w80 = make_2(ws_decode_raw + 619840, 1, 3072);
  zero_bytes(ws_decode_raw + 619840, 12288);
  MemRef2 w81 = make_2(ws_decode_raw + 632128, 1, 3072);
  MemRef2 w82 = make_2(ws_decode_raw + 644416, 1, 1024);
  MemRef1 w83 = make_1(ws_decode_raw + 645440, 1);
  MemRef2 w84 = make_2(ws_decode_raw + 645504, 1, 3072);
  zero_bytes(ws_decode_raw + 645504, 12288);
  MemRef2 w85 = make_2(ws_decode_raw + 657792, 1, 3072);
  MemRef2 w86 = make_2(ws_decode_raw + 670080, 1, 3072);
  MemRef1 w87 = make_1(ws_decode_raw + 673152, 1);
  MemRef2 w88 = make_2(ws_decode_raw + 673216, 1, 1024);
  zero_bytes(ws_decode_raw + 673216, 4096);
  MemRef2 w89 = make_2(ws_decode_raw + 677312, 1, 1024);
  MemRef2 w90 = make_2(ws_decode_raw + 681408, 1, 1024);
  MemRef1 w91 = make_1(ws_decode_raw + 682432, 1);
  MemRef2 w92 = make_2(ws_decode_raw + 682496, 1, 2048);
  zero_bytes(ws_decode_raw + 682496, 8192);
  MemRef2 w93 = make_2(ws_decode_raw + 690688, 1, 2048);
  MemRef2 w94 = make_2(ws_decode_raw + 698880, 1, 1024);
  MemRef1 w95 = make_1(ws_decode_raw + 699904, 1);
  MemRef2 w96 = make_2(ws_decode_raw + 699968, 1, 1024);
  zero_bytes(ws_decode_raw + 699968, 4096);
  MemRef2 w97 = make_2(ws_decode_raw + 704064, 1, 1024);
  MemRef2 w98 = make_2(ws_decode_raw + 708160, 1, 1024);
  MemRef1 w99 = make_1(ws_decode_raw + 709184, 1);
  MemRef2 w100 = make_2(ws_decode_raw + 709248, 1, 1024);
  zero_bytes(ws_decode_raw + 709248, 4096);
  MemRef2 w101 = make_2(ws_decode_raw + 713344, 1, 1024);
  MemRef2 w102 = make_2(ws_decode_raw + 717440, 1, 2048);
  MemRef1 w103 = make_1(ws_decode_raw + 719488, 1);
  MemRef2 w104 = make_2(ws_decode_raw + 719552, 1, 1024);
  zero_bytes(ws_decode_raw + 719552, 4096);
  MemRef2 w105 = make_2(ws_decode_raw + 723648, 1, 1024);
  MemRef2 w106 = make_2(ws_decode_raw + 727744, 1, 1024);
  MemRef1 w107 = make_1(ws_decode_raw + 728768, 1);
  MemRef2 w108 = make_2(ws_decode_raw + 728832, 1, 3072);
  zero_bytes(ws_decode_raw + 728832, 12288);
  MemRef2 w109 = make_2(ws_decode_raw + 741120, 1, 3072);
  MemRef2 w110 = make_2(ws_decode_raw + 753408, 1, 1024);
  MemRef1 w111 = make_1(ws_decode_raw + 754432, 1);
  MemRef2 w112 = make_2(ws_decode_raw + 754496, 1, 3072);
  zero_bytes(ws_decode_raw + 754496, 12288);
  MemRef2 w113 = make_2(ws_decode_raw + 766784, 1, 3072);
  MemRef2 w114 = make_2(ws_decode_raw + 779072, 1, 3072);
  MemRef1 w115 = make_1(ws_decode_raw + 782144, 1);
  MemRef2 w116 = make_2(ws_decode_raw + 782208, 1, 1024);
  zero_bytes(ws_decode_raw + 782208, 4096);
  MemRef2 w117 = make_2(ws_decode_raw + 786304, 1, 1024);
  MemRef2 w118 = make_2(ws_decode_raw + 790400, 1, 1024);
  MemRef1 w119 = make_1(ws_decode_raw + 791424, 1);
  MemRef2 w120 = make_2(ws_decode_raw + 791488, 1, 2048);
  zero_bytes(ws_decode_raw + 791488, 8192);
  MemRef2 w121 = make_2(ws_decode_raw + 799680, 1, 2048);
  MemRef2 w122 = make_2(ws_decode_raw + 807872, 1, 1024);
  MemRef1 w123 = make_1(ws_decode_raw + 808896, 1);
  MemRef2 w124 = make_2(ws_decode_raw + 808960, 1, 1024);
  zero_bytes(ws_decode_raw + 808960, 4096);
  MemRef2 w125 = make_2(ws_decode_raw + 813056, 1, 1024);
  MemRef2 w126 = make_2(ws_decode_raw + 817152, 1, 1024);
  MemRef1 w127 = make_1(ws_decode_raw + 818176, 1);
  MemRef2 w128 = make_2(ws_decode_raw + 818240, 1, 1024);
  zero_bytes(ws_decode_raw + 818240, 4096);
  MemRef2 w129 = make_2(ws_decode_raw + 822336, 1, 1024);
  MemRef2 w130 = make_2(ws_decode_raw + 826432, 1, 2048);
  MemRef1 w131 = make_1(ws_decode_raw + 828480, 1);
  MemRef2 w132 = make_2(ws_decode_raw + 828544, 1, 1024);
  zero_bytes(ws_decode_raw + 828544, 4096);
  MemRef2 w133 = make_2(ws_decode_raw + 832640, 1, 1024);
  MemRef2 w134 = make_2(ws_decode_raw + 836736, 1, 1024);
  MemRef1 w135 = make_1(ws_decode_raw + 837760, 1);
  MemRef2 w136 = make_2(ws_decode_raw + 837824, 1, 3072);
  zero_bytes(ws_decode_raw + 837824, 12288);
  MemRef2 w137 = make_2(ws_decode_raw + 850112, 1, 3072);
  MemRef2 w138 = make_2(ws_decode_raw + 862400, 1, 1024);
  MemRef1 w139 = make_1(ws_decode_raw + 863424, 1);
  MemRef2 w140 = make_2(ws_decode_raw + 863488, 1, 3072);
  zero_bytes(ws_decode_raw + 863488, 12288);
  MemRef2 w141 = make_2(ws_decode_raw + 875776, 1, 3072);
  MemRef2 w142 = make_2(ws_decode_raw + 888064, 1, 3072);
  MemRef1 w143 = make_1(ws_decode_raw + 891136, 1);
  MemRef2 w144 = make_2(ws_decode_raw + 891200, 1, 1024);
  zero_bytes(ws_decode_raw + 891200, 4096);
  MemRef2 w145 = make_2(ws_decode_raw + 895296, 1, 1024);
  MemRef2 w146 = make_2(ws_decode_raw + 899392, 1, 1024);
  MemRef1 w147 = make_1(ws_decode_raw + 900416, 1);
  MemRef2 w148 = make_2(ws_decode_raw + 900480, 1, 151936);
  zero_bytes(ws_decode_raw + 900480, 607744);
  MemRef2 w149 = make_2(ws_decode_raw + 1508224, 1, 151936);
  uint64_t preparation_cycles = nr_cycles() - preparation_begin;
  if (trace) { nr_puts("[model] decode begin position=");
    nr_hex32(position); nr_puts(" input_token=");
    nr_hex32((unsigned)input_ids[0]); nr_puts("\r\n"); }
  qwen_profile_reset();
  uint64_t begin = nr_cycles();
  _mlir_ciface_forward_decode(&result, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15, &p16, &p17, &p18, &p19, &p20, &p21, &p22, &p23, &p24, &p25, &p26, &p27, &p28, &p29, &p30, &p31, &p32, &p33, &p34, &p35, &p36, &p37, &p38, &p39, &p40, &p41, &p42, &p43, &p44, &p45, &p46, &p47, &p48, &p49, &p50, &p51, &p52, &p53, &p54, &p55, &p56, &p57, &p58, &p59, &p60, &p61, &p62, &p63, &p64, &p65, &p66, &p67, &p68, &p69, &p70, &p71, &p72, &p73, &p74, &p75, &ids, &pos, &k[0], &v[0], &pos, &k[1], &v[1], &pos, &k[2], &v[2], &pos, &k[3], &v[3], &w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31, &w32, &w33, &w34, &w35, &w36, &w37, &w38, &w39, &w40, &w41, &w42, &w43, &w44, &w45, &w46, &w47, &w48, &w49, &w50, &w51, &w52, &w53, &w54, &w55, &w56, &w57, &w58, &w59, &w60, &w61, &w62, &w63, &w64, &w65, &w66, &w67, &w68, &w69, &w70, &w71, &w72, &w73, &w74, &w75, &w76, &w77, &w78, &w79, &w80, &w81, &w82, &w83, &w84, &w85, &w86, &w87, &w88, &w89, &w90, &w91, &w92, &w93, &w94, &w95, &w96, &w97, &w98, &w99, &w100, &w101, &w102, &w103, &w104, &w105, &w106, &w107, &w108, &w109, &w110, &w111, &w112, &w113, &w114, &w115, &w116, &w117, &w118, &w119, &w120, &w121, &w122, &w123, &w124, &w125, &w126, &w127, &w128, &w129, &w130, &w131, &w132, &w133, &w134, &w135, &w136, &w137, &w138, &w139, &w140, &w141, &w142, &w143, &w144, &w145, &w146, &w147, &w148, &w149);
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
  if (trace) qwen_profile_report(position + 1 - 1);
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

