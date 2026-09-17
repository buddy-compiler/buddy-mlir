/* Runs on RA with the shared NR runtime. Expected IDs are an oracle only:
 * all printed IDs and decoded bytes come from the actual board text path. */
#include "nr_runtime.h"
#include "tokenizer_resource.h"
#include "tokenizer_probe_fixtures.h"

extern const unsigned char qwen_tokenizer_blob_start[];
extern const unsigned char qwen_tokenizer_blob_end[];

typedef struct {
  uint8_t bytes[4096];
  size_t size;
  int failed;
} Decoded;

static void collect(void *context, const uint8_t *bytes, size_t count) {
  Decoded *out = (Decoded *)context;
  if (count > sizeof(out->bytes) - out->size) { out->failed = 1; return; }
  for (size_t i = 0; i < count; ++i) out->bytes[out->size++] = bytes[i];
}

static void hex_bytes(const uint8_t *bytes, size_t count) {
  const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < count; ++i) {
    char pair[2] = {digits[bytes[i] >> 4], digits[bytes[i] & 15]};
    nr_write(pair, sizeof(pair));
  }
}

int launch(void) {
  nr_puts("[tokenizer] BEGIN actual template/encoder/incremental decoder\r\n");
  QwenTokenizerResource resource;
  const size_t resource_size = (size_t)(qwen_tokenizer_blob_end - qwen_tokenizer_blob_start);
  uint64_t begin = nr_cycles();
  if (qwen_tokenizer_open(&resource, qwen_tokenizer_blob_start, resource_size)) {
    nr_puts("verify tokenizer resource: FAIL\r\n"); return 1;
  }
  nr_puts("[tokenizer] resource_bytes="); nr_hex64(resource_size);
  nr_puts(" open_cycles="); nr_hex64(nr_cycles() - begin); nr_puts("\r\n");
  unsigned failures = 0;
  for (unsigned i = 0; i < TOKENIZER_PROBE_CASE_COUNT; ++i) {
    const TokenizerProbeFixture *fixture = &tokenizer_probe_fixtures[i];
    uint8_t prompt[4096];
    uint32_t ids[512];
    size_t prompt_size = 0, count = 0;
    uint64_t start = nr_cycles();
    int status = qwen_chat_single_turn(prompt, sizeof(prompt), &prompt_size,
                                       0, 0, 0, fixture->user, fixture->user_size,
                                       fixture->thinking);
    if (!status) status = qwen_encode(&resource, prompt, prompt_size, ids,
                                     sizeof(ids)/sizeof(ids[0]), &count);
    uint64_t encode_cycles = nr_cycles() - start;
    nr_puts("[tokenizer] case="); nr_hex32(i);
    nr_puts(" count="); nr_hex32((uint32_t)count); nr_puts(" ids=");
    for (size_t j = 0; j < count; ++j) {
      if (j) nr_puts(",");
      nr_hex32(ids[j]);
    }
    nr_puts("\r\n");
    int match = !status && count == fixture->expected_count;
    for (size_t j = 0; j < count && j < fixture->expected_count; ++j)
      if (ids[j] != fixture->expected_ids[j]) match = 0;
    QwenUtf8Decoder state = {{0, 0, 0, 0}, 0, 0};
    Decoded decoded = {{0}, 0, 0};
    start = nr_cycles();
    for (size_t j = 0; j < count; ++j)
      if (qwen_decode_token(&resource, &state, ids[j], 0, collect, &decoded)) match = 0;
    qwen_decode_finish(&state, collect, &decoded);
    uint64_t decode_cycles = nr_cycles() - start;
    if (decoded.failed || decoded.size != fixture->decoded_size) match = 0;
    for (size_t j = 0; j < decoded.size && j < fixture->decoded_size; ++j)
      if (decoded.bytes[j] != fixture->decoded[j]) match = 0;
    nr_puts("[tokenizer] case="); nr_hex32(i); nr_puts(" decoded_hex=");
    hex_bytes(decoded.bytes, decoded.size); nr_puts("\r\n");
    nr_puts("[tokenizer] case="); nr_hex32(i);
    nr_puts(" encode_cycles="); nr_hex64(encode_cycles);
    nr_puts(" decode_cycles="); nr_hex64(decode_cycles); nr_puts("\r\n");
    nr_puts("verify tokenizer case "); nr_hex32(i);
    nr_puts(match ? ": PASS\r\n" : ": FAIL\r\n");
    failures += !match;
  }
  nr_puts("[tokenizer] checked="); nr_hex32(TOKENIZER_PROBE_CASE_COUNT);
  nr_puts(" failures="); nr_hex32(failures); nr_puts("\r\n");
  nr_puts(failures ? "verify tokenizer suite: FAIL\r\n" : "verify tokenizer suite: PASS\r\n");
  return failures ? 1 : 0;
}
