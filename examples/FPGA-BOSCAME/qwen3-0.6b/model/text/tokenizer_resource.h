#ifndef QWEN_TOKENIZER_RESOURCE_H
#define QWEN_TOKENIZER_RESOURCE_H
#include <stddef.h>
#include <stdint.h>

/* A whole request is bounded in bytes; each ordinary (non-added-token) span
 * must fit this many Unicode scalars before AND during NFC decomposition. */
#define QWEN_TEXT_MAX_CODEPOINTS 8192u
#define QWEN_TEXT_MAX_PIECE_BYTES (QWEN_TEXT_MAX_CODEPOINTS * 4u)
#define QWEN_TEXT_MAX_INPUT_BYTES QWEN_TEXT_MAX_PIECE_BYTES

/* No files, allocation, global state or UART ownership. The caller supplies
 * immutable DDR resources and a byte-output callback (NR console in firmware).
 * All input ranges are bounded; blob is little-endian and may be unaligned. */
typedef struct {
  const uint8_t *blob;
  size_t size;
  uint32_t slots, records_offset, pieces_offset, pieces_size;
  /* Merge table, keyed by token id and sorted by (left, right): the encoder
   * binary-searches it, so it needs the bounds, not the count of merges. */
  uint32_t merge_offset, merge_count;
  /* Added tokens, packed longest-first so an exact match is deterministic. The
   * encoder scans these before the regex, because the chat template's
   * <|im_start|> and friends have to become their own ids rather than literal
   * text. */
  uint32_t added_offset, added_count;
} QwenTokenizerResource;
typedef void (*QwenEmit)(void *context, const uint8_t *bytes, size_t count);
typedef struct {
  uint8_t pending[4];
  uint8_t count, expected;
} QwenUtf8Decoder;

int qwen_tokenizer_open(QwenTokenizerResource *out, const void *blob, size_t size);
int qwen_token_piece(const QwenTokenizerResource *resource, uint32_t token,
                     const uint8_t **bytes, size_t *count, int *is_special);
void qwen_decode_bytes(QwenUtf8Decoder *state, const uint8_t *bytes, size_t count,
                       QwenEmit emit, void *context);
int qwen_decode_token(const QwenTokenizerResource *resource, QwenUtf8Decoder *state,
                      uint32_t token, int skip_special, QwenEmit emit, void *context);
void qwen_decode_finish(QwenUtf8Decoder *state, QwenEmit emit, void *context);

/* Encoding: exact unnormalized added tokens, NFC, the declared split regex,
 * byte-level mapping, then BPE merges. Bounded, reentrant and allocation-free;
 * scratch is on the caller's stack (less than 256 KiB for this implementation).
 * Malformed UTF-8 is REJECTED. NUL is a valid scalar. Returns -1 on malformed
 * input, resource/input/output capacity error or uncovered text, with *written
 * reset to zero. On error discard partially written output. */
int qwen_encode(const QwenTokenizerResource *resource, const uint8_t *text,
                size_t size, uint32_t *out, size_t capacity, size_t *written);
int qwen_bpe(const QwenTokenizerResource *resource, const uint8_t *bytes,
             size_t size, uint32_t *out, size_t capacity, size_t *written);

/* Supported chat-template subset: optional system, exactly one user message,
 * no tools/history, generation prompt. Content is UTF-8, may contain NUL bytes.
 * enable_thinking=0 adds the official empty thinking span. No BOS is added.
 * Capacity error returns -1; caller must discard partially written output. */
int qwen_chat_single_turn(uint8_t *out, size_t capacity, size_t *written,
                          const uint8_t *system, size_t system_size, int has_system,
                          const uint8_t *user, size_t user_size, int enable_thinking);
#endif
