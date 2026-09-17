/* Host-only differential-test access to actual static normalization/regex code.
 * Production and this harness compile the same implementation. */
#include "../text/tokenizer_encode.c"

size_t qwen_test_resource_size(void) { return sizeof(QwenTokenizerResource); }
int qwen_test_nfc(const uint32_t *input, uint32_t count, uint32_t *output,
                  uint32_t capacity) {
  return nfc(input, count, output, capacity);
}
int qwen_test_piece_lengths(const uint32_t *input, uint32_t count,
                            uint32_t *lengths, uint32_t capacity) {
  uint32_t at = 0, used = 0;
  while (at < count) {
    uint32_t length = match_piece(input, count, at);
    if (!length || length > count - at || used == capacity) return -1;
    lengths[used++] = length;
    at += length;
  }
  return (int)used;
}
