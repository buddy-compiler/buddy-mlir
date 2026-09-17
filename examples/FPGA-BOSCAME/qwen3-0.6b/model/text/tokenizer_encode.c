/* Freestanding Qwen3 encoding: exact added-token splitting, NFC, the declared
 * Unicode regex, then byte-level BPE. Tables are generated from the reference
 * tokenizers engine. Storage is bounded and stack-owned; malformed UTF-8 or
 * capacity overflow is rejected rather than truncating user input. */
#include "tokenizer_resource.h"
#include "unicode_tables.h"

/* No <string.h> on the freestanding RISC-V target; the NR runtime provides
 * memcpy, so declare exactly what is used. */
void *memcpy(void *destination, const void *source, size_t count);
int memcmp(const void *left, const void *right, size_t count);

#define QWEN_MAX_CODEPOINTS QWEN_TEXT_MAX_CODEPOINTS
#define QWEN_MAX_TOKENS QWEN_TEXT_MAX_PIECE_BYTES

/* ------------------------------------------------------------------ tables */

static int range_has(const QwenRange *table, uint32_t count, uint32_t code) {
  uint32_t low = 0, high = count;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2;
    if (code < table[mid].first) {
      high = mid;
    } else if (code > table[mid].last) {
      low = mid + 1;
    } else {
      return 1;
    }
  }
  return 0;
}

static int is_letter(uint32_t code) {
  return range_has(qwen_letter_ranges, qwen_letter_count, code);
}
static int is_number(uint32_t code) {
  return range_has(qwen_number_ranges, qwen_number_count, code);
}
static int is_space(uint32_t code) {
  return range_has(qwen_space_ranges, qwen_space_count, code);
}

static uint32_t combining_of(uint32_t code) {
  uint32_t low = 0, high = qwen_combining_count;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2;
    if (qwen_combining[mid].code < code) {
      low = mid + 1;
    } else if (qwen_combining[mid].code > code) {
      high = mid;
    } else {
      return qwen_combining[mid].combining;
    }
  }
  return 0;
}

/* The generated table stores full canonical decompositions. A singleton is
 * one scalar, not a two-item record with a synthetic U+0000 suffix. */
static int decompose(uint32_t code, uint32_t *out, uint32_t capacity) {
  if (code >= 0xac00u && code < 0xac00u + 11172u) {
    uint32_t index = code - 0xac00u;
    uint32_t trail = index % 28u;
    uint32_t count = trail ? 3u : 2u;
    if (capacity < count) return -1;
    out[0] = 0x1100u + index / 588u;
    out[1] = 0x1161u + (index % 588u) / 28u;
    if (trail) out[2] = 0x11a7u + trail;
    return (int)count;
  }
  uint32_t low = 0, high = qwen_canonical_count;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2;
    const QwenDecomposition *entry = &qwen_canonical[mid];
    if (entry->code < code) low = mid + 1;
    else if (entry->code > code) high = mid;
    else {
      if (capacity < entry->count) return -1;
      for (uint32_t i = 0; i < entry->count; ++i) out[i] = entry->sequence[i];
      return (int)entry->count;
    }
  }
  if (!capacity) return -1;
  out[0] = code;
  return 1;
}

static uint32_t compose(uint32_t first, uint32_t second) {
  if (first >= 0x1100u && first < 0x1100u + 19u &&
      second >= 0x1161u && second < 0x1161u + 21u)
    return 0xac00u + ((first - 0x1100u) * 21u + second - 0x1161u) * 28u;
  if (first >= 0xac00u && first < 0xac00u + 11172u &&
      (first - 0xac00u) % 28u == 0 &&
      second > 0x11a7u && second < 0x11a7u + 28u)
    return first + second - 0x11a7u;
  uint32_t low = 0, high = qwen_composition_count;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2;
    const QwenComposition *row = &qwen_composition[mid];
    if (row->first < first || (row->first == first && row->second < second)) {
      low = mid + 1;
    } else if (row->first > first || (row->first == first && row->second > second)) {
      high = mid;
    } else {
      return row->composed;
    }
  }
  return 0;
}

/* --------------------------------------------------------------------- NFC */

static int nfc(const uint32_t *in, uint32_t count, uint32_t *out,
               uint32_t capacity) {
  uint32_t expanded = 0;
  for (uint32_t i = 0; i < count; ++i) {
    int added = decompose(in[i], out + expanded, capacity - expanded);
    if (added < 0) return -1;
    expanded += (uint32_t)added;
  }
  /* Stable canonical ordering within each starter segment. */
  for (uint32_t i = 1; i < expanded; ++i) {
    uint32_t value = out[i], klass = combining_of(value), j = i;
    while (j > 0 && klass != 0) {
      uint32_t previous = combining_of(out[j - 1]);
      if (previous == 0 || previous <= klass) break;
      out[j] = out[j - 1];
      --j;
    }
    out[j] = value;
  }
  /* Compose with the last starter, including across intervening lower-class
   * marks. Only an unconsumed mark blocks later composition. */
  uint32_t written = 0, starter = 0, previous_class = 0;
  int has_starter = 0;
  for (uint32_t i = 0; i < expanded; ++i) {
    uint32_t code = out[i], klass = combining_of(code), merged = 0;
    if (has_starter && (previous_class == 0 || previous_class < klass))
      merged = compose(out[starter], code);
    if (merged) {
      out[starter] = merged;
    } else {
      if (klass == 0) { starter = written; has_starter = 1; }
      out[written++] = code;
      previous_class = klass;
    }
  }
  return (int)written;
}

/* --------------------------------------------------------------------- UTF-8 */

static int utf8_decode(const uint8_t *text, size_t size, uint32_t *out,
                       uint32_t capacity) {
  uint32_t written = 0;
  for (size_t i = 0; i < size;) {
    uint8_t lead = text[i];
    uint32_t code, length;
    if (written == capacity) return -1;
    if (lead < 0x80u) { code = lead; length = 1; }
    else if (lead >= 0xc2u && lead <= 0xdfu) { code = lead & 0x1fu; length = 2; }
    else if (lead >= 0xe0u && lead <= 0xefu) { code = lead & 0x0fu; length = 3; }
    else if (lead >= 0xf0u && lead <= 0xf4u) { code = lead & 0x07u; length = 4; }
    else return -1;
    if (length > size - i) return -1;
    for (uint32_t k = 1; k < length; ++k) {
      uint8_t byte = text[i + k];
      if (byte < 0x80u || byte > 0xbfu) return -1;
      if (k == 1 && ((lead == 0xe0u && byte < 0xa0u) ||
                     (lead == 0xedu && byte > 0x9fu) ||
                     (lead == 0xf0u && byte < 0x90u) ||
                     (lead == 0xf4u && byte > 0x8fu))) return -1;
      code = (code << 6) | (byte & 0x3fu);
    }
    out[written++] = code;
    i += length;
  }
  return (int)written;
}

static size_t utf8_encode(const uint32_t *in, uint32_t count, uint8_t *out,
                          size_t capacity) {
  size_t written = 0;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t code = in[i];
    uint8_t buffer[4];
    size_t length;
    if (code < 0x80u) {
      buffer[0] = (uint8_t)code;
      length = 1;
    } else if (code < 0x800u) {
      buffer[0] = (uint8_t)(0xc0u | (code >> 6));
      buffer[1] = (uint8_t)(0x80u | (code & 0x3fu));
      length = 2;
    } else if (code < 0x10000u) {
      buffer[0] = (uint8_t)(0xe0u | (code >> 12));
      buffer[1] = (uint8_t)(0x80u | ((code >> 6) & 0x3fu));
      buffer[2] = (uint8_t)(0x80u | (code & 0x3fu));
      length = 3;
    } else {
      buffer[0] = (uint8_t)(0xf0u | (code >> 18));
      buffer[1] = (uint8_t)(0x80u | ((code >> 12) & 0x3fu));
      buffer[2] = (uint8_t)(0x80u | ((code >> 6) & 0x3fu));
      buffer[3] = (uint8_t)(0x80u | (code & 0x3fu));
      length = 4;
    }
    if (length > capacity - written) return (size_t)-1;
    memcpy(out + written, buffer, length);
    written += length;
  }
  return written;
}

/* -------------------------------------------------------- pre-tokenization */

/* The Qwen split pattern, tried in the declared order:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+
 *   |\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 * Each helper returns the matched length in code points, or 0 for no match. */
static int ascii_lower(uint32_t code) {
  /* Oniguruma's (?i:'s) also matches the Unicode long s. */
  if (code == 0x017fu) return 's';
  return code >= 'A' && code <= 'Z' ? (int)(code + 32) : (int)code;
}

static int match_suffix(const uint32_t *text, uint32_t count, uint32_t at,
                        const char *suffix) {
  uint32_t k = 0;
  while (suffix[k] != '\0') {
    if (at + k >= count || ascii_lower(text[at + k]) != suffix[k]) {
      return 0;
    }
    ++k;
  }
  return (int)k;
}

static uint32_t match_contraction(const uint32_t *text, uint32_t count,
                                  uint32_t at) {
  static const char *const forms[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  if (at >= count || text[at] != '\'') {
    return 0;
  }
  uint32_t best = 0;
  for (unsigned i = 0; i < sizeof(forms) / sizeof(forms[0]); ++i) {
    int length = match_suffix(text, count, at + 1, forms[i] + 1);
    if (length > 0 && (uint32_t)length + 1 > best) {
      best = (uint32_t)length + 1;
    }
  }
  return best;
}

static uint32_t match_optional_letter_run(const uint32_t *text, uint32_t count,
                                          uint32_t at) {
  uint32_t index = at;
  if (index < count && is_letter(text[index])) {
    /* the optional prefix is absent */
  } else if (index < count && text[index] != '\r' && text[index] != '\n'
             && !is_number(text[index]) && index + 1 < count
             && is_letter(text[index + 1])) {
    index += 1;
  } else {
    return 0;
  }
  uint32_t length = 0;
  while (index + length < count && is_letter(text[index + length])) {
    ++length;
  }
  return length > 0 ? index - at + length : 0;
}

static uint32_t match_symbol_run(const uint32_t *text, uint32_t count,
                                 uint32_t at) {
  uint32_t index = at;
  int leading_space = 0;
  if (index < count && text[index] == ' ') {
    leading_space = 1;
    index += 1;
  }
  uint32_t symbol = 0;
  while (index + symbol < count && !is_space(text[index + symbol])
         && !is_letter(text[index + symbol]) && !is_number(text[index + symbol])) {
    ++symbol;
  }
  if (symbol == 0) {
    return 0;
  }
  index += symbol;
  uint32_t breaks = 0;
  while (index + breaks < count
         && (text[index + breaks] == '\r' || text[index + breaks] == '\n')) {
    ++breaks;
  }
  return index - at + breaks - (leading_space && symbol == 0 ? 1u : 0u);
}

static uint32_t match_breaks(const uint32_t *text, uint32_t count, uint32_t at) {
  /* Greedy \s* can include newlines and spaces BETWEEN newlines; the final
   * [\r\n]+ backtracks to the last newline in this whitespace run. */
  uint32_t last_break = 0;
  for (uint32_t i = at; i < count && is_space(text[i]); ++i)
    if (text[i] == '\r' || text[i] == '\n') last_break = i - at + 1;
  return last_break;
}

static uint32_t match_trailing_spaces(const uint32_t *text, uint32_t count,
                                      uint32_t at) {
  uint32_t length = 0;
  while (at + length < count && is_space(text[at + length])) {
    ++length;
  }
  if (length == 0) {
    return 0;
  }
  /* \s+(?!\S) is NOT "whitespace at the end of the string". With a greedy
   * \s+ and a negative lookahead, a maximal run of n whitespace characters
   * followed by a non-space matches n-1 of them (the lookahead succeeds because
   * the character it peeks at is itself whitespace), and only at end of string
   * does the whole run match. Reading it the naive way is what made
   * "  leading" tokenise as one piece instead of " " + " leading". */
  if (at + length >= count) {
    return length;
  }
  return length > 1 ? length - 1 : 0;
}

static uint32_t match_spaces(const uint32_t *text, uint32_t count, uint32_t at) {
  uint32_t length = 0;
  while (at + length < count && is_space(text[at + length])) {
    ++length;
  }
  return length;
}

static uint32_t match_piece(const uint32_t *text, uint32_t count, uint32_t at) {
  uint32_t length;
  if ((length = match_contraction(text, count, at)) != 0) return length;
  if ((length = match_optional_letter_run(text, count, at)) != 0) return length;
  if (at < count && is_number(text[at])) return 1;
  if ((length = match_symbol_run(text, count, at)) != 0) return length;
  if ((length = match_breaks(text, count, at)) != 0) return length;
  if ((length = match_trailing_spaces(text, count, at)) != 0) return length;
  /* The final \s+ also catches a single space that is followed by more text. */
  return match_spaces(text, count, at);
}

/* --------------------------------------------------------------------- BPE */

/* Added tokens, matched exactly and longest-first as the packer stored them.
 * The packer rejects added tokens that declare single_word/lstrip/rstrip or
 * normalized flags, so plain byte matching is exactly what the resource
 * supports -- nothing here is approximating a flag that was dropped. */
static int match_added(const QwenTokenizerResource *r, const uint8_t *text,
                       size_t size, uint32_t *token, size_t *consumed) {
  for (uint32_t i = 0; i < r->added_count; ++i) {
    const uint8_t *p = r->blob + r->added_offset + (size_t)i * 4u;
    uint32_t id = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)
                  | ((uint32_t)p[3] << 24);
    const uint8_t *piece = 0;
    size_t length = 0;
    int special = 0;
    if (qwen_token_piece(r, id, &piece, &length, &special) != 0) continue;
    if (length && length <= size && memcmp(text, piece, length) == 0) {
      *token = id;
      *consumed = length;
      return 1;
    }
  }
  return 0;
}

static uint32_t byte_token(const QwenTokenizerResource *r, uint8_t byte) {
  const uint8_t *p = r->blob + r->size - 256u * 4u + (size_t)byte * 4u;
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

/* Lowest-rank merge among adjacent pairs, applied until none is left. The merge
 * table is keyed by token ids and sorted by (left, right), so this is a binary
 * search with no string work at all. */
static uint32_t merge_rank(const QwenTokenizerResource *r, uint32_t left,
                           uint32_t right, uint32_t *result) {
  uint32_t low = 0, high = r->merge_count;
  while (low < high) {
    uint32_t mid = low + (high - low) / 2;
    const uint8_t *row = r->blob + r->merge_offset + (size_t)mid * 16u;
    uint32_t rl = (uint32_t)row[0] | ((uint32_t)row[1] << 8)
                  | ((uint32_t)row[2] << 16) | ((uint32_t)row[3] << 24);
    uint32_t rr = (uint32_t)row[4] | ((uint32_t)row[5] << 8)
                  | ((uint32_t)row[6] << 16) | ((uint32_t)row[7] << 24);
    if (rl < left || (rl == left && rr < right)) {
      low = mid + 1;
    } else if (rl > left || (rl == left && rr > right)) {
      high = mid;
    } else {
      *result = (uint32_t)row[8] | ((uint32_t)row[9] << 8)
                | ((uint32_t)row[10] << 16) | ((uint32_t)row[11] << 24);
      return (uint32_t)row[12] | ((uint32_t)row[13] << 8)
             | ((uint32_t)row[14] << 16) | ((uint32_t)row[15] << 24);
    }
  }
  return 0xffffffffu;
}

int qwen_bpe(const QwenTokenizerResource *r, const uint8_t *bytes, size_t size,
             uint32_t *out, size_t capacity, size_t *written) {
  if (!r || !r->blob || (size && !bytes) || !out || !written || *written > capacity) {
    return -1;
  }
  if (size > QWEN_MAX_TOKENS) {
    return -1;
  }
  uint32_t symbols[QWEN_MAX_TOKENS];
  for (size_t i = 0; i < size; ++i) {
    symbols[i] = byte_token(r, bytes[i]);
  }
  uint32_t count = (uint32_t)size;
  for (;;) {
    uint32_t best_rank = 0xffffffffu, best_at = 0, best_result = 0;
    for (uint32_t i = 0; i + 1 < count; ++i) {
      uint32_t result = 0;
      uint32_t rank = merge_rank(r, symbols[i], symbols[i + 1], &result);
      if (rank < best_rank) {
        best_rank = rank;
        best_at = i;
        best_result = result;
      }
    }
    if (best_rank == 0xffffffffu) {
      break;
    }
    symbols[best_at] = best_result;
    for (uint32_t i = best_at + 1; i + 1 < count; ++i) {
      symbols[i] = symbols[i + 1];
    }
    --count;
  }
  if (count > capacity - *written) {
    return -1;
  }
  memcpy(out + *written, symbols, count * sizeof(uint32_t));
  *written += count;
  return 0;
}

static int encode_plain(const QwenTokenizerResource *r, const uint8_t *text,
                        size_t size, uint32_t *out, size_t capacity,
                        size_t *written) {
  uint32_t codepoints[QWEN_MAX_CODEPOINTS];
  uint32_t normalised[QWEN_MAX_CODEPOINTS];
  uint8_t utf8[QWEN_TEXT_MAX_PIECE_BYTES];
  int count = utf8_decode(text, size, codepoints, QWEN_MAX_CODEPOINTS);
  if (count < 0) return -1;
  int composed = nfc(codepoints, (uint32_t)count, normalised, QWEN_MAX_CODEPOINTS);
  if (composed < 0) return -1;
  uint32_t cursor = 0;
  while (cursor < (uint32_t)composed) {
    uint32_t length = match_piece(normalised, (uint32_t)composed, cursor);
    if (!length || length > (uint32_t)composed - cursor) return -1;
    size_t bytes = utf8_encode(normalised + cursor, length, utf8, sizeof(utf8));
    if (bytes == (size_t)-1 || qwen_bpe(r, utf8, bytes, out, capacity, written))
      return -1;
    cursor += length;
  }
  return 0;
}

int qwen_encode(const QwenTokenizerResource *r, const uint8_t *text, size_t size,
                uint32_t *out, size_t capacity, size_t *written) {
  if (!written) return -1;
  *written = 0;
  if (!r || !r->blob || (size && !text) || !out ||
      size > QWEN_TEXT_MAX_INPUT_BYTES) return -1;
  if (!size) return 0;
  /* normalized=false added tokens are recognized on ORIGINAL UTF-8 before
   * normalizing the ordinary spans. This also prevents the regex swallowing
   * a token's opening punctuation. */
  size_t at = 0, plain = 0;
  while (at < size) {
    uint32_t token = 0;
    size_t consumed = 0;
    if (match_added(r, text + at, size - at, &token, &consumed)) {
      if (encode_plain(r, text + plain, at - plain, out, capacity, written) ||
          *written == capacity) goto error;
      out[(*written)++] = token;
      at += consumed;
      plain = at;
    } else {
      ++at;
    }
  }
  if (encode_plain(r, text + plain, size - plain, out, capacity, written)) goto error;
  return 0;
error:
  *written = 0;
  return -1;
}
