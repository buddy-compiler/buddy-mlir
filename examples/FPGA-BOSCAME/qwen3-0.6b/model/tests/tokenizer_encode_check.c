/* Host harness: encode stdin lines with the bare-metal encoder and print ids.
 * Compared against the official tokenizer by tools/check_tokenizer_encode.py. */
#include "tokenizer_resource.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  FILE *f = fopen(argv[1], "rb");
  if (!f) return 2;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *blob = malloc((size_t)size);
  if (fread(blob, 1, (size_t)size, f) != (size_t)size) return 2;
  fclose(f);
  QwenTokenizerResource r;
  if (qwen_tokenizer_open(&r, blob, (size_t)size) != 0) { fprintf(stderr, "open failed\n"); return 2; }
  static uint32_t ids[8192];
  /* NUL-separated records, not lines: the corpus deliberately contains newlines
   * and they must reach the encoder intact. */
  static unsigned char record[65536];
  size_t n = 0;
  int c;
  while ((c = fgetc(stdin)) != EOF) {
    if (c != '\0') {
      if (n + 1 < sizeof(record)) record[n++] = (unsigned char)c;
      continue;
    }
    size_t written = 0;
    if (qwen_encode(&r, record, n, ids, 8192, &written) != 0) {
      printf("ERROR\n");
    } else {
      for (size_t i = 0; i < written; ++i) printf("%s%u", i ? " " : "", ids[i]);
      printf("\n");
    }
    n = 0;
  }
  free(blob);
  return 0;
}
