/* Host ASan/UBSan harness for public resource/encoder boundary handling. */
#include "../text/tokenizer_resource.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  FILE *f = fopen(argv[1], "rb");
  assert(f && fseek(f, 0, SEEK_END) == 0);
  long size = ftell(f);
  assert(size > 0 && fseek(f, 0, SEEK_SET) == 0);
  unsigned char *blob = malloc((size_t)size);
  assert(blob && fread(blob, 1, (size_t)size, f) == (size_t)size);
  fclose(f);
  QwenTokenizerResource r;
  assert(qwen_tokenizer_open(&r, blob, (size_t)size) == 0);
  uint32_t *ids = malloc(32769u * sizeof(uint32_t));
  unsigned char *text = malloc(32769u);
  assert(ids && text);
  memset(text, '1', 32769u);
  size_t written = 999;
  assert(qwen_encode(&r, 0, 0, ids, 0, &written) == 0 && written == 0);
  assert(qwen_encode(&r, text, 8192, ids, 8192, &written) == 0 && written == 8192);
  assert(qwen_encode(&r, text, 8193, ids, 32769, &written) == -1 && written == 0);
  assert(qwen_encode(&r, text, 32769, ids, 32769, &written) == -1 && written == 0);
  assert(qwen_encode(&r, text, 8192, ids, 8191, &written) == -1 && written == 0);
  static const unsigned char bad[][4] = {
      {0xc0,0xaf,0,0}, {0xe0,0x80,0xaf,0}, {0xed,0xa0,0x80,0},
      {0xf0,0x80,0x80,0xaf}, {0xf4,0x90,0x80,0x80}};
  for (size_t i=0; i<sizeof(bad)/sizeof(bad[0]); ++i)
    assert(qwen_encode(&r,bad[i],sizeof(bad[i]),ids,32769,&written)==-1 && written==0);
  assert(qwen_encode(&r,(const uint8_t *)"a\xe4\xb8",3,ids,32769,&written)==-1 && written==0);
  assert(qwen_tokenizer_open(&r,blob,12)==-1 && !r.blob);
  assert(qwen_encode(&r,text,1,ids,32769,&written)==-1 && written==0);
  free(text); free(ids); free(blob);
  puts("PASS host sanitizer boundary checks");
  return 0;
}
