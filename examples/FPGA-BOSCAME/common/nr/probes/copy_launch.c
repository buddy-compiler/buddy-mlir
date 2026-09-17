#include "nr_runtime.h"

static unsigned char input[640] __attribute__((aligned(64)));
static unsigned char output[640] __attribute__((aligned(64)));

int nr_copy_probe(void) {
  static const unsigned sizes[] = {
    0, 1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 513
  };
  unsigned errors = 0, cases = 0;
  for (unsigned i = 0; i < sizeof(input); ++i) input[i] = (unsigned char)(i * 71 + 19);
  for (unsigned source_alignment = 0; source_alignment < 4; ++source_alignment)
    for (unsigned output_alignment = 0; output_alignment < 4; ++output_alignment)
      for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        unsigned size = sizes[s], begin = 32 + output_alignment;
        for (unsigned i = 0; i < sizeof(output); ++i) output[i] = 0xad;
        nr_copy_bytes(output + begin, input + 32 + source_alignment, size);
        for (unsigned i = 0; i < sizeof(output); ++i) {
          unsigned char want = i >= begin && i < begin + size
            ? input[32 + source_alignment + i - begin] : 0xad;
          errors += output[i] != want;
        }
        ++cases;
      }
  nr_puts("verify nr_copy_bytes: "); nr_puts(errors ? "FAIL" : "PASS");
  nr_puts(" cases=0x"); nr_hex32(cases);
  nr_puts(" errors=0x"); nr_hex32(errors); nr_puts("\r\n");
  return errors != 0;
}
#ifndef NR_COPY_PROBE_EMBEDDED
int launch(void) { return nr_copy_probe(); }
#endif
