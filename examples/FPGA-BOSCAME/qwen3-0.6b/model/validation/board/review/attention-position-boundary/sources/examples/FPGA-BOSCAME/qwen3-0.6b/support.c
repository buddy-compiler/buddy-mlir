#include "support.h"
#ifdef HOST_TEST
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define ARENA_ATTR __attribute__((aligned(64)))
void nr_copy_bytes(void *d,const void *s,size_t n) { memcpy(d,s,n); }
void nr_puts(const char *s) { fputs(s, stdout); fflush(stdout); }
void nr_hex32(uint32_t x) { printf("%08x", x); }
void nr_hex64(uint64_t x) { printf("%016llx", (unsigned long long)x); }
uint64_t nr_cycles(void) { return (uint64_t)clock(); }
#else
/* Explicit NOBITS keeps each object small as well as the final raw image. */
__asm__(".section .workspace,\"aw\",@nobits\n.balign 64\nqwen_arena:\n.skip 671088640\n.previous\n");
extern unsigned char arena[] __asm__("qwen_arena");
#endif
/* NOLOAD on NR, outside the b0000000..b7ffffff fault aperture. Each test
 * initializes exactly the elements its kernel may read; no huge upload. */
#ifdef HOST_TEST
static unsigned char arena[640u * 1024u * 1024u] ARENA_ATTR;
#endif
void *workspace(size_t offset) { return arena + offset; }
int check_close(float a, float b, float atol, float rtol) {
  union { float f; uint32_t u; } av={a}, bv={b};
  if ((av.u & 0x7f800000u)==0x7f800000u || (bv.u & 0x7f800000u)==0x7f800000u)
    return a == b; /* Only equal infinities pass; NaNs never do. */
  float d = a - b; if (d < 0) d = -d;
  float v = b < 0 ? -b : b;
  return d <= atol + rtol * v; /* NaNs always fail. */
}
int print_check(const char *name, unsigned errors, float max_error) {
  nr_puts("verify "); nr_puts(name); nr_puts(errors ? ": FAIL errors=" : ": PASS errors=");
  nr_hex32(errors); nr_puts(" max_abs_error_f32_bits=");
  union { float f; uint32_t u; } v = {max_error}; nr_hex32(v.u); nr_puts("\r\n");
  return errors != 0;
}
