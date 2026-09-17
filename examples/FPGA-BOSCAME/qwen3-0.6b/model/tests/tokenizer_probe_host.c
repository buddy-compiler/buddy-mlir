/* Host-only shim for the probe's launch/verifier; not linked into the NR ELF.
 * The cycle field here holds clock() ticks and is NOT an FPGA cycle measurement. */
#include "nr_runtime.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
void nr_puts(const char *text) { fputs(text, stdout); }
void nr_write(const void *bytes, size_t count) { fwrite(bytes, 1, count, stdout); }
void nr_hex32(uint32_t value) { printf("%08" PRIX32, value); }
void nr_hex64(uint64_t value) { printf("%016" PRIX64, value); }
uint64_t nr_cycles(void) { return (uint64_t)clock(); }
int main(void) { return launch(); }
