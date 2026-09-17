#include "../nr_runtime.h"

/* Sparse address/alias check, not a complete DDR capacity or reliability test.
 * This standalone probe overwrites one 64-byte line at each listed address;
 * never combine it with an already-loaded model/weight image. */
static const uintptr_t addresses[] = {
    0xb8000000UL, 0xbc000000UL, 0xc0000000UL, 0xc4000000UL,
    0xc8000000UL, 0xcc000000UL, 0xd0000000UL, 0xd4000000UL,
    0xd8000000UL, 0xdc000000UL, 0xe0000000UL, 0xe4000000UL,
    0xe8000000UL, 0xec000000UL, 0xf0000000UL, 0xf4000000UL,
    0xf8000000UL, 0xfc000000UL, 0xfffff000UL};

static uint64_t pattern(uintptr_t address, unsigned index) {
  return ((uint64_t)address << 32) ^ address ^
         (UINT64_C(0x9e3779b97f4a7c15) * (index + 1u));
}

int launch(void) {
  const unsigned count = sizeof(addresses)/sizeof(addresses[0]);
  nr_puts("[ddr] sparse RA address/alias test; 19 lines, 64 bytes each\r\n");
  for (unsigned i = 0; i < count; ++i) {
    nr_puts("[ddr] write address=0x"); nr_hex64(addresses[i]); nr_puts("\r\n");
    volatile uint64_t *line = (volatile uint64_t *)addresses[i];
    for (unsigned j = 0; j < 8; ++j) line[j] = pattern(addresses[i], j);
    __asm__ volatile("fence rw, rw" ::: "memory");
  }
  /* Read only after every address has been written, so address aliases cannot
   * pass merely because the last store was read immediately. */
  for (unsigned i = 0; i < count; ++i) {
    volatile uint64_t *line = (volatile uint64_t *)addresses[i];
    for (unsigned j = 0; j < 8; ++j) {
      uint64_t actual = line[j], expected = pattern(addresses[i], j);
      if (actual != expected) {
        nr_puts("[ddr] FAIL address=0x"); nr_hex64(addresses[i] + 8u*j);
        nr_puts(" expected=0x"); nr_hex64(expected);
        nr_puts(" actual=0x"); nr_hex64(actual); nr_puts("\r\n");
        return 1;
      }
    }
    nr_puts("[ddr] read PASS address=0x"); nr_hex64(addresses[i]); nr_puts("\r\n");
  }
  nr_puts("verify sparse DDR address probe: PASS\r\n");
  return 0;
}
