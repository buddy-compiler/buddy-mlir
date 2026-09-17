#include "../nr_runtime.h"

/* No model, tokenizer, or host inference: prove that NH receives physical UART
 * bytes and publishes them to RA. Host sends exactly the line below. */
int launch(void) {
  static const char expected[] = "BOSC NR UART RX\n";
  nr_puts("[rx] ready: send BOSC NR UART RX followed by LF\r\n");
  uintptr_t mark = nr_heap_mark();
  unsigned char *scratch = malloc(129);
  scratch[128] = 0x5a;
  if (scratch[128] != 0x5a) return 1;
  nr_heap_reset(mark);
  if (nr_heap_mark() != mark) return 1;
  unsigned int index = 0;
  while (index < sizeof(expected) - 1) {
    int value = nr_getchar();
    if (value < 0) continue;
    nr_puts("[rx] byte=0x");
    nr_hex32((uint32_t)value);
    nr_puts("\r\n");
    if (value != (unsigned char)expected[index++]) {
      nr_puts("verify UART RX: FAIL\r\n");
      return 1;
    }
  }
  nr_puts("verify UART RX: PASS\r\n");
  return 0;
}
