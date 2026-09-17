#include "../nr_runtime.h"

int launch(void) {
  static const char row[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\r\n";
  nr_puts("[tx] BEGIN\r\n");
  /* 67,584 bytes exceeds the old one-shot 65,536-byte console. */
  for (unsigned i = 0; i < 1024; ++i) nr_write(row, sizeof(row)-1);
  nr_puts("[tx] END\r\nverify NR console wrap: PASS\r\n");
  return 0;
}
