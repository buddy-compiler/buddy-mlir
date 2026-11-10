//===- uart_rx_launch.c - NR probe: UART RX launch ------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "../nr_runtime.h"

// No model, tokenizer, or host inference: prove that NH receives physical UART
// bytes and publishes them to RA. Host sends exactly the line below.
int launch(void) {
  static const char expected[] = "BOSC NR UART RX\n";
  nr_puts("[rx] ready: send BOSC NR UART RX followed by LF\r\n");
  uintptr_t mark = nr_heap_mark();
  unsigned char *scratch = malloc(129);
  scratch[128] = 0x5a;
  if (scratch[128] != 0x5a)
    return 1;
  nr_heap_reset(mark);
  if (nr_heap_mark() != mark)
    return 1;
  unsigned int index = 0;
  while (index < sizeof(expected) - 1) {
    int value = nr_getchar();
    if (value < 0)
      continue;
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
