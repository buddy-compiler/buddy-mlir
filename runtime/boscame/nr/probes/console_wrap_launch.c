//===- console_wrap_launch.c - NR probe: console wrap launch --------------===//
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

int launch(void) {
  static const char row[] =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\r\n";
  nr_puts("[tx] BEGIN\r\n");
  // 67,584 bytes exceeds the old one-shot 65,536-byte console.
  for (unsigned i = 0; i < 1024; ++i)
    nr_write(row, sizeof(row) - 1);
  nr_puts("[tx] END\r\nverify NR console wrap: PASS\r\n");
  return 0;
}
