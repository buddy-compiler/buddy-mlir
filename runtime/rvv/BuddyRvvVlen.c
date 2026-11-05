//===- BuddyRvvVlen.c - Check the VLEN a model is built for --------------===//
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
//
// Linked into the model library of a buddy-codegen model built for RISC-V
// Vector of an exact VLEN: BUDDY_RISCV_VLEN, or 1024 for the SpacemiT K3 A100
// code of "prefill_ime". Its vector code is for that VLEN only: llc
// is told the exact VLEN (+zvlNb, -riscv-v-vector-bits-max) and
// -lower-vir-to-vector sizes its scalable vectors for it, so on a CPU of
// another VLEN the model computes wrong results instead of failing. When the
// library is loaded, it reads the VLEN of the CPU and stops the process if
// it differs. On the K3, a process that is not on the A100 cores (VLEN 1024)
// runs on the X100 cores (VLEN 256).
//
// BUDDY_RVV_VLEN: the VLEN in bits the model is built for.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <unistd.h>

#ifndef BUDDY_RVV_VLEN
#error "BUDDY_RVV_VLEN: the VLEN in bits the model is built for"
#endif

__attribute__((constructor)) static void buddyRvvVlenCheck(void) {
  unsigned long vlenb;
  // vlenb (CSR 0xc22) by number: the assembler knows its name only with V
  __asm__ volatile("csrr %0, 0xc22" : "=r"(vlenb));
  if (vlenb * 8 == BUDDY_RVV_VLEN)
    return;
  fprintf(stderr,
          "buddy model: built for RISC-V Vector VLEN %d, but this CPU has "
          "VLEN %lu: the results would be wrong.\n"
          "  On the SpacemiT K3, run the model on the A100 cores (VLEN "
          "1024): `echo 0 > /proc/set_ai_thread` in the shell that starts "
          "it (docs/K3DeepSeekR1.md).\n",
          BUDDY_RVV_VLEN, vlenb * 8);
  // not exit(): it would run the destructors under the loader's lock
  _exit(1);
}
