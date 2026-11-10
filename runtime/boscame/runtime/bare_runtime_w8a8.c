//===- bare_runtime_w8a8.c - Bare-runtime W8A8 helpers --------------------===//
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
// Lowered W8A8 helper ABI of the bare single-hart runtime.
//
// The optimized Qwen3 W8A8 pipeline emits a call to
// `_mlir_ciface_buddy_w8a8_rvv_accumulate_n64`, so the helper is
// compiler-called support code rather than part of the startup path.  The
// sequence is kept in inline assembly (see the comment on the function) and the
// neighbouring-vector prime/`BARE_RUNTIME_AME_STORE_I32` variants are
// unchanged.
//

#include "bare_runtime_internal.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

//
// Fixed-width RVV dequantize/accumulate leaf used by the optimized Qwen3
// W8A8 lowering.  Keep this sequence in inline assembly: on the current FPGA
// the compiler-generated equivalent can spill vectors through an unsupported
// vlenb CSR read.  This is the same e32,m1 sequence used by the validated
// hand-written Qwen3 kernel. Fence producer/consumer boundaries and vector
// memory accesses. A build without V uses the scalar branch below.
//
__attribute__((noinline)) void _mlir_ciface_buddy_w8a8_rvv_accumulate_n64(
    uint64_t xout_address, uint64_t crow_address, uint64_t ws_row_address,
    uint32_t xscale_bits) {
#if defined(__riscv)
  __asm__ volatile("fence rw, rw" ::: "memory");
#endif
#if BARE_RUNTIME_TRACE_LIVE
  static int diagnosed = 0;
  if (!diagnosed) {
    volatile float probe;
    diagnosed = 1;
    print_uart("[w8a8-rvv] xout=0x");
    print_uart_addr(xout_address);
    print_uart(" crow=0x");
    print_uart_addr(crow_address);
    print_uart(" ws=0x");
    print_uart_addr(ws_row_address);
    print_uart("\r\n");
    probe = *(const float *)(uintptr_t)xout_address;
    (void)probe;
    print_uart("[w8a8-rvv] scalar xout ok\r\n");
    probe = *(const float *)(uintptr_t)crow_address;
    (void)probe;
    print_uart("[w8a8-rvv] scalar crow ok\r\n");
    print_uart("[w8a8-rvv] crow lanes=0x");
    print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[0]);
    print_uart(" 0x");
    print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[16]);
    print_uart(" 0x");
    print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[32]);
    print_uart(" 0x");
    print_uart_addr((uint32_t)((const int32_t *)(uintptr_t)crow_address)[48]);
    print_uart("\r\n");
    probe = *(const float *)(uintptr_t)ws_row_address;
    (void)probe;
    print_uart("[w8a8-rvv] scalar ws ok\r\n");
  }
#endif
#if defined(__riscv_vector) && !defined(BARE_RUNTIME_AME_STORE_I32)
  float *xout = (float *)(uintptr_t)xout_address;
  const float *crow = (const float *)(uintptr_t)crow_address;
  const float *ws_row = (const float *)(uintptr_t)ws_row_address;
  int done = 0;

  // The FPGA needs one complete vector issue group after an AME store before
  // floating-point RVV accumulation becomes reliable.  Prime every opcode
  // with a zero scale so the sequence writes xout back unchanged.
  __asm__ volatile("vsetivli zero, 16, e32, m1, ta, ma\n\t"
                   "vle32.v v10, (%[ws])\n\t"
                   "fence rw, rw\n\t"
                   "vle32.v v9, (%[acc])\n\t"
                   "fence rw, rw\n\t"
                   "vle32.v v8, (%[xout])\n\t"
                   "fence rw, rw\n\t"
                   "vmv.v.i v11, 0\n\t"
                   "vfmul.vv v12, v9, v11\n\t"
                   "vfmul.vv v13, v12, v10\n\t"
                   "vfadd.vv v8, v8, v13\n\t"
                   "fence rw, rw\n\t"
                   "vse32.v v8, (%[xout])\n\t"
                   "fence rw, rw" ::[xout] "r"(xout),
                   [acc] "r"(crow), [ws] "r"(ws_row)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

  while (done < 64) {
    size_t vl;
    float *xout_ptr = xout + done;
    const float *acc_ptr = crow + done;
    const float *ws_ptr = ws_row + done;
    const size_t avl = (size_t)(64 - done);

    __asm__ volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "fence rw, rw\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "fence rw, rw\n\t"
        "vle32.v v9, (%[acc])\n\t"
        "fence rw, rw\n\t"
        "vle32.v v8, (%[xout])\n\t"
        "fence rw, rw\n\t"
        "vmv.v.x v11, %[xs_bits]\n\t"
        "vfmul.vv v12, v9, v11\n\t"
        "vfmul.vv v13, v12, v10\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "fence rw, rw\n\t"
        "vse32.v v8, (%[xout])\n\t"
        "fence rw, rw\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [xout] "r"(xout_ptr), [acc] "r"(acc_ptr),
          [ws] "r"(ws_ptr), [xs_bits] "r"((unsigned long)xscale_bits)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

    if (vl == 0)
      break;
    done += (int)vl;
  }
#else
  float *xout = (float *)(uintptr_t)xout_address;
#if defined(BARE_RUNTIME_AME_STORE_I32)
  // NR FPGA0 msce32 stores signed integer accumulator bits. Convert their
  // numeric value before applying the model's unchanged FP32 scales.
  typedef int32_t AmeI32Word __attribute__((__may_alias__));
  const AmeI32Word *crow = (const AmeI32Word *)(uintptr_t)crow_address;
#else
  const float *crow = (const float *)(uintptr_t)crow_address;
#endif
  const float *ws_row = (const float *)(uintptr_t)ws_row_address;
  union {
    uint32_t u;
    float f;
  } xscale = {xscale_bits};
  for (int i = 0; i < 64; ++i)
    xout[i] += ((float)crow[i] * xscale.f) * ws_row[i];
#endif
#if defined(__riscv)
  __asm__ volatile("fence rw, rw" ::: "memory");
#endif
}
