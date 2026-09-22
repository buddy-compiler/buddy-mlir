/* Generated silent observer; computation remains in the linked Triton kernel. */
#include "support.h"
#include "nr_runtime.h"
#ifndef NR_HANG_DIAGNOSTICS
#error "hang-watch requires NR_HANG_DIAGNOSTICS"
#endif
static uint64_t qwen_hang_calls;
void qwen_hang_reset(unsigned position) {
  qwen_hang_calls = 0;
  nr_diag_reset(position);
}
extern void __real__mlir_ciface_kernel_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2);
void __wrap__mlir_ciface_kernel_matmul_1x1024x2048(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
  const uint64_t call_index = qwen_hang_calls++;
  const int selected = call_index == 14ULL;
  if (selected) {
    if (a0) {
      nr_diag_memref(0, (uintptr_t)a0, (uintptr_t)a0->aligned,
                     a0->offset, a0->sizes[0], a0->sizes[1],
                     a0->strides[0], a0->strides[1]);
    } else {
      nr_diag_memref(0, 0, 0, 0, 0, 0, 0, 0);
    }
    if (a1) {
      nr_diag_memref(1, (uintptr_t)a1, (uintptr_t)a1->aligned,
                     a1->offset, a1->sizes[0], a1->sizes[1],
                     a1->strides[0], a1->strides[1]);
    } else {
      nr_diag_memref(1, 0, 0, 0, 0, 0, 0, 0);
    }
    if (a2) {
      nr_diag_memref(2, (uintptr_t)a2, (uintptr_t)a2->aligned,
                     a2->offset, a2->sizes[0], a2->sizes[1],
                     a2->strides[0], a2->strides[1]);
    } else {
      nr_diag_memref(2, 0, 0, 0, 0, 0, 0, 0);
    }
    nr_diag_mark(NR_DIAG_KERNEL_ENTER, call_index);
  }
  __real__mlir_ciface_kernel_matmul_1x1024x2048(a0, a1, a2);
  if (selected) nr_diag_mark(NR_DIAG_KERNEL_RETURN, call_index);
}
