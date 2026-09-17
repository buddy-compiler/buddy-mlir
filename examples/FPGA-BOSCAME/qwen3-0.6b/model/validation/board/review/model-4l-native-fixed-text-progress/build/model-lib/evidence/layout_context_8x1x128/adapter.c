#include "support.h"
typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;
extern void triton_layout_context_8x1x128(int64_t, MemRef0 *, int64_t, MemRef0 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_layout_context_8x1x128(MemRef3 *a0, MemRef3 *a1) {
  void *p0 = (unsigned char *)a0->aligned + a0->offset * 4;
  MemRef0 m0 = {p0, p0, 0};
  void *p1 = (unsigned char *)a1->aligned + a1->offset * 4;
  MemRef0 m1 = {p1, p1, 0};
  for (int32_t x=0; x<8; ++x)
    for (int32_t y=0; y<1; ++y)
      for (int32_t z=0; z<1; ++z)
        triton_layout_context_8x1x128(0, &m0, 0, &m1, 8, 1, 1, x, y, z);
}
