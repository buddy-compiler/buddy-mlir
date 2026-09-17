#include "support.h"
typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;
extern void triton_attention_qk_position_16x16x512x128(int64_t, MemRef0 *, int64_t, MemRef0 *, int64_t, MemRef0 *, int64_t, MemRef0 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_attention_qk_position_16x16x512x128(MemRef3 *a0, MemRef3 *a1, MemRef1 *a2, MemRef3 *a3) {
  void *p0 = (unsigned char *)a0->aligned + a0->offset * 4;
  MemRef0 m0 = {p0, p0, 0};
  void *p1 = (unsigned char *)a1->aligned + a1->offset * 4;
  MemRef0 m1 = {p1, p1, 0};
  void *p2 = (unsigned char *)a2->aligned + a2->offset * 4;
  MemRef0 m2 = {p2, p2, 0};
  void *p3 = (unsigned char *)a3->aligned + a3->offset * 4;
  MemRef0 m3 = {p3, p3, 0};
  for (int32_t x=0; x<1; ++x)
    for (int32_t y=0; y<32; ++y)
      for (int32_t z=0; z<16; ++z)
        triton_attention_qk_position_16x16x512x128(0, &m0, 0, &m1, 0, &m2, 0, &m3, 1, 32, 16, x, y, z);
}
