#include "support.h"
typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;
extern void triton_kv_cache_update_position_1x8x128_cap128(int64_t, MemRef0 *, int64_t, MemRef0 *, int64_t, MemRef0 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
void _mlir_ciface_kernel_kv_cache_update_position_1x8x128_cap128(MemRef3 *a0, MemRef1 *a1, MemRef3 *a2) {
  void *p0 = (unsigned char *)a0->aligned + a0->offset * 4;
  MemRef0 m0 = {p0, p0, 0};
  void *p1 = (unsigned char *)a1->aligned + a1->offset * 4;
  MemRef0 m1 = {p1, p1, 0};
  void *p2 = (unsigned char *)a2->aligned + a2->offset * 4;
  MemRef0 m2 = {p2, p2, 0};
  for (int32_t x=0; x<1; ++x)
    for (int32_t y=0; y<8; ++y)
      for (int32_t z=0; z<1; ++z)
        triton_kv_cache_update_position_1x8x128_cap128(0, &m0, 0, &m1, 0, &m2, 1, 8, 1, x, y, z);
}
