//===- vision_shim.cpp - plain-C wrapper around the compiled vision kernel
//-===//
// Exposes qwen3vl_vision() with a flat float* ABI so a ctypes driver can call
// the compiled vision encoder without dealing with the MLIR memref ABI.
//===----------------------------------------------------------------------===//
#include "buddy/Core/Container.h"
#include <cstdint>
#include <cstring>

struct MR2 {
  float *allocated, *aligned;
  intptr_t offset, sizes[2], strides[2];
  float *d() const { return aligned + offset; }
};
// Imported graph returns (ds0, ds1, ds2, pooled) — merger output is last.
struct VisRet {
  MR2 ds0, ds1, ds2, pooled;
};

extern "C" void _mlir_ciface_forward(VisRet *, MemRef<float, 1> *,
                                     MemRef<float, 2> *);

// pixel: (392,1536); each output: (98,2048).
extern "C" void qwen3vl_vision(const float *W, long NW, const float *pixel,
                               float *ds0, float *ds1, float *ds2,
                               float *pooled) {
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<float, 1> w(W, ws);
  intptr_t ps[2] = {392, 1536};
  MemRef<float, 2> px(pixel, ps);
  VisRet r;
  _mlir_ciface_forward(&r, &w, &px);
  const size_t n = 98 * 2048 * sizeof(float);
  memcpy(ds0, r.ds0.d(), n);
  memcpy(ds1, r.ds1.d(), n);
  memcpy(ds2, r.ds2.d(), n);
  memcpy(pooled, r.pooled.d(), n);
}
