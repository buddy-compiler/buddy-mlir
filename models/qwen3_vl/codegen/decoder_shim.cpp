//===- decoder_shim.cpp - plain-C wrapper around the compiled decoder
//------===//
// Exposes qwen3vl_decoder() with a flat float* ABI for the ctypes / C++ driver.
// Matches qwen3_vl_codegen.py import-decoder-rt: forward(inputs_embeds, cos,
// sin, cmask, ds0, ds1, ds2) with weights as arg0; cos/sin/cmask are runtime
// inputs.
//===----------------------------------------------------------------------===//
#include "buddy/Core/Container.h"
#include <cstdint>
#include <cstring>

struct MR3 {
  float *allocated, *aligned;
  intptr_t offset, sizes[3], strides[3];
  float *d() const { return aligned + offset; }
};

// buddy orders the main-graph inputs by first-use, not Python arg order:
//   weights, cos, sin, inputs_embeds, cmask, ds0, ds1, ds2
// (cos/sin are touched first via .unsqueeze). Must match exactly or we
// segfault.
extern "C" void _mlir_ciface_forward(MR3 *, MemRef<float, 1> *, // weights
                                     MemRef<float, 2> *,        // cos
                                     MemRef<float, 2> *,        // sin
                                     MemRef<float, 3> *,        // inputs_embeds
                                     MemRef<float, 4> *,        // cmask
                                     MemRef<float, 3> *,        // ds0
                                     MemRef<float, 3> *,        // ds1
                                     MemRef<float, 3> *);       // ds2

// ie/d*: (1,N,H); cos/sin: (N,HD); cmask: (1,1,N,N); logits out: (1,N,V).
extern "C" void qwen3vl_decoder(const float *W, long NW, const float *ie,
                                const float *cos, const float *sin,
                                const float *cmask, const float *d0,
                                const float *d1, const float *d2, float *logits,
                                long N, long V, long H, long HD) {
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<float, 1> w(W, ws);
  intptr_t s3[3] = {1, (intptr_t)N, (intptr_t)H};
  MemRef<float, 3> mie(ie, s3), m0(d0, s3), m1(d1, s3), m2(d2, s3);
  intptr_t s2[2] = {(intptr_t)N, (intptr_t)HD};
  MemRef<float, 2> mcos(cos, s2), msin(sin, s2);
  intptr_t s4[4] = {1, 1, (intptr_t)N, (intptr_t)N};
  MemRef<float, 4> mcmask(cmask, s4);
  MR3 r;
  _mlir_ciface_forward(&r, &w, &mcos, &msin, &mie, &mcmask, &m0, &m1, &m2);
  memcpy(logits, r.d(), (size_t)N * V * sizeof(float));
}
