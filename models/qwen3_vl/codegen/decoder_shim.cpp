//===- decoder_shim.cpp - plain-C wrapper around the compiled decoder -----===//
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
// Exposes qwen3vl_decoder() with a flat f16 (uint16_t) ABI for the C++ driver.
// Matches qwen3_vl_codegen.py import-decoder-rt: forward(inputs_embeds, cos,
// sin, cmask, ds0, ds1, ds2) with weights as arg0; cos/sin/cmask are runtime
// inputs.
//
// Legacy full-sequence path (BUDDY_QWEN3_VL_KV_DECODE=OFF). The default KV
// build links decoder_kv_shim.cpp instead (prefill/decode entry points).
//
//===----------------------------------------------------------------------===//
#include "buddy/Core/Container.h"
#include <cstdint>
#include <cstring>

using F16 = uint16_t;

struct MR3 {
  F16 *allocated, *aligned;
  intptr_t offset, sizes[3], strides[3];
  F16 *d() const { return aligned + offset; }
};

// buddy orders the main-graph inputs by first-use, not Python arg order:
//   weights, cos, sin, inputs_embeds, cmask, ds0, ds1, ds2
// (cos/sin are touched first via .unsqueeze). Must match exactly or we
// segfault.
extern "C" void _mlir_ciface_forward(MR3 *, MemRef<F16, 1> *, // weights
                                     MemRef<F16, 2> *,        // cos
                                     MemRef<F16, 2> *,        // sin
                                     MemRef<F16, 3> *,        // inputs_embeds
                                     MemRef<F16, 4> *,        // cmask
                                     MemRef<F16, 3> *,        // ds0
                                     MemRef<F16, 3> *,        // ds1
                                     MemRef<F16, 3> *);       // ds2

// ie/d*: (1,N,H); cos/sin: (N,HD); cmask: (1,1,N,N); logits out: (1,N,V).
extern "C" void qwen3vl_decoder(const F16 *W, long NW, const F16 *ie,
                                const F16 *cos, const F16 *sin,
                                const F16 *cmask, const F16 *d0, const F16 *d1,
                                const F16 *d2, F16 *logits, long N, long V,
                                long H, long HD) {
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<F16, 1> w(W, ws);
  intptr_t s3[3] = {1, (intptr_t)N, (intptr_t)H};
  MemRef<F16, 3> mie(ie, s3), m0(d0, s3), m1(d1, s3), m2(d2, s3);
  intptr_t s2[2] = {(intptr_t)N, (intptr_t)HD};
  MemRef<F16, 2> mcos(cos, s2), msin(sin, s2);
  intptr_t s4[4] = {1, 1, (intptr_t)N, (intptr_t)N};
  MemRef<F16, 4> mcmask(cmask, s4);
  MR3 r;
  _mlir_ciface_forward(&r, &w, &mcos, &msin, &mie, &mcmask, &m0, &m1, &m2);
  memcpy(logits, r.d(), (size_t)N * V * sizeof(F16));
}
