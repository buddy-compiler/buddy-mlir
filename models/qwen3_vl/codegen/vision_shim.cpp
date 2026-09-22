//===- vision_shim.cpp - plain-C wrapper around the compiled vision kernel ===//
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
// Exposes qwen3vl_vision() with a flat f16 (uint16_t) ABI so the C++ runner
// can call the compiled vision encoder without the MLIR memref ABI.
//
//===----------------------------------------------------------------------===//
#include "buddy/Core/Container.h"
#include <cstdint>
#include <cstring>

using F16 = uint16_t;

struct MR2 {
  F16 *allocated, *aligned;
  intptr_t offset, sizes[2], strides[2];
  F16 *d() const { return aligned + offset; }
};
// Imported graph returns (ds0, ds1, ds2, pooled) — merger output is last.
struct VisRet {
  MR2 ds0, ds1, ds2, pooled;
};

extern "C" void _mlir_ciface_forward(VisRet *, MemRef<F16, 1> *,
                                     MemRef<F16, 2> *);

// pixel: (392,1536); each output: (98,2048). All f16.
extern "C" void qwen3vl_vision(const F16 *W, long NW, const F16 *pixel,
                               F16 *ds0, F16 *ds1, F16 *ds2, F16 *pooled) {
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<F16, 1> w(W, ws);
  intptr_t ps[2] = {392, 1536};
  MemRef<F16, 2> px(pixel, ps);
  VisRet r;
  _mlir_ciface_forward(&r, &w, &px);
  const size_t n = 98 * 2048 * sizeof(F16);
  memcpy(ds0, r.ds0.d(), n);
  memcpy(ds1, r.ds1.d(), n);
  memcpy(ds2, r.ds2.d(), n);
  memcpy(pooled, r.pooled.d(), n);
}
