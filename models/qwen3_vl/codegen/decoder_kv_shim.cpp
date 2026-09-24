//===- decoder_kv_shim.cpp - Prefill/decode KV wrappers for Qwen3-VL ------===//
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
// Flat f16 ABI wrappers around the two buddy-compiled decoder graphs:
//
//   qwen3vl_decoder_prefill  -> _mlir_ciface_forward_prefill
//   qwen3vl_decoder_decode   -> _mlir_ciface_forward_decode
//
// Prefill consumes the full padded sequence (inputs_embeds[1,N,H], cos/sin,
// causal mask, deepstack) and returns 56 GQA cache tensors plus logits.
// Decode consumes one new token plus those caches and returns updated caches
// plus a single-row logits vector.
//
// Cache layout (matches qwen3_vl_codegen.DecoderTracePrefillKV):
//   28 layers x {K, V} = 56 MemRef<f16,4>, each shaped [1, n_kv, N, head_dim]
//   with n_kv=8, head_dim=128 for Qwen3-VL-2B. K/V are interleaved per layer
//   (kv0=K0, kv1=V0, kv2=K1, ...).
//
// MLIR returns a large aggregate by pointer; we placement-new MemRefs into a
// stack buffer sized like PrefillReturns / DecodeReturns so the ABI matches
// the generated C wrapper without requiring a C++ default constructor on
// MemRef (which would allocate). After the call we copy (or alias-skip) back
// into the caller's flat buffers.
//
// Prefill and decode both export `subgraph0` / `_mlir_ciface_subgraph0`;
// link_decoder_kv_shim.sh renames those symbols before linking into one .so.
//
//===----------------------------------------------------------------------===//

#include "buddy/Core/Container.h"
#include <cstdint>
#include <cstring>
#include <vector>

using F16 = uint16_t;
using I64 = long long;

// Aggregate matching the MLIR multi-result of forward_prefill:
//   (kv0..kv55, logits) where logits is [1, N, V].
struct PrefillReturns {
  MemRef<F16, 4> kv0;
  MemRef<F16, 4> kv1;
  MemRef<F16, 4> kv2;
  MemRef<F16, 4> kv3;
  MemRef<F16, 4> kv4;
  MemRef<F16, 4> kv5;
  MemRef<F16, 4> kv6;
  MemRef<F16, 4> kv7;
  MemRef<F16, 4> kv8;
  MemRef<F16, 4> kv9;
  MemRef<F16, 4> kv10;
  MemRef<F16, 4> kv11;
  MemRef<F16, 4> kv12;
  MemRef<F16, 4> kv13;
  MemRef<F16, 4> kv14;
  MemRef<F16, 4> kv15;
  MemRef<F16, 4> kv16;
  MemRef<F16, 4> kv17;
  MemRef<F16, 4> kv18;
  MemRef<F16, 4> kv19;
  MemRef<F16, 4> kv20;
  MemRef<F16, 4> kv21;
  MemRef<F16, 4> kv22;
  MemRef<F16, 4> kv23;
  MemRef<F16, 4> kv24;
  MemRef<F16, 4> kv25;
  MemRef<F16, 4> kv26;
  MemRef<F16, 4> kv27;
  MemRef<F16, 4> kv28;
  MemRef<F16, 4> kv29;
  MemRef<F16, 4> kv30;
  MemRef<F16, 4> kv31;
  MemRef<F16, 4> kv32;
  MemRef<F16, 4> kv33;
  MemRef<F16, 4> kv34;
  MemRef<F16, 4> kv35;
  MemRef<F16, 4> kv36;
  MemRef<F16, 4> kv37;
  MemRef<F16, 4> kv38;
  MemRef<F16, 4> kv39;
  MemRef<F16, 4> kv40;
  MemRef<F16, 4> kv41;
  MemRef<F16, 4> kv42;
  MemRef<F16, 4> kv43;
  MemRef<F16, 4> kv44;
  MemRef<F16, 4> kv45;
  MemRef<F16, 4> kv46;
  MemRef<F16, 4> kv47;
  MemRef<F16, 4> kv48;
  MemRef<F16, 4> kv49;
  MemRef<F16, 4> kv50;
  MemRef<F16, 4> kv51;
  MemRef<F16, 4> kv52;
  MemRef<F16, 4> kv53;
  MemRef<F16, 4> kv54;
  MemRef<F16, 4> kv55;
  MemRef<F16, 3> logits;
};

// Same cache layout as PrefillReturns; logits is [1, 1, V] for one new token.
struct DecodeReturns {
  MemRef<F16, 4> kv0;
  MemRef<F16, 4> kv1;
  MemRef<F16, 4> kv2;
  MemRef<F16, 4> kv3;
  MemRef<F16, 4> kv4;
  MemRef<F16, 4> kv5;
  MemRef<F16, 4> kv6;
  MemRef<F16, 4> kv7;
  MemRef<F16, 4> kv8;
  MemRef<F16, 4> kv9;
  MemRef<F16, 4> kv10;
  MemRef<F16, 4> kv11;
  MemRef<F16, 4> kv12;
  MemRef<F16, 4> kv13;
  MemRef<F16, 4> kv14;
  MemRef<F16, 4> kv15;
  MemRef<F16, 4> kv16;
  MemRef<F16, 4> kv17;
  MemRef<F16, 4> kv18;
  MemRef<F16, 4> kv19;
  MemRef<F16, 4> kv20;
  MemRef<F16, 4> kv21;
  MemRef<F16, 4> kv22;
  MemRef<F16, 4> kv23;
  MemRef<F16, 4> kv24;
  MemRef<F16, 4> kv25;
  MemRef<F16, 4> kv26;
  MemRef<F16, 4> kv27;
  MemRef<F16, 4> kv28;
  MemRef<F16, 4> kv29;
  MemRef<F16, 4> kv30;
  MemRef<F16, 4> kv31;
  MemRef<F16, 4> kv32;
  MemRef<F16, 4> kv33;
  MemRef<F16, 4> kv34;
  MemRef<F16, 4> kv35;
  MemRef<F16, 4> kv36;
  MemRef<F16, 4> kv37;
  MemRef<F16, 4> kv38;
  MemRef<F16, 4> kv39;
  MemRef<F16, 4> kv40;
  MemRef<F16, 4> kv41;
  MemRef<F16, 4> kv42;
  MemRef<F16, 4> kv43;
  MemRef<F16, 4> kv44;
  MemRef<F16, 4> kv45;
  MemRef<F16, 4> kv46;
  MemRef<F16, 4> kv47;
  MemRef<F16, 4> kv48;
  MemRef<F16, 4> kv49;
  MemRef<F16, 4> kv50;
  MemRef<F16, 4> kv51;
  MemRef<F16, 4> kv52;
  MemRef<F16, 4> kv53;
  MemRef<F16, 4> kv54;
  MemRef<F16, 4> kv55;
  MemRef<F16, 3> logits;
};

static void copyF16(F16 *dst, const F16 *src, size_t n) {
  std::memcpy(dst, src, n * sizeof(F16));
}

// Generated C iface: result aggregate first, then operands in first-use order
// (weights, cos, sin, embeds, cmask, ds0..ds2) — not Python argument order.
extern "C" void _mlir_ciface_forward_prefill(
    PrefillReturns *result, MemRef<F16, 1> *weights, MemRef<F16, 2> *cos,
    MemRef<F16, 2> *sin, MemRef<F16, 3> *embeds, MemRef<F16, 4> *cmask,
    MemRef<F16, 3> *ds0, MemRef<F16, 3> *ds1, MemRef<F16, 3> *ds2);

// Decode operands follow first-use in the traced graph: weights, cos/sin (1
// row), embeds (1 token), cache_position, then interleaved K/V caches with
// deepstack zeros spliced at the same relative positions as prefill's ds*.
extern "C" void _mlir_ciface_forward_decode(
    DecodeReturns *result, MemRef<F16, 1> *arg0, MemRef<F16, 2> *arg1,
    MemRef<F16, 2> *arg2, MemRef<F16, 3> *arg3, MemRef<I64, 1> *arg4,
    MemRef<F16, 4> *arg5, MemRef<F16, 4> *arg6, MemRef<F16, 4> *arg7,
    MemRef<F16, 3> *arg8, MemRef<F16, 4> *arg9, MemRef<F16, 4> *arg10,
    MemRef<F16, 3> *arg11, MemRef<F16, 4> *arg12, MemRef<F16, 4> *arg13,
    MemRef<F16, 3> *arg14, MemRef<F16, 4> *arg15, MemRef<F16, 4> *arg16,
    MemRef<F16, 4> *arg17, MemRef<F16, 4> *arg18, MemRef<F16, 4> *arg19,
    MemRef<F16, 4> *arg20, MemRef<F16, 4> *arg21, MemRef<F16, 4> *arg22,
    MemRef<F16, 4> *arg23, MemRef<F16, 4> *arg24, MemRef<F16, 4> *arg25,
    MemRef<F16, 4> *arg26, MemRef<F16, 4> *arg27, MemRef<F16, 4> *arg28,
    MemRef<F16, 4> *arg29, MemRef<F16, 4> *arg30, MemRef<F16, 4> *arg31,
    MemRef<F16, 4> *arg32, MemRef<F16, 4> *arg33, MemRef<F16, 4> *arg34,
    MemRef<F16, 4> *arg35, MemRef<F16, 4> *arg36, MemRef<F16, 4> *arg37,
    MemRef<F16, 4> *arg38, MemRef<F16, 4> *arg39, MemRef<F16, 4> *arg40,
    MemRef<F16, 4> *arg41, MemRef<F16, 4> *arg42, MemRef<F16, 4> *arg43,
    MemRef<F16, 4> *arg44, MemRef<F16, 4> *arg45, MemRef<F16, 4> *arg46,
    MemRef<F16, 4> *arg47, MemRef<F16, 4> *arg48, MemRef<F16, 4> *arg49,
    MemRef<F16, 4> *arg50, MemRef<F16, 4> *arg51, MemRef<F16, 4> *arg52,
    MemRef<F16, 4> *arg53, MemRef<F16, 4> *arg54, MemRef<F16, 4> *arg55,
    MemRef<F16, 4> *arg56, MemRef<F16, 4> *arg57, MemRef<F16, 4> *arg58,
    MemRef<F16, 4> *arg59, MemRef<F16, 4> *arg60, MemRef<F16, 4> *arg61,
    MemRef<F16, 4> *arg62, MemRef<F16, 4> *arg63, MemRef<F16, 4> *arg64);

static MemRef<F16, 4> *kvPtr(PrefillReturns &r, int i) {
  return reinterpret_cast<MemRef<F16, 4> *>(&r.kv0) + i;
}
static MemRef<F16, 4> *kvPtr(DecodeReturns &r, int i) {
  return reinterpret_cast<MemRef<F16, 4> *>(&r.kv0) + i;
}

// Bind result MemRefs to the caller's contiguous KV/logits storage so MLIR
// can write through them (or allocate elsewhere — we copy back either way).
static void initPrefillReturns(PrefillReturns &r, F16 *kv_out, F16 *logits,
                               intptr_t skv[4], intptr_t sl[3],
                               size_t kvElems) {
  MemRef<F16, 4> *slot = reinterpret_cast<MemRef<F16, 4> *>(&r.kv0);
  for (int i = 0; i < 56; ++i)
    new (slot + i) MemRef<F16, 4>(kv_out + (size_t)i * kvElems, skv);
  new (&r.logits) MemRef<F16, 3>(logits, sl);
}

static void initDecodeReturns(DecodeReturns &r, F16 *kv, F16 *logits1,
                              intptr_t skv[4], intptr_t sl[3], size_t kvElems) {
  MemRef<F16, 4> *slot = reinterpret_cast<MemRef<F16, 4> *>(&r.kv0);
  for (int i = 0; i < 56; ++i)
    new (slot + i) MemRef<F16, 4>(kv + (size_t)i * kvElems, skv);
  new (&r.logits) MemRef<F16, 3>(logits1, sl);
}

/// Prefill the full sequence once and fill the GQA caches.
///
/// Shapes (f16 unless noted):
///   W[NW], ie/d*[1,N,H], cos/sin[N,HD], cmask[1,1,N,N]
///   kv_out: 56 contiguous blocks of [1,NKV,N,HD]
///   logits[1,N,V]
extern "C" void qwen3vl_decoder_prefill(const F16 *W, long NW, const F16 *ie,
                                        const F16 *cos, const F16 *sin,
                                        const F16 *cmask, const F16 *d0,
                                        const F16 *d1, const F16 *d2,
                                        F16 *kv_out, F16 *logits, long N,
                                        long V, long H, long HD, long NKV) {
  const size_t kvElems = (size_t)NKV * (size_t)N * (size_t)HD;
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<F16, 1> w(W, ws);
  intptr_t s3[3] = {1, (intptr_t)N, (intptr_t)H};
  MemRef<F16, 3> mie(ie, s3), m0(d0, s3), m1(d1, s3), m2(d2, s3);
  intptr_t s2[2] = {(intptr_t)N, (intptr_t)HD};
  MemRef<F16, 2> mcos(cos, s2), msin(sin, s2);
  intptr_t s4[4] = {1, 1, (intptr_t)N, (intptr_t)N};
  MemRef<F16, 4> mcmask(cmask, s4);
  intptr_t skv[4] = {1, (intptr_t)NKV, (intptr_t)N, (intptr_t)HD};
  // Uninitialized storage + placement-new: MemRef's default ctor allocates.
  alignas(PrefillReturns) unsigned char prefillStorage[sizeof(PrefillReturns)];
  PrefillReturns &r = *reinterpret_cast<PrefillReturns *>(prefillStorage);
  intptr_t sl[3] = {1, (intptr_t)N, (intptr_t)V};
  initPrefillReturns(r, kv_out, logits, skv, sl, kvElems);
  _mlir_ciface_forward_prefill(&r, &w, &mcos, &msin, &mie, &mcmask, &m0, &m1,
                               &m2);
  // Always copy out: prefill may allocate fresh buffers for results.
  for (int i = 0; i < 56; ++i)
    copyF16(kv_out + (size_t)i * kvElems, kvPtr(r, i)->getData(), kvElems);
  copyF16(logits, r.logits.getData(), (size_t)N * (size_t)V);
}

/// One-token decode step against the caches filled by prefill.
///
/// Shapes:
///   ie1/d*[1,1,H], cos1/sin1[1,HD], cmask1[1,1,1,N] (0 for j<=pos else -inf)
///   pos: absolute sequence index for this step
///   kv: same 56-block layout as prefill (read/write)
///   logits1[1,1,V]
///
/// W may be the panel-packed decode weight blob when the graph was lowered
/// with -matmul-vectorization-decode-packed.
extern "C" void qwen3vl_decoder_decode(const F16 *W, long NW, const F16 *ie1,
                                       const F16 *cos1, const F16 *sin1,
                                       const F16 *cmask1, const F16 *d0,
                                       const F16 *d1, const F16 *d2, long pos,
                                       F16 *kv, F16 *logits1, long N, long V,
                                       long H, long HD, long NKV) {
  intptr_t ws[1] = {(intptr_t)NW};
  MemRef<F16, 1> w(W, ws);
  intptr_t s2[2] = {1, (intptr_t)HD};
  MemRef<F16, 2> mcos(cos1, s2), msin(sin1, s2);
  intptr_t s3[3] = {1, 1, (intptr_t)H};
  MemRef<F16, 3> mie(ie1, s3), m0(d0, s3), m1(d1, s3), m2(d2, s3);
  intptr_t s4m[4] = {1, 1, 1, (intptr_t)N};
  MemRef<F16, 4> mcmask(cmask1, s4m);
  I64 posStorage = (I64)pos;
  intptr_t s1[1] = {1};
  MemRef<I64, 1> mpos(&posStorage, s1);
  const size_t kvElems = (size_t)NKV * (size_t)N * (size_t)HD;
  intptr_t skv[4] = {1, (intptr_t)NKV, (intptr_t)N, (intptr_t)HD};
  intptr_t sl1[3] = {1, 1, (intptr_t)V};
  // Operand MemRefs share the caller's kv storage with the result slots.
  std::vector<MemRef<F16, 4>> mkv;
  mkv.reserve(56);
  for (int i = 0; i < 56; ++i)
    mkv.emplace_back(kv + (size_t)i * kvElems, skv);
  alignas(DecodeReturns) unsigned char decodeStorage[sizeof(DecodeReturns)];
  DecodeReturns &r = *reinterpret_cast<DecodeReturns *>(decodeStorage);
  initDecodeReturns(r, kv, logits1, skv, sl1, kvElems);
  // Argument order must match forward_decode's first-use memref list exactly.
  _mlir_ciface_forward_decode(
      &r, &w, &mcos, &msin, &mie, &mpos, &mkv[0], &mkv[1], &mcmask, &m0,
      &mkv[2], &mkv[3], &m1, &mkv[4], &mkv[5], &m2, &mkv[6], &mkv[7], &mkv[8],
      &mkv[9], &mkv[10], &mkv[11], &mkv[12], &mkv[13], &mkv[14], &mkv[15],
      &mkv[16], &mkv[17], &mkv[18], &mkv[19], &mkv[20], &mkv[21], &mkv[22],
      &mkv[23], &mkv[24], &mkv[25], &mkv[26], &mkv[27], &mkv[28], &mkv[29],
      &mkv[30], &mkv[31], &mkv[32], &mkv[33], &mkv[34], &mkv[35], &mkv[36],
      &mkv[37], &mkv[38], &mkv[39], &mkv[40], &mkv[41], &mkv[42], &mkv[43],
      &mkv[44], &mkv[45], &mkv[46], &mkv[47], &mkv[48], &mkv[49], &mkv[50],
      &mkv[51], &mkv[52], &mkv[53], &mkv[54], &mkv[55]);
  // Prefer in-place returns: skip the ~18MB KV bounce when the graph wrote
  // back into the caller's buffers (common with bufferization).
  for (int i = 0; i < 56; ++i) {
    F16 *src = kvPtr(r, i)->getData();
    F16 *dst = kv + (size_t)i * kvElems;
    if (src != dst)
      copyF16(dst, src, kvElems);
  }
  F16 *lsrc = r.logits.getData();
  if (lsrc != logits1)
    copyF16(logits1, lsrc, (size_t)V);
}
