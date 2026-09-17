#!/usr/bin/env python3
"""Generate the Triton operator cases the real Qwen3 graph actually needs.

The shared 72-case set was built for a *fully packed* cache: attention spans
exactly as many keys as the sequence holds (T=16 prefill, T=17 decode). The
imported graph does something different -- it keeps a fixed 512-slot cache and
varies only the mask boundary -- so several shapes and two runtime-position
kernel variants are genuinely new:

  * attention over the full 512-slot cache for both prefill and decode;
  * scale+mask whose causal boundary is the runtime ``cache_position``;
  * KV writes whose destination slot is runtime data, not a constexpr;
  * softmax and GQA expansion over 512 slots.

Every case is emitted with the same contract as the existing set: a
``metadata.json`` describing buffers, plus a ``launch.c`` that fills
deterministic inputs, calls the lowered function and compares element by
element against an independent C oracle. Nothing here re-implements a kernel in
C -- the C code only produces inputs and the expected values.

Cases are written to a directory outside the shared operator tree so the
existing 72-case build is untouched; the shared build finds them through
``QWEN_CASE_ROOTS``.
"""
import argparse
import json
from pathlib import Path

HEAD = {"attention": '''#include "support.h"
#define HEADS 16
#define M %(m)d
#define N %(n)d
#define K %(k)d
extern void _mlir_ciface_kernel_%(name)s(MemRef3*,MemRef3*,MemRef3*);
int launch(void) {
  float *a=workspace(0),*b=workspace(HEADS*M*K*4),*c=workspace(HEADS*(M*K+K*N)*4);
  for(int x=0;x<HEADS*M*K;x++) a[x]=(float)(x%%17-8)*0.125f;
  for(int x=0;x<HEADS*K*N;x++) b[x]=(float)(x%%13-6)*0.0625f;
  for(int x=0;x<HEADS*M*N;x++) c[x]=1234.0f;
  MemRef3 A=make_3(a,HEADS,M,K),B=make_3(b,HEADS,K,N),C=make_3(c,HEADS,M,N);
  _mlir_ciface_kernel_%(name)s(&A,&B,&C);
  unsigned errors=0; float max_error=0;
  for(int h=0;h<HEADS;h++) for(int i=0;i<M;i++) for(int j=0;j<N;j++) {
    double want=0;
    for(int q=0;q<K;q++) want+=(double)a[(h*M+i)*K+q]*(double)b[(h*K+q)*N+j];
    float got=c[(h*M+i)*N+j],d=got-(float)want; if(d<0)d=-d;
    if(d>max_error)max_error=d;
    if(!check_close(got,(float)want,1e-6f,1e-6f))errors++;
  }
  return print_check("%(name)s",errors,max_error);
}
''',
        "math_head": '''#include "support.h"
#ifdef HOST_TEST
typedef double oracle_float;
static float oracle_exp(float x) { return (float)exp((double)x); }
static float oracle_sqrt(float x) { return (float)sqrt((double)x); }
#else
typedef float oracle_float;
static float oracle_exp(float x) { return expf(x); }
static float oracle_sqrt(float x) { return sqrtf(x); }
#endif
static void compare(float actual, float expected, float atol, float rtol,
                    int *errors, float *maximum) {
  if (actual == expected) return;
  if (!__builtin_isfinite(actual) || !__builtin_isfinite(expected)) {
    ++*errors; *maximum = __builtin_inff(); return;
  }
  if (!check_close(actual, expected, atol, rtol)) ++*errors;
  float difference = actual - expected;
  if (difference < 0) difference = -difference;
  if (difference > *maximum) *maximum = difference;
}
'''}


def softmax_case(name, heads, sequence, total):
    head = HEAD["math_head"]
    body = f'''extern void _mlir_ciface_kernel_{name}(MemRef3 *, MemRef2 *, MemRef2 *, MemRef3 *);
int launch(void) {{
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, {heads}, {sequence}, {total});
  float *maxima = (float *)workspace({heads * sequence * total * 4}u);
  MemRef2 m_maxima = make_2(maxima, {heads * sequence}, 1);
  float *sums = (float *)workspace({heads * sequence * total * 4 + heads * sequence * 4}u);
  MemRef2 m_sums = make_2(sums, {heads * sequence}, 1);
  float *out = (float *)workspace({heads * sequence * total * 4 + heads * sequence * 8}u);
  MemRef3 m_out = make_3(out, {heads}, {sequence}, {total});
  int rows = {heads * sequence};
  for (int r=0; r<rows; ++r) for (int k=0; k<{total}; ++k) {{
    int i=r*{total}+k; x[i]=80.0f+(float)((i*7)%53-26)*0.25f; out[i]=9999;
    if (k=={total}-1) x[i]=-__builtin_inff();
  }}
  _mlir_ciface_kernel_{name}(&m_x, &m_maxima, &m_sums, &m_out);
  for (int r=0; r<rows; ++r) {{
    float maximum_input=-__builtin_inff(); oracle_float sum=0;
    for (int k=0; k<{total}; ++k) if(x[r*{total}+k]>maximum_input) maximum_input=x[r*{total}+k];
    for (int k=0; k<{total}; ++k) sum+=oracle_exp(x[r*{total}+k]-maximum_input);
    oracle_float output_sum=0;
    for (int k=0; k<{total}; ++k) {{
      int i=r*{total}+k;
      float expected=(float)(oracle_exp(x[i]-maximum_input)/sum);
      compare(out[i],expected,2e-6f,5e-5f,&errors,&maximum); output_sum+=out[i];
    }}
    compare((float)output_sum,1.0f,3e-6f,3e-6f,&errors,&maximum);
  }}
  return print_check("{name}", errors, maximum);
}}
'''
    return head + body


def mask_position_case(name, heads, sequence, total, base_position):
    head = HEAD["math_head"]
    body = f'''extern void _mlir_ciface_kernel_{name}(MemRef3 *, MemRef1 *, MemRef3 *);
int launch(void) {{
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, {heads}, {sequence}, {total});
  int *position = (int *)workspace({heads * sequence * total * 4}u);
  MemRef1 m_position = make_1(position, {sequence});
  float *out = (float *)workspace({heads * sequence * total * 4 + sequence * 4}u);
  MemRef3 m_out = make_3(out, {heads}, {sequence}, {total});
  int count = {heads * sequence * total};
  for(int i=0; i<count; ++i) {{ x[i]=(float)((i*11)%61-30)*0.25f; out[i]=9999; }}
  for(int s=0; s<{sequence}; ++s) position[s] = {base_position} + s;
  _mlir_ciface_kernel_{name}(&m_x, &m_position, &m_out);
  for (int h=0; h<{heads}; ++h) for(int s=0; s<{sequence}; ++s) for(int t=0; t<{total}; ++t) {{
    int i=(h*{sequence}+s)*{total}+t;
    int boundary = {base_position} + s;
    float expected = (t <= boundary) ? x[i]/oracle_sqrt(128.0f) : -__builtin_inff();
    compare(out[i],expected,1e-6f,2e-6f,&errors,&maximum);
  }}
  return print_check("{name}", errors, maximum);
}}
'''
    return head + body


def kv_position_case(name, sequence, heads_kv, capacity):
    head = HEAD["math_head"]
    body = f'''extern void _mlir_ciface_kernel_{name}(MemRef3 *, MemRef1 *, MemRef3 *);
int launch(void) {{
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, {sequence}, {heads_kv}, 128);
  int *position = (int *)workspace({sequence * heads_kv * 128 * 4}u);
  MemRef1 m_position = make_1(position, {sequence});
  float *cache = (float *)workspace({sequence * heads_kv * 128 * 4 + sequence * 4}u);
  MemRef3 m_cache = make_3(cache, {heads_kv}, {capacity}, 128);
  for(int i=0; i<{sequence * heads_kv * 128}; ++i) x[i]=(float)((i*5)%29-14)*0.25f;
  for(int i=0; i<{heads_kv * capacity * 128}; ++i) cache[i]=-777.0f;
  /* Two tokens land at non-trivial slots so a wrong slot or a wrong stride
     changes the untouched-region check below. */
  for(int s=0; s<{sequence}; ++s) position[s] = 3 + s*7;
  _mlir_ciface_kernel_{name}(&m_x, &m_position, &m_cache);
  /* Reference: copy the untouched cache, then write at the same slots. */
  static float expected[{heads_kv} * {capacity} * 128];
  for(int i=0; i<{heads_kv} * {capacity} * 128; ++i) expected[i]=-777.0f;
  for(int s=0; s<{sequence}; ++s) for(int h=0; h<{heads_kv}; ++h) for(int c=0; c<128; ++c)
    expected[(h*{capacity} + position[s])*128 + c] = x[(s*{heads_kv}+h)*128 + c];
  for(int i=0; i<{heads_kv} * {capacity} * 128; ++i)
    compare(cache[i], expected[i], 1e-6f, 1e-6f, &errors, &maximum);
  return print_check("{name}", errors, maximum);
}}
'''
    return head + body


def gqa_case(name, kv_heads, query_heads, total):
    head = HEAD["math_head"]
    body = f'''extern void _mlir_ciface_kernel_{name}(MemRef3 *, MemRef3 *);
int launch(void) {{
  int errors = 0; float maximum = 0;
  float *x = (float *)workspace(0u);
  MemRef3 m_x = make_3(x, {kv_heads}, {total}, 128);
  float *out = (float *)workspace({kv_heads * total * 128 * 4}u);
  MemRef3 m_out = make_3(out, {query_heads}, {total}, 128);
  for(int i=0; i<{kv_heads * total * 128}; ++i) x[i]=(float)((i*3)%41-20)*0.125f;
  for(int i=0; i<{query_heads * total * 128}; ++i) out[i]=9999.0f;
  _mlir_ciface_kernel_{name}(&m_x, &m_out);
  for(int q=0; q<{query_heads}; ++q) for(int t=0; t<{total}; ++t) for(int c=0; c<128; ++c) {{
    int kv = q/2;
    float expected = x[(kv*{total}+t)*128+c];
    compare(out[(q*{total}+t)*128+c], expected, 1e-6f, 1e-6f, &errors, &maximum);
  }}
  return print_check("{name}", errors, maximum);
}}
'''
    return head + body


def attention_case(name, m, n, k):
    return HEAD["attention"] % {"name": name, "m": m, "n": n, "k": k}


def attention_metadata(name, kind, shape, description):
    """Attention cases use the shared ``shape`` schema, not buffer offsets.

    The existing attention examples describe themselves with a single
    [heads, M, N, K] tuple; matching that schema is what lets the shared
    Triton case builder specialize them without a special case for ours.
    """
    return {
        "name": name, "kind": kind, "shape": list(shape), "dtype": "f32",
        "description": description, "target": "nr-fpga",
        "checked_elements": shape[0] * shape[1] * shape[2],
    }


def metadata(name, kind, dtype, description, entry, buffers, validation):
    offset = 0
    records = []
    for label, shape, item in buffers:
        records.append({"name": label, "shape": list(shape), "dtype": item,
                        "offset": offset})
        count = 1
        for dim in shape:
            count *= dim
        offset += count * (4 if item in ("f32", "i32") else 1)
    return {
        "name": name, "kind": kind, "dtype": dtype, "description": description,
        "entry": entry, "workspace_bytes": offset, "buffers": records,
        "validation": validation,
    }


def build_cases(capacity):
    """Return name -> (metadata, launch_source)."""
    cases = {}
    total = capacity

    def add(name, meta, source):
        cases[name] = (meta, source)

    for label, m in (("16", 16), ("1", 1)):
        add(f"attention_qk_16x{m}x{total}x128",
            attention_metadata(f"attention_qk_16x{m}x{total}x128", "attention_qk",
                               (16, m, total, 128),
                               "Per-head Q x K^T over the full decode cache"),
            attention_case(f"attention_qk_16x{m}x{total}x128", m, total, 128))
        add(f"attention_pv_16x{m}x128x{total}",
            attention_metadata(f"attention_pv_16x{m}x128x{total}", "attention_pv",
                               (16, m, 128, total),
                               "Per-head probabilities x V over the full decode cache"),
            attention_case(f"attention_pv_16x{m}x128x{total}", m, 128, total))
        add(f"softmax_16x{m}x{total}",
            metadata(f"softmax_16x{m}x{total}", "softmax", "f32",
                     "Stable softmax over the full valid cache window",
                     f"kernel_softmax_16x{m}x{total}",
                     [("x", (16, m, total), "f32"), ("maxima", (16 * m,), "f32"),
                      ("sums", (16 * m,), "f32"), ("out", (16, m, total), "f32")],
                     "Row sums checked to 1 and every element against a double-precision oracle; -inf padding included."),
            softmax_case(f"softmax_16x{m}x{total}", 16, m, total))
        base = 0 if m == 16 else 16
        add(f"attention_scale_mask_position_16x{m}x{total}",
            metadata(f"attention_scale_mask_position_16x{m}x{total}",
                     "attention_scale_mask_position", "f32",
                     "Scale by 1/sqrt(128) and mask past the runtime cache position",
                     f"kernel_attention_scale_mask_position_16x{m}x{total}",
                     [("x", (16, m, total), "f32"), ("position", (m,), "i32"),
                      ("out", (16, m, total), "f32")],
                     "Masked values must be -inf and unmasked values scaled; boundary taken from the runtime position tensor."),
            mask_position_case(f"attention_scale_mask_position_16x{m}x{total}",
                               16, m, total, base))
        add(f"kv_cache_update_position_{m}x8x128_cap{capacity}",
            metadata(f"kv_cache_update_position_{m}x8x128_cap{capacity}",
                     "kv_cache_update_position", "f32",
                     "Write new K/V into the cache at a run-time destination slot",
                     f"kernel_kv_cache_update_position_{m}x8x128_cap{capacity}",
                     [("x", (m, 8, 128), "f32"), ("position", (m,), "i32"),
                      ("cache", (8, capacity, 128), "f32")],
                     "Whole cache compared, so both the written slots and the untouched region are checked."),
            kv_position_case(f"kv_cache_update_position_{m}x8x128_cap{capacity}",
                             m, 8, capacity))

    add(f"gqa_repeat_8x{total}x128_to_16x{total}x128",
        metadata(f"gqa_repeat_8x{total}x128_to_16x{total}x128", "gqa_repeat", "f32",
                 "Expand 8 KV heads to 16 query heads over the full cache",
                 f"kernel_gqa_repeat_8x{total}x128_to_16x{total}x128",
                 [("x", (8, total, 128), "f32"), ("out", (16, total, 128), "f32")],
                 "Every output element against the kv_head = q_head/2 reference."),
        gqa_case(f"gqa_repeat_8x{total}x128_to_16x{total}x128", 8, 16, total))
    return cases


def case_makefile():
    """Makefile that reuses the shared toolchain config from a deeper directory.

    ``common.mk`` derives ROOT/COMMON/REPO_ROOT/TOOLS from ``..``, which is only
    correct for a case sitting directly under ``qwen3-0.6b/``. Model cases live
    under ``model/``, so this makefile pins those four variables with
    ``override`` (which outranks the plain assignments inside ``common.mk``)
    rather than editing the shared file or relocating the cases.
    """
    qwen = Path(__file__).resolve().parents[2]
    return "\n".join([
        "# Generated by model/tools/model_kernel_cases.py -- do not edit.",
        f"override ROOT := {qwen}",
        f"override COMMON := {qwen.parent / 'common'}",
        f"override REPO_ROOT := {qwen.parents[2]}",
        f"override TOOLS := {qwen / 'tools'}",
        f"include {qwen / 'common.mk'}",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=512)
    args = parser.parse_args()

    cases = build_cases(args.capacity)
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    for name, (meta, source) in sorted(cases.items()):
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (directory / "launch.c").write_text(source)
        (directory / "makefile").write_text(case_makefile())
        names.append(name)
    (args.output / "cases.json").write_text(json.dumps(
        {"capacity": args.capacity, "count": len(names), "cases": names},
        indent=2) + "\n")
    print(f"generated {len(names)} model cases in {args.output}")
    for name in names:
        print("  ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())