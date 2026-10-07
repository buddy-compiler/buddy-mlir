# ===- spacemit-k3-w4-kernel-bench.py ------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Board microbenchmarks of the w4g32 kernels (graph/transform/k3_w4.py) of
# DeepSeek-R1-Distill-Qwen-1.5B on the SpacemiT K3 A100 cores. Run from the
# buddy-mlir source tree, with build/python_packages on PYTHONPATH:
#
#   python3 spacemit-k3-w4-kernel-bench.py matmul OUT --rows 1,4,8 \
#       --toolchain $BUILD_RISCV_GNU_TOOLCHAIN_DIR --llvm-bin $LLVM_MLIR_BUILD_DIR/bin
#   python3 spacemit-k3-w4-kernel-bench.py attention OUT ...
#
# then on the board, as an AI process:
#
#   sh -c 'echo 0 > /proc/set_ai_thread && exec OUT/run'           # matmul
#   sh -c 'echo 0 > /proc/set_ai_thread && exec OUT/run 900'       # attention
#
# matmul: the q/k/v, o, gate/up and down kernels of one layer for each row
# count, each call streaming its weights from DRAM (several copies, 32 MB or
# more), best of 60 calls, and the weight bandwidth reached.
# attention: the decode attention at a position, best of 300 calls; writes
# its output to o_<name>_<position> for `cmp` against another build.
#
# ===---------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys

sys.path.insert(0, "tools/buddy-codegen")
import compile_pipeline  # noqa: E402
from buddy.compiler.graph.transform import k3_w4  # noqa: E402

ATTRS = (
    "-march=riscv64 -mattr=+m,+d,+v,+zfh,+zvfh,+zvl1024b "
    "-mtriple=riscv64-unknown-linux-gnu -riscv-v-vector-bits-max=1024"
)
THREADS = 8
# name, kind, k, ns, bias, fused RMSNorm (DeepSeek-R1-1.5B layer)
MATMULS = [
    ("qkv", "multi", 1536, [1536, 256, 256], True, True),
    ("o", "plain", 1536, [1536], False, False),
    ("gate_up", "glu", 1536, [8960], False, True),
    ("down", "plain", 8960, [1536], False, False),
]

COMMON = r"""#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { float *a, *d; int64_t off, size[2], stride[2]; } M2;
typedef struct { float *a, *d; int64_t off, size[3], stride[3]; } M3;
typedef struct { float *a, *d; int64_t off, size[4], stride[4]; } M4;
typedef struct { void *a, *d; int64_t off, size, stride; } M1;
typedef struct { M2 r[3]; } R3;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *buf(size_t n) { char *p = aligned_alloc(128, (n + 127) / 128 * 128); for (size_t i = 0; i < n; i++) p[i] = (char)(i * 7); return p; }
static float *rnd(size_t n) { float *p = aligned_alloc(128, n * 4); for (size_t i = 0; i < n; i++) p[i] = (float)rand() / (float)RAND_MAX - 0.5f; return p; }
"""

ATTN_MAIN = r"""
typedef struct { M2 o; M3 lse; M4 kc, vc; } R;
int main(int argc, char **argv) {
  int64_t start = argc > 1 ? atoll(argv[1]) : 900;
  float *q = rnd(12 * 128), *k = rnd(256), *v = rnd(256), *kc = rnd(2 * 1024 * 128), *vc = rnd(2 * 1024 * 128);
  float invf[64]; for (int i = 0; i < 64; i++) invf[i] = 1.0f / (i + 1);
  M2 Q = {q, q, 0, {1, 1536}, {1536, 1}}, K = {k, k, 0, {1, 256}, {256, 1}}, V = {v, v, 0, {1, 256}, {256, 1}};
  M4 KC = {kc, kc, 0, {1, 2, 1024, 128}, {2 * 1024 * 128, 1024 * 128, 128, 1}}, VC = KC; VC.a = VC.d = vc;
  int64_t pos = start; M1 P = {&pos, &pos, 0, 1, 1}; M1 F = {invf, invf, 0, 64, 1};
  R r; double best = 1e9;
  for (int i = 0; i < 300; i++) {
    double t0 = now(); _mlir_ciface_%NAME%(&r, &Q, &K, &V, &KC, &VC, &P, &F); double t = now() - t0;
    if (t < best) best = t;
    if (i < 299) free(r.o.a), free(r.lse.a);
  }
  printf("%NAME% position %lld: %.1f us\n", (long long)start, best * 1e6);
  char path[96]; snprintf(path, sizeof path, "o_%NAME%_%lld", (long long)start);
  FILE *fo = fopen(path, "wb"); fwrite(r.o.d, 4, 1536, fo); fclose(fo);
  return 0;
}
"""


def matmul_sources(rows):
    specs, decls, body = [], [], ["int main(void) {"]
    for name0, kind, k, ns, bias, rms in MATMULS:
        for m in rows:
            name = f"{name0}_m{m}"
            s = k3_w4.kernel_spec(kind, m, k, ns, bias, THREADS)
            if rms:
                s = dict(s, name=s["name"] + "_rms", norm_eps=1e-6)
            specs.append(s)
            wbytes = k * sum(ns) * 9 // 16 * (2 if kind == "glu" else 1)
            res = "R3" if len(ns) == 3 else "M2"
            args = ["M2 *", "M1 *"] + ["M1 *"] * (bias + rms)
            decls.append(
                f"void _mlir_ciface_{s['name']}({res} *, {', '.join(args)});"
            )
            copies = max(1, (32 << 20) // wbytes + 1)
            call = f"_mlir_ciface_{s['name']}(&r, &X, &W[i % {copies}]"
            call += (", &B" if bias else "") + (", &N" if rms else "") + ");"
            body += [
                f"  {{ float *x = buf({m} * {k} * 4); M2 X = {{x, x, 0, {{{m}, {k}}}, {{{k}, 1}}}};",
                f"    M1 W[{copies}]; for (int c = 0; c < {copies}; c++) {{ void *w = buf({wbytes}); M1 t = {{w, w, 0, {wbytes}, 1}}; W[c] = t; }}",
                f"    float *b = buf({sum(ns)} * 4); M1 B = {{b, b, 0, {sum(ns)}, 1}};",
                f"    float *nw = buf({k} * 4); M1 N = {{nw, nw, 0, {k}, 1}};",
                f"    {res} r; double best = 1e9;",
                f"    for (int i = 0; i < 60; i++) {{ double t0 = now(); {call} double t = now() - t0; if (t < best) best = t; }}",
                f'    printf("{name:10s} weights %6.2f MB %8.1f us %5.1f GB/s\\n", {wbytes} / 1e6, best * 1e6, {wbytes} / best / 1e9); }}',
            ]
    body.append("  return 0;\n}")
    return specs, COMMON + "\n".join(decls) + "\n" + "\n".join(body) + "\n"


def attention_sources(heads_per_item_threads):
    spec = {
        "kind": "attn",
        "m": 1,
        "heads": 12,
        "kv_heads": 2,
        "dim": 128,
        "scale": 128**-0.5,
        "ctx": 1024,
        "name": "attn_decode",
        "threads": heads_per_item_threads,
    }
    decl = (
        "void _mlir_ciface_attn_decode(void *, M2 *, M2 *, M2 *, M4 *,"
        " M4 *, M1 *, M1 *);\n"
    )
    return [spec], COMMON + decl + ATTN_MAIN.replace("%NAME%", "attn_decode")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["matmul", "attention"])
    p.add_argument("out")
    p.add_argument(
        "--rows",
        default="1,4,8",
        help="matmul row counts (1 or multiples of 4)",
    )
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--pipeline", default="kernels_a100")
    p.add_argument("--buddy-opt", default="build/bin/buddy-opt")
    p.add_argument(
        "--llvm-bin",
        required=True,
        help="host LLVM bin dir (mlir-translate, llc, clang)",
    )
    p.add_argument(
        "--toolchain", required=True, help="RISC-V GNU toolchain install dir"
    )
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.kind == "matmul":
        specs, driver = matmul_sources([int(r) for r in args.rows.split(",")])
    else:
        specs, driver = attention_sources(args.threads)
    with open(f"{args.out}/driver.c", "w") as f:
        f.write(driver)
    with open(f"{args.out}/k.mlir", "w") as f:
        f.write(k3_w4.gen_kernels(specs, "main"))
    stages = compile_pipeline.build_stages(
        args.pipeline, args.threads, ATTRS, "w4g32"
    )
    compile_pipeline.run_pipeline(
        stages,
        f"{args.out}/k.mlir",
        f"{args.out}/k.o",
        args.buddy_opt,
        args.llvm_bin,
    )
    t = args.toolchain
    subprocess.run(
        [
            f"{args.llvm_bin}/clang",
            "--target=riscv64-unknown-linux-gnu",
            f"--sysroot={t}/sysroot",
            f"--gcc-toolchain={t}",
            "-O2",
            "-march=rv64gcv",
            f"{args.out}/driver.c",
            f"{args.out}/k.o",
            "runtime/threadpool/BuddyThreadPool.c",
            "-lm",
            "-lpthread",
            "-o",
            f"{args.out}/run",
        ],
        check=True,
    )
    print(f"built {args.out}/run")


if __name__ == "__main__":
    main()
