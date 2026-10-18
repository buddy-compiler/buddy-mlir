# ===- spacemit-k3-insn-bench.py -----------------------------------------------
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
# Per-instruction cost of RVV instructions at e32 LMUL 4 (one
# vector<128xf32> at VLEN 1024) on a SpacemiT K3 A100 core. Writes an
# assembly file and a C driver into OUT and cross-compiles them:
#
#   python3 spacemit-k3-insn-bench.py OUT \
#       --clang $LLVM_MLIR_BUILD_DIR/bin/clang \
#       --toolchain $BUILD_RISCV_GNU_TOOLCHAIN_DIR
#
# then on the board, as an AI process:
#
#   sh -c 'echo 0 > /proc/set_ai_thread && exec ./insn-bench'
#
# Each kernel is a loop of 16 instructions; "_dep" kernels chain them through
# one register (latency), the others use independent destinations
# (throughput). Add kernels to KERNELS to measure other instructions.
#
# ===---------------------------------------------------------------------------

import argparse
import os
import subprocess

M4 = "vsetvli t0, zero, e32, m4, ta, ma"
MF2 = "vsetivli t0, 16, e32, mf2, ta, ma"


def _ind(op):
    """16 instances writing v0 / v4 / v8 in turn."""
    return [op.format(d=f"v{4 * (i % 3)}") for i in range(16)]


KERNELS = {
    # name: (vsetvl, body)
    "vfmul_m4": (M4, _ind("vfmul.vv {d}, v16, v20")),
    "vfmacc_m4": (M4, _ind("vfmacc.vv {d}, v16, v20")),
    "vfmacc_vf_m4": (M4, _ind("vfmacc.vf {d}, fa0, v20")),
    "vfredusum_m4": (
        M4,
        [f"vfredusum.vs v{i % 12}, v16, v24" for i in range(16)],
    ),
    "vfredusum_m4_dep": (M4, ["vfredusum.vs v24, v16, v24"] * 16),
    "vfredmax_m4": (
        M4,
        [f"vfredmax.vs v{i % 12}, v16, v24" for i in range(16)],
    ),
    "vrgather_vi_m4": (M4, _ind("vrgather.vi {d}, v16, 3")),
    "vslideup_vi_m4": (M4, _ind("vslideup.vi {d}, v16, 3")),
    "vslideup_vi_mf2": (
        MF2,
        [f"vslideup.vi v{i % 12 + 1}, v24, 3" for i in range(16)],
    ),
    "vfmv_f_s_m4": (
        M4,
        [f"vfmv.f.s fa{i % 8}, v{4 * (i % 3)}" for i in range(16)],
    ),
    "vfmv_v_f_m4": (M4, _ind("vfmv.v.f {d}, fa0")),
    "vl4re32_l1": (M4, [f"vl4re32.v v{4 * (i % 4)}, (a1)" for i in range(16)]),
    "vs4r_l1": (M4, [f"vs4r.v v{4 * (i % 4)}, (a1)" for i in range(16)]),
}

DRIVER_HEAD = r"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e9 + t.tv_nsec;
}
"""


def write_sources(out):
    with open(os.path.join(out, "insn.S"), "w") as f:
        f.write("    .text\n    .option arch, +v, +zfh, +zvfh\n")
        for name, (vl, body) in KERNELS.items():
            f.write(
                f"    .globl k_{name}\n    .p2align 2\nk_{name}:\n    {vl}\n1:\n"
            )
            f.write("".join(f"    {b}\n" for b in body))
            f.write("    addi a0, a0, -1\n    bnez a0, 1b\n    ret\n")
    with open(os.path.join(out, "driver.c"), "w") as f:
        f.write(DRIVER_HEAD)
        for name in KERNELS:
            f.write(f"void k_{name}(long iterations, void *buffer);\n")
        f.write("int main(void) {\n  void *b = aligned_alloc(4096, 1 << 16);\n")
        for name, (_, body) in KERNELS.items():
            n = len(body)
            f.write(
                f"  {{ long r = 200000; k_{name}(1000, b); double t0 = now();"
                f" k_{name}(r, b); double t = now() - t0;"
                f' printf("%-18s %7.2f ns per instruction\\n", "{name}",'
                f" t / (r * {n})); }}\n"
            )
        f.write("  return 0;\n}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out")
    p.add_argument("--clang", default="clang")
    p.add_argument("--toolchain", help="RISC-V GNU toolchain install dir")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_sources(args.out)
    cmd = [
        args.clang,
        "--target=riscv64-unknown-linux-gnu",
        "-O2",
        "-march=rv64gcv_zfh_zvfh",
    ]
    if args.toolchain:
        cmd += [
            f"--sysroot={args.toolchain}/sysroot",
            f"--gcc-toolchain={args.toolchain}",
        ]
    cmd += [
        os.path.join(args.out, "driver.c"),
        os.path.join(args.out, "insn.S"),
        "-o",
        os.path.join(args.out, "insn-bench"),
    ]
    subprocess.run(cmd, check=True)
    print(f"built {os.path.join(args.out, 'insn-bench')}")


if __name__ == "__main__":
    main()
