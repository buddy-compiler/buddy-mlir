# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# The -lower-vir-to-vector of the "kernels" pipelines of compile_pipeline.py
# for the target of their llc options: scalable vectors on RVV (for the
# exact VLEN when the options give it, else for any VLEN), fixed
# vectors elsewhere; VLEN options that disagree are an error.

import os
import subprocess
import sys

from buddy.compiler.graph.transform import k3_w4
from buddy_mlir import ir

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402

BUDDY_OPT = sys.argv[1]
RVV = "-march=riscv64 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v"


def lower_vir(pipeline_type, llc_attrs):
    """The -lower-vir-to-vector of the pipeline, or its error."""
    try:
        stages = compile_pipeline.build_stages(
            pipeline_type, num_threads=4, llc_attrs=llc_attrs, variant="w4g32"
        )
    except ValueError as e:
        return f"error: {e}"
    for _, args in stages:
        for a in args:
            if a.startswith("-lower-vir-to-vector"):
                return a
    raise AssertionError(f"no -lower-vir-to-vector in {pipeline_type}")


K3 = RVV + " -mattr=+zvl1024b -riscv-v-vector-bits-max=1024"
for name, pipeline_type, llc_attrs in (
    ("A100", "kernels_a100", K3),
    ("A100, no VLEN", "kernels_a100", RVV),
    (
        "A100 IME",
        "kernels_ime",
        RVV + ",+zvl1024b -riscv-v-vector-bits-max=1024",
    ),
    ("RVV 1024", "kernels", RVV + ",+zvl1024b -riscv-v-vector-bits-max=1024"),
    ("RVV 128", "kernels", RVV + ",+zvl128b -riscv-v-vector-bits-max=128"),
    ("RVV", "kernels", RVV),
    ("RVV zvl256b", "kernels", RVV + ",+zvl256b"),
    ("RVV host", "kernels", "-mcpu=native -mattr=+m,+d,+v"),
    ("x86", "kernels", "-mcpu=native"),
    ("SVE", "kernels", "-mtriple=aarch64-unknown-linux-gnu -mattr=+sve"),
    ("RVV max 0", "kernels", RVV + ",+zvl128b -riscv-v-vector-bits-max=0"),
    (
        "RVV min 512 max 512",
        "kernels",
        RVV + ",+zvl256b -riscv-v-vector-bits-min=512 "
        "-riscv-v-vector-bits-max=512",
    ),
    ("RVV min 512", "kernels", RVV + ",+zvl256b -riscv-v-vector-bits-min=512"),
    (
        "A100, VLEN 256",
        "kernels_a100",
        RVV + ",+zvl256b -riscv-v-vector-bits-max=256",
    ),
    ("zvl 256 and 512", "kernels", RVV + ",+zvl256b -mattr=+zvl512b"),
    (
        "min 256, zvl512b",
        "kernels",
        RVV + ",+zvl512b -riscv-v-vector-bits-min=256",
    ),
    (
        "max 256, zvl1024b",
        "kernels",
        RVV + ",+zvl1024b -riscv-v-vector-bits-max=256",
    ),
):
    print(f"{name}: {lower_vir(pipeline_type, llc_attrs)}")
# CHECK: A100: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: A100, no VLEN: error: the A100 code needs the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); the build's llc options give no exact VLEN
# CHECK-NEXT: A100 IME: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: RVV 1024: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: RVV 128: -lower-vir-to-vector=rvv-vlen=128 use-scalable=true
# CHECK-NEXT: RVV: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: RVV zvl256b: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: RVV host: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: x86: -lower-vir-to-vector{{$}}
# CHECK-NEXT: SVE: -lower-vir-to-vector{{$}}
# CHECK-NEXT: RVV max 0: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: RVV min 512 max 512: -lower-vir-to-vector=rvv-vlen=512 use-scalable=true
# CHECK-NEXT: RVV min 512: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: A100, VLEN 256: error: the A100 code needs the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); the build's llc options give VLEN 256
# CHECK-NEXT: zvl 256 and 512: error: llc options of several VLEN: +zvl[256, 512]b
# CHECK-NEXT: min 256, zvl512b: error: llc options of a -riscv-v-vector-bits-min=256 below the VLEN of +zvl512b
# CHECK-NEXT: max 256, zvl1024b: error: llc options of a -riscv-v-vector-bits-max=256 below the VLEN of +zvl1024b

# The decode attention of the K3 kernels (DeepSeek-R1: 12 heads, 2 KV heads,
# head 128, 2 heads per work item; with q / k norms, as Qwen3), a vir.set_vl
# with results whose rotate_half is a vir.shuffle of the two rows of a head.
# For an exact VLEN, scalable vectors of the pressure's LMUL 4,
# vector<[8]xf32>: one per head on the A100 (VLEN 1024; the rows are parts of
# it, taken and put in place), a row of whole pieces below (they swap, no
# instruction); no masks. For no exact VLEN, rejected (no code is valid on
# every VLEN).
import re  # noqa: E402

SPEC = {
    "kind": "attn",
    "m": 1,
    "heads": 12,
    "kv_heads": 2,
    "dim": 128,
    "scale": 128**-0.5,
    "ctx": 1024,
    "threads": 8,
    "name": "attn",
}


def exact(vlen):
    return RVV + f",+zvl{vlen}b -riscv-v-vector-bits-max={vlen}"


for norm in (False, True):
    spec = dict(SPEC, qk_norm_eps=1e-6) if norm else SPEC
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            k3_w4._attn_decode_vir_fn(k3_w4._Types(), spec)
        text = str(module)
    for name, pipeline_type, llc_attrs in (
        ("A100", "kernels_a100", exact(1024)),
        ("RVV 512", "kernels", exact(512)),
        ("RVV 256", "kernels", exact(256)),
        ("RVV 128", "kernels", exact(128)),
        ("RVV", "kernels", RVV),
    ):
        out = subprocess.run(
            [BUDDY_OPT, lower_vir(pipeline_type, llc_attrs)],
            input=text,
            capture_output=True,
            text=True,
        )
        name = f"{'q/k norm ' if norm else ''}{name}"
        if out.returncode:
            errors = sorted(
                {
                    l.split("error: ")[1]
                    for l in out.stderr.splitlines()
                    if "error: " in l
                }
            )
            print(f"{name}: {errors}")
            continue
        types = sorted(set(re.findall(r"vector<\[\d+\]xf32>", out.stdout)))
        print(
            f"{name}: {' '.join(types)}, extract/insert "
            f"{'scalable.extract' in out.stdout}, splice "
            f"{'splice' in out.stdout}, masks {'mask' in out.stdout}"
        )
# CHECK: A100: vector<[4]xf32> vector<[8]xf32>, extract/insert True, splice False, masks False
# CHECK-NEXT: RVV 512: vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: RVV 256: vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: RVV 128: vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: RVV: ['a vir.set_vl with results on scalable vectors needs rvv-vlen, the exact VLEN']
# CHECK-NEXT: q/k norm A100: vector<[4]xf32> vector<[8]xf32>, extract/insert True, splice False, masks False
# CHECK-NEXT: q/k norm RVV 512: vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: q/k norm RVV 256: vector<[16]xf32> vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: q/k norm RVV 128: vector<[16]xf32> vector<[8]xf32>, extract/insert False, splice False, masks False
# CHECK-NEXT: q/k norm RVV: ['a vir.set_vl with results on scalable vectors needs rvv-vlen, the exact VLEN']
