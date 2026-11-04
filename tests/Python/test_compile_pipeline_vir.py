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


for name, pipeline_type, llc_attrs in (
    ("A100", "kernels_a100", RVV),
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
    ("A100, zvl256b", "kernels_a100", RVV + ",+zvl256b"),
    ("zvl 256 and 512", "kernels", RVV + ",+zvl256b -mattr=+zvl512b"),
    (
        "min 512, zvl256b",
        "kernels",
        RVV + ",+zvl256b -riscv-v-vector-bits-min=512",
    ),
    (
        "max 256, zvl1024b",
        "kernels",
        RVV + ",+zvl1024b -riscv-v-vector-bits-max=256",
    ),
):
    print(f"{name}: {lower_vir(pipeline_type, llc_attrs)}")
# CHECK: A100: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: A100 IME: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: RVV 1024: -lower-vir-to-vector=rvv-vlen=1024 use-scalable=true
# CHECK-NEXT: RVV 128: -lower-vir-to-vector=rvv-vlen=128 use-scalable=true
# CHECK-NEXT: RVV: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: RVV zvl256b: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: RVV host: -lower-vir-to-vector=use-scalable=true{{$}}
# CHECK-NEXT: x86: -lower-vir-to-vector{{$}}
# CHECK-NEXT: SVE: -lower-vir-to-vector{{$}}
# CHECK-NEXT: A100, zvl256b: error: llc options of several VLEN: +zvl[256, 1024]b
# CHECK-NEXT: zvl 256 and 512: error: llc options of several VLEN: +zvl[256, 512]b
# CHECK-NEXT: min 512, zvl256b: error: llc options of several minimum VLEN: +zvl256b, -riscv-v-vector-bits-min=512
# CHECK-NEXT: max 256, zvl1024b: error: llc options of a maximum VLEN below the minimum 1024: -riscv-v-vector-bits-max=256

# The decode attention of the K3 kernels (DeepSeek-R1: 12 heads, 2 KV heads,
# head 128, 2 heads per work item), a vir.set_vl with results: on the A100,
# one vector<[8]xf32> (128 lanes, LMUL 4) per head and no masks; for no exact
# VLEN, rejected (no code is valid on all of them).
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
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        k3_w4._attn_decode_vir_fn(k3_w4._Types(), SPEC)
    text = str(module)
for name, pipeline_type, llc_attrs in (
    ("A100", "kernels_a100", RVV),
    ("RVV", "kernels", RVV),
):
    out = subprocess.run(
        [BUDDY_OPT, lower_vir(pipeline_type, llc_attrs)],
        input=text,
        capture_output=True,
        text=True,
    )
    if out.returncode:
        errors = sorted(
            {
                l.split("error: ")[1]
                for l in out.stderr.splitlines()
                if "error: " in l
            }
        )
        print(f"{name}: {errors}")
    else:
        print(
            f"{name}: vector<[8]xf32> {'vector<[8]xf32>' in out.stdout}, "
            f"masks {'mask' in out.stdout}"
        )
# CHECK: A100: vector<[8]xf32> True, masks False
# CHECK-NEXT: RVV: ['a vir.set_vl with results on scalable vectors needs rvv-vlen, the exact VLEN']
