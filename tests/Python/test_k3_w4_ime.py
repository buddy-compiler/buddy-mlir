# RUN: %PYTHON %s buddy-opt buddy-translate 2>&1 | FileCheck %s
#
# "prefill_ime" in graph/transform/k3_w4.py: the prefill tiles on the matrix
# engine of the SpacemiT K3 A100 cores, and the prefill attention's matrix
# loops. The host cannot run smt.vmadot / smt.vfwmadot, so this checks what
# it can: the IME weight layout, the two modules (the main one declares the
# IME tiles and the attention's matrix loops, the IME one defines them and
# the tiles' step), and the "kernels_ime" pipeline down to the IME
# intrinsics and a riscv64 object. (test_k3_w4_kernels.py runs the IME
# attention with its matrix loops emulated; docs/K3DeepSeekR1.md gives the
# check of the kernels' results on a board.)

import os
import shutil
import subprocess
import sys
import tempfile

import numpy
from buddy.compiler.graph.transform import k3_w4

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "tools", "buddy-codegen")
)
import compile_pipeline  # noqa: E402

BUDDY_OPT, BUDDY_TRANSLATE = sys.argv[1], sys.argv[2]
rng = numpy.random.default_rng(0)

# The IME layout, read back: per 8-column block, per group, byte c * 16 + j
# holds W[32g + j][8b + c] (low nibble) and W[32g + 16 + j][8b + c] (high
# nibble), then the 8 f16 scales of the block.
k, n = 96, 128
q, scale = k3_w4.quantize_q4(rng.standard_normal((k, n)).astype(numpy.float32))
raw = k3_w4.pack_ime(q, scale).view(numpy.uint8)
groups = k // 32
blocks = raw.reshape(n // 8, groups, k3_w4.IME_BLOCK)
nib = blocks[:, :, :128].reshape(n // 8, groups, 8, 16).astype(numpy.int16)
lo = ((nib & 0x0F) ^ 8) - 8
hi = ((nib >> 4) ^ 8) - 8
back = numpy.concatenate([lo, hi], axis=3)  # block, group, column, row
back = back.transpose(1, 3, 0, 2).reshape(groups, 32, n)
scales = blocks[:, :, 128:].copy().view(numpy.float16).transpose(1, 0, 2)
print(
    "IME layout: values",
    numpy.array_equal(back, q),
    "scales",
    numpy.array_equal(scales.reshape(groups, n), scale),
    "size",
    raw.size == k * n * 9 // 16,
)
# CHECK: IME layout: values True scales True size True

specs = [
    k3_w4.kernel_spec("plain", 64, 96, [128], False, 8, ime=True),
    dict(
        k3_w4.kernel_spec("multi", 64, 96, [128] * 3, True, 8, ime=True),
        norm_eps=1e-6,
    ),
    k3_w4.kernel_spec("glu", 64, 96, [256], False, 8, ime=True),
    k3_w4.kernel_spec("plain", 1, 96, [128], False, 8),
    {
        "name": "k3_attn_m64_ime",
        "kind": "attn",
        "m": 64,
        "heads": 4,
        "kv_heads": 2,
        "dim": 128,
        "scale": 128**-0.5,
        "ctx": 128,
        "ime": True,
    },
]
specs[1]["name"] += "_rms"
main = k3_w4.gen_kernels(specs, "main")
ime = k3_w4.gen_kernels(specs, "ime")


def functions(text, private):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("func.func") and ("private" in line) == private:
            name = line.split("@")[1].split("(")[0]
            out.append(
                name + (" (declaration)" if not line.endswith("{") else "")
            )
    return sorted(out)


print("main:", ", ".join(functions(main, False)))
print("main private:", ", ".join(functions(main, True)))
print("ime private:", ", ".join(functions(ime, True)))
print(
    "IME ops in main:",
    main.count("ime."),
    "in ime:",
    ime.count("ime.intr.vmadot.hp"),
    "vmadot.hp,",
    ime.count("ime.intr.vfmadot"),
    "vfmadot",
)
# CHECK: main: k3_attn_m64_ime, k3_q4_glu_m64_k96_n256_ime, k3_q4_multi_m64_k96_n128_128_128_b_ime_rms, k3_q4_plain_m1_k96_n128, k3_q4_plain_m64_k96_n128_ime
# CHECK-NEXT: main private: buddy_spacemit_tcm_pair (declaration), k3_attn_ime_pv (declaration), k3_attn_ime_qk (declaration), k3_q4_glu_m64_k96_n256_ime__tile (declaration), k3_q4_multi_m64_k96_n128_128_128_b_ime_rms__tile (declaration), k3_q4_plain_m1_k96_n128__tile, k3_q4_plain_m64_k96_n128_ime__tile (declaration)
# CHECK-NEXT: ime private: buddy_spacemit_tcm_here (declaration), k3_attn_ime_pv, k3_attn_ime_qk, k3_ime_hp_step, k3_q4_glu_m64_k96_n256_ime__tile, k3_q4_multi_m64_k96_n128_128_128_b_ime_rms__tile, k3_q4_plain_m64_k96_n128_ime__tile
# CHECK-NEXT: IME ops in main: 0 in ime: 8 vmadot.hp, 16 vfmadot

# The activations of an IME call go through the TCM of each core pair, in
# passes over K that fit in it (768 KiB): k 1536 in one pass, k 8960 in two
# of 140 groups, each a copy (buddy_spacemit_tcm_pair) and a parallel loop
# of tiles (5 parallel loops with the quantization).
print(
    "pass groups: k 1536",
    k3_w4._ime_pass_groups(48),
    "k 8960",
    k3_w4._ime_pass_groups(280),
)
down = k3_w4.gen_kernels(
    [k3_w4.kernel_spec("plain", 64, 8960, [128], False, 8, ime=True)], "main"
)
print(
    "k 8960: copies",
    down.count("call @buddy_spacemit_tcm_pair"),
    "tile calls",
    down.count("call @k3_q4_plain_m64_k8960_n128_ime__tile"),
    "omp loops",
    down.count("scf.parallel"),
)
# CHECK: pass groups: k 1536 48 k 8960 140
# CHECK-NEXT: k 8960: copies 2 tile calls 2 omp loops 5

# The "kernels_ime" pipeline: -lower-ime target=k3, buddy-translate, llc for
# the A100, the exact VLEN included whatever the build gives the other
# kernels (an equal one is not repeated, another one is refused).
stages = compile_pipeline.build_stages("kernels_ime", 8, "", "w4g32")
print("tools:", " ".join(tool for tool, _ in stages))
print("llc:", " ".join(stages[-1][1]))
# CHECK: tools: buddy-opt buddy-translate llvm-as llc
# CHECK-NEXT: llc: -code-model=large -mattr=+xsmtvdotii,+zvl1024b -mcpu=spacemit-a100 -misched-prera-direction=topdown -riscv-v-vector-bits-max=1024 -filetype=obj -relocation-model=pic -O3
# "kernels_a100" (the other kernels of a model on the A100 cores): the passes
# of "kernels", the llc options of the A100.
a100 = compile_pipeline.build_stages("kernels_a100", 8, "", "w4g32")
plain = compile_pipeline.build_stages("kernels", 8, "", "w4g32")
print("kernels_a100 tools:", " ".join(tool for tool, _ in a100))
print("same passes:", a100[0] == plain[0], "same llc:", a100[-1] == stages[-1])
# CHECK-NEXT: kernels_a100 tools: buddy-opt mlir-translate llvm-as llc
# CHECK-NEXT: same passes: True same llc: True
for given in ("-riscv-v-vector-bits-max=1024", "-riscv-v-vector-bits-max=256"):
    try:
        llc = compile_pipeline.build_stages("kernels_ime", 8, given, "w4g32")[
            -1
        ][1]
        print(f"{given}: given once {llc.count(given) == 1}")
    except ValueError as e:
        print(f"{given}: ValueError: {e}")
# CHECK-NEXT: -riscv-v-vector-bits-max=1024: given once True
# CHECK-NEXT: -riscv-v-vector-bits-max=256: ValueError: the A100 IME kernels need VLEN 1024, the build gives -riscv-v-vector-bits-max=256
lowered = subprocess.run(
    [BUDDY_OPT, *stages[0][1]],
    input=ime,
    capture_output=True,
    text=True,
    check=True,
).stdout
llvm_ir = subprocess.run(
    [BUDDY_TRANSLATE, "--buddy-to-llvmir"],
    input=lowered,
    capture_output=True,
    text=True,
    check=True,
).stdout
print(
    "vmadot.hp calls:",
    llvm_ir.count("call <vscale x 4 x half> @llvm.riscv.ime.vmadot.hp"),
    "vfmadot calls:",
    llvm_ir.count("call <vscale x 4 x float> @llvm.riscv.ime.vfmadot"),
)
# CHECK: vmadot.hp calls: 8 vfmadot calls: 16

work = tempfile.mkdtemp()
with open(os.path.join(work, "ime.ll"), "w") as f:
    f.write(llvm_ir)
# the build's options without any VLEN: the pipeline brings its own
attrs = ["-mtriple=riscv64-unknown-linux-gnu", "-mattr=+m,+d,+v,+zfh,+zvfh"]
obj = os.path.join(work, "ime.o")
subprocess.run(
    [
        shutil.which("llc"),
        *attrs,
        *stages[-1][1][1:],
        os.path.join(work, "ime.ll"),
        "-o",
        obj,
    ],
    check=True,
)
print("riscv64 object:", os.path.getsize(obj) > 0)
# CHECK: riscv64 object: True
