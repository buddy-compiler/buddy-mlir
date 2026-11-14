# RUN: %PYTHON %s buddy-opt buddy-translate llc 2>&1 | FileCheck %s
#
# The IME kernels of models/yolov8 ("ime" specs, codegen/yolov8_ime.py): the
# convolutions on the matrix engine of the SpacemiT K3 A100 cores. The host
# cannot run smt.vfwmadot, so this checks what it can: the packed weight
# layout read back, and that every kind of kernel (each convolution variant:
# 2 x 2 on the space-to-depth image, 3 x 3 stride 1 in row tiles and in flat
# tiles, stride 2, 1 x 1 with a residual, a partial last tile, the
# transposed NCHW head tiles, weights in the TCM and their invalidation;
# and the space-to-depth,
# column-parity split, upsample, concatenation, SPPF and head decode)
# lowers as models/yolov8/codegen/lower_to_obj.sh lowers them, down to a
# riscv64 object for the A100. models/yolov8/README.md gives the check of
# the results on a board.

import os
import subprocess
import sys
import tempfile

import numpy

sys.path.insert(
    0, os.path.join(os.environ["BUDDY_SRC_ROOT"], "models", "yolov8", "codegen")
)
from yolov8_ime import Conv, gen_kernels, pack_conv  # noqa: E402

BUDDY_OPT, BUDDY_TRANSLATE, LLC = sys.argv[1:4]
rng = numpy.random.default_rng(0)

# Weights [nt][cb][ky][kx][nbl][c][k] for output channel (nt * Nb + nb) * 8
# + c and input channel cb * 8 + k; 51 output channels: the last tile is
# partial, its channels past 51 zero.
conv = Conv("t", ic=16, oc=51, k=3, s=1, hin=10, win=10, silu=True)
w = rng.standard_normal((51, 16, 3, 3)).astype(numpy.float32)
b = rng.standard_normal(51).astype(numpy.float32)
wp, bp = pack_conv(conv, w, b)
p = wp.view(numpy.float16).reshape(conv.nt, 2, 3, 3, conv.nbl, 8, 8)
oc = numpy.arange(conv.nt * conv.nb * 8)
nt, nb, c = oc // (conv.nb * 8), oc // 8 % conv.nb, oc % 8
read = p[nt, :, :, :, nb, c, :]  # [oc][cb][ky][kx][k]
read = read.transpose(0, 1, 4, 2, 3).reshape(len(oc), 16, 3, 3)
expect = numpy.zeros_like(read)
expect[:51] = w.astype(numpy.float16)
print("weights", "ok" if (read == expect).all() else "WRONG")
bias = bp.reshape(-1, 8, 8)  # [output block][pixel][channel]
expect = numpy.zeros(len(bias) * 8, numpy.float32)
expect[:51] = b
print("bias", "ok" if (bias == expect.reshape(-1, 1, 8)).all() else "WRONG")
# CHECK: weights ok
# CHECK: bias ok

tcm = Conv("yolo_conv_tcm", 64, 64, 3, 1, 20, 20, silu=True)
tcm.tcm_off = 0
kernels = [
    ("s2d", ("yolo_s2d", 3, 64, 64)),
    ("conv", Conv("yolo_conv_k2", 16, 16, 2, 1, 32, 32, silu=True)),
    ("conv", Conv("yolo_conv_rows", 32, 32, 3, 1, 40, 40, silu=True)),
    ("conv", Conv("yolo_conv_flat", 32, 32, 3, 1, 10, 10, silu=True)),
    ("split2", ("yolo_split2", 32, 40, 40)),
    ("conv", Conv("yolo_conv_s2", 32, 64, 3, 2, 40, 40, silu=True)),
    ("conv", Conv("yolo_conv_res", 32, 32, 1, 1, 20, 20, True, True)),
    ("conv", Conv("yolo_conv_nchw", 64, 51, 1, 1, 10, 10, False, nchw=True)),
    ("conv", tcm),
    ("upsample", ("yolo_upsample", 32, 10, 10)),
    ("cat", ("yolo_cat", (32, 64), 20, 20)),
    ("sppf", ("yolo_sppf", 32, 10, 10)),
    ("decode", ("yolo_decode", [(10, 10, 32), (5, 5, 64)], 1, 17)),
]
with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, "kernels.mlir")
    with open(src, "w") as f:
        f.write(gen_kernels(kernels))
    passes = [
        "-lower-ime=target=k3",
        "-convert-scf-to-openmp",
        "-expand-strided-metadata",
        "-convert-vector-to-scf",
        "-lower-affine",
        "-expand-strided-metadata",
        "-memref-expand",
        "-arith-expand",
        "-convert-vector-to-llvm",
        "-convert-ub-to-llvm",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-scf-to-cf",
        "-convert-cf-to-llvm",
        "-llvm-request-c-wrappers",
        "-convert-openmp-to-llvm",
        "-convert-arith-to-llvm",
        "-convert-math-to-llvm",
        "-convert-math-to-libm",
        "-convert-func-to-llvm",
        "-reconcile-unrealized-casts",
    ]
    llvm = subprocess.run(
        [BUDDY_OPT, src, *passes], check=True, capture_output=True, text=True
    ).stdout
    ll = subprocess.run(
        [BUDDY_TRANSLATE, "--buddy-to-llvmir"],
        input=llvm,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for name in [k[1].name if k[0] == "conv" else k[1][0] for k in kernels]:
        if f"@_mlir_ciface_{name}(" not in ll:
            print("missing", name)
    print("vfmadot", "@llvm.riscv.ime.vfmadot" in ll)
    # The runner forgets the TCM copies when it loads weights.
    print(
        "invalidate",
        "define void @buddy_yolov8_tcm_invalidate()" in ll
        and "store i64 0, ptr @yolo_conv_tcm_tcm" in ll,
    )
    obj = os.path.join(tmp, "kernels.o")
    subprocess.run(
        [
            LLC,
            "-mtriple=riscv64",
            "-mattr=+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+xsmtvdotii,+zvl1024b",
            "-mcpu=spacemit-a100",
            "-misched-prera-direction=topdown",
            "-riscv-v-vector-bits-max=1024",
            "-filetype=obj",
            "-relocation-model=pic",
            "-O3",
            "-o",
            obj,
        ],
        input=ll,
        check=True,
        text=True,
    )
    print("object", os.path.getsize(obj) > 0)
# CHECK-NOT: missing
# CHECK: vfmadot True
# CHECK: invalidate True
# CHECK: object True
