# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# The llc options of models/yolov8/codegen/lower_to_obj.sh: with the IME
# kernels ("ime" specs), the build's, then those of the A100 code of the
# other models (compile_pipeline.a100_llc_args), which need the build's to
# give the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); without them, the build's
# only. The tools are stubs that log their arguments.

import os
import stat
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(
    os.environ["BUDDY_SRC_ROOT"],
    "models",
    "yolov8",
    "codegen",
    "lower_to_obj.sh",
)
RVV = "-mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v,+zfh,+zvfh"

STUB = """#!/usr/bin/env bash
echo "$(basename "$0") $*" >> "$STUB_LOG"
cat > /dev/null
prev=
for a in "$@"; do
  [[ "$prev" == -o && "$a" != - ]] && : > "$a"
  prev="$a"
done
"""


def run(llc_attrs, ime):
    """The options of the last llc run of the script, or its error."""
    with tempfile.TemporaryDirectory() as tmp:
        tools = os.path.join(tmp, "bin")
        os.mkdir(tools)
        for t in (
            "mlir-opt",
            "mlir-translate",
            "llvm-as",
            "llvm-link",
            "llc",
            "buddy-opt",
            "buddy-translate",
        ):
            path = os.path.join(tools, t)
            with open(path, "w") as f:
                f.write(STUB)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        src = os.path.join(tmp, "subgraph0.mlir")
        open(src, "w").close()
        if ime:
            open(os.path.join(tmp, "yolov8_kernels.mlir"), "w").close()
        log = os.path.join(tmp, "log")
        out = subprocess.run(
            [
                "bash",
                SCRIPT,
                os.path.join(tools, "buddy-opt"),
                tools,
                src,
                os.path.join(tmp, "subgraph0.o"),
                "8",
                *llc_attrs.split(),
            ],
            env=dict(os.environ, STUB_LOG=log, PYTHON=sys.executable),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if out.returncode:
            return f"error: {out.stderr.strip()}"
        with open(log) as f:
            llc = [l for l in f.read().splitlines() if l.startswith("llc ")]
        return llc[-1].split(" -filetype=obj")[0]


K3 = RVV + " -mattr=+v,+zvl1024b -riscv-v-vector-bits-max=1024"
for name, llc_attrs, ime in (
    ("ime, K3", K3, True),
    (
        "ime, VLEN 256",
        RVV + " -mattr=+v,+zvl256b -riscv-v-vector-bits-max=256",
        True,
    ),
    ("ime, zvl256b", RVV + ",+zvl256b", True),
    ("ime, no VLEN", RVV, True),
    ("f32, K3", K3, False),
):
    print(f"{name}: {run(llc_attrs, ime)}")
# CHECK: ime, K3: llc -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v,+zfh,+zvfh -mattr=+v,+zvl1024b -riscv-v-vector-bits-max=1024 -mattr=+xsmtvdotii -mcpu=spacemit-a100 -misched-prera-direction=topdown{{$}}
# CHECK-NEXT: ime, VLEN 256: error: lower_to_obj.sh: the A100 code needs the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); the build's llc options give VLEN 256
# CHECK-NEXT: ime, zvl256b: error: lower_to_obj.sh: the A100 code needs the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); the build's llc options give no exact VLEN
# CHECK-NEXT: ime, no VLEN: error: lower_to_obj.sh: the A100 code needs the exact VLEN 1024 (BUDDY_RISCV_VLEN=1024); the build's llc options give no exact VLEN
# CHECK-NEXT: f32, K3: llc -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v,+zfh,+zvfh -mattr=+v,+zvl1024b -riscv-v-vector-bits-max=1024{{$}}
