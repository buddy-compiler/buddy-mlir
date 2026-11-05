# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# tools/buddy-codegen/cmake/buddy_rvv_vlen.cmake, run with cmake -P. The VLEN
# of an RVV model build is set in one place, BUDDY_RISCV_VLEN:
# buddy_rvv_vlen_llc_attrs adds its VLEN options to the llc options (a
# model's LLC_ATTRS included), which must not set one; buddy_rvv_vlen_required
# reads the exact VLEN of the final llc options (as
# compile_pipeline.exact_rvv_vlen) for the VLEN check of the model's
# libraries (runtime/rvv/BuddyRvvVlen.c), "" for code valid on any VLEN.

import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.environ["BUDDY_SRC_ROOT"]
CMAKE = shutil.which("cmake")
RVV = "-march=riscv64 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v"


def run(call):
    script = f"""
include("{SRC}/tools/buddy-codegen/cmake/buddy_rvv_vlen.cmake")
{call}
message(STATUS "'${{out}}'")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cmake") as f:
        f.write(script)
        f.flush()
        out = subprocess.run(
            [CMAKE, "-P", f.name], capture_output=True, text=True
        )
    if out.returncode:
        lines = [l.strip() for l in out.stderr.splitlines() if l.strip()]
        return "error: " + " ".join(
            l for l in lines if not l.startswith("CMake Error")
        )
    return out.stdout.strip().removeprefix("-- ")


def llc_attrs(attrs, vlen=""):
    return run(
        f'buddy_rvv_vlen_llc_attrs(out LLC_ATTRS "{attrs}" VLEN "{vlen}" '
        f'CONTEXT "model")'
    )


def required(attrs):
    return run(
        f'buddy_rvv_vlen_required(out LLC_ATTRS "{attrs}" '
        f'PYTHON "{sys.executable}" '
        f'CODEGEN_DIR "{SRC}/tools/buddy-codegen" CONTEXT "model")'
    )


for name, attrs, vlen in (
    ("BUDDY_RISCV_VLEN=1024", RVV, "1024"),
    ("no BUDDY_RISCV_VLEN", RVV, ""),
    ("LLC_ATTRS, BUDDY_RISCV_VLEN=256", "-mcpu=native -mattr=+m,+d,+v", "256"),
    ("LLC_ATTRS with a zvl", "-mcpu=native -mattr=+m,+d,+v,+zvl256b", "1024"),
    ("LLC_ATTRS with a max", "-mcpu=native -riscv-v-vector-bits-max=256", ""),
):
    print(f"llc options, {name}: {llc_attrs(attrs, vlen)}")
# CHECK: llc options, BUDDY_RISCV_VLEN=1024: '-march=riscv64 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v -mattr=+zvl1024b -riscv-v-vector-bits-max=1024'
# CHECK-NEXT: llc options, no BUDDY_RISCV_VLEN: '-march=riscv64 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v'
# CHECK-NEXT: llc options, LLC_ATTRS, BUDDY_RISCV_VLEN=256: '-mcpu=native -mattr=+m,+d,+v -mattr=+zvl256b -riscv-v-vector-bits-max=256'
# CHECK-NEXT: llc options, LLC_ATTRS with a zvl: error: {{.*}}model: the llc options "-mcpu=native -mattr=+m,+d,+v,+zvl256b" set a VLEN (+zvl256b); the VLEN of a build is BUDDY_RISCV_VLEN.
# CHECK-NEXT: llc options, LLC_ATTRS with a max: error: {{.*}}model: the llc options "-mcpu=native -riscv-v-vector-bits-max=256" set a VLEN (-riscv-v-vector-bits-max); the VLEN of a build is BUDDY_RISCV_VLEN.

for name, attrs in (
    ("VLEN 1024", RVV + " -mattr=+zvl1024b -riscv-v-vector-bits-max=1024"),
    (
        "VLEN 256",
        "-mcpu=native -mattr=+m,+d,+v -mattr=+zvl256b -riscv-v-vector-bits-max=256",
    ),
    ("no exact VLEN", RVV),
    ("x86", "-mcpu=native"),
    ("zvl 256 and 512", RVV + ",+zvl256b -mattr=+zvl512b"),
):
    print(f"check, {name}: {required(attrs)}")
# CHECK-NEXT: check, VLEN 1024: '1024'
# CHECK-NEXT: check, VLEN 256: '256'
# CHECK-NEXT: check, no exact VLEN: ''
# CHECK-NEXT: check, x86: ''
# CHECK-NEXT: check, zvl 256 and 512: error: {{.*}}model: the llc options "{{.*}}": ValueError: llc options of several VLEN: +zvl[256, 512]b
