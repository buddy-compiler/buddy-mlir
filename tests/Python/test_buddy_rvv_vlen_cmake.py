# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# tools/buddy-codegen/cmake/buddy_rvv_vlen.cmake: the VLEN that the VLEN
# check of a model's libraries (runtime/rvv/BuddyRvvVlen.c) is built for,
# from the final llc options of the model (those of BUDDY_RISCV_VLEN, or a
# model's own LLC_ATTRS) read as compile_pipeline.exact_rvv_vlen reads them,
# and 1024 with "prefill_ime"; "" for code valid on any VLEN. Options that
# disagree stop the configuration.

import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.environ["BUDDY_SRC_ROOT"]
CMAKE = shutil.which("cmake")
RVV = "-march=riscv64 -mtriple=riscv64-unknown-linux-gnu -mattr=+m,+d,+v"


def required(llc_attrs, prefill_ime=False):
    script = f"""
include("{SRC}/tools/buddy-codegen/cmake/buddy_rvv_vlen.cmake")
buddy_rvv_vlen_required(vlen
  LLC_ATTRS "{llc_attrs}"
  PREFILL_IME "{"ON" if prefill_ime else "OFF"}"
  PYTHON "{sys.executable}"
  CODEGEN_DIR "{SRC}/tools/buddy-codegen"
  CONTEXT "model")
message(STATUS "VLEN '${{vlen}}'")
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


for name, llc_attrs, ime in (
    (
        "BUDDY_RISCV_VLEN=1024",
        RVV + ",+zvl1024b -riscv-v-vector-bits-max=1024",
        False,
    ),
    # a model's LLC_ATTRS (native builds), whatever BUDDY_RISCV_VLEN is
    (
        "LLC_ATTRS of VLEN 256",
        "-mcpu=native -mattr=+m,+d,+v,+zvl256b -riscv-v-vector-bits-max=256",
        False,
    ),
    ("RVV of no exact VLEN", RVV, False),
    ("RVV, max 0", RVV + ",+zvl256b -riscv-v-vector-bits-max=0", False),
    ("x86", "-mcpu=native", False),
    ("prefill_ime", RVV, True),
    (
        "prefill_ime, VLEN 1024",
        RVV + ",+zvl1024b -riscv-v-vector-bits-max=1024",
        True,
    ),
    (
        "prefill_ime, VLEN 256",
        RVV + ",+zvl256b -riscv-v-vector-bits-max=256",
        True,
    ),
    ("zvl 256 and 512", RVV + ",+zvl256b -mattr=+zvl512b", False),
):
    print(f"{name}: {required(llc_attrs, ime)}")
# CHECK: BUDDY_RISCV_VLEN=1024: VLEN '1024'
# CHECK-NEXT: LLC_ATTRS of VLEN 256: VLEN '256'
# CHECK-NEXT: RVV of no exact VLEN: VLEN ''
# CHECK-NEXT: RVV, max 0: VLEN ''
# CHECK-NEXT: x86: VLEN ''
# CHECK-NEXT: prefill_ime: VLEN '1024'
# CHECK-NEXT: prefill_ime, VLEN 1024: VLEN '1024'
# CHECK-NEXT: prefill_ime, VLEN 256: error: {{.*}}model: "prefill_ime" needs the VLEN of the K3 A100 cores, 1024; the llc options give 256.
# CHECK-NEXT: zvl 256 and 512: error: {{.*}}model: the llc options "{{.*}}": ValueError: llc options of several VLEN: +zvl[256, 512]b
