# REQUIRES: riscv-vector
# RUN: %PYTHON %s buddy-opt 2>&1 | FileCheck %s
#
# -lower-vir-to-vector without rvv-vlen or use-scalable sizes fixed vectors
# for the host's registers. A RISC-V host with V is lowered for as
# rvv-vlen=<its VLEN> (vlenb, read at run time), not for a 128-bit default.

import os
import shutil
import subprocess
import sys
import tempfile

BUDDY_OPT = sys.argv[1]
cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")

with tempfile.TemporaryDirectory() as tmp:
    src, exe = os.path.join(tmp, "vlenb.c"), os.path.join(tmp, "vlenb")
    with open(src, "w") as f:
        f.write(
            "#include <stdio.h>\n"
            "int main(void) {\n"
            "  unsigned long b;\n"
            '  __asm__ volatile("csrr %0, 0xc22" : "=r"(b));\n'
            '  printf("%lu\\n", b * 8);\n'
            "  return 0;\n"
            "}\n"
        )
    subprocess.run([cc, "-O2", src, "-o", exe], check=True)
    vlen = int(subprocess.run([exe], capture_output=True, text=True).stdout)

MLIR = """
func.func @scale(%a: memref<?xf32>, %n: index) {
  vir.set_vl %n : index {
    %x = vir.load %a[] : memref<?xf32> -> !vir.vec<?xf32>
    %y = arith.mulf %x, %x : !vir.vec<?xf32>
    vir.store %y, %a[] : !vir.vec<?xf32> -> memref<?xf32>
    vector.yield
  }
  return
}
"""


def lower(options):
    flag = "-lower-vir-to-vector" + (f"={options}" if options else "")
    return subprocess.run(
        [BUDDY_OPT, flag],
        input=MLIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


print(
    "host VLEN reported: "
    f"{f'RISC-V Vector host: VLEN {vlen} ---' in lower('verbose=true')}"
)
print(f"as rvv-vlen=<host VLEN>: {lower('') == lower(f'rvv-vlen={vlen}')}")
# CHECK: host VLEN reported: True
# CHECK-NEXT: as rvv-vlen=<host VLEN>: True
