# REQUIRES: riscv-vector
# RUN: %PYTHON %s 2>&1 | FileCheck %s
#
# runtime/rvv/BuddyRvvVlen.c, linked into the model library of a model built
# for RISC-V Vector of an exact VLEN: when the library loads, it stops a
# process whose CPU has another VLEN. A library built for each VLEN from 128
# to 65536 is loaded in a process of its own: only the one of the host's
# VLEN loads; the others exit with 1 and say why, with the SpacemiT K3 way
# to the A100 cores only when built for their VLEN, 1024.

import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.join(
    os.environ["BUDDY_SRC_ROOT"], "runtime", "rvv", "BuddyRvvVlen.c"
)
cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")

loads, stops, messages, k3_hints = [], [], [], []
with tempfile.TemporaryDirectory() as tmp:
    for vlen in (128 << i for i in range(10)):
        lib = os.path.join(tmp, f"vlen{vlen}.so")
        subprocess.run(
            [
                cc,
                "-O2",
                "-fPIC",
                "-shared",
                f"-DBUDDY_RVV_VLEN={vlen}",
                SRC,
                "-o",
                lib,
            ],
            check=True,
        )
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import ctypes; ctypes.CDLL({lib!r}); print('loaded')",
            ],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip() == "loaded":
            loads.append(vlen)
        else:
            stops.append(out.returncode)
            messages.append(
                f"built for RISC-V Vector VLEN {vlen}, but this CPU has VLEN"
                in out.stderr
            )
            k3_hints.append(("SpacemiT K3" in out.stderr) == (vlen == 1024))

print(f"loads for {len(loads)} VLEN")
print(
    f"stops for {len(stops)} VLEN, exit codes {sorted(set(stops))}, messages {all(messages)}"
)
print(f"K3 hint for VLEN 1024 only: {all(k3_hints)}")
# CHECK: loads for 1 VLEN
# CHECK-NEXT: stops for 9 VLEN, exit codes [1], messages True
# CHECK-NEXT: K3 hint for VLEN 1024 only: True
