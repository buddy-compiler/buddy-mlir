# RUN: %PYTHON %s | FileCheck %s

from unittest.mock import patch

from Inputs.runtime_test_utils import llc_target_args

for machine in ("riscv64", "x86_64", "aarch64"):
    with patch("platform.machine", return_value=machine):
        print(f"{machine}: {llc_target_args()}")

# CHECK: riscv64: ['-mtriple=riscv64-unknown-linux-gnu', '-target-abi=lp64d', '-mattr=+f,+d']
# CHECK-NEXT: x86_64: []
# CHECK-NEXT: aarch64: []
