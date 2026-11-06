# ===- check_reference.py --------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

"""Execute the small historical Linalg demos through standard LLVM lowering.

Inputs: demo .mlir path, work directory, buddy-opt, mlir-cpu-runner (or
equivalent), and the shared runtime library path from the BOSCAMEDialect
makefile / lit setup.
Outputs: exit 0 when printed values match EXPECTED for that demo stem;
otherwise a mismatch diagnostic.
When used: BOSCAMEDialect generic-* / linalg-to-boscame-matmul reference
checks (standard Linalg→LLVM path, not the BOSCAME FPGA planner).

Rewrites the private print callback, runs the reference entry, and compares
prints. Floating demos use printF32; integer demos widen to i64 before printing.
"""

import re
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "generic-broadcast-column": [
        value for value in range(1, 5) for _ in range(4)
    ],
    "generic-broadcast-element": [7] * 16,
    "generic-broadcast-row": [1, 2, 3, 4] * 4,
    "generic-elementwise-add": [3] * 16,
    "generic-matmul": [8] * 16,
    "generic-transpose": [
        row * 4 + column + 1 for column in range(4) for row in range(4)
    ],
    "generic-unary-square-f32-qwen": [4, 9, 16, 25],
    "linalg-to-boscame-matmul": [8] * 16,
}

PASSES = [
    "--convert-linalg-to-loops",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-math-to-llvm",
    "--convert-func-to-llvm",
    "--finalize-memref-to-llvm",
    "--reconcile-unrealized-casts",
]


def main():
    source, temporary, opt, runner, runtime = sys.argv[1:]
    source = Path(source)
    work = Path(temporary)
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    floating = source.stem == "generic-unary-square-f32-qwen"
    scalar = "f32" if floating else "i32"
    callback = "print_square" if floating else "print_C"
    printer = "printF32" if floating else "printI64"
    argument_type = "f32" if floating else "i64"
    wrapper = (
        f"func.func private @{printer}({argument_type})\n"
        "  func.func private @printNewline()\n"
        f"  func.func private @{callback}("
        + ", ".join(f"%v{i}: {scalar}" for i in range(4))
        + ") {\n"
    )
    for i in range(4):
        value = f"%v{i}"
        if not floating:
            wrapper += f"    %wide{i} = arith.extsi {value} : i32 to i64\n"
            value = f"%wide{i}"
        wrapper += (
            f"    func.call @{printer}({value}) : ({argument_type}) -> ()\n"
        )
        wrapper += "    func.call @printNewline() : () -> ()\n"
    wrapper += "    return\n  }"
    text, count = re.subn(
        rf"func.func private @{callback}\([^)]*\)", wrapper, text, count=1
    )
    assert count == 1, source
    text = (
        text.rsplit("}", 1)[0]
        + """
  func.func @reference_entry() {
    %status = func.call @main() : () -> i32
    return
  }
}
"""
    )
    instrumented = work / "reference.mlir"
    instrumented.write_text(text)
    lowered = subprocess.run(
        [opt, str(instrumented), *PASSES],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "linalg." not in lowered, lowered
    path = work / "llvm.mlir"
    path.write_text(lowered)
    result = subprocess.run(
        [
            runner,
            str(path),
            "-e",
            "reference_entry",
            "-entry-point-result=void",
            f"-shared-libs={runtime}",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [float(value) for value in result.stdout.split()]
    assert actual == EXPECTED[source.stem], (source, actual, result.stderr)
    print(f"PASS {source.stem}: {len(actual)} exact reference values")


if __name__ == "__main__":
    main()
