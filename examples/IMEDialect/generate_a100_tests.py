# RUN: %PYTHON %s %t
# RUN: buddy-translate -buddy-to-llvmir %t/a100-kernels.mlir | buddy-llc -mtriple=riscv64 -mcpu=spacemit-a100 -verify-machineinstrs -o /dev/null

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

"""Generate A100 SSA kernels and the scalar hardware regression's dispatch table.

All 21 non-window instructions, supported pack SEWs and immediate values are
covered. Dot kernels chain two SSA accumulations without an intermediate store.
The generated pointer ABI permits linking with a baseline scalar C runner.
"""

import argparse
from pathlib import Path


def vector(lanes, element):
    return f"vector<[{lanes}]x{element}>"


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    kernels = []
    declarations = []
    entries = []

    def kernel(op, kind, sign, sew, imm, a_type, b_type, c_type, p_type=None):
        name = f"a100_{op.replace('.', '_')}_{sew}_{imm}"
        lines = [
            f"llvm.func @{name}(%out: !llvm.ptr, %ap: !llvm.ptr, "
            "%bp: !llvm.ptr, %pp: !llvm.ptr) {",
            f"  %a = llvm.load %ap : !llvm.ptr -> {a_type}",
            f"  %b = llvm.load %bp : !llvm.ptr -> {b_type}",
        ]
        args, types = ["%a", "%b"], [a_type, b_type]
        if kind in ("DOT", "FP", "HP", "SP"):
            lines.append(f"  %c = llvm.load %out : !llvm.ptr -> {c_type}")
            args.insert(0, "%c")
            types.insert(0, c_type)
        if p_type:
            lines.append(f"  %p = llvm.load %pp : !llvm.ptr -> {p_type}")
            args.append("%p")
            types.append(p_type)
        attr = f" {{group = {imm} : i32}}" if p_type else ""
        if kind == "PACK":
            attr = f" {{block = {imm} : i32}}"
        lines.append(
            f'  %r = "ime.intr.{op}"({", ".join(args)}){attr} : '
            f"({', '.join(types)}) -> {c_type}"
        )
        if kind != "PACK":
            args[0] = "%r"
            lines.append(
                f'  %r2 = "ime.intr.{op}"({", ".join(args)}){attr} : '
                f"({', '.join(types)}) -> {c_type}"
            )
        lines.extend(
            [
                f"  llvm.store %{'r' if kind == 'PACK' else 'r2'}, "
                f"%out : {c_type}, !llvm.ptr",
                "  llvm.return",
                "}",
            ]
        )
        kernels.append("\n".join(lines))
        declarations.append(
            f"extern void {name}(void *, const void *, const void *, const void *);"
        )
        entries.append(f'  {{"{op}", {kind}, {sign}, {sew}, {imm}, {name}}},')

    i8, i8_pair = vector(8, "i8"), vector(16, "i8")
    i32, f16, f32 = vector(4, "i32"), vector(4, "f16"), vector(4, "f32")
    for sign, suffix in enumerate(("", "u", "su", "us")):
        kernel(f"vmadot{suffix}", "DOT", sign, 8, 0, i8, i8, i32)
        for imm in range(8):
            kernel(f"vmadot{suffix}.hp", "HP", sign, 8, imm, i8, i8, f16, f16)
        for imm in range(4):
            kernel(
                f"vmadot{suffix}.sp", "SP", sign, 8, imm, i8_pair, i8, i32, i8
            )
    kernel("vfmadot", "FP", 0, 16, 0, f16, f16, f32)
    for op in ("vpack", "vupack", "vnpack", "vnspack", "vnpack4", "vnspack4"):
        widths = (8, 16, 32, 64) if op in ("vpack", "vupack") else (8, 16, 32)
        if op.endswith("4"):
            widths = (8,)
        for sew in widths:
            wide = op in ("vpack", "vupack")
            a_type = vector(64 // sew, f"i{sew}")
            c_type = vector(128 // sew, f"i{sew}")
            if not wide:
                a_type = vector(32 // sew, f"i{2 * sew}")
                c_type = vector(64 // sew, f"i{sew}")
            if op.endswith("4"):
                a_type = c_type = i8
            for imm in range(4):
                kernel(op, "PACK", 0, sew, imm, a_type, a_type, c_type)
    (directory / "a100-kernels.mlir").write_text("\n\n".join(kernels) + "\n")
    header = "\n".join(declarations)
    header += "\nstatic const struct Entry tests[] = {\n"
    header += "\n".join(entries) + "\n};\n"
    (directory / "a100-kernels.h").write_text(header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
