# RUN: %PYTHON %s 2>&1 | FileCheck %s
# ===- ame-encoder-registers.py --------------------------------------------
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
#
# Execute scheduled multi-K matrix words after the FPGA encoder.
#
# Lit Python test (%PYTHON). Builds a small scheduled multi-K matmul assembly,
# runs it through ame_to_word.py then restrict_fpga_assembly.py, and FileCheck-
# matches the encoded matrix/GPR fields so register allocation and the FPGA
# encoder stay in lockstep. Depends on examples/BOSCAMEFPGA/tools on sys.path.
#
# ===---------------------------------------------------------------------------

import re
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "examples/BOSCAMEFPGA/tools"),
)
from ame_to_word import transform  # noqa: E402


def execute(encoded):
    registers = {}
    products = iter((2, 3, 5, -4))
    stores = []
    for line in encoded.splitlines():
        match = re.search(r"\.word 0x([0-9a-f]{8}) # (\w[\w.]*)", line)
        if not match:
            continue
        word, mnemonic = int(match[1], 16), match[2]
        destination = (word >> 7) & 15
        if mnemonic in ("mlae8.m", "mlbe8.m"):
            registers["tr", destination] = next(products)
        elif mnemonic == "mlce32.m":
            registers["acc", destination] = 7
        elif mnemonic == "mqma.b.mm":
            lhs, rhs = (word >> 15) & 15, (word >> 20) & 15
            registers["acc", destination] += (
                registers["tr", lhs] * registers["tr", rhs]
            )
        elif mnemonic == "msce32.m":
            stores.append(registers["acc", destination])
    return stores


def main():
    # LLVM preloads the next K pair before consuming the first pair. The
    # signed products differ, so collapsing tr2/tr3 into tr0/tr1 doubles the
    # wrong product. Keep a nonzero accumulator and use acc2 to test its field.
    scheduled = """
mlce32.m acc2, (s0), a1
mlae8.m tr0, (s1), a1
mlbe8.m tr1, (s2), a1
mlae8.m tr2, (s3), a1
mlbe8.m tr3, (s4), a1
mqma.b.mm acc2, tr0, tr1
mqma.b.mm acc2, tr2, tr3
msce32.m acc2, (s0), a1
"""
    expected = 7 + 2 * 3 + 5 * -4
    for mode in ("fixed", "direct"):
        encoded = transform(
            scheduled, gpr_mode=mode, preserve_matrix_registers=True
        )
        assert execute(encoded) == [expected]
        if mode == "fixed":
            assert "sd\ta0," in encoded, "fixed GPR preservation disappeared"

    for source, operand in (
        (scheduled.replace("acc2", "acc0"), "tr0"),
        (
            "mlce32.m acc0, (s0), a1\n"
            "mlce32.m acc2, (s1), a1\n"
            "msce32.m acc0, (s0), a1\n",
            "acc0",
        ),
    ):
        try:
            transform(source)
        except ValueError as error:
            assert f"fixed matrix mapping clobbers {operand}" in str(error)
        else:
            raise AssertionError("unsafe fixed matrix schedule was accepted")

    serialized = """
mlce32.m acc7, (s0), a1
mlae8.m tr2, (s1), a1
mlbe8.m tr3, (s2), a1
mqma.b.mm acc7, tr2, tr3
mlae8.m tr2, (s3), a1
mlbe8.m tr3, (s4), a1
mqma.b.mm acc7, tr2, tr3
msce32.m acc7, (s0), a1
"""
    assert execute(transform(serialized)) == [expected]
    # CHECK: PASS matrix encoder: scheduled multi-K products and accumulator fields
    print(
        "PASS matrix encoder: scheduled multi-K products and accumulator fields"
    )


if __name__ == "__main__":
    main()
