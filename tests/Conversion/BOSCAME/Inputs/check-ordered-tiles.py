# ===- check-ordered-tiles.py ----------------------------------------------
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
# Check real backend allocation/order and execute its encoded matrix fields.
#
# The encoding check evaluates one traversal of the emitted loop body. It tests
# register identity and encoder preservation, not full hardware tile arithmetic.
#
# Helper imported by ordered-tile lit tests. Parses backend-emitted assembly,
# rejects unexpected vector/AME forms, checks msettile*/msettype order around
# matrix ops, and executes encoded matrix fields for one loop-body traversal.
# It proves register identity and encoder preservation, not full hardware MAC
# arithmetic. Expects examples/BOSCAMEFPGA/tools on sys.path.
#
# ===---------------------------------------------------------------------------

import re
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[4] / "examples/BOSCAMEFPGA/tools"),
)
from ame_to_word import transform as encode  # noqa: E402
from restrict_fpga_assembly import transform as restrict  # noqa: E402


def check_assembly(assembly):
    instructions = [
        re.sub(r"\s+", " ", line.split("#", 1)[0].strip())
        for line in assembly.splitlines()
        if line.split("#", 1)[0].strip()
        and not line.lstrip().startswith((".", "#"))
        and not line.split("#", 1)[0].strip().endswith(":")
    ]
    assert not re.search(r"\b(?:mmve[\w.]*|vset\w*|v[ls]\w*\.v)\b", assembly)
    matrix = []
    configurations = {
        "mwsub.h.mm": ["msettilem", "msettilen", "msettype"],
        "mlae8.m": ["msettilem", "msettilek", "msettype"],
        "mlbe8.m": ["msettilen", "msettilek", "msettype"],
        "mqma.b.mm": ["msettilem", "msettilen", "msettilek", "msettype"],
        "msce32.m": ["msettilem", "msettilen", "msettype"],
    }
    for index, instruction in enumerate(instructions):
        mnemonic = instruction.split()[0]
        assert mnemonic != "mlce32.m", "zero_acc introduced a C load"
        if mnemonic not in configurations:
            continue
        config = configurations[mnemonic]
        assert [
            line.split()[0]
            for line in instructions[index - len(config) : index]
        ] == config, (
            mnemonic,
            "configuration was separated from its operation",
        )
        assert instructions[index - len(config) - 1] == "fence rw, rw"
        assert instructions[index + 1] == "fence rw, rw"
        matrix.append((mnemonic, re.findall(r"(?:acc|tr)[0-7]", instruction)))

    assert [mnemonic for mnemonic, _ in matrix] == [
        "mwsub.h.mm",
        "mwsub.h.mm",
        "mlae8.m",
        "mlbe8.m",
        "mqma.b.mm",
        "mlbe8.m",
        "mqma.b.mm",
        "msce32.m",
        "msce32.m",
    ], matrix
    z0, z1, a, b0, r0, b1, r1, c0, c1 = [regs for _, regs in matrix]
    assert len(set(z0)) == len(set(z1)) == 1
    assert z0[0] != z1[0], "two live accumulator values share a register"
    assert r0 == [z0[0], a[0], b0[0]]
    assert r1 == [z1[0], a[0], b1[0]]
    assert a[0] != b0[0] and a[0] != b1[0]
    assert c0 == [z0[0]] and c1 == [z1[0]]


def check_encoding(assembly, mode):
    encoded = encode(assembly, mode, preserve_matrix_registers=True)
    restrict(encoded)
    registers, stores = {}, []
    inputs = iter((2, -3, 5))
    for word, mnemonic in re.findall(
        r"\.word 0x([0-9a-f]{8}) # (\w[\w.]*)", encoded
    ):
        instruction = int(word, 16)
        destination = (instruction >> 7) & 15
        lhs, rhs = (instruction >> 15) & 15, (instruction >> 20) & 15
        if mnemonic == "mwsub.h.mm":
            assert destination == lhs == rhs
            registers["acc", destination] = 0
        elif mnemonic in ("mlae8.m", "mlbe8.m"):
            registers["tr", destination] = next(inputs)
        elif mnemonic == "mqma.b.mm":
            registers["acc", destination] += (
                registers["tr", lhs] * registers["tr", rhs]
            )
        elif mnemonic == "msce32.m":
            stores.append(registers["acc", destination])
    assert stores == [-6, 10], (mode, stores)


if __name__ == "__main__":
    source = Path(sys.argv[1]).read_text()
    check_assembly(source)
    for gpr_mode in ("fixed", "direct"):
        check_encoding(source, gpr_mode)
    print(
        "PASS ordered tiles: distinct ACCs, A reuse, bundled config, preserved words"
    )
