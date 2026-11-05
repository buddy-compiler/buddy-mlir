# RUN: %PYTHON %s
# ===- ame-zero-acc.py -----------------------------------------------------
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
# Check self-clear encodings and exact tile semantics, including production sizes.
#
# The 1x1/2x3/8x8 Acc0..7 board evidence is in reference/examples/clear_acc.
# Larger rectangles here are simulation evidence, not additional board results.
#
# Lit Python test (%PYTHON). Validates mwsub.mm / mwsub.h.mm self-clear encodings
# for Acc0..7 and simulates exact tile clear semantics for production tile sizes.
# Board-backed 1x1/2x3/8x8 evidence lives under reference/examples/clear_acc;
# larger rectangles here are simulator-only. Uses ame_to_word, boscame_buffer_sim,
# nr_isa, and restrict_fpga_assembly from examples/BOSCAMEFPGA/tools.
#
# ===---------------------------------------------------------------------------

import struct
import sys
import tempfile
from functools import partial
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/BOSCAMEFPGA/tools")
)
from ame_to_word import convert_line  # noqa: E402
from ame_to_word import transform as encode
from boscame_buffer_sim import Machine, SimError  # noqa: E402
from nr_isa import is_ame_acc_clear, validate_ame_word  # noqa: E402
from restrict_fpga_assembly import transform as restrict  # noqa: E402


def rejects(action, message):
    try:
        action()
    except (ValueError, SimError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("accepted invalid clear")


def encodings():
    for mnemonic, base in (
        ("mwsub.mm", 0x2C084077),
        ("mwsub.h.mm", 0x2C081077),
    ):
        for accumulator in range(8):
            assembly = f"{mnemonic} acc{accumulator}, acc{accumulator}, acc{accumulator}"
            expected = (
                base
                | (accumulator << 20)
                | (accumulator << 15)
                | (accumulator << 7)
            )
            assert is_ame_acc_clear(expected)
            validate_ame_word(expected)
            for mode in ("fixed", "direct"):
                encoded = encode(assembly, mode, preserve_matrix_registers=True)
                assert f"0x{expected:08x}" in encoded
                fenced = restrict(encoded)
                instructions = [
                    line.strip() for line in fenced.splitlines() if line.strip()
                ]
                index = next(
                    i for i, line in enumerate(instructions) if ".word" in line
                )
                assert (
                    instructions[index - 1]
                    == instructions[index + 1]
                    == "fence\trw, rw"
                )
            # Fixed legacy mapping writes acc0 but must preserve all three equal fields.
            assert f"0x{base:08x}" in encode(assembly)
            for destination in range(8):
                for lhs in range(8):
                    for rhs in range(8):
                        if destination == lhs == rhs:
                            continue
                        word = (
                            base
                            | (destination << 7)
                            | (lhs << 15)
                            | (rhs << 20)
                        )
                        rejects(partial(validate_ame_word, word), "unverified")
                        rejects(
                            partial(
                                convert_line,
                                f"{mnemonic} acc{destination}, acc{lhs}, acc{rhs}",
                            ),
                            "dst == src1 == src2",
                        )
        for bit in (24, 25, 11, 10, 19):
            word = base ^ (1 << bit)
            rejects(partial(validate_ame_word, word), "unverified")
    rejects(lambda: encode("mwsub.w.mm acc0, acc0, acc0"), "unsupported NR")
    rejects(lambda: encode("mwsub.mm tr0, tr0, tr0"), "expected acc")
    # A self-clear defines the mapped register without a preceding load.
    encode("mwsub.mm acc7, acc7, acc7\nmsce32.m acc7, (a0), a1")
    rejects(
        lambda: encode(
            "mwsub.mm acc0, acc0, acc0\nmwsub.mm acc7, acc7, acc7\nmsce32.m acc0, (a0), a1"
        ),
        "fixed matrix mapping clobbers acc0",
    )


def module(rows, columns):
    pitch, inner, snapshots = 80, 64, 6
    tile = f"vector<{rows}x{columns}xi32>"
    memory = (
        f"memref<{rows}x{columns}xi32, strided<[{pitch}, 1]>, #bosc_ame.spm<0>>"
    )
    lines = [
        'module attributes {bosc_ame.target = "nr-fpga"} {',
        f"  func.func @clear(%out: memref<{snapshots * 64 * pitch}xi32>) {{",
        "    %poison = arith.constant 1513931685 : i32",
        "    %left = arith.constant -128 : i8",
        "    %right = arith.constant 127 : i8",
        f"    %c = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : memref<64x{pitch}xi32, #bosc_ame.spm<0>>",
        f"    %cv = memref.reinterpret_cast %c to offset: [0], sizes: [{rows}, {columns}], strides: [{pitch}, 1] : memref<64x{pitch}xi32, #bosc_ame.spm<0>> to {memory}",
        f"    %a = bosc_ame.spm.view {{bank = 1 : i32, offset = 0 : i64}} : memref<{rows}x{inner}xi8, #bosc_ame.spm<1>>",
        f"    %rawb = bosc_ame.spm.view {{bank = 1 : i32, offset = 4096 : i64}} : memref<{columns}x{inner}xi8, #bosc_ame.spm<1>>",
        f"    %b = memref.reinterpret_cast %rawb to offset: [0], sizes: [{inner}, {columns}], strides: [1, {inner}] : memref<{columns}x{inner}xi8, #bosc_ame.spm<1>> to memref<{inner}x{columns}xi8, strided<[1, {inner}]>, #bosc_ame.spm<1>>",
    ]

    def store(value, index):
        offset = index * 64 * pitch
        destination = f"memref<{rows}x{columns}xi32, strided<[{pitch}, 1], offset: {offset}>>"
        lines.extend(
            [
                f"    %d{index} = memref.reinterpret_cast %out to offset: [{offset}], sizes: [{rows}, {columns}], strides: [{pitch}, 1] : memref<{snapshots * 64 * pitch}xi32> to {destination}",
                f"    bosc_ame.tile.store_acc %{value}, %d{index} {{bosc_ame.direct_ddr}} : {tile}, {destination}",
            ]
        )

    def zero(value):
        lines.append(f"    %{value} = bosc_ame.tile.zero_acc : {tile}")

    def mma(value, accumulator):
        lines.extend(
            [
                f"    %a{value} = bosc_ame.tile.load_a %a : memref<{rows}x{inner}xi8, #bosc_ame.spm<1>> -> vector<{rows}x{inner}xi8>",
                f"    %b{value} = bosc_ame.tile.load_b %b : memref<{inner}x{columns}xi8, strided<[1, {inner}]>, #bosc_ame.spm<1>> -> vector<{inner}x{columns}xi8>",
                f"    %{value} = bosc_ame.tile.mma %{accumulator}, %a{value}, %b{value} : {tile}, vector<{rows}x{inner}xi8>, vector<{inner}x{columns}xi8> -> {tile}",
            ]
        )

    lines.append(f"    %seed = bosc_ame.tile.load_acc %cv : {memory} -> {tile}")
    store("seed", 0)
    zero("z1")
    store("z1", 1)
    zero("z2")
    store("z2", 2)
    zero("z3")
    mma("product", "z3")
    store("product", 3)
    zero("z4")
    store("z4", 4)
    zero("z5")
    mma("first", "z5")
    mma("second", "first")
    store("second", 5)
    lines.extend(["    return", "  }", "}"])
    return "\n".join(lines) + "\n"


def semantics():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clear.mlir"
        for rows, columns in (
            (1, 1),
            (2, 3),
            (8, 8),
            (4, 40),
            (32, 64),
            (64, 64),
        ):
            path.write_text(module(rows, columns))
            machine = Machine(path, "clear")
            machine.spm[0][: 64 * 80 * 4] = struct.pack("<i", 1513931685) * (
                64 * 80
            )
            machine.spm[1][: rows * 64] = b"\x80" * (rows * 64)
            machine.spm[1][4096 : 4096 + columns * 64] = b"\x7f" * (
                columns * 64
            )
            guard = b"\x69" * 64
            poison = b"\x96" * (6 * 64 * 80 * 4)
            output = bytearray(poison + guard)
            machine.launch({"%out": output})
            expected = bytearray(poison + guard)
            for snapshot, value in enumerate(
                (1513931685, 0, 0, -128 * 127 * 64, 0, 2 * -128 * 127 * 64)
            ):
                for row in range(rows):
                    for column in range(columns):
                        struct.pack_into(
                            "<i",
                            expected,
                            ((snapshot * 64 + row) * 80 + column) * 4,
                            value,
                        )
            assert output == expected, (rows, columns)
            assert all(
                action == "write" for _, _, _, action in machine.accesses
            )
            assert not machine.dma_log
            # The register clears cannot touch the poisoned C window or A/B.
            assert machine.spm[0][: 64 * 80 * 4] == struct.pack(
                "<i", 1513931685
            ) * (64 * 80)
            assert machine.spm[1][: rows * 64] == b"\x80" * (rows * 64)
            assert machine.spm[1][4096 : 4096 + columns * 64] == b"\x7f" * (
                columns * 64
            )
        invalid = module(1, 1).replace(
            "tile.zero_acc : vector<1x1xi32>", "tile.zero_acc : vector<1x1xi16>"
        )
        path.write_text(invalid)
        rejects(
            lambda: Machine(path, "clear").launch(
                {"%out": bytearray(6 * 64 * 80 * 4)}
            ),
            "i32 result",
        )
    print(
        "PASS zero_acc: restricted Acc0..7 words, exact guards and polluted/repeated/post-MMA/production tiles"
    )


if __name__ == "__main__":
    encodings()
    semantics()
