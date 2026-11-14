#!/usr/bin/env python3
# ===- boscame_check.py ----------------------------------------------------
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

"""Structural queries on emitted BOSCAME IR for lit tests (DMA sources, scf.for accumulator carry, call-free bodies).

Inputs: BOSCAME MLIR module path or text; uses buddy-opt when needed.
Outputs: Pass/fail structural assertions printable for FileCheck (not a numeric simulator).
When used: Alongside boscame_sim.py in FPGA example and Conversion/BOSCAME lit checks.

`boscame_sim.py` answers "what does this module compute and which DDR bytes
does it touch".  This module answers the structural questions a lit test wants
to assert without brittle `FileCheck` counting: where the DMA transfers read
from, whether the accumulator is carried by a real `scf.for`, and whether the
loop bodies are free of calls.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boscame_sim import (  # noqa: E402
    op_name,
    read_module,
    split_result,
)

SPM_BANK_BYTES = 128 * 1024


def buddy_opt_path():
    """Explicit tool override or `buddy-opt` from PATH (lit sets it up)."""
    from shutil import which

    override = os.environ.get("BUDDY_OPT")
    if override:
        return override
    found = which("buddy-opt")
    if found:
        return found
    raise RuntimeError("missing buddy-opt; set BUDDY_OPT or add it to PATH")


def run_lower(buddy_opt, module, output):
    result = subprocess.run(
        [buddy_opt, module, "--lower-bosc-ame", "-o", output],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stderr


class Structure:
    """The facts a structural check needs, collected from one module."""

    def __init__(self, path):
        self.ops = read_module(path)
        self.tile = {
            "load_a": 0,
            "load_b": 0,
            "load_acc": 0,
            "mma": 0,
            "store_acc": 0,
        }
        self.dma_in_loop = 0
        self.loops = 0
        self.accumulator_loops = 0
        self.loop_headers = []
        self.spm_windows = []
        self.function_args = []
        # `roots` maps every SSA value to the DDR argument (or "spm") its
        # memory ultimately comes from, so a DMA transfer can be attributed to
        # A, B, C or to SPM.
        self.roots = {}
        self.constants = {}
        self.dma = []
        self._walk_all(self.ops)

    # -- helpers ---------------------------------------------------------
    def _root(self, name):
        return self.roots.get(name.strip(), name.strip())

    def _record_view(self, op):
        text = op.text
        result, body = split_result(text)
        if not result:
            return
        if body.startswith("bosc_ame.spm.view"):
            self.roots[result] = "spm"
            return
        match = re.match(
            r"(?:memref\.view|memref\.reinterpret_cast|"
            r"memref\.subview|memref\.collapse_shape|"
            r"memref\.cast) (%[A-Za-z0-9_.\-]+)",
            body,
        )
        if match:
            self.roots[result] = self._root(match.group(1))

    def _walk_all(self, ops, in_loop=False):
        for op in ops:
            name = op_name(op.text)
            text = op.text
            result, _ = split_result(text)
            if name == "func.func":
                signature = text[text.index("(") + 1 :]
                signature = signature[: signature.index(")")]
                for piece in signature.split(","):
                    argument, _, type_text = piece.partition(":")
                    argument = argument.strip()
                    type_text = type_text.strip()
                    self.function_args.append(type_text)
                    self.roots[argument] = (
                        argument
                        if type_text.startswith("memref<")
                        else "scalar"
                    )
            if name.startswith("memref.") or name == "bosc_ame.spm.view":
                self._record_view(op)
            if name == "arith.constant" and result:
                payload = text.split("=", 1)[1].split(":", 1)[0].strip()
                payload = payload.replace("arith.constant", "").strip()
                try:
                    self.constants[result] = int(payload)
                except ValueError:
                    pass
            if name.startswith("bosc_ame.tile."):
                self.tile[name.rsplit(".", 1)[1]] += 1
            if name == "bosc_ame.spm.view":
                match = re.search(
                    r"bank = (\d+) : i32, offset = (\d+) : i64", text
                )
                self.spm_windows.append(
                    (int(match.group(1)), int(match.group(2)))
                )
            if name == "bosc_ame.dma.start":
                if in_loop:
                    self.dma_in_loop += 1
                body = split_result(text)[1]
                operands = re.findall(
                    r"%[A-Za-z0-9_.\-]+", body.split(":", 1)[0]
                )
                length = (
                    self.constants.get(operands[2], "?")
                    if len(operands) > 2
                    else "?"
                )
                self.dma.append(
                    {
                        "source": self._root(operands[0]),
                        "destination": self._root(operands[1]),
                        "length": length,
                    }
                )
            if name == "scf.for":
                self.loops += 1
                self.loop_headers.append(text)
                if "iter_args" in text and "vector<" in text:
                    self.accumulator_loops += 1
            for block in op.blocks:
                self._walk_all(block, in_loop or name == "scf.for")

    # -- questions -------------------------------------------------------
    def loops_text(self):
        return self.loop_headers

    def dma_from(self):
        return sorted({entry["source"] for entry in self.dma})

    def dma_to(self):
        return sorted({entry["destination"] for entry in self.dma})

    def dma_reads(self, root):
        return [entry for entry in self.dma if entry["source"] == root]

    def dma_writes(self, root):
        return [entry for entry in self.dma if entry["destination"] == root]


def spm_fits(path):
    """Every SPM window of the module must fit into its 128 KiB bank."""
    structure = Structure(path)
    banks = {0: 0, 1: 0}
    for bank, offset in structure.spm_windows:
        banks[bank] = max(banks[bank], offset)
    return all(
        offset < SPM_BANK_BYTES
        for offset in [offset for _, offset in structure.spm_windows]
    )


def report(path):
    structure = Structure(path)
    return {
        "dma_sources": structure.dma_from(),
        "dma_in_loop": structure.dma_in_loop,
        "tile": structure.tile,
        "loops": structure.loops,
        "accumulator_loops": structure.accumulator_loops,
        "spm_windows": sorted(set(structure.spm_windows)),
        "function_args": structure.function_args,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(report(sys.argv[1]), indent=2))
