#!/usr/bin/env python3
# ===- check_layernorm_boundary.py -----------------------------------------
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

"""Check the fixed LayerNorm example DMA or direct-DDR external-access boundary contract.

Inputs: Emitted IR / build artifacts for the 197x768 LayerNorm example.
Outputs: Pass/fail on exact external access coverage for the planner or frozen DMA route.
When used: layernorm_197x768 example checks after planning/emission.

The planner defaults to its DDR/register route. Its frozen compatibility
route instead stages exactly three loads and one store through DMA. Execute
the actual emitted IR and require exact external access coverage for either
route; accepting a zero instruction count alone would not check the contract.
"""

import argparse
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from boscame_buffer_sim import Machine
from boscame_check import buddy_opt_path
from boscame_sim import SimError

ROWS, WIDTH = 197, 768
SYMBOL = "triton_layernorm_197x768"


def check(module):
    source = module.read_text()
    dma_count = len(re.findall(r"\bbosc_ame\.dma\.start\b", source))
    if dma_count == 4:
        route, x_reads = "DMA", 1
    elif dma_count == 0:
        route, x_reads = "direct DDR", 3
        direct_loads = re.findall(
            r"\bvector\.load\b[^\n]*\{bosc_ame\.direct_ddr\}", source
        )
        direct_stores = re.findall(
            r"\bvector\.store\b[^\n]*\{bosc_ame\.direct_ddr\}", source
        )
        if len(direct_loads) != 5 or len(direct_stores) != 1:
            raise ValueError(
                "direct-DDR route requires five marked vector loads and one store"
            )
    else:
        raise ValueError(
            "expected four boundary DMA operations or the checked direct-DDR route"
        )
    with tempfile.TemporaryDirectory(prefix="layernorm-boundary-") as temporary:
        loops = Path(temporary) / "kernel.loops.mlir"
        subprocess.run(
            [
                buddy_opt_path(),
                str(module),
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--canonicalize",
                "-o",
                str(loops),
            ],
            check=True,
        )
        machine = Machine(loops, SYMBOL)
    guard = bytes([0xC7]) * 1024
    buffers = {
        "%arg0": bytearray(struct.pack("<f", 1.0) * (ROWS * WIDTH) + guard),
        "%arg1": bytearray(struct.pack("<f", 1.0) * WIDTH + guard),
        "%arg2": bytearray(bytes(WIDTH * 4) + guard),
        "%arg3": bytearray(bytes(ROWS * WIDTH * 4) + guard),
    }
    original = {
        name: bytes(data) for name, data in buffers.items() if name != "%arg3"
    }
    for row in reversed(range(ROWS)):
        start, end = row * WIDTH * 4, (row + 1) * WIDTH * 4
        ranges = {
            "%arg0": (start, end, "read", x_reads),
            "%arg1": (0, WIDTH * 4, "read", 1),
            "%arg2": (0, WIDTH * 4, "read", 1),
            "%arg3": (start, end, "write", 1),
        }
        access_start, dma_start = len(machine.accesses), len(machine.dma_log)
        machine.launch(
            buffers, {"%arg4": ROWS, "%arg5": 1, "%arg6": 1, "%arg7": row}
        )
        accesses = machine.accesses[access_start:]
        coverage = {name: [0] * WIDTH for name in ranges}
        for name, begin, finish, kind in accesses:
            if name not in ranges:
                raise ValueError(
                    f"row {row}: unexpected external buffer {name}"
                )
            lower, upper, expected_kind, _ = ranges[name]
            if (
                kind != expected_kind
                or not lower <= begin < finish <= upper
                or begin % 4
                or finish % 4
            ):
                raise ValueError(
                    f"row {row}: invalid external access {(name, begin, finish, kind)}"
                )
            for lane in range((begin - lower) // 4, (finish - lower) // 4):
                coverage[name][lane] += 1
        for name, (_, _, _, count) in ranges.items():
            if coverage[name] != [count] * WIDTH:
                raise ValueError(
                    f"row {row}: incomplete or repeated coverage for {name}"
                )
        transfers = machine.dma_log[dma_start:]
        if route == "DMA":
            expected_accesses = [
                (name, lower, upper, kind)
                for name, (lower, upper, kind, _) in ranges.items()
            ]
            if accesses != expected_accesses or len(transfers) != 4:
                raise ValueError(f"row {row}: boundary DMA ranges differ")
            for transfer, (name, lower, upper, kind) in zip(
                transfers, expected_accesses
            ):
                (
                    source_name,
                    destination_name,
                    source_address,
                    destination_address,
                    size,
                    _,
                ) = transfer
                if kind == "read":
                    valid = (
                        source_name == name
                        and destination_name is None
                        and source_address == lower
                    )
                else:
                    valid = (
                        destination_name == name
                        and source_name is None
                        and destination_address == lower
                    )
                if not valid or size != upper - lower:
                    raise ValueError(
                        f"row {row}: unexpected DMA endpoint {transfer}"
                    )
        elif transfers:
            raise ValueError(f"row {row}: DMA in direct-DDR route")
    if any(
        bytes(buffers[name]) != snapshot for name, snapshot in original.items()
    ):
        raise ValueError("input or affine parameters were modified")
    if buffers["%arg3"][-len(guard) :] != guard:
        raise ValueError("output guard was modified")
    print(
        f"BOUNDARY CHECK PASS: {ROWS} programs, {route}, exact DDR coverage X x{x_reads} / Weight x1 / Bias x1 / Out x1, guards intact"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    args = parser.parse_args()
    try:
        check(args.module)
    except (
        OSError,
        ValueError,
        RuntimeError,
        SimError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"BOUNDARY CHECK FAIL: {error}\n")


if __name__ == "__main__":
    main()
