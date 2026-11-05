#!/usr/bin/env python3
# ===- check_experimental_board.py -----------------------------------------
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

"""Compare post-execution DDR dumps against independent experimental BOSCAME oracles.

Inputs: Board DDR dump after the app terminal verdict, matching ELF for symbol addresses, oracle config.
Outputs: Per-symbol pass/fail against oracle bytes; does not validate UVHS image upload alone.
When used: After fpga_run completes an experimental kernel; not a substitute for loader readback.

The dump must be read after the application's terminal verdict. The runner's
loader readback only verifies image upload and cannot validate these results.
Symbol addresses come from the exact ELF used to produce the uploaded image.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Symbol:
    address: int
    size: int


def load_symbols(elf, nm):
    result = subprocess.run(
        [
            str(nm),
            "--format=posix",
            "--print-size",
            "--defined-only",
            "--extern-only",
            "--radix=x",
            str(elf),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    symbols = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        name, _, address, size = fields
        symbol = Symbol(int(address, 16), int(size, 16))
        if name in symbols:
            raise ValueError(f"duplicate ELF symbol: {name}")
        symbols[name] = symbol
    return symbols


class DDR:
    def __init__(self, data, symbols, base):
        self.data, self.symbols, self.base = data, symbols, base
        self.ranges = []

    def read(self, name, dtype, count, offset=0):
        if name not in self.symbols:
            raise ValueError(f"missing ELF symbol: {name}")
        symbol = self.symbols[name]
        length = np.dtype(dtype).itemsize * count
        if offset < 0 or offset + length > symbol.size:
            raise ValueError(f"ELF symbol {name} is too small")
        start = symbol.address - self.base + offset
        if start < 0 or start + length > len(self.data):
            raise ValueError(f"DDR dump does not cover {name}")
        if offset == 0:
            for previous, begin, end in self.ranges:
                if (
                    symbol.address < end
                    and symbol.address + symbol.size > begin
                ):
                    raise ValueError(
                        f"overlapping ELF symbols: {previous}, {name}"
                    )
            self.ranges.append(
                (name, symbol.address, symbol.address + symbol.size)
            )
        return np.frombuffer(self.data, dtype=dtype, count=count, offset=start)

    def scalar(self, name):
        return int(self.read(name, "<u4", 1)[0])

    def guarded(self, name, dtype, count, sentinel, failures):
        words, before_bytes = 256, 1024
        before = self.read(name, "<u4", words)
        values = self.read(name, dtype, count, before_bytes)
        after = self.read(name, "<u4", words, before_bytes + values.nbytes)
        for side, guard in (("before", before), ("after", after)):
            errors = int(np.count_nonzero(guard != sentinel))
            if errors:
                failures.append(
                    f"{name}.{side}: {errors} sentinel words changed"
                )
        return values


def compare_bits(name, actual, expected, failures):
    actual_bytes = actual.view(np.uint8).reshape(-1)
    expected_bytes = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
    errors = int(np.count_nonzero(actual_bytes != expected_bytes))
    if errors:
        failures.append(f"{name}: {errors} input bytes changed")


def layernorm_inputs(round_index):
    rows = np.arange(197, dtype=np.int64)[:, None]
    columns = np.arange(768, dtype=np.int64)
    x = (((rows * 37 + columns * 13 + round_index * 71) % 511) - 255).astype(
        np.float32
    ) * np.float32(0.015625) + (rows % 7).astype(np.float32) * np.float32(0.25)
    weight = (((columns * 11 + round_index * 19) % 65) - 24).astype(
        np.float32
    ) * np.float32(0.0625)
    bias = (((columns * 7 + round_index * 31) % 49) - 24).astype(
        np.float32
    ) * np.float32(0.03125)
    return x, weight, bias


def layernorm_oracle(x, weight, bias):
    # cumsum preserves the scalar f32 reduction order of the original path.
    padded = np.pad(x, ((0, 0), (0, 256)))
    mean = np.divide(
        np.cumsum(padded, axis=1, dtype=np.float32)[:, -1:],
        np.float32(768),
        dtype=np.float32,
    )
    centered = np.subtract(padded, mean, dtype=np.float32)
    centered[:, 768:] = np.float32(0)
    squares = np.multiply(centered, centered, dtype=np.float32)
    variance = np.divide(
        np.cumsum(squares, axis=1, dtype=np.float32)[:, -1:],
        np.float32(768),
        dtype=np.float32,
    )
    shifted = np.add(variance, np.float32(1.0e-12), dtype=np.float32)
    # The freestanding board rsqrtf rounds sqrtf and division separately.
    inverse = np.divide(
        np.float32(1), np.sqrt(shifted, dtype=np.float32), dtype=np.float32
    )
    scaled = np.multiply(centered[:, :768], inverse, dtype=np.float32)
    return np.add(
        np.multiply(scaled, weight, dtype=np.float32), bias, dtype=np.float32
    )


def check_layernorm(ddr, failures):
    rows, width, count = 197, 768, 197 * 768
    actual = {}
    for name, size in (
        ("x", count),
        ("weight", width),
        ("bias", width),
        ("out", count),
    ):
        actual[name] = ddr.guarded(
            "layernorm_" + name, "<f4", size, 0x4F400001, failures
        )
    x, weight, bias = layernorm_inputs(1)
    compare_bits("layernorm_x", actual["x"], x, failures)
    compare_bits("layernorm_weight", actual["weight"], weight, failures)
    compare_bits("layernorm_bias", actual["bias"], bias, failures)
    wanted = layernorm_oracle(x, weight, bias)
    output = actual["out"].reshape(rows, width)
    delta = np.abs(np.subtract(output, wanted, dtype=np.float32))
    tolerance = np.multiply(
        np.float32(2.0e-5),
        np.add(np.float32(1), np.abs(wanted), dtype=np.float32),
        dtype=np.float32,
    )
    errors = int(np.count_nonzero(~np.isfinite(output) | (delta > tolerance)))
    if errors:
        failures.append(f"layernorm_out: {errors} values exceed f32 tolerance")
    return {
        "shape": [rows, width],
        "final_round_values_checked": count,
        "output_mismatches": errors,
        "exact_final_round_values": int(
            np.count_nonzero(output.view(np.uint32) == wanted.view(np.uint32))
        ),
        "max_absolute_error": (
            float(np.max(delta)) if np.all(np.isfinite(delta)) else None
        ),
        "tolerance": "2e-5 * (1 + abs(expected))",
        "oracle": "sequential f32, 1024 logical lanes with padded centered lanes zeroed before variance, f32 sqrt then division",
    }


def gemm_inputs(round_index):
    i = np.arange(128, dtype=np.int64)[:, None]
    j = np.arange(1024, dtype=np.int64)[:, None]
    q = np.arange(128, dtype=np.int64)
    a = (((i * 13 + q * 7 + round_index * 71) % 256) - 128).astype(np.int8)
    b = (((j * 11 + q * 5 + round_index * 37) % 256) - 128).astype(np.int8)
    c = (0x7FFFFF00 + ((i * 3 + j.T + round_index * 97) % 512)).astype(
        np.uint32
    )
    return a, b, c


def check_gemm(ddr, failures):
    actual = {}
    for name, dtype, count in (
        ("a", "i1", 128 * 128),
        ("b", "i1", 1024 * 128),
        ("c", "<u4", 128 * 1024),
    ):
        actual[name] = ddr.guarded(
            name + "_storage", dtype, count, 0x6E5A1234, failures
        )
    a, b, initial_c = gemm_inputs(1)
    compare_bits("a_storage", actual["a"], a, failures)
    compare_bits("b_storage", actual["b"], b, failures)
    # B's physical rows are logical columns of the multiply. Compute in i64
    # and wrap explicitly, independent of both AME and the C board oracle.
    product = a.astype(np.int64) @ b.astype(np.int64).T
    wanted = ((product + initial_c.astype(np.int64)) & 0xFFFFFFFF).astype(
        np.uint32
    )
    output = actual["c"].reshape(128, 1024)
    errors = int(np.count_nonzero(output != wanted))
    if errors:
        failures.append(f"c_storage: {errors} i32 values differ")
    return {
        "shape": [128, 1024, 128],
        "final_round_values_checked": 128 * 1024,
        "output_mismatches": errors,
        "exact_final_round_values": 128 * 1024 - errors,
        "oracle": "signed i8 multiply, independent i64 accumulation, explicit i32 wrap",
    }


def check_dump(kind, data, symbols, base=0x80000000):
    failures = []
    report = {"kind": kind, "passed": False, "failures": failures}
    ddr = DDR(data, symbols, base)
    expected = (
        {
            "layernorm_round": 1,
            "layernorm_checked": 2 * 197 * 768,
            "layernorm_errors": 0,
            "layernorm_guard_errors": 0,
        }
        if kind == "layernorm"
        else {
            "e6_diag_round": 1,
            "e6_diag_checked_rows": 256,
            "e6_diag_errors": 0,
            "e6_diag_guard_errors": 0,
        }
    )
    try:
        diagnostics = {name: ddr.scalar(name) for name in expected}
        report["board_diagnostics"] = diagnostics
        for name, wanted in expected.items():
            if diagnostics[name] != wanted:
                failures.append(
                    f"{name}: expected {wanted}, got {diagnostics[name]}"
                )
        check = check_layernorm if kind == "layernorm" else check_gemm
        report.update(check(ddr, failures))
    except ValueError as error:
        failures.append(str(error))
    report["passed"] = not failures
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("layernorm", "gemm"))
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument(
        "--nm",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "llvm/build/bin/llvm-nm",
    )
    parser.add_argument(
        "--ddr-base", type=lambda value: int(value, 0), default=0x80000000
    )
    parser.add_argument(
        "--report", type=Path, help="write the JSON validation report"
    )
    args = parser.parse_args()
    try:
        data = args.dump.read_bytes()
        symbols = load_symbols(args.elf, args.nm)
        report = check_dump(args.kind, data, symbols, args.ddr_base)
        report["elf_sha256"] = hashlib.sha256(args.elf.read_bytes()).hexdigest()
        report["dump_sha256"] = hashlib.sha256(data).hexdigest()
        report["dump_bytes"] = len(data)
        report["ddr_base"] = hex(args.ddr_base)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report = {"kind": args.kind, "passed": False, "failures": [str(error)]}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
