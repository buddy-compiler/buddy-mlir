# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
# ===- typed-dma-contracts.py ----------------------------------------------
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
# Typed byte capacities, descriptor semantics, and conservative alias checks.
#
# Lit Python test with FileCheck on stderr/stdout. Checks typed DMA byte
# capacities, descriptor field semantics, and conservative alias rejection for
# illegal same-bank / overlapping views. Complements the .mlir SpmDmaCheck tests.
#
# ===---------------------------------------------------------------------------

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "examples/BOSCAMEFPGA/tools"
sys.path.insert(0, str(TOOLS))
from boscame_check import buddy_opt_path  # noqa: E402

OPT = buddy_opt_path()
ABI = 'module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"}'


def dma(src, dst, token, st, dt, channel=0):
    return f"%{token} = bosc_ame.dma.start %{src}, %{dst}, %bytes {{channel = {channel} : i32}} : ({st}, {dt}, index) -> !bosc_ame.dma_token<{channel}>"


def wait(token, channel=0):
    return f"bosc_ame.dma.wait %{token} : !bosc_ame.dma_token<{channel}>"


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)

    def check(name, text, expect=None, opt_flags=()):
        source = work / (name + ".mlir")
        source.write_text(text)
        result = subprocess.run(
            [OPT, *opt_flags, str(source), "--bosc-ame-spm-dma-check"],
            capture_output=True,
            text=True,
        )
        if expect:
            assert result.returncode != 0, name + " silently succeeded"
            assert expect in result.stderr, (name, result.stderr)
        else:
            assert result.returncode == 0, (name, result.stderr)
            lowered = subprocess.run(
                [OPT, str(source), "--lower-bosc-ame"],
                capture_output=True,
                text=True,
            )
            assert lowered.returncode == 0, (name, lowered.stderr)
            assert "llvm.call @bosc_ame_dma_start" in lowered.stdout
        return source

    for element, width in (
        ("i8", 1),
        ("i16", 2),
        ("i32", 4),
        ("i64", 8),
        ("f16", 2),
        ("bf16", 2),
        ("f32", 4),
        ("f64", 8),
    ):
        count = 64 // width
        dt = f"memref<{count}x{element}>"
        sp = f"memref<{count}x{element}, #bosc_ame.spm<0>>"
        text = f"""{ABI} {{
          func.func @typed(%src: {dt}) {{
            %bytes = arith.constant 64 : index
            %spm = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {sp}
            {dma("src", "spm", "t", dt, sp)}
            {wait("t")}
            return
          }}
        }}"""
        check(element, text)
        check(
            element + "-overflow",
            text.replace("constant 64 : index", "constant 128 : index"),
            "exceeds the src byte range of 64 bytes",
        )
        check(
            element + "-dropped",
            text.replace(wait("t"), ""),
            "token that is never waited",
        )

    # The second request reads the first request's DDR destination. The
    # reinterpret offset REPLACES 16; the subview offset ADDS to 16.
    dt = "memref<16xf32>"
    sp = "memref<16xf32, #bosc_ame.spm<0>>"
    offset = "memref<16xf32, strided<[1], offset: 16>>"
    replaced = f"""%parent = memref.subview %root[16] [16] [1] : memref<64xf32> to {offset}
        %ddr = memref.reinterpret_cast %parent to offset: [0], sizes: [16], strides: [1] : {offset} to {dt}
        %other = memref.subview %root[0] [16] [1] : memref<64xf32> to {dt}"""
    added = """%parent = memref.reinterpret_cast %root to offset: [16], sizes: [32], strides: [1] : memref<64xf32> to memref<32xf32, strided<[1], offset: 16>>
        %ddr = memref.subview %parent[16] [16] [1] : memref<32xf32, strided<[1], offset: 16>> to memref<16xf32, strided<[1], offset: 32>>
        %other = memref.subview %root[32] [16] [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: 32>>"""
    dynamic = """%scale = arith.constant 16 : index
        %offset = arith.muli %row, %scale : index
        %ddr = memref.reinterpret_cast %root to offset: [%offset], sizes: [16], strides: [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: ?>>
        %other = memref.subview %root[32] [16] [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: 32>>"""
    for name, views, ddr_type, other_type in (
        ("reinterpret", replaced, dt, dt),
        (
            "subview",
            added,
            "memref<16xf32, strided<[1], offset: 32>>",
            "memref<16xf32, strided<[1], offset: 32>>",
        ),
        (
            "dynamic-alias",
            dynamic,
            "memref<16xf32, strided<[1], offset: ?>>",
            "memref<16xf32, strided<[1], offset: 32>>",
        ),
    ):
        text = f"""{ABI} {{
          func.func @alias(%root: memref<64xf32>, %row: index) {{
            %bytes = arith.constant 64 : index
            %spm = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {sp}
            %spm2 = bosc_ame.spm.view {{bank = 0 : i32, offset = 64 : i64}} : {sp}
            {views}
            {dma("spm", "ddr", "t", sp, ddr_type)}
            {dma("other", "spm2", "u", other_type, sp, 1)}
            {wait("t")}
            {wait("u", 1)}
            return
          }}
        }}"""
        check(name, text, "overlaps the in-flight request")
    unknown_spm = f"""{ABI} {{
      func.func @bounds(%src: {dt}, %row: index) {{
        %bytes = arith.constant 64 : index
        %scale = arith.constant 16 : index
        %offset = arith.muli %row, %scale : index
        %base = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {sp}
        %spm = memref.reinterpret_cast %base to offset: [%offset], sizes: [16], strides: [1] : {sp} to memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<0>>
        {dma("src", "spm", "t", dt, "memref<16xf32, strided<[1], offset: ?>, #bosc_ame.spm<0>>")}
        {wait("t")}
        return
      }}
    }}"""
    check(
        "dynamic-spm", unknown_spm, "SPM DMA range must be statically bounded"
    )
    scalar_cast = """module {
      func.func @scalar() -> f32 {
        %zero = arith.constant 0 : index
        %base = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>
        %cast = memref.memory_space_cast %base : memref<16xf32, #bosc_ame.spm<0>> to memref<16xf32>
        %v = memref.load %cast[%zero] : memref<16xf32>
        return %v : f32
      }
    }"""
    check(
        "scalar-cast",
        scalar_cast,
        "scalar load/store, ordinary memref.copy or Linalg may not access SPM",
    )
    # SPM ownership rules apply even without an active DMA request.
    for name, body in (
        (
            "alloc-spm",
            "%spm = memref.alloc() : memref<16xf32, #bosc_ame.spm<0>>",
        ),
        (
            "copy-spm",
            "%spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>\n%normal = memref.alloc() : memref<16xf32>\nmemref.copy %spm, %normal : memref<16xf32, #bosc_ame.spm<0>> to memref<16xf32>",
        ),
        (
            "linalg-spm",
            "%spm = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64} : memref<16xf32, #bosc_ame.spm<0>>\n%one = arith.constant 1.0 : f32\nlinalg.fill ins(%one : f32) outs(%spm : memref<16xf32, #bosc_ame.spm<0>>)",
        ),
    ):
        check(
            name,
            "module { func.func @invalid() { " + body + "\nreturn } }",
            "SPM",
        )
    outside = unknown_spm.replace(
        "%offset = arith.muli %row, %scale : index",
        "%offset = arith.constant 32768 : index",
    )
    check("outside-bank", outside, "SPM DMA range must be statically bounded")
    global_text = f"""{ABI} {{
      memref.global "private" @storage : {dt} = uninitialized
      func.func @global() {{
        %bytes = arith.constant 64 : index
        %src = memref.get_global @storage : {dt}
        %spm = bosc_ame.spm.view {{bank = 0 : i32, offset = 0 : i64}} : {sp}
        {dma("src", "spm", "t", dt, sp)}
        {wait("t")}
        return
      }}
    }}"""
    check(
        "unaligned-global",
        global_text,
        "endpoint global must prove 64 B alignment",
    )
    # memref.global itself rejects non-power-of-two alignments. Disable only
    # parsing verification here so the DMA checker can exercise its own
    # multiple-of-64 proof on this malformed contract.
    check(
        "misaligned-global",
        global_text.replace(
            "= uninitialized", "= uninitialized alignment = 65"
        ),
        "endpoint global must prove 64 B alignment",
        ("--mlir-very-unsafe-disable-verifier-on-parsing",),
    )
    check(
        "aligned-global",
        global_text.replace(
            "= uninitialized", "= uninitialized alignment = 64"
        ),
    )
    # CHECK: PASS typed DMA: eight element types, byte capacities and missing waits
    print(
        "PASS typed DMA: eight element types, byte capacities and missing waits"
    )
    # CHECK: PASS footprints: reinterpret replacement, subview addition, dynamic DDR alias, bounded SPM, scalar SPM cast
    print(
        "PASS footprints: reinterpret replacement, subview addition, dynamic DDR alias, bounded SPM, scalar SPM cast"
    )


if __name__ == "__main__":
    main()
