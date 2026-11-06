# RUN: %PYTHON %s %t
# ===- private-ddr-execution.py --------------------------------------------
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
# Native private A/B/P endpoints preserve CPU/device address and lifetime.
#
# Lit Python test with a temporary directory argument. Covers native private A/B/P
# DDR endpoints: address identity between host and device views, lifetime across
# acquire/release, and that CPU-visible bytes match after the simulated kernel.
#
# ===---------------------------------------------------------------------------

import re
import struct
import subprocess
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/BOSCAMEFPGA/tools"))
from boscame_buffer_sim import Machine  # noqa: E402
from boscame_check import buddy_opt_path  # noqa: E402
from boscame_sim import SimError  # noqa: E402

SOURCE = r"""
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  func.func @private_chain(%out: memref<2x16xi32>) {
    %frame = bosc_ame.ddr.frame : memref<1536xi8>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c1344 = arith.constant 1344 : index
    %a = memref.view %frame[%c64][] : memref<1536xi8> to memref<2x64xi8>
    %bt = memref.view %frame[%c256][] : memref<1536xi8> to memref<16x64xi8>
    %p = memref.view %frame[%c1344][] : memref<1536xi8> to memref<2x16xi32>
    %b = memref.transpose %bt (d0, d1) -> (d1, d0) : memref<16x64xi8> to memref<64x16xi8, strided<[1, 64]>>
    %av = arith.constant dense<-3> : vector<64xi8>
    %bv = arith.constant dense<7> : vector<64xi8>
    %pv = arith.constant dense<17> : vector<16xi32>
    scf.for %i = %c0 to %c2 step %c1 {
      vector.store %av, %a[%i, %c0] : memref<2x64xi8>, vector<64xi8>
      vector.store %pv, %p[%i, %c0] : memref<2x16xi32>, vector<16xi32>
      %old = vector.load %p[%i, %c0] : memref<2x16xi32>, vector<16xi32>
      vector.store %old, %out[%i, %c0] : memref<2x16xi32>, vector<16xi32>
    }
    scf.for %i = %c0 to %c16 step %c1 {
      vector.store %bv, %bt[%i, %c0] : memref<16x64xi8>, vector<64xi8>
    }
    scf.for %k = %c0 to %c2 step %c1 {
      %cv = bosc_ame.tile.load_acc %p {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
      %aa = bosc_ame.tile.load_a %a {bosc_ame.direct_ddr} : memref<2x64xi8> -> vector<2x64xi8>
      %bb = bosc_ame.tile.load_b %b {bosc_ame.direct_ddr} : memref<64x16xi8, strided<[1, 64]>> -> vector<64x16xi8>
      %next = bosc_ame.tile.mma %cv, %aa, %bb : vector<2x16xi32>, vector<2x64xi8>, vector<64x16xi8> -> vector<2x16xi32>
      bosc_ame.tile.store_acc %next, %p {bosc_ame.direct_ddr} : vector<2x16xi32>, memref<2x16xi32>
    }
    scf.for %i = %c0 to %c2 step %c1 {
      %v = vector.load %p[%i, %c0] : memref<2x16xi32>, vector<16xi32>
      vector.store %v, %out[%i, %c0] : memref<2x16xi32>, vector<16xi32>
    }
    bosc_ame.ddr.release %frame : memref<1536xi8>
    return
  }
}
"""


def compiler_tool(name):
    opt = Path(buddy_opt_path()).resolve()
    sibling = opt.with_name(name)
    if sibling.is_file():
        return str(sibling)
    found = which(name)
    if found:
        return found
    llvm = ROOT / "llvm/build/bin"
    if (llvm / name).is_file():
        return str(llvm / name)
    raise FileNotFoundError(
        f"{name}: set BUDDY_OPT / PATH (lit should inject buddy_tools_dir + llvm_tools_dir)"
    )


def check_object(work, stem, lowered):
    """Exercise the shipped NR chain: policy calls used to force vlenb spills."""
    tool = compiler_tool

    def run(command, source=None):
        result = subprocess.run(
            list(map(str, command)),
            input=source,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    path = work / (stem + ".llvm.mlir")
    path.write_text(lowered)
    ll = work / (stem + ".ll")
    assembly = work / (stem + ".s")
    run([tool("buddy-translate"), "--buddy-to-llvmir", path, "-o", ll])
    run(
        [
            tool("llc"),
            ll,
            "-O2",
            "-filetype=asm",
            "-mtriple=riscv64",
            "-target-abi=lp64d",
            "-mattr=+m,+a,+f,+d,+c,+v,+xboscame",
            "-code-model=medium",
            "-o",
            assembly,
        ]
    )
    text = assembly.read_text()
    assert "vlenb" not in text, text
    assert "call\tbosc_ame_ddr_read_begin" not in text
    assert "call\tbosc_ame_ddr_write_begin" not in text
    tools = ROOT / "examples/BOSCAMEFPGA/tools"
    encoded = run(
        [
            sys.executable,
            tools / "ame_to_word.py",
            "--gpr-mode=fixed",
            "--preserve-matrix-registers",
        ],
        text,
    )
    restricted = run(
        [
            sys.executable,
            tools / "restrict_fpga_assembly.py",
            "--expand-f32-sub",
            "--expand-vector-merges",
            "--order-vector-copies",
        ],
        encoded,
    )
    nr = work / (stem + ".nr.s")
    nr.write_text(restricted)
    run(
        [
            tool("clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64imafdcv_zicsr_zifencei",
            "-mabi=lp64d",
            "-mno-relax",
            "-c",
            nr,
            "-o",
            work / (stem + ".o"),
        ]
    )


def check_mixed(work, shape, element, engine):
    typ = "memref<" + "x".join(map(str, shape)) + "x" + element + ">"
    spm = typ[:-1] + ", #bosc_ame.spm<0>>"
    source = f"""module {{
      func.func @mixed(%input: {typ}, %output: {typ}) {{
        %p = memref.alloc() alignment = 64 : {typ}
        %spm = bosc_ame.spm.view {{bank = 0 : i32, offset = 256 : i64}} : {spm}
        memref.copy %input, %p : {typ} to {typ}
        memref.copy %p, %spm : {typ} to {spm}
        memref.copy %spm, %p : {spm} to {typ}
        memref.copy %p, %output : {typ} to {typ}
        memref.dealloc %p : {typ}
        return
      }}
    }}"""
    path = work / (engine + ".source.mlir")
    path.write_text(source)
    compiled = subprocess.run(
        [
            str(buddy_opt_path()),
            str(path),
            "--linalg-to-boscame=plan-version=2 enable-spm-dma=true "
            "dma-abi=args_aligned64_nonalias target=nr-fpga",
        ],
        text=True,
        capture_output=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert "bosc_ame.ddr.frame" in compiled.stdout
    if engine == "matrix":
        assert compiled.stdout.count("bosc_ame.matrix.copy") == 2
    elif engine == "dma":
        assert compiled.stdout.count("bosc_ame.dma.start") == 6
    else:
        assert "bosc_ame.dma.start" in compiled.stdout
        assert "vector.load" in compiled.stdout
    path = work / (engine + ".emitted.mlir")
    path.write_text(compiled.stdout)
    lowered = subprocess.run(
        [str(buddy_opt_path()), str(path), "--lower-bosc-ame"],
        text=True,
        capture_output=True,
    )
    assert lowered.returncode == 0, lowered.stderr
    check_object(work, engine, lowered.stdout)
    machine = Machine(path, "mixed")
    import math

    size = math.prod(shape) * (4 if element == "i32" else 1)
    source = bytearray((i * 37 + 19) % 256 for i in range(size + 64))
    before = bytes(source)
    for _ in range(2):
        output = bytearray([0xE7]) * (size + 64)
        machine.launch({"%arg0": source, "%arg1": output})
        assert output[:size] == before[:size]
        assert output[size:] == bytes([0xE7]) * 64
        assert source == before and not machine.private_top
    assert len(machine.frame_log) == 4 and machine.frame_accesses


def check_runtime(work):
    runtime = ROOT / "runtime/boscame/runtime"
    for name in ("test_private_ddr", "test_private_ddr_e6"):
        for optimization in ("-O0", "-O2"):
            executable = work / (name + optimization)
            flags = (
                ["-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=0"]
                if name == "test_private_ddr" and optimization == "-O2"
                else []
            )
            subprocess.run(
                [
                    "cc",
                    optimization,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *flags,
                    str(runtime / (name + ".c")),
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            subprocess.run([str(executable)], check=True)
    invalid = subprocess.run(
        [
            "cc",
            "-fsyntax-only",
            "-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=2",
            str(runtime / "test_private_ddr.c"),
        ],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0
    assert "BOSC_AME_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1" in invalid.stderr
    for mode in (0, 1):
        obj = work / f"private-ddr-runtime-{mode}.o"
        subprocess.run(
            [
                compiler_tool("clang"),
                "--target=riscv64-unknown-elf",
                "-march=rv64imafdc_zicsr_zifencei",
                "-mabi=lp64d",
                "-mcmodel=medany",
                "-mno-relax",
                "-O2",
                "-ffreestanding",
                "-DBOSC_AME_DMA_ASSERT(x)=",
                f"-DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6={mode}",
                "-c",
                str(runtime / "bosc_ame_dma.c"),
                "-o",
                str(obj),
            ],
            check=True,
        )
        disassembly = subprocess.run(
            [compiler_tool("llvm-objdump"), "-d", str(obj)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (work / f"private-ddr-runtime-{mode}.disassembly").write_text(
            disassembly
        )
        atomics = re.findall(r"\t(?:lr|sc|amo\w+)\.[a-z.]+\s", disassembly)
        if mode:
            assert not atomics
            assert re.search(r"csrrci\s+\w+, mstatus, 0x8", disassembly)
            assert re.search(r"csrs\s+mstatus, \w+", disassembly)
        else:
            assert atomics
            assert not re.search(r"csrrci\s+\w+, mstatus", disassembly)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    check_runtime(work)
    check_mixed(work, (3, 16), "i32", "dma")
    check_mixed(work, (3, 20), "i32", "matrix")
    check_mixed(work, (65,), "i8", "segmented")
    path = work / "private.mlir"
    path.write_text(SOURCE)
    lower = subprocess.run(
        [str(buddy_opt_path()), str(path), "--lower-bosc-ame"],
        text=True,
        capture_output=True,
    )
    assert lower.returncode == 0, lower.stderr
    assert "bosc_ame_private_ddr_acquire" in lower.stdout
    assert "bosc_ame_private_ddr_release" in lower.stdout
    assert "llvm.inline_asm has_side_effects" in lower.stdout
    assert "llvm.call @bosc_ame_ddr_read_begin" not in lower.stdout
    assert "llvm.call @bosc_ame_ddr_write_begin" not in lower.stdout
    assert "llvm.riscv.bosc.ordered.load.acc" in lower.stdout
    assert "llvm.riscv.bosc.ordered.store.acc" in lower.stdout
    (work / "private.llvm.mlir").write_text(lower.stdout)
    check_object(work, "private", lower.stdout)
    machine = Machine(path, "private_chain")
    for poison in (0xA5, 0x5A, 0xE7):
        machine.private_poison = poison
        output = bytearray([0x39]) * (128 + 64)
        machine.launch({"%out": output})
        expected = struct.pack("<i", 17 - 2 * 64 * 3 * 7) * 32
        assert output[:128] == expected
        assert output[128:] == bytes([0x39]) * 64
        assert machine.private_top == 0
    assert len(machine.frame_log) == 6
    assert all(not frame["alive"] for frame in machine.private_frames)
    assert {kind for _, _, _, kind in machine.ame_frame_accesses} == {
        "read",
        "write",
    }
    limits = {
        frame["name"]: (frame["begin"], frame["end"])
        for frame in machine.private_frames
    }
    for name, begin, end, _ in machine.frame_accesses:
        lo, hi = limits[name]
        assert lo <= begin < end <= hi

    # A descriptor cannot authorize bytes beyond its actual backing frame.
    broken = work / "bad-view.mlir"
    broken.write_text(SOURCE.replace("1344 : index", "1472 : index"))
    try:
        Machine(broken, "private_chain").launch({"%out": bytearray(128)})
    except SimError as error:
        assert "frame view" in str(error)
    else:
        raise AssertionError("out-of-frame private view accepted")
    print("native private DDR A/B/P execution and frame guards passed")


if __name__ == "__main__":
    main()
