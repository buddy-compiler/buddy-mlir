"""Check actual emitted tiles and machine code against the W8A8 schedule ABI."""
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

opt, translate, llc, directory = sys.argv[1:]
out = Path(directory)
out.mkdir(parents=True, exist_ok=True)


def run(command, source=None):
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{command}\n{result.stderr}")
    return result.stdout


lower = "--lower-linalg-to-boscame=target=qwen3-fpga triton-w8a8-fast-path=true"
export = ["--lower-bosc-ame", "--convert-linalg-to-loops", "--lower-affine",
          "--convert-scf-to-cf", "--expand-strided-metadata", "--lower-affine",
          "--convert-cf-to-llvm", "--convert-arith-to-llvm", "--convert-math-to-llvm",
          "--convert-func-to-llvm", "--finalize-memref-to-llvm", "--reconcile-unrealized-casts"]
for m, n, k in [(32, 64, 192), (32, 128, 128), (1, 64, 128),
                (1, 128, 192), (1, 192, 1024)]:
    name = f"wide_{m}_{n}_{k}"
    source = f"""module {{
func.func @{name}(%A: memref<{m}x{k}xi8>, %B: memref<{k}x{n}xi8, strided<[1, {k}]>>) -> f32 {{
  %C = memref.alloc() : memref<{m}x{n}xf32>
  %z = arith.constant 0.0 : f32
  %i = arith.constant 0 : index
  linalg.fill ins(%z : f32) outs(%C : memref<{m}x{n}xf32>)
  linalg.matmul ins(%A, %B : memref<{m}x{k}xi8>, memref<{k}x{n}xi8, strided<[1, {k}]>>)
                outs(%C : memref<{m}x{n}xf32>)
  %r = memref.load %C[%i, %i] : memref<{m}x{n}xf32>
  memref.dealloc %C : memref<{m}x{n}xf32>
  return %r : f32
}}
}}"""
    (out / f"{name}.mlir").write_text(source)
    lowered = run([opt, lower, "--canonicalize"], source)
    (out / f"{name}.lowered.mlir").write_text(lowered)
    constants = dict(re.findall(r"(%\w+) = arith.constant (\d+) : index", lowered))
    buf = re.search(r"(%\w+) = memref.alloc", lowered)[1]
    tiles = []
    for row, col in re.findall(r"memref.subview " + re.escape(buf) + r"\[([^,]+), ([^\]]+)\]", lowered):
        tiles.append((int(constants.get(row, row)), int(constants.get(col, col))))
    expected = {(row, col) for row in range(0, m, 16) for col in range(0, n, 16)}
    assert set(tiles) == expected, (name, tiles, expected)
    assert lowered.count("bosc_ame.mlce32.m") == len(expected), name
    assert lowered.count("bosc_ame.msce32.m") == len(expected), name
    # Check the actual first seed, not a subsequent batch matched by FileCheck.
    first_seed = lowered.index("bosc_ame.mlce32.m")
    assert re.search(r"bosc_ame.msettype %c65602_i64", lowered[:first_seed]), name
    assert lowered.index("bosc_ame.msettype %c65552_i64") > first_seed, name
    assert lowered.rindex("llvm.fence") > lowered.rindex("bosc_ame.msce32.m"), name
    llvm_mlir = run([opt, *export], lowered)
    ir = run([translate, "--buddy-to-llvmir"], llvm_mlir)
    (out / f"{name}.ll").write_text(ir)
    for new_pm, level in product([False, True], [0, 2, 3]):
        artifact = f"{name}-{'newpm' if new_pm else 'legacy'}-O{level}"
        flags = [llc, "-mtriple=riscv64", "-mattr=+m,+f,+d,+v,+xboscame",
                 f"-enable-new-pm={str(new_pm).lower()}",
                 f"-O{level}", "-verify-machineinstrs", "-o", "-"]
        asm = run(flags, ir)
        (out / f"{artifact}.s").write_text(asm)
        assert set(re.findall(r"mlae8\.m\s+(tr\d)", asm)) == ({"tr0", "tr2"} if m == 32 else {"tr0"}), name
        assert set(re.findall(r"mlbe8\.m\s+(tr\d)", asm)) == {"tr4", "tr5", "tr6", "tr7"}, name
        assert "mmve" not in asm, name
        for acc, a, b in re.findall(r"mqma\.b\.mm\s+acc(\d), tr(\d), tr(\d)", asm):
            lane = int(acc)
            assert int(b) == 4 + lane % 4, (name, acc, a, b)
            assert int(a) == (2 if m == 32 and lane >= 4 else 0), (name, acc, a, b)
        # The matrix allocator leaves only physical registers and no matrix
        # stack objects before the general allocator starts.
        mir = run(flags[:-2] + ["-stop-after=riscv-fpga-matrix-slots", "-o", "-"], ir)
        (out / f"{artifact}.mir").write_text(mir)
        body = mir.split("body:", 1)[1]
        assert not re.search(r"%\d+:(?:tile|acc)reg\b", body), name
        assert not re.search(r"BOSC_AME_FPGA_MQMA_B_MM[^\n]*%\d+", body), name
        assert not re.search(r"(?:BOSC_AME|FPGA_LOAD)[^\n]*%stack", body), name
        assert "PseudoFPGA_LOAD" not in body, name
print("FPGA geometry, configuration and slots: O0/O2/O3 passed with both pass managers")
