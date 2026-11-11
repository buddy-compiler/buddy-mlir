#!/usr/bin/env python3
# ===- run_xtame_bench.py ------------------------------------------------------
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
# ===-----------------------------------------------------------------------===#

"""Instruction-count benchmark for the AME matrix-multiply family.

Every file in ``ops/`` holds one kernel on ``memref`` operands: ``linalg.matmul``,
``linalg.batch_matmul`` or ``linalg.quantized_matmul``. For each case the driver
builds three programs from that file plus a generated ``@main`` harness:

  * ``xtame``    - lowered by ``-lower-linalg-to-xtame``;
  * ``scalar``   - the same kernel lowered to loops, i.e. the product without
                   the AME extension;
  * ``overhead`` - the same harness without calling the kernel, so the fill,
                   zero and dump cost that both programs pay can be subtracted.

The harness fills the inputs with index-derived values (the batch index takes
part in the value, so a kernel that reads the wrong batch slice is caught),
runs the kernel ``--reps`` times inside a region timed with ``rdcycle``, and
writes C, the cycle count and a keep-alive buffer to stdout. The driver compares
C with a Python product and then counts the instructions each program executes
under QEMU.

Only instructions are counted: QEMU has no timing model for the XuanTie core, so
neither the ``rdcycle`` value nor the wall clock of the QEMU process says
anything about hardware. Without ``--qemu`` the same binaries run natively and
the cycles per kernel become meaningful.

A case the pass cannot lower is reported as ``no-xtame`` and skipped, the same
way ``benchmarks/LinalgBench`` reports ``no-vir``.

Usage:
  QEMU_RISCV64=/path/to/qemu-riscv64 python3 \
      benchmarks/XtameBench/run_xtame_bench.py
  python3 benchmarks/XtameBench/run_xtame_bench.py matmul_20x20x20
  python3 benchmarks/XtameBench/run_xtame_bench.py --csv results.csv
"""

import argparse
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")
BUDDY_OPT = os.path.join(ROOT, "build/bin/buddy-opt")
BUDDY_TRANSLATE = os.path.join(ROOT, "build/bin/buddy-translate")
BUDDY_LLC = os.path.join(ROOT, "build/bin/buddy-llc")
LLVM_MC = os.path.join(ROOT, "llvm/build/bin/llvm-mc")

# Zero points the harness passes to linalg.quantized_matmul. The reference uses
# the same values, so a case fails if the kernel ignores them.
A_ZERO = 3
B_ZERO = -5

TARGET_PIPELINE = [
    "-inline",
    "-symbol-dce",
    "-convert-linalg-to-loops",
    "-lower-affine",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-expand-strided-metadata",
    "-lower-affine",
    "-convert-arith-to-llvm",
    "-convert-math-to-llvm",
    "-convert-func-to-llvm",
    "-finalize-memref-to-llvm",
    "-reconcile-unrealized-casts",
]

XTAME_PIPELINE = ["-lower-linalg-to-xtame", "-lower-xt-ame"] + TARGET_PIPELINE

# Freestanding runtime: exit for _start, write(2) for dump, rdcycle for
# read_cycles. Generated here so no assembly is checked in twice.
STUB = """  .text
  .globl _start
_start:
  call main
  li a7, 93
  ecall
  .globl dump
dump:
  mv a2, a1
  mv a1, a0
  li a0, 1
  li a7, 64
  ecall
  ret
  .globl read_cycles
read_cycles:
  rdcycle a0
  ret
"""

FUNC_RE = re.compile(r"func\.func\s+@(?P<name>\w+)\s*\((?P<args>[^)]*)\)", re.S)
MEMREF_RE = re.compile(r"memref<(?P<dims>\d+x\w+)>")
KIND_RE = re.compile(r"\blinalg\.(?P<kind>\w+)")


class Case:
    """One kernel file from ops/."""

    def __init__(self, path):
        self.path = path
        self.stem = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            text = f.read()
        match = FUNC_RE.search(text)
        if not match:
            raise ValueError(f"{path}: no func.func with a name")
        self.kernel = match.group("name")
        self.body = text[
            text.index("module {") + len("module {") : text.rindex("}")
        ]
        kind = KIND_RE.search(text)
        self.kind = kind.group("kind") if kind else "?"

        self.params = []
        for param in match.group("args").split(","):
            param = param.strip()
            if not param:
                continue
            name, _, mtype = param.partition(":")
            self.params.append((name.strip(), mtype.strip()))
        self.memrefs = [
            (n, t) for n, t in self.params if t.startswith("memref<")
        ]
        self.scalars = [(n, t) for n, t in self.params if t == "i32"]

        self.batch = 1
        self.m = self.k = self.n = None
        self.reason = None
        dims = [MEMREF_RE.match(t) for _, t in self.memrefs]
        if len(self.memrefs) != 3 or len(self.params) != 3 + len(self.scalars):
            self.reason = (
                f"expected three memrefs and i32 scalars, got {self.params}"
            )
        elif not all(dims):
            self.reason = "operand types are not static memrefs"
        else:
            shapes = [
                tuple(int(x) for x in d.group("dims").split("x")[:-1])
                for d in dims
            ]
            elems = [d.group("dims").split("x")[-1] for d in dims]
            if elems != ["i8", "i8", "i32"]:
                self.reason = f"operands are {elems}, expected i8, i8, i32"
            elif (
                len(shapes[0]) == 2
                and len(shapes[1]) == 2
                and len(shapes[2]) == 2
            ):
                (self.m, self.k), (k2, self.n), (cm, cn) = shapes
                if k2 != self.k or cm != self.m or cn != self.n:
                    self.reason = (
                        "operand shapes are not MxK times KxN into MxN"
                    )
            elif (
                len(shapes[0]) == 3
                and len(shapes[1]) == 3
                and len(shapes[2]) == 3
            ):
                self.batch, self.m, self.k = shapes[0]
                batch2, k2, self.n = shapes[1]
                batch3, cm, cn = shapes[2]
                if (batch2, k2, batch3, cm, cn) != (
                    self.batch,
                    self.k,
                    self.batch,
                    self.m,
                    self.n,
                ):
                    self.reason = "batch operand shapes do not agree"
            else:
                self.reason = "operands must all be rank 2 or all rank 3"
        if self.reason is None:
            if self.m < 4 or self.k < 1 or self.n < 1 or self.batch < 1:
                self.reason = (
                    "the AME tile needs at least four rows, one K and one N"
                )
            elif self.scalars and len(self.scalars) != 2:
                self.reason = f"{len(self.scalars)} scalar operands, expected 2"
            elif self.scalars and self.kind != "quantized_matmul":
                self.reason = (
                    f"scalar operands are not handled for linalg.{self.kind}"
                )

    @property
    def supported(self):
        return self.reason is None

    @property
    def out_bytes(self):
        return self.batch * self.m * self.n * 4

    @property
    def call_args(self):
        """Harness values in the order of the kernel parameters."""
        memrefs = ["%a", "%b", "%c"]
        scalars = ["%azero", "%bzero"]
        operands = []
        mi = si = 0
        for _, mtype in self.params:
            if mtype.startswith("memref<"):
                operands.append(memrefs[mi])
                mi += 1
            else:
                operands.append(scalars[si])
                si += 1
        return ", ".join(operands)


def i8(value):
    """Reinterpret the low 8 bits of an integer as a signed i8."""
    return ((value & 0xFF) ^ 0x80) - 0x80


def a_value(batch, i, k):
    return i8(i * 3 + k * 5 + batch * 7 + 1)


def b_value(batch, k, l):
    return i8(k * 7 + l * 2 + batch * 11 + 3)


def reference(case):
    """The product the kernel has to compute, computed in Python."""
    out = []
    for batch in range(case.batch):
        rows = []
        for i in range(case.m):
            row = []
            for l in range(case.n):
                total = 0
                for kk in range(case.k):
                    a = a_value(batch, i, kk) - (A_ZERO if case.scalars else 0)
                    b = b_value(batch, kk, l) - (B_ZERO if case.scalars else 0)
                    total += a * b
                row.append(total)
            rows.append(row)
        out.append(rows)
    return out[0] if case.batch == 1 else out


def harness(case, reps, call_kernel):
    """The @main body: fill, time, dump."""
    m, k, n, b = case.m, case.k, case.n, case.batch
    batched = b > 1

    aType = f"memref<{b}x{m}x{k}xi8>" if batched else f"memref<{m}x{k}xi8>"
    bType = f"memref<{b}x{k}x{n}xi8>" if batched else f"memref<{k}x{n}xi8>"
    cType = f"memref<{b}x{m}x{n}xi32>" if batched else f"memref<{m}x{n}xi32>"
    aIdx = "%bi, %i, %kk" if batched else "%i, %kk"
    bIdx = "%bi, %kk, %l" if batched else "%kk, %l"
    cIdx = "%bi, %i, %l" if batched else "%i, %l"
    aHead = "%c0, %c0, %c0" if batched else "%c0, %c0"
    bHead = "%c0, %c0, %c0" if batched else "%c0, %c0"

    call = (
        ""
        if not call_kernel
        else (
            f"\n        func.call @{case.kernel}({case.call_args}) "
            f": ({', '.join(t for _, t in case.params)}) -> ()"
        )
    )

    return f"""
  func.func @main() -> i32 {{
    %a = memref.alloca() : {aType}
    %b = memref.alloca() : {bType}
    %c = memref.alloca() : {cType}
    %cycles = memref.alloca() : memref<1xi64>
    %sink = memref.alloca() : memref<2xi32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cb = arith.constant {b} : index
    %cm = arith.constant {m} : index
    %ckk = arith.constant {k} : index
    %cn = arith.constant {n} : index
    %reps = arith.constant {reps} : index
    %i32zero = arith.constant 0 : i32
    %i32one = arith.constant 1 : i32
    %i32two = arith.constant 2 : i32
    %i32three = arith.constant 3 : i32
    %i32five = arith.constant 5 : i32
    %i32seven = arith.constant 7 : i32
    %i32eleven = arith.constant 11 : i32
    %azero = arith.constant {A_ZERO} : i32
    %bzero = arith.constant {B_ZERO} : i32
    %nbytes = arith.constant {case.out_bytes} : i64
    %eight = arith.constant 8 : i64

    scf.for %bi = %c0 to %cb step %c1 {{
      %bi32 = arith.index_cast %bi : index to i32
      %batchA = arith.muli %bi32, %i32seven : i32
      scf.for %i = %c0 to %cm step %c1 {{
        %i32 = arith.index_cast %i : index to i32
        %row = arith.muli %i32, %i32three : i32
        %aBase = arith.addi %row, %batchA : i32
        scf.for %kk = %c0 to %ckk step %c1 {{
          %kk32 = arith.index_cast %kk : index to i32
          %scaled = arith.muli %kk32, %i32five : i32
          %sum = arith.addi %aBase, %scaled : i32
          %biased = arith.addi %sum, %i32one : i32
          %value = arith.trunci %biased : i32 to i8
          memref.store %value, %a[{aIdx}] : {aType}
        }}
      }}
    }}

    scf.for %bi = %c0 to %cb step %c1 {{
      %bi32 = arith.index_cast %bi : index to i32
      %batchB = arith.muli %bi32, %i32eleven : i32
      scf.for %kk = %c0 to %ckk step %c1 {{
        %kk32 = arith.index_cast %kk : index to i32
        %row = arith.muli %kk32, %i32seven : i32
        %bBase = arith.addi %row, %batchB : i32
        scf.for %l = %c0 to %cn step %c1 {{
          %l32 = arith.index_cast %l : index to i32
          %scaled = arith.muli %l32, %i32two : i32
          %sum = arith.addi %bBase, %scaled : i32
          %biased = arith.addi %sum, %i32three : i32
          %value = arith.trunci %biased : i32 to i8
          memref.store %value, %b[{bIdx}] : {bType}
        }}
      }}
    }}

    %sinkA = memref.load %a[{aHead}] : {aType}
    %sinkA32 = arith.extsi %sinkA : i8 to i32
    memref.store %sinkA32, %sink[%c0] : memref<2xi32>
    %sinkB = memref.load %b[{bHead}] : {bType}
    %sinkB32 = arith.extsi %sinkB : i8 to i32
    memref.store %sinkB32, %sink[%c1] : memref<2xi32>

    %t0 = llvm.call @read_cycles() : () -> i64
    scf.for %rep = %c0 to %reps step %c1 {{
      scf.for %bi = %c0 to %cb step %c1 {{
        scf.for %i = %c0 to %cm step %c1 {{
          scf.for %l = %c0 to %cn step %c1 {{
            memref.store %i32zero, %c[{cIdx}] : {cType}
          }}
        }}
      }}{call}
    }}
    %t1 = llvm.call @read_cycles() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    memref.store %delta, %cycles[%c0] : memref<1xi64>

    %caddr = memref.extract_aligned_pointer_as_index %c : {cType} -> index
    %caddr64 = arith.index_cast %caddr : index to i64
    %cptr = llvm.inttoptr %caddr64 : i64 to !llvm.ptr
    llvm.call @dump(%cptr, %nbytes) : (!llvm.ptr, i64) -> ()

    %kaddr = memref.extract_aligned_pointer_as_index %cycles : memref<1xi64> -> index
    %kaddr64 = arith.index_cast %kaddr : index to i64
    %kptr = llvm.inttoptr %kaddr64 : i64 to !llvm.ptr
    llvm.call @dump(%kptr, %eight) : (!llvm.ptr, i64) -> ()

    %saddr = memref.extract_aligned_pointer_as_index %sink : memref<2xi32> -> index
    %saddr64 = arith.index_cast %saddr : index to i64
    %sptr = llvm.inttoptr %saddr64 : i64 to !llvm.ptr
    llvm.call @dump(%sptr, %eight) : (!llvm.ptr, i64) -> ()
    func.return %i32zero : i32
  }}
"""


def module_text(case, reps, call_kernel):
    return (
        f"module {{\n  llvm.func @dump(!llvm.ptr, i64)\n"
        f"  llvm.func @read_cycles() -> i64\n"
        f"{case.body}\n{harness(case, reps, call_kernel)}\n}}\n"
    )


def tool(name, path):
    if not os.path.exists(path):
        sys.exit(
            f"{name} not found at {path}; build Buddy-MLIR first "
            f"(README.md, Build and Test LLVM/MLIR/CLANG)"
        )
    return path


def lower(source, pipeline, work, tag):
    src = os.path.join(work, tag + ".mlir")
    with open(src, "w") as f:
        f.write(source)
    return subprocess.run(
        [tool("buddy-opt", BUDDY_OPT), src] + pipeline,
        capture_output=True,
        text=True,
    )


def build(case, variant, reps, work):
    """Lower, assemble and link one program. Returns (executable, detail)."""
    tag = f"{case.stem}_{variant}"
    asm = os.path.join(work, tag + ".s")
    obj = os.path.join(work, tag + ".o")
    stub_s = os.path.join(work, "start.s")
    stub_o = os.path.join(work, "start.o")
    exe = os.path.join(work, tag)

    pipeline = XTAME_PIPELINE if variant == "xtame" else TARGET_PIPELINE
    mlir = lower(
        module_text(case, reps, variant != "overhead"), pipeline, work, tag
    )
    if mlir.returncode != 0:
        return None, mlir.stderr
    ll = subprocess.run(
        [tool("buddy-translate", BUDDY_TRANSLATE), "--buddy-to-llvmir"],
        input=mlir.stdout,
        capture_output=True,
        text=True,
    )
    if ll.returncode != 0:
        return None, ll.stderr
    asm_out = subprocess.run(
        [
            tool("buddy-llc", BUDDY_LLC),
            "-filetype=asm",
            "-mtriple=riscv64-unknown-linux-gnu",
            "-mattr=+m,+a,+c,+d,+xtheadame",
            "-o",
            asm,
        ],
        input=ll.stdout,
        capture_output=True,
        text=True,
    )
    if asm_out.returncode != 0:
        return None, asm_out.stderr

    with open(stub_s, "w") as f:
        f.write(STUB)
    for s, o in ((asm, obj), (stub_s, stub_o)):
        subprocess.run(
            [
                tool("llvm-mc", LLVM_MC),
                "-triple=riscv64-unknown-linux-gnu",
                "-mattr=+xtheadame",
                "-filetype=obj",
                s,
                "-o",
                o,
            ],
            check=True,
        )
    subprocess.run(
        ["ld.lld", "-m", "elf64lriscv", "-static", "-o", exe, stub_o, obj],
        check=True,
    )
    return exe, asm


BLOCK_INSN = re.compile(r"^0x([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)")
TRACE = re.compile(r"^Trace \d+: 0x[0-9a-f]+ \[([0-9a-f]+)/([0-9a-f]+)/")


def parse_trace_log(path):
    """Instructions per translation block, its AME instructions, executions.

    Instructions the RISC-V disassembler does not know are printed as
    ``illegal``; on this model those are the AME ones.
    """
    insns = {}
    ame = {}
    executions = {}
    current = None
    duplicate = False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("----------------"):
                current = None
                continue
            match = BLOCK_INSN.match(line)
            if match:
                address, _, mnemonic = match.groups()
                if current is None:
                    current = int(address, 16)
                    # QEMU can print a block again when it retranslates it;
                    # count each block once and use executions for the rest.
                    duplicate = current in insns
                    if not duplicate:
                        insns[current] = 0
                        ame[current] = 0
                if not duplicate:
                    insns[current] += 1
                    if mnemonic.startswith("illegal"):
                        ame[current] += 1
                continue
            match = TRACE.match(line)
            if match:
                start = int(match.group(2), 16)
                executions[start] = executions.get(start, 0) + 1
    return insns, ame, executions


def measure(exe, args):
    """Return (total instructions, AME instructions, completed process)."""
    if not args.qemu:
        return None, None, subprocess.run([exe], capture_output=True)

    log = os.path.join(args.work_dir, "trace.log")
    # nochain makes QEMU log every translation block instead of only the entry
    # of a chained run, otherwise the per-block counts are undercounts.
    subprocess.run(
        [
            args.qemu,
            "-cpu",
            args.cpu,
            "-d",
            "in_asm,exec,nochain",
            "-D",
            log,
            exe,
        ],
        capture_output=True,
        check=True,
    )
    cmd = [args.qemu, "-cpu", args.cpu]
    plugin = (
        os.path.join(args.plugin_dir, "libinsn.so") if args.plugin_dir else ""
    )
    if plugin and os.path.exists(plugin):
        cmd += ["-d", "plugin", "-plugin", plugin]
    run = subprocess.run(cmd + [exe], capture_output=True)

    total = None
    for stream in (run.stderr, run.stdout):
        match = re.search(rb"total insns: (\d+)", stream)
        if match:
            total = int(match.group(1))
    insns, ame, executions = parse_trace_log(log)
    ame_total = sum(
        ame.get(start, 0) * count for start, count in executions.items()
    )
    block_total = sum(
        insns.get(start, 0) * count for start, count in executions.items()
    )
    if total is None:
        total = block_total
    elif block_total and abs(block_total - total) > max(4, total // 200):
        print(f"    note: block sum {block_total} != plugin count {total}")
    return total, ame_total, run


def check_output(raw, case):
    expected = case.out_bytes
    if len(raw) < expected + 8:
        sys.exit(f"expected {expected + 8} bytes of output, got {len(raw)}")
    flat = struct.unpack(f"<{expected // 4}i", raw[:expected])
    size = case.m * case.n
    got = [
        [
            [flat[(b * case.m + i) * case.n + l] for l in range(case.n)]
            for i in range(case.m)
        ]
        for b in range(case.batch)
    ]
    got = got[0] if case.batch == 1 else got
    cycles = struct.unpack("<q", raw[expected : expected + 8])[0]
    return got == reference(case), cycles


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "cases",
        nargs="*",
        help="case names or substrings; default: every file in ops/",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="kernel iterations per program (default 1)",
    )
    parser.add_argument(
        "--qemu",
        default=os.environ.get("QEMU_RISCV64"),
        help="qemu-riscv64 with the XuanTie matrix extension; "
        "omit to run the binaries natively",
    )
    parser.add_argument(
        "--cpu", default=os.environ.get("QEMU_CPU", "xt-c9501fdvmt")
    )
    parser.add_argument(
        "--plugin-dir",
        default=os.environ.get("QEMU_PLUGIN_DIR"),
        help="directory holding libinsn.so; default: the "
        "tests/plugin next to the QEMU binary",
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.qemu and not args.plugin_dir:
        guess = os.path.join(
            os.path.dirname(os.path.abspath(args.qemu)), "tests/plugin"
        )
        args.plugin_dir = guess if os.path.isdir(guess) else ""
    args.work_dir = args.work_dir or os.path.join(
        ROOT, "build/benchmarks/XtameBench"
    )
    os.makedirs(args.work_dir, exist_ok=True)

    paths = sorted(
        os.path.join(OPS, name)
        for name in os.listdir(OPS)
        if name.endswith(".mlir")
    )
    if args.cases:
        paths = [
            path
            for path in paths
            if any(part in os.path.basename(path) for part in args.cases)
        ]
    if not paths:
        sys.exit("no case matches")

    rows = []
    mode = f"QEMU {args.cpu}" if args.qemu else "native"
    print(f"mode: {mode}, reps: {args.reps}\n")
    print(
        f"{'case':>25} {'status':>9} {'AME ops':>8} {'AME kernel':>11} "
        f"{'scalar kernel':>14} {'work ratio':>11}"
    )
    for path in paths:
        case = Case(path)
        row = {
            "case": case.stem,
            "kind": case.kind,
            "reps": args.reps,
            "status": "ok",
            "ame_insns": None,
            "ame_kernel": None,
            "scalar_kernel": None,
            "correct": None,
            "cycles": None,
        }
        if not case.supported:
            print(f"{case.stem:>25} {'no-xtame':>9}   {case.reason}")
            row["status"] = "no-xtame"
            rows.append(row)
            continue

        xtame_exe, detail = build(case, "xtame", args.reps, args.work_dir)
        if xtame_exe is None:
            first = [line for line in detail.splitlines() if "error" in line]
            print(
                f"{case.stem:>25} {'no-xtame':>9}   "
                f"{first[0] if first else detail[:80]}"
            )
            row["status"] = "no-xtame"
            rows.append(row)
            continue

        scalar_exe, _ = build(case, "scalar", args.reps, args.work_dir)
        over_exe, _ = build(case, "overhead", args.reps, args.work_dir)

        ame_total, ame_insns, run = measure(xtame_exe, args)
        ok, cycles = check_output(run.stdout, case)
        scalar_total, _, _ = measure(scalar_exe, args)
        over_total, _, _ = measure(over_exe, args)

        row["ame_insns"] = ame_insns
        row["correct"] = ok
        if args.qemu:
            # Both programs pay the same fill, zero and dump cost; subtracting
            # it leaves the instructions that do the kernel.
            ame_kernel = ame_total - over_total
            scalar_kernel = scalar_total - over_total
            ratio = (
                f"{scalar_kernel / ame_kernel:.1f}x"
                if ame_kernel > 0 and scalar_kernel > ame_kernel
                else "-"
            )
            print(
                f"{case.stem:>25} {'ok' if ok else 'WRONG':>9} {ame_insns:>8} "
                f"{ame_kernel:>11} {scalar_kernel:>14} {ratio:>11}"
            )
            row["ame_kernel"] = ame_kernel
            row["scalar_kernel"] = scalar_kernel
        else:
            print(
                f"{case.stem:>25} {'ok' if ok else 'WRONG':>9} "
                f"{'-':>8} {'-':>11} {'-':>14} {'-':>11}  cycles={cycles}"
            )
            row["cycles"] = cycles
        rows.append(row)

    print(
        "\nAME ops      AME instructions executed (mcfg*, mlde*, mldte*, "
        "mmacc*, mste*)\n"
        "AME kernel   executed instructions of the AME program with the shared "
        "fill/zero/dump cost removed\n"
        "scalar kernel  the same with the kernel lowered to loops\n"
        "work ratio   scalar kernel / AME kernel; instruction counts, not time"
    )

    if args.csv:
        keys = [
            "case",
            "kind",
            "reps",
            "status",
            "ame_insns",
            "ame_kernel",
            "scalar_kernel",
            "correct",
            "cycles",
        ]
        with open(args.csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(
                    ",".join(
                        "" if row.get(key) is None else str(row[key])
                        for key in keys
                    )
                    + "\n"
                )
        print(f"\nwrote {args.csv}")

    ran = [row for row in rows if row["status"] == "ok"]
    skipped = [row for row in rows if row["status"] != "ok"]
    if any(row["correct"] is False for row in ran):
        print("\nsome results do not match the Python product")
        return 1
    print(
        f"\n{len(ran)} case{'s' if len(ran) != 1 else ''} "
        f"agree{'s' if len(ran) == 1 else ''} with the Python product"
        + (f", {len(skipped)} reported no-xtame" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
