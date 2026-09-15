"""Fixed-slot allocation must reject overwrites, not silently spill or copy."""
import subprocess
import sys

llc = sys.argv[1]
acc = "<vscale x 32 x i32>"
tile = "<vscale x 128 x i8>"
load = "@llvm.riscv.bosc.fpga.mlce32.m.nxv32i32.i64"
store = "@llvm.riscv.bosc.fpga.msce32.m.nxv32i32.i64"
load_a = "@llvm.riscv.bosc.fpga.mlae8.m.nxv128i8.i64"
load_b = "@llvm.riscv.bosc.fpga.mlbe8.m.nxv128i8.i64"
mma = "@llvm.riscv.bosc.fpga.mqma.b.mm.nxv32i32.nxv128i8"
common_load = "@llvm.riscv.bosc.mlce32.m.nxv32i32.i64"
decls = f"""
target triple = "riscv64"
declare {acc} {load}(ptr, i64, i32 immarg)
declare void {store}({acc}, ptr, i64)
declare {tile} {load_a}(ptr, i64, i32 immarg)
declare {tile} {load_b}(ptr, i64, i32 immarg)
declare {acc} {mma}({acc}, {tile}, {tile})
declare {acc} {common_load}(ptr, i64)
declare void @external()
attributes #0 = {{ "target-features"="+xboscame-fpga" }}
"""


def seed(name, slot):
    return f"%{name} = call {acc} {load}(ptr %p, i64 64, i32 {slot})\n"


def save(name):
    return f"call void {store}({acc} %{name}, ptr %p, i64 64)\n"


def phi(slot):
    return ("br i1 %cond, label %left, label %right\nleft:\n" + seed("l", 0)
            + "br label %merge\nright:\n" + seed("r", slot)
            + f"br label %merge\nmerge:\n%v = phi {acc} [%l, %left], [%r, %right]\n"
            + save("v"))


cases = [
    ("compatible_phi", phi(0), None),
    ("conflicting_phi", phi(1), "requires incompatible fixed slots"),
    ("live_across_call", seed("s", 0) + "call void @external()\n" + save("s"),
     "matrix values cannot live across calls"),
    ("dead_definition", seed("s", 0) + seed("dead", 0) + save("s"),
     "simultaneously live values overlap"),
    ("invalid_bank", f"%a = call {tile} {load_a}(ptr %p, i64 64, i32 1)\n",
     "slot is outside its A/B/ACC bank"),
    ("missing_slot", f"%s = call {acc} {common_load}(ptr %p, i64 64)\n" + save("s"),
     "matrix value has no explicit load slot"),
    ("destructive_seed", seed("s", 0)
     + f"%a = call {tile} {load_a}(ptr %p, i64 64, i32 0)\n"
     + f"%b = call {tile} {load_b}(ptr %p, i64 64, i32 4)\n"
     + f"%v = call {acc} {mma}({acc} %s, {tile} %a, {tile} %b)\n"
     + save("v") + save("s"), "simultaneously live values overlap"),
]
for level in [0, 2, 3]:
    for name, body, diagnostic in cases:
        ir = decls + f"define void @{name}(ptr %p, i1 %cond) #0 {{\n{body}ret void\n}}\n"
        result = subprocess.run(
            [llc, f"-O{level}", "-mattr=+m,+f,+d,+v,+xboscame",
             "-verify-machineinstrs", "-o", "-"],
            input=ir, text=True, capture_output=True)
        if diagnostic:
            assert result.returncode != 0 and diagnostic in result.stderr, (
                name, level, result.returncode, result.stderr)
        else:
            assert result.returncode == 0, (name, level, result.stderr)
            assert "mmve" not in result.stdout, (name, level)
print("FPGA slot liveness and diagnostics: O0/O2/O3 passed")
