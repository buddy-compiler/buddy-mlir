"""Run the AME functional test in AME_gem5 syscall-emulation mode.

Invoke this configuration with ``gem5.opt``, not the host Python interpreter.
The AME_gem5 ISA fixes AMENLEN=4, AMEUDSZ=32, 32 M registers, and 4 ACC
registers in C++; the compiled ELF must use the matching XAME gem5 profile.
"""

import argparse
import json
import sys
from pathlib import Path

import m5
from m5.objects import (
    AddrRange,
    Process,
    RiscvAtomicSimpleCPU,
    Root,
    SEWorkload,
    SimpleMemory,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary", required=True, help="linked RISC-V ELF executable"
)
parser.add_argument("--max-ticks", type=int, default=10_000_000_000)
parser.add_argument("--max-insts", type=int, default=5_000_000)
args = parser.parse_args()
if args.max_ticks < 1 or args.max_insts < 1:
    parser.error("execution limits must be positive")
binary = Path(args.binary).resolve()
if not binary.is_file():
    parser.error(f"ELF executable not found: {binary}")

report_path = Path(m5.options.outdir, "result.json")
# A simulator fault must not leave a previous successful report in place.
report_path.unlink(missing_ok=True)

system = System()
system.clk_domain = SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("64MiB")]
system.cache_line_size = 64
system.cpu = RiscvAtomicSimpleCPU(max_insts_any_thread=args.max_insts)
system.membus = SystemXBar()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()
system.memory = SimpleMemory(range=system.mem_ranges[0], latency="20ns")
system.memory.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

system.workload = SEWorkload.init_compatible(str(binary))
system.cpu.workload = Process(cmd=[str(binary)])
system.cpu.createThreads()
isa = system.cpu.isa[0]
isa.riscv_profile = "RVI20U64"
isa.extra_extensions = ["M", "Zicsr", "Zifencei"]

root = Root(full_system=False, system=system)
m5.instantiate()
event = m5.simulate(args.max_ticks)
result = {
    "binary": str(binary),
    "cpu": "atomic",
    "mode": "syscall-emulation",
    "profile": "xame-gem5",
    "tile_side": 4,
    "unit_bits": 32,
    "m_registers": 32,
    "acc_registers": 4,
    "max_ticks": args.max_ticks,
    "max_insts": args.max_insts,
    "tick": m5.curTick(),
    "cause": event.getCause(),
    "guest_exit_code": event.getCode(),
}
result["passed"] = (
    result["cause"] == "exiting with last active thread context"
    and result["guest_exit_code"] == 0
)
report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("AME gem5 result: " + json.dumps(result))
sys.exit(0 if result["passed"] else 1)
