"""Check the compiled terminal trap ABI without running on FPGA hardware."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


FPGA = Path(__file__).resolve().parents[1]
NR = FPGA / "common" / "nr"
LLVM = Path(os.environ.get("LLVM_BIN", FPGA.parents[1] / "llvm/build-2d26/bin"))
FLAGS = ["--target=riscv64-unknown-elf", "-march=rv64gc_zicbom",
         "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding",
         "-fno-builtin", "-fno-pie", "-fno-vectorize", "-fno-slp-vectorize"]


@unittest.skipUnless((LLVM / "clang").is_file() and
                     (LLVM / "llvm-objdump").is_file(),
                     "RISC-V LLVM tools required (set LLVM_BIN)")
class NrTrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        obj = directory / "crt.o"
        subprocess.run([str(LLVM / "clang"), *FLAGS, "-Wa,-L", "-c",
                        str(NR / "crt.S"), "-o", str(obj)], check=True,
                       capture_output=True, text=True)
        cls.disassembly = subprocess.run(
            [str(LLVM / "llvm-objdump"), "-dr", "--no-show-raw-insn", str(obj)],
            check=True, capture_output=True, text=True).stdout
        ir = directory / "nr_runtime.ll"
        subprocess.run([str(LLVM / "clang"), *FLAGS, "-fno-inline", "-S",
                        "-emit-llvm", str(NR / "nr_runtime.c"), "-o", str(ir)],
                       check=True, capture_output=True, text=True)
        cls.runtime_ir = ir.read_text()
        subprocess.run([str(LLVM / "clang"), *FLAGS, "-c", str(ir), "-o",
                        str(directory / "nr_runtime.o")], check=True,
                       capture_output=True, text=True)

    def trap_entry(self, core):
        body = self.disassembly.split(f"<.L{core}_trap>:", 1)[1]
        if core == "nh":
            body = body.split("<.Lra_trap>:", 1)[0]
        instructions = []
        for line in body.splitlines():
            match = re.match(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)\s*(.*?)\s*$",
                             line)
            if match:
                operand = match[2].split("<", 1)[0].strip()
                if match[1] != "nop":
                    instructions.append((match[1], re.sub(r"\s+", "", operand)))
        return body, instructions

    def test_both_entries_capture_before_stack_reset_without_stack_access(self):
        for core, stack in (("nh", "__stack_top"), ("ra", "__ra_stack_top")):
            with self.subTest(core=core):
                body, instructions = self.trap_entry(core)
                self.assertEqual(instructions[:2], [("mv", "a3,ra"),
                                                   ("mv", "a4,sp")])
                self.assertEqual(instructions[2][0], "auipc")
                self.assertTrue(instructions[2][1].startswith("sp,"))
                self.assertIn(stack, body)
                self.assertIn(f"nr_{core}_trap", body)
                for item in (("csrr", "a0,mcause"), ("csrr", "a1,mepc"),
                             ("csrr", "a2,mtval")):
                    self.assertIn(item, instructions[3:])
                # Neither the interrupted stack nor a captured pointer is read.
                self.assertTrue(all(op in {"mv", "auipc", "addi", "csrr", "jr"}
                                    for op, _ in instructions))
                self.assertFalse(any(operands.startswith(("a3,", "a4,"))
                                     for _, operands in instructions[2:]))
                self.assertEqual(instructions[-1], ("jr", "t1"))

    def test_c_handlers_print_all_five_integer_arguments_in_abi_order(self):
        for core, printer in (("nh", "host_hex"), ("ra", "nr_hex64")):
            with self.subTest(core=core):
                match = re.search(
                    rf"define[^\n]*@nr_{core}_trap\(([^\n]*)\)[^\n]*\{{\n(.*?)\n\}}",
                    self.runtime_ir, re.S)
                self.assertIsNotNone(match)
                arguments = re.findall(r"i64\b[^,]*\s(%[\w.]+)", match[1])
                self.assertEqual(len(arguments), 5)
                printed = re.findall(
                    rf"call (?:fastcc )?void @{printer}\(i64[^)]*\s(%[\w.]+)\)",
                    match[2])
                self.assertEqual(printed, arguments)

    def test_standalone_probe_fatal_path_supplies_new_context_arguments(self):
        directory = Path(self.temporary.name)
        obj = directory / 'rvv_ops.o'
        flags = [flag for flag in FLAGS if not flag.startswith('-march=')]
        subprocess.run([str(LLVM / 'clang'), *flags, '-march=rv64gcv_zicbom', '-Wa,-L',
                        '-c', str(NR/'probes/rvv_ops.S'), '-o', str(obj)],
                       check=True, capture_output=True, text=True)
        disassembly = subprocess.run([str(LLVM/'llvm-objdump'), '-dr', '--no-show-raw-insn', str(obj)],
                                     check=True, capture_output=True, text=True).stdout
        fatal = disassembly.split('<.Lfatal>:', 1)[1]
        self.assertRegex(fatal, r'addi\s+sp, sp, (?:0x20|32)\s*\n'
                               r'\s*[0-9a-f]+:\s+mv\s+a3, ra\s*\n'
                               r'\s*[0-9a-f]+:\s+mv\s+a4, sp')
        self.assertIn('nr_ra_trap', fatal)


if __name__ == "__main__":
    unittest.main()
