"""Check that the RA AME cache hooks stay opt-in at the machine-code boundary."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "common" / "nr" / "nr_runtime.c"
INCLUDE = ROOT / "common" / "nr"
UART_INCLUDE = ROOT / "common" / "uart"


class AmeCacheDiagnosticTest(unittest.TestCase):
    def compile_ir(self, enabled):
        clang = shutil.which("clang")
        if not clang:
            self.skipTest("clang is required for the target-codegen check")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nr_runtime.ll"
            command = [
                clang,
                "--target=riscv64-unknown-elf",
                "-march=rv64gc",
                "-mabi=lp64d",
                "-std=c11",
                "-ffreestanding",
                "-O0",
                "-S",
                "-emit-llvm",
                f"-I{INCLUDE}",
                f"-I{UART_INCLUDE}",
                str(RUNTIME),
                "-o",
                str(output),
            ]
            if enabled:
                command.insert(-2, "-DNR_RA_AME_CACHE_DIAGNOSTIC=1")
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return output.read_text()

    def test_default_build_does_not_emit_ra_cbo(self):
        disabled = self.compile_ir(False)
        enabled = self.compile_ir(True)
        # The NH console already emits one flush/invalidate each.  Enabling the
        # RA diagnostic walker adds exactly one of each at -O0; the default must
        # not acquire either instruction.
        self.assertEqual(disabled.count("cbo.flush"), 1)
        self.assertEqual(disabled.count("cbo.inval"), 1)
        self.assertEqual(enabled.count("cbo.flush"), 2)
        self.assertEqual(enabled.count("cbo.inval"), 2)


if __name__ == "__main__":
    unittest.main()
