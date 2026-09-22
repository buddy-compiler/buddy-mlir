#!/usr/bin/env python3
"""Create address-identical startup controls from the audited 28-layer prime ELF.

This deliberately accepts one exact historical ELF. It is a diagnostic artifact
producer, not a production build option. The two inlined startup calls are
replaced in place; both graph-completion calls and every other ELF byte remain.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys


SOURCE_SHA256 = "fed6d5bf1a3612e4e653ebb0e781ce27f2f30ef3d794bfc9fd66a6d6d763ad0d"
AME_FENCE_ADDRESS = 0x8017041A
STARTUP = (
    ("run_prefill", 0x80150C78, 0x7A21F0EF),
    ("run_decode", 0x80162412, 0x0080E0EF),
)
COMPLETION = (
    ("run_prefill", 0x801558D6, 0x3451A0EF),
    ("run_decode", 0x80167072, 0x3A8090EF),
)
REPLACEMENTS = {"layout": 0x00000013, "fence": 0x0330000F}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def jal_target(address, word):
    if word & 0x7F != 0x6F or (word >> 7) & 31 != 1:
        raise ValueError(f"{address:#x}: expected a 32-bit JAL with rd=ra")
    immediate = (((word >> 31) & 1) << 20
                 | ((word >> 21) & 0x3FF) << 1
                 | ((word >> 20) & 1) << 11
                 | ((word >> 12) & 0xFF) << 12)
    if immediate & (1 << 20):
        immediate -= 1 << 21
    return address + immediate


class Elf:
    def __init__(self, data):
        self.data = data
        if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
            raise ValueError("expected little-endian ELF64")
        if struct.unpack_from("<HH", data, 16) != (2, 243):
            raise ValueError("expected a RISC-V executable ELF")
        table_offset = struct.unpack_from("<Q", data, 40)[0]
        entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
        if entry_size != 64 or not count or names_index >= count:
            raise ValueError("unsupported ELF section table")
        self.section_table = self.slice(table_offset, entry_size * count)
        self.sections = [struct.unpack_from("<IIQQQQIIQQ", self.section_table, i * 64)
                         for i in range(count)]
        strings = self.section_data(self.sections[names_index])
        self.section_names = [self.string(strings, h[0]) for h in self.sections]
        self.symbols = {}
        self.symbol_tables = []
        for h in self.sections:
            if h[1] != 2:  # SHT_SYMTAB
                continue
            if h[9] != 24 or h[5] % 24 or h[6] >= count:
                raise ValueError("invalid ELF symbol table")
            raw = self.section_data(h)
            names = self.section_data(self.sections[h[6]])
            self.symbol_tables.append((raw, names))
            for offset in range(0, len(raw), 24):
                name, info, other, section, value, size = struct.unpack_from("<IBBHQQ", raw, offset)
                label = self.string(names, name)
                self.symbols.setdefault(label, []).append((info, other, section, value, size))

    def slice(self, offset, length):
        if offset < 0 or length < 0 or offset + length > len(self.data):
            raise ValueError("ELF range outside file")
        return self.data[offset:offset + length]

    def section_data(self, header):
        return self.slice(header[4], header[5])

    @staticmethod
    def string(strings, offset):
        if offset >= len(strings) or b"\0" not in strings[offset:]:
            raise ValueError("invalid ELF string")
        return strings[offset:strings.index(b"\0", offset)].decode("utf-8")

    def function(self, name):
        matches = [s for s in self.symbols.get(name, []) if s[0] & 15 == 2]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one function symbol: {name}")
        return matches[0]

    def code_offset(self, address, length=4):
        matches = [(name, h[4] + address - h[3])
                   for name, h in zip(self.section_names, self.sections)
                   if h[1] == 1 and h[2] & 4 and h[3] <= address
                   and address + length <= h[3] + h[5]]
        if len(matches) != 1 or matches[0][0] != ".text":
            raise ValueError(f"{address:#x}: expected one .text mapping")
        self.slice(matches[0][1], length)
        return matches[0][1]

    def check_call(self, function, address, expected_word):
        symbol = self.function(function)
        if not symbol[3] <= address or address + 4 > symbol[3] + symbol[4]:
            raise ValueError(f"{address:#x}: outside {function}")
        offset = self.code_offset(address)
        word = struct.unpack_from("<I", self.data, offset)[0]
        if word != expected_word or jal_target(address, word) != AME_FENCE_ADDRESS:
            raise ValueError(f"{address:#x}: startup/completion call differs from audited ELF")
        return {"function": function, "address": hex(address), "file_offset": hex(offset),
                "word": f"0x{word:08x}", "bytes": self.slice(offset, 4).hex(),
                "target": hex(AME_FENCE_ADDRESS)}


def patch_elf(data, variant):
    if sha256(data) != SOURCE_SHA256:
        raise ValueError("source ELF SHA256 does not match the audited production-prime2 ELF")
    if variant not in REPLACEMENTS:
        raise ValueError("variant must be layout or fence")
    original = Elf(data)
    if original.function("ame_fence")[3] != AME_FENCE_ADDRESS:
        raise ValueError("ame_fence address changed")
    calls = [original.check_call(*entry) for entry in STARTUP]
    completion = [original.check_call(*entry) for entry in COMPLETION]
    patched = bytearray(data)
    replacement = struct.pack("<I", REPLACEMENTS[variant])
    allowed_offsets = set()
    for call in calls:
        offset = int(call["file_offset"], 16)
        patched[offset:offset + 4] = replacement
        allowed_offsets.update(range(offset, offset + 4))
        call["replacement_word"] = f"0x{REPLACEMENTS[variant]:08x}"
        call["replacement_bytes"] = replacement.hex()
    patched = bytes(patched)
    modified = Elf(patched)
    differences = [{"file_offset": hex(i), "before": f"{a:02x}", "after": f"{b:02x}"}
                   for i, (a, b) in enumerate(zip(data, patched)) if a != b]
    if len(data) != len(patched) or not differences:
        raise ValueError("patch did not preserve ELF size or made no changes")
    if any(int(d["file_offset"], 16) not in allowed_offsets for d in differences):
        raise ValueError("patch changed bytes outside the two startup calls")
    if (original.section_table != modified.section_table
            or original.symbol_tables != modified.symbol_tables
            or original.symbols != modified.symbols):
        raise ValueError("ELF sections or symbols changed")
    if [modified.check_call(*entry) for entry in COMPLETION] != completion:
        raise ValueError("graph-completion synchronization changed")
    return patched, {"patches": calls, "byte_differences": differences,
                     "graph_completion_calls": completion,
                     "invariants": {"file_size_unchanged": True,
                                    "section_headers_identical": True,
                                    "symbol_tables_identical": True,
                                    "all_other_elf_bytes_identical": True,
                                    "graph_completion_calls_unchanged": True},
                     "source_elf_sha256": sha256(data),
                     "elf_sha256": sha256(patched)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-elf", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(REPLACEMENTS), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new, previously nonexistent output directory")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[5])
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output directory must not already exist")
        source = args.source_elf.resolve(strict=True)
        patched, report = patch_elf(source.read_bytes(), args.variant)
        root = args.repo_root.resolve(strict=True)
        llvm = root / "llvm/build-2d26/bin"
        audit_tool = root / "examples/FPGA-BOSCAME/tools/check_nr_elf.py"
        for path in (llvm / "llvm-objcopy", llvm / "llvm-objdump", llvm / "llvm-nm", audit_tool):
            if not path.is_file():
                raise ValueError(f"missing audit/build tool: {path}")
        args.output.mkdir(parents=True, exist_ok=False)
        output = args.output.resolve()
        elf, binary = output / "qwen_model.elf", output / "qwen_model.bin"
        elf.write_bytes(patched)
        report.update({"schema_version": 1, "variant": args.variant, "status": "FAIL",
                       "source_elf": str(source), "elf": str(elf), "bin": str(binary),
                       "tool_sha256": sha256(Path(__file__).read_bytes()),
                       "limits": ["diagnostic only; no production builder changes",
                                  "removing startup work changes its AME state, data accesses, and timing",
                                  "the two startup guards and primed stores remain unchanged"],
                       "commands": []})
        manifest = output / "startup-ab.json"

        def run(command):
            result = subprocess.run([str(c) for c in command], capture_output=True, text=True)
            report["commands"].append({"argv": [str(c) for c in command],
                                       "returncode": result.returncode,
                                       "stdout": result.stdout, "stderr": result.stderr})
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            if result.returncode:
                raise ValueError(f"command failed: {command[0]}: {result.stderr[-1000:]}")
            return result

        run([llvm / "llvm-objcopy", "-O", "binary", elf, binary])
        report["bin_sha256"] = sha256(binary.read_bytes())
        run([sys.executable, audit_tool, elf, "--objdump", llvm / "llvm-objdump",
             "--output", output / "elf-audit.json"])
        undefined = run([llvm / "llvm-nm", "--undefined-only", elf]).stdout.strip()
        report["undefined_symbols"] = undefined.splitlines()
        if undefined:
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            raise ValueError("patched ELF has undefined symbols")
        report["status"] = "PASS"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "variant", "elf", "bin",
                                                "source_elf_sha256", "elf_sha256", "bin_sha256")}, indent=2))
        return 0
    except (OSError, ValueError, struct.error) as error:
        print(f"startup A/B preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
