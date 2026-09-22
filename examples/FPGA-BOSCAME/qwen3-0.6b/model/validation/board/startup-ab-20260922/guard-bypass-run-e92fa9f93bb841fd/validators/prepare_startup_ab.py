#!/usr/bin/env python3
"""Create address-identical startup controls from the audited 28-layer prime ELF.

This deliberately accepts one exact historical ELF. It is a diagnostic artifact
producer, not a production build option. Replace either the two inlined startup
calls or their guard loads in place. Both graph-completion calls and every other
ELF byte remain unchanged.
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
GUARD_BYPASS_REPLACEMENT = 0x00100513  # addi a0, zero, 1
# These guards are the only startup state checks.  The compressed bnez is
# deliberately left in place: forcing a0=1 takes its already audited skip
# edge, bypassing both the AME call and the primed flag store.
GUARD_BYPASS = (
    ("run_prefill", 0x80150C72, 0x0004C503, 0x80150C76, 0xE509,
     0x80150C80, 0x80150C78, 0x80150C7C),
    ("run_decode", 0x8016240C, 0x0004C503, 0x80162410, 0xE509,
     0x8016241A, 0x80162412, 0x80162416),
)


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

    def instruction(self, address, size=4):
        offset = self.code_offset(address, size)
        return struct.unpack_from("<I" if size == 4 else "<H", self.data,
                                  offset)[0]

    def check_guard(self, function, address, expected_word, branch_address,
                    expected_branch, branch_target, skipped_call,
                    skipped_store):
        symbol = self.function(function)
        if not symbol[3] <= address or branch_target + 2 > symbol[3] + symbol[4]:
            raise ValueError(f"{address:#x}: outside {function}")
        offset = self.code_offset(address)
        word = self.instruction(address)
        if word != expected_word:
            raise ValueError(f"{address:#x}: startup guard differs from audited ELF")
        if self.instruction(branch_address, 2) != expected_branch:
            raise ValueError(f"{branch_address:#x}: guard branch differs from audited ELF")
        if branch_target != branch_address + c_bnez_offset(expected_branch):
            raise ValueError(f"{branch_address:#x}: unexpected guard branch target")
        # The taken edge immediately overwrites a0 with the cycle counter
        # base (c.lui a0, ...), so the forced value is only a branch predicate.
        if self.instruction(branch_target, 2) != 0x752D:
            raise ValueError(f"{branch_target:#x}: guard skip edge does not overwrite a0")
        startup_word = next(word for name, _, word in STARTUP if name == function)
        call = self.check_call(function, skipped_call, startup_word)
        store = self.instruction(skipped_store)
        if store != (0x01A48023 if function == "run_prefill" else 0x01B48023):
            raise ValueError(f"{skipped_store:#x}: primed store differs from audited ELF")
        return {"function": function, "address": hex(address),
                "file_offset": hex(offset), "word": f"0x{word:08x}",
                "bytes": self.slice(offset, 4).hex(),
                "branch": {"address": hex(branch_address),
                           "word": f"0x{expected_branch:04x}",
                           "target": hex(branch_target)},
                "skipped_call": call, "skipped_store": {
                    "address": hex(skipped_store), "word": f"0x{store:08x}"}}


def c_bnez_offset(halfword):
    """Decode the signed PC-relative offset of a C.BNEZ instruction."""
    if halfword & 0x3 != 0x1 or (halfword >> 13) != 0x7:
        raise ValueError(f"{halfword:#x}: expected C.BNEZ")
    # CB-format immediate: imm[8|7:6|5|4:3|2:1|0] =
    # inst[12|6:5|2|11:10|4:3|0].
    immediate = (((halfword >> 12) & 1) << 8
                 | ((halfword >> 5) & 0x3) << 6
                 | ((halfword >> 2) & 1) << 5
                 | ((halfword >> 10) & 0x3) << 3
                 | ((halfword >> 3) & 0x3) << 1)
    if immediate & 0x100:
        immediate -= 0x200
    return immediate


def patch_elf(data, variant):
    if sha256(data) != SOURCE_SHA256:
        raise ValueError("source ELF SHA256 does not match the audited production-prime2 ELF")
    if variant not in (*REPLACEMENTS, "guard-bypass"):
        raise ValueError("variant must be layout, fence, or guard-bypass")
    original = Elf(data)
    if original.function("ame_fence")[3] != AME_FENCE_ADDRESS:
        raise ValueError("ame_fence address changed")
    calls = [original.check_call(*entry) for entry in STARTUP]
    completion = [original.check_call(*entry) for entry in COMPLETION]
    patched = bytearray(data)
    allowed_offsets = set()
    patches = []
    if variant in REPLACEMENTS:
        replacement = struct.pack("<I", REPLACEMENTS[variant])
        for call in calls:
            offset = int(call["file_offset"], 16)
            patched[offset:offset + 4] = replacement
            allowed_offsets.update(range(offset, offset + 4))
            call["replacement_word"] = f"0x{REPLACEMENTS[variant]:08x}"
            call["replacement_bytes"] = replacement.hex()
            patches.append(call)
    else:
        guards = [original.check_guard(*entry) for entry in GUARD_BYPASS]
        replacement = struct.pack("<I", GUARD_BYPASS_REPLACEMENT)
        for guard in guards:
            offset = int(guard["file_offset"], 16)
            patched[offset:offset + 4] = replacement
            allowed_offsets.update(range(offset, offset + 4))
            guard["replacement_word"] = f"0x{GUARD_BYPASS_REPLACEMENT:08x}"
            guard["replacement_bytes"] = replacement.hex()
            patches.append(guard)
    patched = bytes(patched)
    modified = Elf(patched)
    differences = [{"file_offset": hex(i), "before": f"{a:02x}", "after": f"{b:02x}"}
                   for i, (a, b) in enumerate(zip(data, patched)) if a != b]
    if len(data) != len(patched) or not differences:
        raise ValueError("patch did not preserve ELF size or made no changes")
    if any(int(d["file_offset"], 16) not in allowed_offsets for d in differences):
        raise ValueError("patch changed bytes outside the two startup patch sites")
    if (original.section_table != modified.section_table
            or original.symbol_tables != modified.symbol_tables
            or original.symbols != modified.symbols):
        raise ValueError("ELF sections or symbols changed")
    if [modified.check_call(*entry) for entry in COMPLETION] != completion:
        raise ValueError("graph-completion synchronization changed")
    if variant == "guard-bypass" and [modified.check_call(*entry) for entry in STARTUP] != calls:
        raise ValueError("guard-bypass changed the startup call instructions")
    invariants = {"file_size_unchanged": True,
                  "section_headers_identical": True,
                  "symbol_tables_identical": True,
                  "all_other_elf_bytes_identical": True,
                  "graph_completion_calls_unchanged": True}
    if variant == "guard-bypass":
        invariants["startup_calls_unchanged"] = True
    return patched, {"patches": patches, "byte_differences": differences,
                     "graph_completion_calls": completion,
                     "invariants": invariants,
                     "source_elf_sha256": sha256(data),
                     "elf_sha256": sha256(patched)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-elf", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(REPLACEMENTS) + ("guard-bypass",), required=True)
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
                                  ("guard-bypass forces both audited guards taken and therefore skips the "
                                   "startup calls and primed stores; graph completion calls remain unchanged"
                                   if args.variant == "guard-bypass" else
                                   "the two startup guards and primed stores remain unchanged")],
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
