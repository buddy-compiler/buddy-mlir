"""Check fail-closed input selection and RISC-V startup-call validation."""
import importlib.util
from pathlib import Path
import struct
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/prepare_startup_ab.py"
SOURCE_ELF = (TOOL.parents[1] / "build/console-fix-20260921/production-prime2/"
              "image/qwen_model.elf")
SPEC = importlib.util.spec_from_file_location("prepare_startup_ab", TOOL)
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)


class StartupABTests(unittest.TestCase):
    def source_data(self):
        if not SOURCE_ELF.is_file():
            self.skipTest(f"audited source ELF missing: {SOURCE_ELF}")
        return SOURCE_ELF.read_bytes()

    def test_rejects_unrecognized_image_before_parsing(self):
        for data in (b"", b"\x7fELF\x02\x01" + bytes(64)):
            for variant in (*AB.REPLACEMENTS, "guard-bypass"):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    AB.patch_elf(data, variant)

    def test_all_four_calls_resolve_to_the_same_ame_fence(self):
        for _, address, word in AB.STARTUP + AB.COMPLETION:
            self.assertEqual(AB.jal_target(address, word), AB.AME_FENCE_ADDRESS)

    def test_negative_jal_offset_and_wrong_rd(self):
        self.assertEqual(AB.jal_target(0x1000, 0xFFDFF0EF), 0xFFC)
        with self.assertRaisesRegex(ValueError, "rd=ra"):
            AB.jal_target(0x1000, 0xFFDFF06F)
        for word in AB.REPLACEMENTS.values():
            with self.assertRaisesRegex(ValueError, "JAL"):
                AB.jal_target(0x1000, word)

    def test_completion_calls_are_not_patch_targets(self):
        self.assertTrue({v[1] for v in AB.STARTUP}.isdisjoint(v[1] for v in AB.COMPLETION))
        self.assertEqual(len(AB.STARTUP), 2)

    def test_guard_branch_offsets_and_a0_skip_overwrite(self):
        for function, address, word, branch, branch_word, target, call, store in AB.GUARD_BYPASS:
            self.assertEqual(word, 0x0004C503)
            self.assertEqual(branch_word, 0xE509)
            self.assertEqual(AB.c_bnez_offset(branch_word), 10)
            self.assertEqual(target, branch + 10)
            self.assertEqual(call, target - 8)
            self.assertEqual(store, target - 4)
        self.assertEqual(AB.GUARD_BYPASS_REPLACEMENT, 0x00100513)

    def test_guard_bypass_only_replaces_the_two_lbu_in_audited_elf(self):
        data = self.source_data()
        patched, report = AB.patch_elf(data, "guard-bypass")
        self.assertEqual(len(patched), len(data))
        offsets = {int(item["file_offset"], 16)
                   for item in report["byte_differences"]}
        expected = set()
        for _, address, _, _, _, _, _, _ in AB.GUARD_BYPASS:
            offset = AB.Elf(data).code_offset(address)
            # The high byte remains zero in both 32-bit instructions.
            expected.update(range(offset, offset + 3))
        self.assertEqual(offsets, expected)
        self.assertEqual(len(report["byte_differences"]), 6)
        self.assertTrue(report["invariants"]["startup_calls_unchanged"])
        self.assertEqual(report["invariants"]["graph_completion_calls_unchanged"], True)
        patched_elf = AB.Elf(patched)
        for _, address, _, _, _, _, _, _ in AB.GUARD_BYPASS:
            self.assertEqual(patched_elf.instruction(address), AB.GUARD_BYPASS_REPLACEMENT)
        for entry in AB.STARTUP + AB.COMPLETION:
            self.assertEqual(patched_elf.instruction(entry[1]), entry[2])

    def test_guard_audit_rejects_each_modified_instruction(self):
        data = self.source_data()
        original = AB.Elf(data)
        for entry in AB.GUARD_BYPASS:
            for address, message in ((entry[1], "startup guard"),
                                     (entry[3], "guard branch"),
                                     (entry[5], "overwrite a0"),
                                     (entry[6], "call differs"),
                                     (entry[7], "primed store")):
                with self.subTest(function=entry[0], address=hex(address)):
                    changed = bytearray(data)
                    changed[original.code_offset(address)] ^= 4
                    with self.assertRaisesRegex(ValueError, message):
                        AB.Elf(bytes(changed)).check_guard(*entry)
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        AB.patch_elf(bytes(changed), "guard-bypass")

    def test_existing_variants_keep_original_patch_sites_and_invariants(self):
        data = self.source_data()
        original = AB.Elf(data)
        for variant, word in AB.REPLACEMENTS.items():
            with self.subTest(variant=variant):
                patched, report = AB.patch_elf(data, variant)
                expected = bytearray(data)
                for _, address, _ in AB.STARTUP:
                    struct.pack_into("<I", expected, original.code_offset(address), word)
                self.assertEqual(patched, bytes(expected))
                self.assertEqual(report["invariants"], {
                    "file_size_unchanged": True,
                    "section_headers_identical": True,
                    "symbol_tables_identical": True,
                    "all_other_elf_bytes_identical": True,
                    "graph_completion_calls_unchanged": True})


if __name__ == "__main__":
    unittest.main()
