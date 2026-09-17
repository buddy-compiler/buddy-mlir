"""Reject mismatched upload pairs before they can overwrite model runtime data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/prepare_model_run.py'
spec = importlib.util.spec_from_file_location('prepare_model_run', path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DeploymentContracts(unittest.TestCase):
    def test_boot_bytes_and_only_canonical_zero_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted, image = root/'from-elf.bin', root/'upload.bin'
            expected = bytes(range(70))
            extracted.write_bytes(expected)
            for data in (expected, expected+b'\0'*58):
                image.write_bytes(data)
                self.assertEqual(prepare.check_image_bytes(image,extracted),70)
            for data in (expected[:-1], expected+b'\0', expected+b'\0'*59,
                         expected+b'\0'*57+b'x', b'x'+expected[1:]):
                image.write_bytes(data)
                with self.assertRaises(ValueError):
                    prepare.check_image_bytes(image,extracted)

    def test_resource_must_fit_its_own_arena_not_entire_workspace(self):
        symbols = {'__workspace_start':0xb8000000,'__workspace_end':0xc0000000,
                   'weight_arena_raw':0xb8000000,'weight_arena_end':0xb8000100}
        self.assertEqual(prepare.check_arena(symbols,'weight_arena',256),
                         (0xb8000000,0xb8000100))
        # Both sizes lie inside the overall workspace but are the wrong image's
        # parameter payload. 320 would overwrite the following KV arena.
        for size in (128,320,0):
            with self.assertRaisesRegex(ValueError,'capacity|invalid linked arena'):
                prepare.check_arena(symbols,'weight_arena',size)

    def test_resource_requires_explicit_end_label_and_workspace_bounds(self):
        symbols = {'__workspace_start':0xb8000000,'__workspace_end':0xc0000000,
                   'tokenizer_blob_raw':0xb8000000}
        with self.assertRaisesRegex(ValueError,'rebuild image'):
            prepare.check_arena(symbols,'tokenizer_blob',64)
        symbols['tokenizer_blob_end']=0xc0000040
        with self.assertRaisesRegex(ValueError,'invalid linked arena'):
            prepare.check_arena(symbols,'tokenizer_blob',0x8000040)


if __name__ == '__main__':
    unittest.main()
