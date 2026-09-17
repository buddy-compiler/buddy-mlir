"""Reject the old RISC-V linker that corrupted wrapped function addresses."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('builder', TOOLS / 'build_nr_w8a8_image.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ModelLinker(unittest.TestCase):
    def test_old_linker_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'ld.lld'
            p.write_text('#!/bin/sh\nprintf "LLD 15.0.5\\n"\n')
            p.chmod(0o755)
            with self.assertRaisesRegex(ValueError, 'requires LLD >= 20'):
                builder.resolve_linker(p, Path(directory))

    def test_symlink_name_preserved_for_multicall_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'lld'
            p.write_text('#!/bin/sh\ncase "$0" in */ld.lld) printf "LLD 20.1.8\\n";; *) exit 1;; esac\n')
            p.chmod(0o755)
            link = Path(directory) / 'ld.lld'
            link.symlink_to(p)
            info = builder.resolve_linker(link, Path(directory))
            self.assertEqual(info['path'], str(link))
            self.assertEqual(info['binary_path'], str(p))
            self.assertEqual(info['version'], 'LLD 20.1.8')
            self.assertEqual(len(info['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
