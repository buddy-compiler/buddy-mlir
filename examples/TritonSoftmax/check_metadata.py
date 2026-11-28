#!/usr/bin/env python3
"""构建期元信息契约测试；不导入Triton，不运行任何目标代码。"""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from run_rax import manifest
from softmax_metadata import (
    EXPORT_PROFILE,
    kernel_attributes,
    target_header,
    validate_export,
)


class MetadataContract(unittest.TestCase):
    def setUp(self):
        self.object_bytes = b"metadata-contract-object"
        self.record = copy.deepcopy(EXPORT_PROFILE)
        self.record["object_sha256"] = hashlib.sha256(
            self.object_bytes
        ).hexdigest()

    def test_valid_record_generates_rhal_and_header(self):
        attrs = kernel_attributes(self.record)
        self.assertEqual(attrs["object_sha256"], self.record["object_sha256"])
        self.assertEqual(attrs["required_isa"], self.record["required_isa"])
        self.assertIn(self.record["required_isa"], target_header(self.record))
        text = manifest(metadata=self.record, isa=self.record["required_isa"])
        for key, value in attrs.items():
            self.assertIn(f'{key} = "{value}"', text)
        self.assertIn(self.record["query_entry"], text)

    def test_missing_fields_are_rejected(self):
        for key in EXPORT_PROFILE:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                del record[key]
                with self.assertRaisesRegex(ValueError, key):
                    kernel_attributes(record)

    def test_changed_fields_are_rejected(self):
        for key, value in EXPORT_PROFILE.items():
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                if isinstance(value, bool):
                    record[key] = not value
                elif isinstance(value, int):
                    record[key] = value + 1
                elif isinstance(value, str):
                    record[key] = value + ".changed"
                else:
                    record[key] = []
                with self.assertRaisesRegex(ValueError, key):
                    kernel_attributes(record)

    def test_bool_cannot_replace_integer(self):
        for key, value in [
            ("axis", True),
            ("grid", [2, True, True]),
            ("shape", [2, True]),
        ]:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                record[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_export(record)

    def test_object_content_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "softmax.obj"
            obj.write_bytes(self.object_bytes)
            validate_export(self.record, obj)
            obj.write_bytes(b"x" * len(self.object_bytes))
            with self.assertRaisesRegex(ValueError, "对象内容不一致"):
                validate_export(self.record, obj)

    def test_missing_or_invalid_digest_is_rejected(self):
        for digest in [None, "", "g" * 64, "0" * 63]:
            with self.subTest(digest=digest):
                record = copy.deepcopy(self.record)
                record["object_sha256"] = digest
                with self.assertRaisesRegex(ValueError, "object_sha256"):
                    target_header(record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
