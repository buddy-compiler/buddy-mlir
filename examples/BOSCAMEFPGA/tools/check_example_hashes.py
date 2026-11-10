#!/usr/bin/env python3
# ===- check_example_hashes.py ---------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

"""Generate or compare portable SHA-256 goldens for a clean example build tree.

Inputs: Subcommand generate|compare, --root BUILD, --golden hashes.json, optional artifact globs.
Outputs: hashes.json (generate) or compare diagnostics (compare); goldens store relative paths/sizes/hashes only.
When used: CI / local reproducibility checks after rebuilding an FPGA example.

    check_example_hashes.py generate --root BUILD --golden hashes.json
    check_example_hashes.py compare --root BUILD --golden hashes.json

The default artifact set is kernel.boscame.mlir, kernel.llvm.mlir, kernel.ll,
kernel.nr.s, and every top-level ELF and binary image. Pass relative file names
or glob patterns after the options to select a different set when generating.
Goldens contain only paths relative to BUILD, sizes, and hashes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

FORMAT = "boscame-example-sha256-v1"
DEFAULT_ARTIFACTS = (
    "kernel.boscame.mlir",
    "kernel.llvm.mlir",
    "kernel.ll",
    "kernel.nr.s",
    "*.elf",
    "*.bin",
)


def artifact_path(root, name):
    """Keep manifest entries relative and inside the selected build root."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(
            f"artifact path must be relative to the root: {name!r}"
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact resolves outside the root: {name!r}")
    return path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def generate(root, golden, patterns):
    files = {}
    for pattern in patterns or DEFAULT_ARTIFACTS:
        artifact_path(root, pattern)
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise ValueError(f"no artifacts match {pattern!r}")
        for path in matches:
            name = path.relative_to(root).as_posix()
            files[name] = fingerprint(artifact_path(root, name))
    golden.write_text(
        json.dumps(
            {"format": FORMAT, "files": dict(sorted(files.items()))}, indent=2
        )
        + "\n"
    )
    print(f"wrote {len(files)} artifact hashes to {golden}")


def compare(root, golden):
    manifest = json.loads(golden.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("unsupported or malformed hash manifest")
    if not manifest["files"]:
        raise ValueError("hash manifest contains no artifacts")
    failures = 0
    for name, expected in sorted(manifest["files"].items()):
        path = artifact_path(root, name)
        if not path.is_file():
            print(f"MISSING {name}")
            failures += 1
            continue
        actual = fingerprint(path)
        if actual != expected:
            print(f"DIFFERENT {name}: expected {expected}, got {actual}")
            failures += 1
        else:
            print(f"MATCH {name}")
    if failures:
        print(f"FAIL: {failures} artifact(s) differ or are missing")
        return 1
    print(f"PASS: all {len(manifest['files'])} artifacts match")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "compare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, required=True)
        subparser.add_argument("--golden", type=Path, required=True)
        if command == "generate":
            subparser.add_argument("artifacts", nargs="*")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        if args.command == "generate":
            generate(root, args.golden, args.artifacts)
            return 0
        return compare(root, args.golden)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
