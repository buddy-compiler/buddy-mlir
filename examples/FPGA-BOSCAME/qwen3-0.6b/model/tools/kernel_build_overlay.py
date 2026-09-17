#!/usr/bin/env python3
"""Create a read-only-use index of kernel build trees without copying objects.

Use this fresh overlay only as --triton-build for graph import/archive/linking.
Do not direct the kernel compiler at it: its entries link to immutable existing
builds. Distinct sources may add new names but may never override a kernel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = {}
    for source in args.source:
        for frontend in sorted(source.resolve().glob("*/frontend.json")):
            name = frontend.parent.name
            if name in entries and entries[name] != frontend.parent:
                parser.error("duplicate kernel name across sources: " + name)
            entries[name] = frontend.parent
    if not entries:
        parser.error("no kernel manifests found")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, directory in sorted(entries.items()):
        (args.output / name).symlink_to(os.path.relpath(directory, args.output.resolve()),
                                     target_is_directory=True)
        report[name] = {"directory": str(directory), "frontend_sha256":
            hashlib.sha256((directory / "frontend.json").read_bytes()).hexdigest()}
    (args.output / "overlay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(entries)} kernels -> {args.output}")


if __name__ == "__main__":
    main()
