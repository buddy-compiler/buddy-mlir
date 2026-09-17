#!/usr/bin/env python3
"""Create a read-only-use index of kernel build trees without copying objects.

Use this fresh overlay only as --triton-build for graph import/archive/linking.
Do not direct the kernel compiler at it: its entries link to immutable existing
builds. Distinct sources may add new names but may never silently override a
kernel. --replace-case explicitly selects an experimental build of an existing
case, recording its previous source. Archive/link validation remains required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--replace-case", type=Path, action="append", default=[])
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
    replacements = {}
    for directory in args.replace_case:
        directory = directory.resolve()
        name = directory.name
        if name not in entries or name in replacements:
            parser.error("replacement must name one existing, unreplaced case: " + name)
        before = json.loads((entries[name] / "frontend.json").read_text())
        after = json.loads((directory / "frontend.json").read_text())
        for field in ("name", "family", "symbol", "arguments", "signature"):
            if before.get(field) != after.get(field):
                parser.error("replacement changes external ABI/semantics: " + name + ": " + field)
        # Grid, blocking and internal frontend may change. Physical dimensions
        # and arithmetic constants must remain equal; ignore blocking only.
        dimensions = lambda manifest: {k: v for k, v in manifest["constexprs"].items()
                                       if k not in ("BLOCK", "BM", "BN", "BK")}
        if dimensions(before) != dimensions(after):
            parser.error("replacement changes dimensions/arithmetic constants: " + name)
        replacements[name] = str(entries[name])
        entries[name] = directory
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, directory in sorted(entries.items()):
        (args.output / name).symlink_to(os.path.relpath(directory, args.output.resolve()),
                                     target_is_directory=True)
        report[name] = {"directory": str(directory), "frontend_sha256":
            hashlib.sha256((directory / "frontend.json").read_bytes()).hexdigest()}
        if name in replacements:
            report[name]["replaced_directory"] = replacements[name]
    (args.output / "overlay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(entries)} kernels -> {args.output}")


if __name__ == "__main__":
    main()
