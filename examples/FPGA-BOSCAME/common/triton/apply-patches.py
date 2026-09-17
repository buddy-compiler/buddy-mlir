#!/usr/bin/env python3
"""Apply an overlapping patch series without changing existing local edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", type=Path)
    parser.add_argument("triton", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    resource = Path(__file__).resolve().parent
    lock = json.loads((resource / "toolchain-lock.json").read_text())
    patches = []
    files = set()
    for item in lock["patches"]:
        root = resource if item["origin"] == "common/triton" else args.plugin
        path = (root / item["path"]).resolve()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise SystemExit(f"Patch checksum mismatch: {path}")
        patches.append(path)
        files.update(re.findall(r"^\+\+\+ b/(.+)$", data.decode(), re.MULTILINE))

    # Later patches update some of the same lines as earlier ones; checking
    # each reverse patch in isolation incorrectly rejects an applied series.
    # Replay the locked series in a temporary tree and compare full contents.
    # A deployment commit can already contain all patches; replay from the
    # recorded pristine base instead of applying them again to that HEAD.
    base_commit = lock["triton"].get("base_commit", lock["triton"]["commit"])
    with tempfile.TemporaryDirectory(prefix="triton-patch-check-") as directory:
        temporary = Path(directory)
        for name in files:
            pristine = subprocess.check_output(
                ["git", "-C", str(args.triton), "show", f"{base_commit}:{name}"])
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pristine)

        def matches():
            return all((args.triton / name).is_file() and
                       (args.triton / name).read_bytes() == (temporary / name).read_bytes()
                       for name in files)

        applied = 0 if matches() else None
        for index, patch in enumerate(patches, 1):
            subprocess.run(["git", "apply", str(patch)], cwd=temporary, check=True)
            if matches():
                applied = index
        if applied is None:
            raise SystemExit("Triton patched files contain changes outside the locked patch series; existing edits were preserved.")
        if args.check and applied != len(patches):
            raise SystemExit(f"Only {applied}/{len(patches)} locked Triton patches are applied; run setup without --check.")
        for patch in patches[applied:]:
            subprocess.run(["git", "-C", str(args.triton), "apply", str(patch)], check=True)
            print(f"Applied: {patch.name}")
        print(f"Verified {len(patches)} locked Triton patches")


if __name__ == "__main__":
    main()
