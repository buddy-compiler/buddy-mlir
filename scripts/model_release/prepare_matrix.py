#!/usr/bin/env python3
# ===- prepare_matrix.py - Build the publish matrix from a release tag ----===//
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
# ===----------------------------------------------------------------------===//
#
# Reads models/publish.json, filters it, and prints a JSON object that the
# publish-models workflow consumes:
#
#   {
#     "version": "0.0.6",
#     "hf_tag": "v0.0.6",
#     "include": [ {family, repo, arch, snapshot, variants:[...]}, ... ]
#   }
#
# The same release tag drives both the CLI version and the Hugging Face tag,
# so a single `--tag release/v0.0.6` keeps them in sync.
#
# ===----------------------------------------------------------------------===//

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def tag_to_version(tag: str) -> str:
    """Derive a version from a release tag, branch, or commit hash.

    `release/v0.0.6` / `nightly/v0.0.7.dev20260826` -> version without `v`.
    A raw commit hash is shortened to 7 characters so it is usable as both a
    package version and a Hugging Face tag during manual testing.
    """
    tag = tag.strip()
    for prefix in ("refs/tags/", "release/", "nightly/"):
        if tag.startswith(prefix):
            tag = tag[len(prefix) :]
            break
    if tag.startswith("v"):
        tag = tag[1:]
    if not tag:
        raise ValueError("could not derive a version from the tag")
    if re.fullmatch(r"[0-9a-f]{40}", tag):
        tag = tag[:7]
    # Keep the value usable as a git tag / HF tag for ad-hoc refs.
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-.")
    if not tag:
        raise ValueError("could not derive a version from the tag")
    return tag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="buddy-mlir git tag/ref")
    ap.add_argument(
        "--registry",
        type=Path,
        default=Path("models/publish.json"),
        help="Publish registry JSON (default: models/publish.json)",
    )
    ap.add_argument(
        "--families",
        default="",
        help="Comma-separated model families to build (empty = all enabled)",
    )
    ap.add_argument(
        "--namespace",
        default="",
        help="Override the Hugging Face namespace from the registry",
    )
    ap.add_argument(
        "--arch",
        default="",
        help="Override the target arch label from the registry",
    )
    ap.add_argument(
        "--output",
        default="-",
        help="Output path ('-' for stdout)",
    )
    args = ap.parse_args()

    try:
        version = tag_to_version(args.tag)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    hf_tag = f"v{version}"

    registry = json.loads(args.registry.read_text())
    namespace = args.namespace or registry["namespace"]
    arch = args.arch or registry.get("arch", "riscv64")

    requested = {f.strip() for f in args.families.split(",") if f.strip()}
    known = {t["family"] for t in registry["targets"]}
    unknown = requested - known
    if unknown:
        print(
            f"error: unknown model families: {', '.join(sorted(unknown))}",
            file=sys.stderr,
        )
        return 1
    enabled = {t["family"] for t in registry["targets"] if t.get("enabled")}

    selected = []
    for target in registry["targets"]:
        family = target["family"]
        if requested:
            if family not in requested:
                continue
        elif family not in enabled:
            continue
        if not target.get("variants"):
            print(
                f"warning: {family!r} has no variants; skipping",
                file=sys.stderr,
            )
            continue
        selected.append(
            {
                "family": family,
                "repo": f"{namespace}/{target['repo']}",
                "hf_repo": target["repo"],
                "arch": arch,
                "snapshot": target.get("snapshot"),
                "variants": [
                    {"variant": v, "spec": spec}
                    for v, spec in target["variants"].items()
                ],
            }
        )

    if not selected:
        print("error: no model targets selected", file=sys.stderr)
        return 1

    payload = {"version": version, "hf_tag": hf_tag, "include": selected}
    text = json.dumps(payload, separators=(",", ":"))
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
