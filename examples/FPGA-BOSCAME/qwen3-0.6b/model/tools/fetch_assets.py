#!/usr/bin/env python3
"""Fetch pinned official tokenizer/config resources, without downloading weights.

The checkpoint itself is deliberately separate. Resources are never taken from
references/. Writes are confined to the caller's output directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = ("config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE")


def fetch(output, revision=REVISION, repository=REPOSITORY):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("use an immutable 40-character commit, not main")
    output.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/" + urllib.parse.quote(repository, safe="/")
    def download(name):
        url = f"{base}/resolve/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError("unexpectedly large metadata file: " + name)
        path = output / name
        # An existing different resource must not silently change provenance.
        if path.exists() and path.read_bytes() != data:
            raise ValueError("existing resource differs; use a new output directory: " + str(path))
        temporary = output / (name + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
        return name, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": url}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = dict(pool.map(download, FILES))
    manifest = {"repository": repository, "revision": revision, "files": records,
                "weights_downloaded": False}
    (output / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--repository", default=REPOSITORY)
    args = parser.parse_args()
    result = fetch(args.output, args.revision, args.repository)
    print(json.dumps(result, indent=2))
