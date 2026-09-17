#!/usr/bin/env python3
"""Fetch the pinned official Qwen3-0.6B checkpoint weights.

Separate from fetch_assets.py: that tool handles the small tokenizer/config
resources. This one handles the ~1.5 GB safetensors file and therefore needs
resumable transfer and post-download size/SHA256 verification.

Provenance rules:
  * the revision must be an immutable 40-char commit, never a branch name;
  * an already-complete file that matches the recorded size and SHA256 is
    reused instead of re-downloaded;
  * an existing file whose content differs is never overwritten silently;
  * no ETag/CDN header is ever recorded as if it were a local SHA256.

Writes are confined to the caller's output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHT_FILE = "model.safetensors"
# From the documented HEAD response (Content-Length). Verified after download;
# a mismatch is a hard error, not a warning.
EXPECTED_BYTES = 1503300328

CHUNK = 8 * 1024 * 1024


def sha256_of(path, progress=None):
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
            if progress:
                progress(total)
    return digest.hexdigest()


def _content_length(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(output, repository=REPOSITORY, revision=REVISION,
             expected_bytes=EXPECTED_BYTES, expected_sha256=None,
             name=WEIGHT_FILE, quiet=False):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("use an immutable 40-character commit, not main")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/" + urllib.parse.quote(repository, safe="/")
    url = f"{base}/resolve/{revision}/{name}"
    final = output / name
    partial = output / (name + ".partial")

    if final.exists():
        actual_bytes = final.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            raise ValueError(
                f"existing {final} has {actual_bytes} bytes, expected "
                f"{expected_bytes}; refusing to reuse or overwrite")
        actual_sha = sha256_of(final)
        if expected_sha256 and actual_sha != expected_sha256:
            raise ValueError(
                f"existing {final} sha256 {actual_sha} != recorded "
                f"{expected_sha256}; refusing to reuse or overwrite")
        return {"file": name, "path": str(final), "bytes": actual_bytes,
                "sha256": actual_sha, "url": url, "repository": repository,
                "revision": revision, "status": "reused",
                "content_length_checked": False}

    resumed_from = partial.stat().st_size if partial.exists() else 0
    if resumed_from:
        if expected_bytes is not None and resumed_from > expected_bytes:
            raise ValueError("partial file larger than expected; remove it")
        if not quiet:
            print(f"resuming {name} at {resumed_from} bytes")
    headers = {"Range": f"bytes={resumed_from}-"} if resumed_from else {}
    request = urllib.request.Request(url, headers=headers)
    written = resumed_from
    # Reported size of the *full* object, cross-checked before trusting it.
    remote_total = None
    with urllib.request.urlopen(request, timeout=120) as response:
        if resumed_from and response.status != 206:
            raise RuntimeError(
                "server ignored Range request; remove the partial file first")
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                remote_total = int(content_range.rsplit("/", 1)[1])
        else:
            remote_total = int(response.headers.get("Content-Length", 0)) or None
        mode = "ab" if resumed_from else "wb"
        with open(partial, mode) as handle:
            last_report = written
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if not quiet and written - last_report >= 64 * 1024 * 1024:
                    last_report = written
                    print(f"  {name}: {written} bytes", flush=True)

    if remote_total is not None and expected_bytes is not None \
            and remote_total != expected_bytes:
        raise ValueError(
            f"remote size {remote_total} != expected {expected_bytes}")
    if expected_bytes is not None and written != expected_bytes:
        raise ValueError(
            f"downloaded {written} bytes, expected {expected_bytes}; "
            f"partial file kept for resume")
    actual_sha = sha256_of(partial)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"sha256 {actual_sha} != recorded {expected_sha256}")
    partial.replace(final)
    return {"file": name, "path": str(final), "bytes": written,
            "sha256": actual_sha, "url": url, "repository": repository,
            "revision": revision, "status": "downloaded",
            "content_length_checked": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--name", default=WEIGHT_FILE)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_BYTES)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="write provenance record here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    record = download(args.output, args.repository, args.revision,
                      args.expected_bytes, args.expected_sha256, args.name,
                      args.quiet)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
