#!/usr/bin/env python3
"""Archive compact A/B evidence only after UART, build, and uploaded image agree."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dequant_ab import inspect_case, sha
from check_dequant_ab import parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.build / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runner = json.loads((args.run / "result.json").read_text())
    binary = args.build / manifest["image"]["path"]
    digest = sha(binary)
    if not (digest == manifest["image"]["sha256"] == runner["sha256"]):
        raise ValueError("image hash disagrees between binary, A/B build manifest, and runner")
    if runner["status"] != "OK" or runner.get("ddr_readback_matches") is not True:
        raise ValueError("upload/readback did not succeed")
    if runner.get("completion_marker_seen") is not True:
        raise ValueError("runner did not capture successful completion")
    log = args.run / "uart.raw.log"
    if log.stat().st_size != runner.get("uart_bytes"):
        raise ValueError("UART byte count differs from runner result")
    result = parse(log.read_text(errors="replace"), platform="fpga", manifest=manifest)
    result.update(run_id=args.run.name, image_sha256=digest,
                  log_sha256=sha(log), manifest_sha256=sha(manifest_path),
                  ddr_readback_matches=True)
    # The first benchmark predates mandatory native-manifest snapshots. Validate
    # current case inputs against recorded immutable snapshot hashes and archive
    # their native manifests as supplemental provenance without rewriting the
    # original A/B build manifest or image.
    native_manifests = {}
    for variant, entry in manifest["variants"].items():
        source = Path(entry["source"])
        inspect_case(source)
        for relative, expected in entry["inputs_sha256"].items():
            if sha(source / relative) != expected:
                raise ValueError(f"source changed since A/B build: {variant}/{relative}")
            snapshot = args.build / "inputs" / variant / relative
            if sha(snapshot) != expected:
                raise ValueError(f"snapshot corrupted: {variant}/{relative}")
        native_manifests[variant] = source / "nr/manifest.json"
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be empty to preserve archived evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    for source, name in ((log, "uart.raw.log"), (manifest_path, "build-manifest.json"),
                         (args.run / "result.json", "runner-result.json"),
                         (args.build / "elf-audit.json", "elf-audit.json")):
        shutil.copy2(source, args.output / name)
    for variant, native_manifest in native_manifests.items():
        shutil.copy2(native_manifest, args.output / f"{variant}-native-manifest.json")
    result["source_hashes_and_abi_checked"] = True
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "shape": result["shape"], "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
