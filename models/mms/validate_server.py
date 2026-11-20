#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.

"""Validate the production MMS HTTP plugin using validate_accuracy.py outputs."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


def request(url, payload=None, path="/v1/audio/vocoder", timeout=300):
    data = (
        None
        if payload is None
        else (
            payload.encode()
            if isinstance(payload, str)
            else json.dumps(payload, allow_nan=False).encode()
        )
    )
    req = Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


@contextmanager
def server(binary, package, directory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ, BUDDY_RAX_PAYLOAD_DIR=str(directory / "payload"))
    env.pop("LD_LIBRARY_PATH", None)
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(binary),
                "--model",
                str(package),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
            env=env,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited: {process.returncode}; see {log.name}"
                    )
                try:
                    code, health = request(url, path="/health")
                    if code == 200 and health.get("status") == "ok":
                        break
                    if health.get("status") == "error":
                        raise RuntimeError(f"model load failed: {health}")
                except (URLError, ConnectionError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"model did not become ready; see {log.name}"
                )
            yield url, health
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buddy-server", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300,
        help="Seconds per HTTP request, including time queued behind serialized inference",
    )
    args = parser.parse_args()
    if not np.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = output / "relocated" / "mms.rax"
    package.parent.mkdir(exist_ok=True)
    shutil.copy2(args.package, package)
    accuracy = json.loads((args.validation_dir / "accuracy.json").read_text())
    cases = []
    # Only use genuine VITS windows with the independently recorded reference.
    for case in accuracy["cases"]:
        name = case["name"]
        latents = np.fromfile(
            args.validation_dir / f"{name}.f32", dtype=np.float32
        )
        expected, rate = sf.read(
            args.validation_dir / f"{name}.reference.wav", dtype="float32"
        )
        cases.append((name, latents, expected, rate))
    if not cases:
        raise ValueError("no real validation windows were provided")
    observations = []
    with server(args.buddy_server.resolve(), package, output) as (url, health):
        model = health["model"]
        send = partial(request, url, timeout=args.request_timeout)

        def compare(case):
            name, latents, expected, rate = case
            started = time.monotonic()
            code, body = send({"model": model, "latents": latents.tolist()})
            wall_ms = (time.monotonic() - started) * 1000
            assert code == 200, (code, body)
            assert body["model"] == model
            assert (
                body["sample_rate"] == rate == 16000 and body["channels"] == 1
            )
            assert (
                np.isfinite(body["inference_ms"]) and body["inference_ms"] >= 0
            )
            actual = np.asarray(body["audio"], dtype=np.float32)
            assert actual.shape == expected.shape and np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            print(
                f"PASS {name}: inference={body['inference_ms']:.1f} ms, HTTP={wall_ms:.1f} ms",
                flush=True,
            )
            return {
                "name": name,
                "inference_ms": body["inference_ms"],
                "http_ms": wall_ms,
                "samples": actual.size,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
            }

        observations.extend(compare(case) for case in cases)
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(compare, cases[:4]))
        repeated = [compare(cases[i % len(cases)]) for i in range(12)]
        payload = {"model": model, "latents": cases[0][1].tolist()}
        invalid = {
            "missing": {},
            "empty": {"latents": []},
            "truncated": dict(payload, latents=payload["latents"][:-1]),
            "extra": dict(
                payload, latents=payload["latents"] + payload["latents"][:1]
            ),
            "nested": dict(payload, latents=[payload["latents"]]),
            "string": dict(payload, latents=["invalid"]),
            "boolean": dict(payload, latents=[True]),
            "float32_overflow": dict(payload, latents=[1e39]),
            "nan": '{"latents":[NaN]}',
            "infinity": '{"latents":[1e999]}',
            "wrong_model": dict(payload, model=model + "-not-loaded"),
            "model_type": dict(payload, model=1),
            "stream": dict(payload, stream=True),
            "text": dict(payload, input="unsupported text-to-speech input"),
            "root_type": "[]",
        }
        for name, data in invalid.items():
            code, body = send(data)
            assert code == 400 and "error" in body, (name, code, body)
        for endpoint in (
            "/completion",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
        ):
            code, body = send({}, endpoint)
            assert code == 400 and "error" in body, (endpoint, code, body)
        # A failed request must not poison the resident model.
        compare(cases[-1])
        code, final_health = send(path="/health")
        assert code == 200 and final_health["status"] == "ok"
    report = {
        "request_timeout_seconds": args.request_timeout,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime default"),
        "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "reference": "validate_accuracy.py: real LibriSpeech text / VITS latent windows",
        "cases": observations,
        "concurrent_requests": len(concurrent),
        "repeated_requests": len(repeated),
        "invalid_requests": list(invalid),
        "unsupported_endpoints": 3,
        "recovery_after_errors": "passed",
        "relocated_package_without_ld_library_path": "passed",
    }
    (output / "server-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
