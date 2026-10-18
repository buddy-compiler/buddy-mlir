#!/usr/bin/env python3
# ===- BuddyServerServiceIntegrationTest.py - Service HTTP checks -------===//
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

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def request(port: int, path: str, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2) as response:
        return response.status, json.load(response)


def wait_until_ready(process: subprocess.Popen, port: int):
    deadline = time.time() + 8
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(
                f"server exited with {process.returncode}: {stderr}"
            )
        try:
            status, payload = request(port, "/health")
            if status == 200 and payload.get("status") == "ok":
                assert payload["model"] == "fake_model", payload
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.03)
    raise AssertionError("server did not become ready")


def stop(process: subprocess.Popen):
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_case(server: str, rax: str, port: int, endpoint: str, body: dict):
    process = subprocess.Popen(
        [server, "--model", rax, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until_ready(process, port)
        status, payload = request(port, endpoint, body)
        assert status == 200, payload
        assert payload["model"] == "fake_model", payload
        return payload
    finally:
        stop(process)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: test.py SERVER RESIDENT_RAX EMBEDDING_RAX MASKED_RAX"
        )
    server, resident_rax, embedding_rax, masked_rax = sys.argv[1:]

    resident = run_case(
        server,
        resident_rax,
        18994,
        "/completion",
        {"prompt": "hello", "max_tokens": 1},
    )
    assert resident["content"] == "fake completion", resident

    embedding = run_case(
        server,
        embedding_rax,
        18995,
        "/v1/embeddings",
        {"model": "fake_model", "input": "hello"},
    )
    assert embedding["data"][0]["embedding"] == [0.25, 0.75], embedding

    masked = run_case(
        server,
        masked_rax,
        18996,
        "/v1/masked-lm",
        {"model": "fake_model", "input": "<mask>", "top_k": 1},
    )
    assert masked["predictions"][0]["tokens"][0]["token"] == "fake", masked

    print("buddy-server model service integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
