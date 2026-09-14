#!/usr/bin/env python3
# ===- BuddyServerTranscriptionIntegrationTest.py - HTTP integration ----===//
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
import os
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
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_for_health(
    port: int, expected: str, process=None, timeout: float = 8.0
):
    deadline = time.time() + timeout
    observed = []
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(
                f"server exited with {process.returncode}: {stderr}"
            )
        try:
            status, payload = request(port, "/health")
            observed.append(payload.get("status"))
            if status == 200 and payload.get("status") == expected:
                return payload, observed
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.03)
    raise AssertionError(f"health never became {expected}; observed={observed}")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def launch(server: str, rax: str, port: int, extra_env: dict):
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.Popen(
        [server, "--model", rax, "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: test.py SERVER RAX PORT PAYLOAD_CACHE")
    server, rax, port_text, payload_cache = sys.argv[1:]
    port = int(port_text)
    common_env = {"BUDDY_RAX_PAYLOAD_DIR": payload_cache}

    process = launch(
        server,
        rax,
        port,
        {**common_env, "BUDDY_FAKE_TRANSCRIPTION_LOAD_DELAY_MS": "500"},
    )
    try:
        _, observed = wait_for_health(port, "loading", process)
        status, payload = request(
            port,
            "/v1/audio/transcriptions",
            {"model": "fake_transcription", "file": "/tmp/sample.wav"},
        )
        assert status == 503, payload
        health, ready_observed = wait_for_health(port, "ok", process)
        assert health["model"] == "fake_transcription", health
        assert "loading" in observed + ready_observed

        status, payload = request(
            port,
            "/v1/audio/transcriptions",
            {
                "model": "fake_transcription",
                "file": "file:/tmp/sample.wav",
                "max_tokens": 1,
            },
        )
        assert status == 200, payload
        assert payload["text"] == "fake transcription", payload
        assert payload["generated_tokens"] == 1, payload

        status, payload = request(
            port,
            "/v1/audio/transcriptions",
            {"model": "wrong", "file": "/tmp/sample.wav"},
        )
        assert status == 400 and payload["error"]["type"] == "model_not_found"
        status, payload = request(port, "/completion", {"prompt": "x"})
        for endpoint, body in (
            ("/v1/chat/completions", {"messages": []}),
            ("/tokenize", {"content": "x"}),
            ("/v1/embeddings", {"input": "x"}),
            ("/v1/masked-lm", {"input": "x"}),
        ):
            other_status, other_payload = request(port, endpoint, body)
            assert other_status == 400, other_payload
            assert other_payload["error"]["type"] == "unsupported_endpoint"
        assert (
            status == 400 and payload["error"]["type"] == "unsupported_endpoint"
        )
    finally:
        stop(process)

    process = launch(
        server,
        rax,
        port + 1,
        {**common_env, "BUDDY_FAKE_TRANSCRIPTION_LOAD_ERROR": "1"},
    )
    try:
        health, _ = wait_for_health(port + 1, "error", process)
        assert "fake load failure" in health["message"], health
    finally:
        stop(process)

    print("buddy-server transcription integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
