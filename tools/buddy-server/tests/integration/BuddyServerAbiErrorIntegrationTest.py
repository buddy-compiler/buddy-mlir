#!/usr/bin/env python3
# ===- BuddyServerAbiErrorIntegrationTest.py - ABI HTTP errors ----------===//
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

CASES = (
    ("invalid_argument", 400, "bad_request", "forced invalid_argument"),
    (
        "typed_invalid_argument",
        400,
        "bad_request",
        "forced typed_invalid_argument",
    ),
    ("not_supported", 400, "unsupported_endpoint", "forced not_supported"),
    ("model_not_ready", 503, "model_not_ready", "forced model_not_ready"),
    (
        "resource_exhausted",
        429,
        "resource_exhausted",
        "forced resource_exhausted",
    ),
    (
        "typed_resource_exhausted",
        429,
        "resource_exhausted",
        "forced typed_resource_exhausted",
    ),
    ("cancelled", 499, "cancelled", "forced cancelled"),
    (
        "deadline_exceeded",
        504,
        "deadline_exceeded",
        "forced deadline_exceeded",
    ),
    ("model_error", 500, "model_error", "forced model_error"),
    ("typed_model_error", 500, "model_error", "forced typed_model_error"),
    ("internal_error", 500, "internal_error", "forced internal_error"),
    ("unknown", 500, "internal_error", "resident model invocation failed"),
)


def make_request(port: int, prompt: str, stream: bool):
    body = json.dumps({"prompt": prompt, "stream": stream}).encode("utf-8")
    return urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )


def health(port: int):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/health", timeout=2
    ) as response:
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
            status, payload = health(port)
            if status == 200 and payload.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.03)
    raise AssertionError("server did not become ready")


def assert_error(payload: dict, message: str, status: int, error_type: str):
    error = payload["error"]
    assert error["message"] == f"buddy-server: {message}", payload
    assert error["type"] == error_type, payload
    assert error["code"] == status, payload


def check_json_error(
    port: int, prompt: str, status: int, error_type: str, message: str
):
    try:
        urllib.request.urlopen(make_request(port, prompt, False), timeout=2)
        raise AssertionError(f"request unexpectedly succeeded for {prompt}")
    except urllib.error.HTTPError as error:
        assert error.code == status, error
        assert error.headers.get_content_type() == "application/json", (
            error.headers
        )
        assert_error(json.load(error), message, status, error_type)


def check_sse_error(
    port: int, prompt: str, status: int, error_type: str, message: str
):
    with urllib.request.urlopen(
        make_request(port, prompt, True), timeout=2
    ) as response:
        assert response.status == 200, response.status
        assert response.headers.get_content_type() == "text/event-stream"
        body = response.read().decode("utf-8")

    assert "HTTP/1.1" not in body, body
    assert "[DONE]" not in body, body
    blocks = [block for block in body.split("\n\n") if block]
    assert len(blocks) == 2, blocks
    assert blocks[0].startswith("data: "), blocks
    partial = json.loads(blocks[0][6:])
    assert partial["content"] == "partial", partial
    lines = blocks[1].splitlines()
    assert lines[0] == "event: error", lines
    assert lines[1].startswith("data: "), lines
    assert_error(json.loads(lines[1][6:]), message, status, error_type)


def stop(process: subprocess.Popen):
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test.py SERVER MODEL_RAX PORT")
    server, model_rax, port_value = sys.argv[1:]
    port = int(port_value)
    process = subprocess.Popen(
        [
            server,
            "--model",
            model_rax,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until_ready(process, port)
        for prompt, status, error_type, message in CASES:
            check_json_error(port, prompt, status, error_type, message)
            check_sse_error(port, prompt, status, error_type, message)
    finally:
        stop(process)

    print("buddy-server ABI HTTP error integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
