#!/usr/bin/env python3
# ===- BuddyServerCapabilityIntegrationTest.py - Capability checks ------===//
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


def request(port: int, path: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    return urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )


def wait_until_ready(process: subprocess.Popen, port: int):
    deadline = time.time() + 8
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(
                f"server exited with {process.returncode}: {stderr}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if (
                    response.status == 200
                    and json.load(response).get("status") == "ok"
                ):
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.03)
    raise AssertionError("server did not become ready")


def check_rejected(port: int, path: str, body: dict):
    try:
        urllib.request.urlopen(request(port, path, body), timeout=2)
        raise AssertionError(
            f"streaming request unexpectedly succeeded for {path}"
        )
    except urllib.error.HTTPError as error:
        assert error.code == 400, error
        assert error.headers.get_content_type() == "application/json", (
            error.headers
        )
        payload = json.load(error)
        assert payload["error"]["type"] == "unsupported_endpoint", payload
        assert payload["error"]["code"] == 400, payload
        assert "streaming is not supported" in payload["error"]["message"], (
            payload
        )


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

        with urllib.request.urlopen(
            request(port, "/completion", {"prompt": "hello"}), timeout=2
        ) as response:
            assert response.status == 200, response.status
            assert json.load(response)["content"] == "fake completion"

        check_rejected(port, "/completion", {"prompt": "hello", "stream": True})
        check_rejected(
            port,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    finally:
        stop(process)

    print("buddy-server streaming capability integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
