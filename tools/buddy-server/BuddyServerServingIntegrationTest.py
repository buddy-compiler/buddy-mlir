#!/usr/bin/env python3
# ===- BuddyServerServingIntegrationTest.py - Serving wire contracts -----===//
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

"""Execute HTTP contract examples plus adversarial HTTP/SSE checks.

The fixture uses scripted logits with the production serving controller. These
checks provide no evidence about real model vocabularies, kernels or accuracy.
"""

import contextlib
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def strict_json(data):
    def reject_constant(value):
        raise AssertionError(f"non-JSON numeric constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, f"duplicate JSON member: {key}")
            result[key] = value
        return result

    return json.loads(
        data, parse_constant=reject_constant, object_pairs_hook=unique_object
    )


def subset(actual, expected, path=""):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), f"{path}: expected object: {actual!r}")
        for key, value in expected.items():
            check(key in actual, f"{path}: missing {key}: {actual!r}")
            subset(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, list):
        check(isinstance(actual, list), f"{path}: expected array: {actual!r}")
        check(
            len(actual) == len(expected),
            f"{path}: wrong array size: {actual!r}",
        )
        for index, value in enumerate(expected):
            subset(actual[index], value, f"{path}/{index}")
    else:
        check(type(actual) is type(expected), f"{path}: wrong type: {actual!r}")
        check(actual == expected, f"{path}: {actual!r} != {expected!r}")


def pointer(value, path):
    for part in path.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def check_type(value, name):
    types = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    check(type(value) is types[name], f"expected {name}, got {value!r}")


def media_type(value):
    return value.split(";", 1)[0].strip().lower()


def validate_contract(contract):
    check(isinstance(contract, dict), "contract must be an object")
    check(
        type(contract.get("format_version")) is int
        and contract["format_version"] == 1,
        "unsupported contract format_version",
    )
    examples = contract.get("examples")
    check(isinstance(examples, list) and examples, "examples must be nonempty")
    setup_fields = {
        "load_state",
        "throw_before_stream",
        "unsupported_parameters",
        "sampled_token_ids",
        "prefill_calls",
        "sample_calls",
        "template_applications",
        "expected_seed_policy",
        "expected_rendered_messages",
    }
    shared_expect_fields = {"status", "content_type", "note"}
    json_expect_fields = {
        "json_subset",
        "strings_contain",
        "required_json_types",
        "absent_json_pointers",
    }
    sse_expect_fields = {
        "sse_json_subsets",
        "done_sentinels",
        "eof",
        "consistent_event_fields",
        "required_event_types",
        "usage_events",
        "concatenated_content",
    }
    ids = set()
    for example in examples:
        check(isinstance(example, dict), "example must be an object")
        identifier = example.get("id")
        check(
            isinstance(identifier, str) and identifier,
            "example id must be a nonempty string",
        )
        check(identifier not in ids, f"duplicate contract id: {identifier}")
        ids.add(identifier)
        check(
            example.get("profile") in ("resident", "embedding"),
            f"{identifier}: unknown contract profile",
        )
        request = example.get("request")
        check(
            isinstance(request, dict),
            f"{identifier}: request must be an object",
        )
        check(
            isinstance(request.get("method"), str)
            and isinstance(request.get("path"), str),
            f"{identifier}: request requires method and path strings",
        )
        setup = example.get("setup", {})
        check(isinstance(setup, dict), f"{identifier}: setup must be an object")
        unknown_setup = setup.keys() - setup_fields
        check(
            not unknown_setup,
            f"{identifier}: unknown setup fields: {sorted(unknown_setup)}",
        )
        expected = example.get("expect")
        check(
            isinstance(expected, dict),
            f"{identifier}: expect must be an object",
        )
        check(
            type(expected.get("status")) is int,
            f"{identifier}: expect requires an integer status",
        )
        sse = "sse_json_subsets" in expected
        allowed = shared_expect_fields | (
            sse_expect_fields if sse else json_expect_fields
        )
        unknown_expect = expected.keys() - allowed
        check(
            not unknown_expect,
            f"{identifier}: unsupported expect fields: {sorted(unknown_expect)}",
        )
        if "eof" in expected:
            check(
                expected["eof"] is True,
                f"{identifier}: only eof:true is supported",
            )


def parse_sse(body):
    text = body.decode("utf-8", errors="strict").replace("\r\n", "\n")
    check(
        text.endswith("\n\n"),
        f"SSE event lacks terminating blank line: {text!r}",
    )
    events = []
    for block in text[:-2].split("\n\n"):
        lines = block.split("\n")
        check(
            all(line.startswith("data: ") for line in lines),
            f"invalid SSE frame: {block!r}",
        )
        data = "\n".join(line[6:] for line in lines)
        events.append(data if data == "[DONE]" else strict_json(data))
    return events


def content(events, chat):
    if chat:
        return "".join(
            choice["delta"].get("content", "")
            for event in events
            if isinstance(event, dict)
            for choice in event.get("choices", [])
        )
    return "".join(
        event.get("content", "") for event in events if isinstance(event, dict)
    )


class Server:
    def __init__(self, executable, rax, scratch, index, setup=None):
        self.setup = setup or {}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.marker = scratch / f"observations-{index}.txt"
        self.marker.write_text("")
        self.log = (scratch / f"server-{index}.log").open("w+")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("BUDDY_CONTRACT_")
        }
        env["BUDDY_CONTRACT_MARKER"] = str(self.marker)
        names = {
            "load_state": "BUDDY_CONTRACT_LOAD_STATE",
            "throw_before_stream": "BUDDY_CONTRACT_THROW_BEFORE_STREAM",
            "unsupported_parameters": "BUDDY_CONTRACT_UNSUPPORTED",
            "sampled_token_ids": "BUDDY_CONTRACT_TOKEN_SCRIPT",
        }
        for key, name in names.items():
            if key in self.setup:
                value = self.setup[key]
                env[name] = (
                    ",".join(map(str, value))
                    if isinstance(value, list)
                    else value
                )
        self.process = subprocess.Popen(
            [
                executable,
                "--model",
                rax,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            stdout=self.log,
            stderr=self.log,
            env=env,
        )

    def __enter__(self):
        expected = self.setup.get("load_state", "ok")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            check(self.process.poll() is None, self.logs())
            try:
                _, _, payload = self.request("GET", "/health")
                if strict_json(payload)["status"] == expected:
                    return self
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.02)
        raise AssertionError(f"server did not reach {expected}: {self.logs()}")

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.log.close()

    def logs(self):
        self.log.flush()
        return Path(self.log.name).read_text()

    def request(self, method, path, body=None):
        data = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        )
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=5
        )
        try:
            connection.request(
                method,
                path,
                body=data,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            headers = dict(response.getheaders())
            if media_type(response.getheader("Content-Type", "")) == (
                "text/event-stream"
            ):
                check(
                    response.getheader("Content-Length") is None,
                    "SSE must terminate by EOF",
                )
                check(
                    response.getheader("Transfer-Encoding") is None
                    and not response.chunked,
                    "SSE must use an unframed body terminated by EOF",
                )
                check(
                    response.will_close,
                    "SSE connection must close after its terminal event",
                )
            # Reading an unframed SSE body to completion requires real EOF.
            payload = response.read()
            return response.status, headers, payload
        finally:
            connection.close()

    def observations(self):
        return self.marker.read_text().splitlines()


def check_observations(server, setup):
    observations = server.observations()
    for key, event in (
        ("prefill_calls", "prefill"),
        ("sample_calls", "sample"),
        ("template_applications", "template"),
    ):
        if key in setup:
            check(
                observations.count(event) == setup[key],
                f"{key}: {observations}",
            )
    if "expected_seed_policy" in setup:
        policy = setup["expected_seed_policy"]
        event = (
            "seed:random"
            if policy["kind"] == "random"
            else f"seed:fixed:{policy['value']}"
        )
        check(
            event in observations,
            f"missing backend observation {event}: {observations}",
        )
    if "expected_rendered_messages" in setup:
        messages = []
        for event in observations:
            if event.startswith("render:"):
                _, role, encoded = event.split(":", 2)
                messages.append(
                    {"role": role, "content": bytes.fromhex(encoded).decode()}
                )
        check(
            messages == setup["expected_rendered_messages"],
            f"rendered messages: {messages}",
        )


def run_example(server, example):
    server.marker.write_text("")
    request = example["request"]
    expected = example["expect"]
    status, headers, body = server.request(
        request["method"], request["path"], request.get("json")
    )
    check(
        status == expected["status"],
        f"{example['id']}: HTTP {status}: {body!r}",
    )
    if "content_type" in expected:
        check(
            media_type(headers.get("Content-Type", ""))
            == media_type(expected["content_type"]),
            headers,
        )
    if "sse_json_subsets" in expected:
        all_events = parse_sse(body)
        check(
            all_events.count("[DONE]") == expected["done_sentinels"], all_events
        )
        if "[DONE]" in all_events:
            check(all_events[-1] == "[DONE]", all_events)
        events = [event for event in all_events if isinstance(event, dict)]
        if example["id"] == "S01":
            subset(events[-1], expected["sse_json_subsets"][0])
            for event in events[:-1]:
                check(event["content"] == "" and event["stop"] is False, event)
        else:
            subset(events, expected["sse_json_subsets"])
        for field in expected.get("consistent_event_fields", []):
            check(
                all(event[field] == events[0][field] for event in events),
                events,
            )
        for field, typename in expected.get("required_event_types", {}).items():
            for event in events:
                check_type(event[field], typename)
        if "usage_events" in expected:
            check(
                sum("usage" in event for event in events)
                == expected["usage_events"],
                events,
            )
        if "concatenated_content" in expected:
            check(
                content(events, request["path"] == "/v1/chat/completions")
                == expected["concatenated_content"],
                events,
            )
    else:
        payload = strict_json(body)
        subset(payload, expected.get("json_subset", {}))
        for path, text in expected.get("strings_contain", {}).items():
            check(text in pointer(payload, path), payload)
        for path, typename in expected.get("required_json_types", {}).items():
            check_type(pointer(payload, path), typename)
        for path in expected.get("absent_json_pointers", []):
            try:
                pointer(payload, path)
            except (KeyError, IndexError):
                continue
            raise AssertionError(f"{path} should be absent: {payload}")
    check_observations(server, example.get("setup", {}))


def completion(server, prompt="A", chat=False, stream=False, **options):
    body = (
        {"messages": [{"role": "user", "content": prompt}]}
        if chat
        else {"prompt": prompt}
    )
    body.update(options)
    body["stream"] = stream
    return server.request(
        "POST", "/v1/chat/completions" if chat else "/completion", body
    )


def check_adversarial(server):
    for chat in (False, True):
        for budget in (0, 1, 2, 3, -1):
            status, _, body = completion(server, chat=chat, n_predict=budget)
            check(status == 200, body)
            normal = strict_json(body)
            status, _, body = completion(
                server,
                chat=chat,
                stream=True,
                n_predict=budget,
                stream_options={"include_usage": True},
            )
            check(status == 200, body)
            events = parse_sse(body)
            expected_content = (
                normal["choices"][0]["message"]["content"]
                if chat
                else normal["content"]
            )
            check(content(events, chat) == expected_content, events)
            usage = next(
                event["usage"]
                for event in events
                if isinstance(event, dict) and "usage" in event
            )
            check(usage == normal["usage"], events)

        status, headers, body = completion(server, "throw-prefill", chat, True)
        check(
            status == 500
            and "text/event-stream" not in headers.get("Content-Type", ""),
            (status, body),
        )
        subset(strict_json(body), {"error": {"code": 500}})
        status, _, body = completion(server, "throw-midstream", chat, True)
        check(status == 200, body)
        events = parse_sse(body)
        check("[DONE]" not in events, events)
        subset(events[-1], {"error": {"code": 500}})
        check(
            "fixture midstream failure" in events[-1]["error"]["message"],
            events,
        )
        check(content(events, chat) == "A", events)
        # Reuse the same locked session after both exception paths.
        status, _, body = completion(server)
        check(status == 200 and strict_json(body)["content"] == "AB", body)

        status, _, body = completion(
            server,
            "final-delta",
            chat,
            True,
            stream_options={"include_usage": True},
        )
        check(status == 200, body)
        events = parse_sse(body)
        check(content(events, chat) == "tail", events)
        check(events.count("[DONE]") == int(chat), events)
        if chat:
            check(len(events) == 5, events)
            subset(
                events[1],
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "tail"},
                            "finish_reason": None,
                        }
                    ]
                },
            )
            subset(
                events[2],
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "length"}
                    ]
                },
            )
        else:
            check(len(events) == 1 and events[0]["stop"] is True, events)

    # A held suffix must be flushed before normal length completion.
    status, _, body = completion(server, stream=True, n_predict=2, stop="ABZ")
    check(status == 200 and content(parse_sse(body), False) == "AB", body)


def check_disconnect(server, before_first):
    server.marker.write_text("")
    prompt = "disconnect-before-first" if before_first else "disconnect"
    body = json.dumps(
        {"prompt": prompt, "stream": True, "n_predict": 100}
    ).encode()
    client = socket.create_connection(("127.0.0.1", server.port), timeout=3)
    try:
        client.sendall(
            b"POST /completion HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        if before_first:
            deadline = time.monotonic() + 2
            while (
                "prefill" not in server.observations()
                and time.monotonic() < deadline
            ):
                time.sleep(0.002)
            check(
                "prefill" in server.observations(),
                "backend did not enter prefill",
            )
            # Establish response delivery, then reset while reading headers,
            # before consuming any SSE event. Network proxies may buffer a
            # socket closed before receiving any response bytes.
            check(client.recv(1) == b"H", "server did not begin HTTP headers")
        else:
            received = b""
            while (
                b"\r\n\r\n" not in received
                or b"\n\n" not in received.split(b"\r\n\r\n", 1)[1]
            ):
                part = client.recv(4096)
                check(part, "SSE connection closed before the first event")
                received += part
            check(received.startswith(b"HTTP/1.1 200"), received)
        # RST makes the next transport write observe the disconnected client.
        client.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
    finally:
        client.close()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        events = server.observations()
        if any(event.startswith("return-cancelled:") for event in events):
            break
        time.sleep(0.005)
    else:
        raise AssertionError(
            f"backend never observed cancellation: {server.observations()}"
        )
    cancelled = next(
        event for event in events if event.startswith("return-cancelled:")
    )
    check(
        int(cancelled.split(":")[1]) < 15,
        f"generation ran to context limit: {events}",
    )
    check("callback-false" in events, events)
    check(
        "reset-position" in events[events.index("callback-false") + 1 :],
        f"session was not reset after cancellation: {events}",
    )
    status, _, body = completion(server)
    check(status == 200 and strict_json(body)["content"] == "AB", body)


def main():
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: test.py SERVER RESIDENT_RAX EMBEDDING_RAX CONTRACT_JSON SCRATCHDIR"
        )
    executable, resident, embedding, contract_path, scratch_path = sys.argv[1:]
    contract = strict_json(Path(contract_path).read_text())
    validate_contract(contract)
    scratch = Path(scratch_path)
    scratch.mkdir(parents=True, exist_ok=True)
    groups = {}
    setup_fields = {
        "load_state",
        "throw_before_stream",
        "unsupported_parameters",
        "sampled_token_ids",
    }
    for example in contract["examples"]:
        setup = {
            key: value
            for key, value in example.get("setup", {}).items()
            if key in setup_fields
            and not (key == "load_state" and value == "ok")
        }
        key = (example["profile"], json.dumps(setup, sort_keys=True))
        groups.setdefault(key, []).append(example)
    executed = 0
    for index, ((profile, serialized_setup), examples) in enumerate(
        groups.items()
    ):
        rax = resident if profile == "resident" else embedding
        with Server(
            executable, rax, scratch, index, json.loads(serialized_setup)
        ) as server:
            for example in examples:
                try:
                    run_example(server, example)
                except Exception as error:
                    raise AssertionError(
                        f"contract {example['id']}: {error}\n{server.logs()}"
                    ) from error
                executed += 1
            if profile == "resident" and serialized_setup == "{}":
                check_adversarial(server)
                check_disconnect(server, before_first=False)
                check_disconnect(server, before_first=True)
    check(
        executed == len(contract["examples"]),
        f"only executed {executed} examples",
    )
    # Exercise the shared stop matcher across both raw token/UTF-8 boundaries.
    with contextlib.ExitStack() as stack:
        for offset, (script, stop, limit) in enumerate(
            (("4,5,6,1,0", "中", 3), ("1,2,1,2,1,0", "ABA", 4))
        ):
            server = stack.enter_context(
                Server(
                    executable,
                    resident,
                    scratch,
                    100 + offset,
                    {"sampled_token_ids": script.split(",")},
                )
            )
            status, _, body = completion(
                server, stream=True, stop=stop, n_predict=limit
            )
            check(status == 200, body)
            events = parse_sse(body)
            check(content(events, False) == "", events)
            subset(
                events[-1],
                {
                    "stop": True,
                    "stop_reason": "stop",
                    "usage": {"completion_tokens": 3},
                },
            )
    print(
        f"buddy-server serving integration tests passed ({executed} contract examples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
