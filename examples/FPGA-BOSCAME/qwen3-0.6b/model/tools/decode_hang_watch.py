#!/usr/bin/env python3
"""Separate framed NH observations from the original RA UART byte stream.

This tool interprets diagnostic observations only. It does not identify a hang
root cause or accept numerical results. Even an unchanged, sequence-consistent
record may be stale on this non-coherent platform.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


BEGIN = b"\x1eNRWATCH1\n"
END = b"\x1f"
HEADER_FIELDS = {
    "sample", "nh_cycles", "stable", "seq", "seq_after", "count", "consumed",
    "pending", "ra_console_wait", "dropped", "waits",
}
STABLE_FIELDS = {"phase", "position", "detail", "target_phase", "target_call"}
MEMREF_FIELDS = {
    "arg", "valid", "descriptor", "aligned", "offset", "rows", "cols",
    "stride0", "stride1",
}
FIELD = re.compile(r"([a-z][a-z0-9_]*)=([0-9a-fA-F]{1,16})")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(line, prefix, required):
    _require(line.startswith(prefix + " "), "unexpected diagnostic row")
    tokens = line[len(prefix) + 1:].split(" ")
    values, raw_hex = {}, {}
    for token in tokens:
        match = FIELD.fullmatch(token)
        _require(match is not None, "invalid hexadecimal diagnostic field: " + token)
        key, value = match.groups()
        _require(key not in values, "duplicate diagnostic field: " + key)
        values[key] = int(value, 16)
        raw_hex[key] = value
    _require(required <= values.keys(), "missing diagnostic fields: " +
             ", ".join(sorted(required - values.keys())))
    return values, raw_hex


def _parse_frame(payload, start, end):
    _require(b"\x1e" not in payload, "nested or malformed NH frame")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("NH frame contains non-ASCII data") from exc
    _require(text.startswith("\r\n") and text.endswith("\r\n"),
             "NH frame must contain complete CRLF rows")
    rows = text[2:-2].split("\r\n")
    header, raw_hex = _fields(rows[0], "[nh-watch]", HEADER_FIELDS)
    _require(header["stable"] in (0, 1), "stable must be zero or one")
    _require(header["ra_console_wait"] in (0, 1),
             "ra_console_wait must be zero or one")
    _require(all(header[k] <= 0xffffffff for k in ("count", "consumed", "pending")),
             "console counters must fit uint32")
    if header["stable"]:
        _require(STABLE_FIELDS <= header.keys(), "stable frame lacks phase fields")
    else:
        _require(not STABLE_FIELDS.intersection(header),
                 "unstable frame must not claim a sampled phase")
    memrefs = []
    for row in rows[1:]:
        values, hex_values = _fields(row, "[nh-watch-memref]", MEMREF_FIELDS)
        _require(values["arg"] in (0, 1, 2) and values["valid"] in (0, 1),
                 "invalid memref operand index or valid flag")
        values["raw_hex"] = hex_values
        memrefs.append(values)
    expect_memrefs = header["stable"] and header["target_phase"] != 0
    _require([m["arg"] for m in memrefs] == ([0, 1, 2] if expect_memrefs else []),
             "frame has missing, duplicated or unexpected memref operands")
    header.update(raw_hex=raw_hex, memrefs=memrefs,
                  raw_byte_begin=start, raw_byte_end=end)
    return header


def _partial_begin_length(tail):
    """A marker cut off by UART capture is not part of the recovered RA log."""
    for size in range(min(len(tail), len(BEGIN) - 1), 0, -1):
        if tail.endswith(BEGIN[:size]):
            return size
    return 0


def decode_bytes(raw, *, console_capacity=65536):
    """Return (RA bytes, JSON-compatible observations), rejecting bad full frames.

    Bytes outside reserved markers are preserved exactly, including NUL/UTF-8.
    A suffix matching an incomplete BEGIN is conservatively treated as a
    truncated frame; that ambiguous tail is identified in the report.
    """
    _require(type(console_capacity) is int and 64 <= console_capacity <= (1 << 30)
             and not console_capacity & (console_capacity - 1),
             'invalid console capacity')
    clean, frames = bytearray(), []
    cursor = 0
    truncated_offset = None
    while cursor < len(raw):
        start = raw.find(BEGIN, cursor)
        if start == -1:
            tail = raw[cursor:]
            partial = _partial_begin_length(tail)
            if partial:
                clean.extend(tail[:-partial])
                truncated_offset = len(raw) - partial
            else:
                clean.extend(tail)
            break
        clean.extend(raw[cursor:start])
        payload_start = start + len(BEGIN)
        end = raw.find(END, payload_start)
        nested = raw.find(BEGIN, payload_start)
        _require(nested == -1 or (end != -1 and nested > end), "nested NH frame")
        if end == -1:
            truncated_offset = start
            break
        frames.append(_parse_frame(raw[payload_start:end], start, end + len(END)))
        cursor = end + len(END)

    issues = []
    previous = None
    for index, frame in enumerate(frames):
        def issue(message):
            issues.append({"frame_index": index, "sample": frame["sample"],
                           "message": message})

        expected_stable = int(frame["seq"] == frame["seq_after"] and
                              frame["seq_after"] % 2 == 0)
        if frame["stable"] != expected_stable:
            issue("stable flag disagrees with sequence observations")
        if frame["pending"] != (frame["count"] - frame["consumed"]) & 0xffffffff:
            issue("pending disagrees with count-consumed modulo 2^32")
        if frame["pending"] > console_capacity:
            issue("pending exceeds console capacity; observation may be stale or inconsistent")
        if frame["sample"] == 0:
            issue("sample counter starts at one")
        if previous is not None:
            if frame["sample"] != previous["sample"] + 1:
                issue("sample counter is not consecutive; samples may be missing or the run changed")
            if frame["dropped"] < previous["dropped"]:
                issue("dropped counter decreased; records may be stale or the run changed")
        elif frame["sample"] != 1:
            issue("first captured sample is not one; earlier observations are missing")
        previous = frame
    dropped = max((f["dropped"] for f in frames), default=0)
    incomplete = bool(dropped or truncated_offset is not None or issues)
    report = {
        "format": "NRWATCH1",
        "console_capacity_bytes": console_capacity,
        "status": ("UART_DIAGNOSTIC_INCOMPLETE" if incomplete else
                   "FRAMES_DECODED" if frames else "NO_DIAGNOSTIC_FRAMES"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "ra_sha256": hashlib.sha256(clean).hexdigest(),
        "ra_bytes": len(clean),
        "frame_count": len(frames),
        "frames": frames,
        "truncated_frame": truncated_offset is not None,
        "truncated_byte_begin": truncated_offset,
        "truncated_bytes": len(raw) - truncated_offset if truncated_offset is not None else 0,
        "observed_dropped_bytes_max": dropped,
        "uart_acceptance_incomplete": incomplete,
        "observation_issues": issues,
        "numerical_acceptance": "NOT_EVALUATED",
        "root_cause": None,
        "limits": [
            "Stable means only matching even sequence observations; data can still be stale.",
            "An unchanged ENTER phase does not prove the kernel is stuck or identify its cause.",
            "Unknown phase values are preserved numerically without interpretation.",
            "Console counters and wait flags are sampled separately from the phase record.",
            "No observed drops does not prove complete UART capture or numerical correctness.",
            "A partial reserved BEGIN at EOF is conservatively excluded from RA bytes.",
        ],
    }
    return bytes(clean), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--console-capacity", type=int, default=65536,
                        help="console bytes from w8a8-image-plan.json; older images use 65536")
    args = parser.parse_args(argv)
    targets = (args.output / "uart.ra.log", args.output / "nh-watch.json")
    _require(args.uart.resolve() not in {p.resolve() for p in targets},
             "output must not overwrite the original UART log")
    clean, report = decode_bytes(args.uart.read_bytes(), console_capacity=args.console_capacity)
    report["uart_source"] = str(args.uart)
    args.output.mkdir(parents=True, exist_ok=True)
    targets[0].write_bytes(clean)
    targets[1].write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()
