"""NH observations cannot alter the reconstructed RA byte stream or prove a hang."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decode_hang_watch import BEGIN, END, decode_bytes, main


def frame(**overrides):
    values = dict(sample=1, nh_cycles=100, stable=1, seq=2, seq_after=2,
                  phase=2, position=16, detail=14, target_phase=2, target_call=14,
                  count=100, consumed=90, pending=10,
                  ra_console_wait=0, dropped=0, waits=0)
    values.update(overrides)
    if not values["stable"]:
        for key in ("phase", "position", "detail", "target_phase", "target_call"):
            values.pop(key)
    body = "\r\n[nh-watch] " + " ".join(f"{k}={v:016X}" for k, v in values.items())
    body += "\r\n"
    if values.get("target_phase", 0):
        for arg in range(3):
            fields = dict(arg=arg, valid=1, descriptor=0x80001000 + arg * 64,
                          aligned=0xb8000000 + arg * 1024, offset=0,
                          rows=1, cols=16, stride0=16, stride1=1)
            body += "[nh-watch-memref] " + " ".join(
                f"{k}={v:016X}" for k, v in fields.items()) + "\r\n"
    return BEGIN + body.encode("ascii") + END


class DecodeHangWatch(unittest.TestCase):
    def test_mid_line_frame_restores_exact_bytes(self):
        expected = b"\x00[kernel] begin matmul call=000000000000000E\r\n\xe4\xb8\xad"
        cut = 21
        raw = expected[:cut] + frame() + expected[cut:]
        clean, report = decode_bytes(raw)
        self.assertEqual(clean, expected)
        self.assertEqual(report["raw_sha256"], sha256(raw).hexdigest())
        self.assertEqual(report["ra_sha256"], sha256(expected).hexdigest())
        self.assertEqual(report["frames"][0]["position"], 16)
        self.assertEqual(report["frames"][0]["target_call"], 14)
        self.assertEqual(report["frames"][0]["memrefs"][1]["aligned"], 0xb8000400)
        self.assertEqual(report["frames"][0]["raw_hex"]["pending"], "000000000000000A")
        self.assertIsNone(report["root_cause"])
        self.assertEqual(report["numerical_acceptance"], "NOT_EVALUATED")

    def test_multiple_frames_empty_stream_and_ordinary_text(self):
        raw = b"pre" + frame() + frame(sample=2, phase=3) + b"post\r\n"
        clean, report = decode_bytes(raw)
        self.assertEqual(clean, b"prepost\r\n")
        self.assertEqual(report["frame_count"], 2)
        self.assertEqual(report["observation_issues"], [])
        for plain in (b"", b"ordinary [nh-watch] unframed text\r\n", b"\xff\x00\x1f"):
            clean, report = decode_bytes(plain)
            self.assertEqual(clean, plain)
            self.assertEqual(report["status"], "NO_DIAGNOSTIC_FRAMES")

    def test_truncated_full_and_partial_begin(self):
        for suffix in (frame()[:-1], BEGIN, BEGIN[:1], BEGIN[:-1],
                       BEGIN + b"\r\n[nh-watch] sample=000"):
            clean, report = decode_bytes(b"RA prefix" + suffix)
            self.assertEqual(clean, b"RA prefix")
            self.assertTrue(report["truncated_frame"])
            self.assertTrue(report["uart_acceptance_incomplete"])
            self.assertEqual(report["truncated_byte_begin"], len(b"RA prefix"))
            self.assertEqual(report["truncated_bytes"], len(suffix))

    def test_drops_make_uart_incomplete_even_if_last_counter_is_stale(self):
        _, report = decode_bytes(frame(dropped=4) + frame(sample=2, dropped=0))
        self.assertEqual(report["observed_dropped_bytes_max"], 4)
        self.assertTrue(report["uart_acceptance_incomplete"])
        self.assertIn("decreased", report["observation_issues"][0]["message"])

    def test_counter_inconsistency_and_sequence_mismatch_reported(self):
        for raw in (frame(pending=11), frame(seq_after=4), frame(sample=2),
                    frame() + frame(sample=3), frame(count=70000, consumed=0, pending=70000)):
            _, report = decode_bytes(raw)
            self.assertTrue(report["observation_issues"])
            self.assertTrue(report["uart_acceptance_incomplete"])
            self.assertIsNone(report["root_cause"])

    def test_uint32_ring_wrap_and_unknown_phase_are_not_errors(self):
        _, report = decode_bytes(frame(count=3, consumed=0xfffffffe, pending=5,
                                       phase=0xabcdef, target_phase=0x9876))
        self.assertEqual(report["frames"][0]["phase"], 0xabcdef)
        self.assertEqual(report["observation_issues"], [])
        self.assertFalse(report["uart_acceptance_incomplete"])

    def test_configured_capacity_does_not_mislabel_larger_console(self):
        raw = frame(count=70000, consumed=0, pending=70000)
        _, report = decode_bytes(raw, console_capacity=524288)
        self.assertEqual(report['observation_issues'], [])
        self.assertEqual(report['console_capacity_bytes'], 524288)
        for capacity in (0, 65, True):
            with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, 'capacity'):
                decode_bytes(raw, console_capacity=capacity)

    def test_unstable_frame_does_not_invent_phase(self):
        _, report = decode_bytes(frame(stable=0, seq=3, seq_after=4))
        self.assertNotIn("phase", report["frames"][0])
        self.assertEqual(report["frames"][0]["memrefs"], [])
        self.assertEqual(report["observation_issues"], [])

    def test_malformed_frames_are_rejected_without_swallowing_ra(self):
        valid = frame()
        bad_frames = (
            BEGIN + BEGIN + END,
            BEGIN + b"\r\nRA line incorrectly hidden in frame\r\n" + END,
            valid.replace(b"pending=000000000000000A", b"pending=wrong"),
            valid.replace(b"[nh-watch] ", b"[nh-watch] sample=1 "),
            valid.replace(b"arg=0000000000000002", b"arg=0000000000000001"),
            valid.replace(b"[nh-watch-memref]", b"RA text", 1),
            valid.replace(b"\r\n[nh-watch]", b"\r\n\xff[nh-watch]"),
        )
        for raw in bad_frames:
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                decode_bytes(b"before" + raw + b"after")

    def test_cli_preserves_source_and_writes_both_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uart, output = root / "uart.raw.log", root / "decoded"
            raw = b"left" + frame() + b"right"
            uart.write_bytes(raw)
            main(["--uart", str(uart), "--output", str(output)])
            self.assertEqual(uart.read_bytes(), raw)
            self.assertEqual((output / "uart.ra.log").read_bytes(), b"leftright")
            report = json.loads((output / "nh-watch.json").read_text())
            self.assertEqual(report["raw_sha256"], sha256(raw).hexdigest())
            with self.assertRaises(ValueError):
                main(["--uart", str(output / "uart.ra.log"), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
