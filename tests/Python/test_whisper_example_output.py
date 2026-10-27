# RUN: %PYTHON %s

# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise the Whisper example checker without downloading a model."""

import subprocess
import sys
import unittest
from pathlib import Path

CHECKER = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "BuddyWhisper"
    / "check-output.py"
)
TRANSCRIPT = "Nor is Mr. Quilter's manner less interesting than his matter."
END_TOKEN = "[Iteration 18] Token: <|endoftext|> | Time: 1s\n"


class WhisperOutputTest(unittest.TestCase):
    def run_checker(self, stdout, *, stderr="", returncode=0):
        program = (
            "import sys; "
            f"sys.stdout.write({stdout!r}); "
            f"sys.stderr.write({stderr!r}); "
            f"sys.exit({returncode})"
        )
        return subprocess.run(
            [sys.executable, str(CHECKER), sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_output(self):
        output = END_TOKEN + f"[Output] {TRANSCRIPT}\n"
        result = self.run_checker(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(output, result.stdout)

    def test_terminal_colors_and_edge_whitespace(self):
        output = END_TOKEN + f"\x1b[33;1m[Output]\x1b[0m  {TRANSCRIPT} \t\n"
        result = self.run_checker(output)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_transcript(self):
        result = self.run_checker(END_TOKEN + "[Output] www\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Actual transcriptions: ['www']", result.stderr)

    def test_empty_or_missing_output(self):
        for output in ("", END_TOKEN, END_TOKEN + "[Output] \n"):
            with self.subTest(output=output):
                self.assertNotEqual(self.run_checker(output).returncode, 0)

    def test_transcript_must_be_in_final_output(self):
        result = self.run_checker(END_TOKEN + f"[Log] {TRANSCRIPT}\n")
        self.assertNotEqual(result.returncode, 0)

    def test_duplicate_final_outputs(self):
        result = self.run_checker(END_TOKEN + f"[Output] {TRANSCRIPT}\n" * 2)
        self.assertNotEqual(result.returncode, 0)

    def test_nonzero_exit_even_with_correct_output(self):
        result = self.run_checker(
            END_TOKEN + f"[Output] {TRANSCRIPT}\n",
            stderr="runtime failure\n",
            returncode=7,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("status 7", result.stderr)
        self.assertIn("runtime failure", result.stderr)

    def test_missing_end_token(self):
        result = self.run_checker(f"[Output] {TRANSCRIPT}\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not generate an end token", result.stderr)

    def test_stderr_is_not_a_transcription(self):
        result = self.run_checker(
            "", stderr=END_TOKEN + f"[Output] {TRANSCRIPT}\n"
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
