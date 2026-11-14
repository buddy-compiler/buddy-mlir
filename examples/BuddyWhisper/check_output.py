# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the bundled audio fixture using a compiled Whisper example."""

import argparse
import re
import subprocess
import sys

EXPECTED_TRANSCRIPT = (
    "Nor is Mr. Quilter's manner less interesting than his matter."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a Whisper executable is required")

    # Forward progress immediately so CTest retains diagnostics on timeout.
    lines = []
    with subprocess.Popen(
        args.command, stdout=subprocess.PIPE, text=True
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        returncode = process.wait()
    if returncode != 0:
        print(f"Whisper exited with status {returncode}.", file=sys.stderr)
        return 1

    # The example colors its log labels, including [Output].
    output = re.sub(r"\x1b\[[0-9;]*m", "", "".join(lines))
    transcripts = [
        line.removeprefix("[Output]").strip()
        for line in output.splitlines()
        if line.startswith("[Output]")
    ]
    if transcripts != [EXPECTED_TRANSCRIPT]:
        print(
            f"Expected one transcription: {EXPECTED_TRANSCRIPT!r}\n"
            f"Actual transcriptions: {transcripts!r}",
            file=sys.stderr,
        )
        return 1
    if "Token: <|endoftext|>" not in output:
        print("Whisper did not generate an end token.", file=sys.stderr)
        return 1

    print("Whisper transcription matches the bundled audio fixture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
