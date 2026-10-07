# ===- ab_cli.py ---------------------------------------------------------------
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
# ===---------------------------------------------------------------------------
#
# A/B comparison of .rax models with buddy-cli: for every prompt file and
# repetition, the models run alternately (A B A B ...); prints the prefill
# time, the decode speed and a hash of the generated text, then the median
# per model and prompt and whether the texts are identical (compared in
# full, not by hash).
#
#   python3 ab_cli.py --cli buddy-cli --model A=a.rax --model B=b.rax \
#       --prompt p64.txt:192 --prompt p458.txt:588 --runs 2 \
#       [--launcher "sh -c 'echo 0 > /proc/set_ai_thread && exec \"$0\" \"$@\"'"]
#
# A prompt is FILE or FILE:N, N being buddy-cli's --max-tokens (prompt plus
# generated tokens: prompt tokens + 128 decodes 128 tokens).
#
# --launcher wraps each command (it receives the buddy-cli argv after it),
# e.g. to switch the process into a mode the platform requires.
#
# A run that fails (non-zero exit status, no timings in the log, no text)
# stops the comparison with its log. With --expect-same, the exit status is
# 1 when the texts of a prompt differ.
#
# ===---------------------------------------------------------------------------

import argparse
import hashlib
import re
import shlex
import statistics
import subprocess
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(args, name, rax, prompt_spec):
    prompt_file, _, max_tokens = prompt_spec.partition(":")
    with open(prompt_file) as f:
        prompt = f.read()
    cmd = [args.cli, "--model", rax, "--prompt", prompt]
    if max_tokens:
        cmd += ["--max-tokens", max_tokens]
    if args.launcher:
        cmd = shlex.split(args.launcher) + cmd
    # buddy-cli writes the generated text to stdout, its logs and timings
    # to stderr
    res = subprocess.run(cmd, capture_output=True, text=True)
    log = ANSI.sub("", res.stderr)
    where = f"{name} ({rax}) on {prompt_spec}"
    if res.returncode != 0:
        raise RuntimeError(
            f"{where} failed with exit status {res.returncode}:\n{log}"
        )
    pre = re.search(r"\[Prefill\]\s*([0-9.]+)s", log)
    dec = re.search(r"\[Decode\]\s*([0-9.]+) tokens/s", log)
    if pre is None or dec is None:
        raise RuntimeError(f"{where}: no prefill / decode timing in:\n{log}")
    if not res.stdout.strip():
        raise RuntimeError(f"{where}: no generated text; log:\n{log}")
    return {
        "model": name,
        "prompt": prompt_spec,
        "prefill": float(pre.group(1)),
        "decode": float(dec.group(1)),
        "text": res.stdout,
        "hash": hashlib.sha256(res.stdout.encode()).hexdigest()[:16],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", default="buddy-cli")
    p.add_argument(
        "--model", action="append", required=True, help="NAME=path.rax"
    )
    p.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="FILE or FILE:max_tokens",
    )
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--launcher", help="command prefix, see the header")
    p.add_argument(
        "--expect-same",
        action="store_true",
        help="exit with status 1 if the texts of a prompt differ",
    )
    args = p.parse_args()
    models = [m.split("=", 1) for m in args.model]

    rows = []
    for prompt in args.prompt:
        for r in range(args.runs):
            for name, rax in models:
                try:
                    row = run(args, name, rax, prompt)
                except RuntimeError as e:
                    sys.exit(f"error: {e}")
                rows.append(row)
                print(
                    f"{prompt} run {r + 1} {name:>10}: prefill {row['prefill']} s,"
                    f" decode {row['decode']} tok/s, text {row['hash']}",
                    flush=True,
                )

    print("\nmedians:")
    differ = False
    for prompt in args.prompt:
        texts = set()
        for name, _ in models:
            mine = [
                x for x in rows if x["prompt"] == prompt and x["model"] == name
            ]
            pre = [x["prefill"] for x in mine]
            dec = [x["decode"] for x in mine]
            texts |= {x["text"] for x in mine}
            print(
                f"{prompt} {name:>10}: prefill {statistics.median(pre):.3f} s"
                f" (spread {max(pre) - min(pre):.3f}), decode"
                f" {statistics.median(dec):.2f} tok/s"
                f" (spread {max(dec) - min(dec):.2f})"
            )
        same = len(texts) == 1
        differ |= not same
        print(f"{prompt}: {'same text' if same else 'TEXTS DIFFER'}")
    if args.expect_same and differ:
        sys.exit(1)


if __name__ == "__main__":
    main()
