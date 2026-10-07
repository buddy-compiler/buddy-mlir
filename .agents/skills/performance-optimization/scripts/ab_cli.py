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
# per model and prompt and whether the texts agree.
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
# ===---------------------------------------------------------------------------

import argparse
import hashlib
import re
import shlex
import statistics
import subprocess

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
    pre = re.search(r"\[Prefill\]\s*([0-9.]+)s", log)
    dec = re.search(r"\[Decode\]\s*([0-9.]+) tokens/s", log)
    text = res.stdout
    return {
        "model": name,
        "prompt": prompt_spec,
        "prefill": float(pre.group(1)) if pre else None,
        "decode": float(dec.group(1)) if dec else None,
        "hash": hashlib.md5(text.encode()).hexdigest()[:8],
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
    args = p.parse_args()
    models = [m.split("=", 1) for m in args.model]

    rows = []
    for prompt in args.prompt:
        for r in range(args.runs):
            for name, rax in models:
                row = run(args, name, rax, prompt)
                rows.append(row)
                print(
                    f"{prompt} run {r + 1} {name:>10}: prefill {row['prefill']} s,"
                    f" decode {row['decode']} tok/s, text {row['hash']}",
                    flush=True,
                )

    print("\nmedians:")
    for prompt in args.prompt:
        hashes = set()
        for name, _ in models:
            mine = [
                x for x in rows if x["prompt"] == prompt and x["model"] == name
            ]
            pre = [x["prefill"] for x in mine if x["prefill"] is not None]
            dec = [x["decode"] for x in mine if x["decode"] is not None]
            hashes |= {x["hash"] for x in mine}
            pre_s = (
                f"{statistics.median(pre):.3f} s (spread {max(pre) - min(pre):.3f})"
                if pre
                else "-"
            )
            dec_s = (
                f"{statistics.median(dec):.2f} tok/s (spread {max(dec) - min(dec):.2f})"
                if dec
                else "-"
            )
            print(f"{prompt} {name:>10}: prefill {pre_s}, decode {dec_s}")
        print(
            f"{prompt}: {'same text' if len(hashes) == 1 else 'TEXTS DIFFER'}"
        )


if __name__ == "__main__":
    main()
