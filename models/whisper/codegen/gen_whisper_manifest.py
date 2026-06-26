#!/usr/bin/env python3
# ===- gen_whisper_manifest.py - RHAL .mlir manifest for Whisper -----------===//
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
#
# Generates the RHAL dialect .mlir manifest that rax-pack consumes for the
# Whisper encoder-decoder model.
#
# Unlike the LLM manifest (tools/buddy-codegen/gen_manifest.py), Whisper has a
# single `forward` entrypoint, no KV cache, and an audio-feature input.  buddy-cli
# does not interpret the rhal.func/buffer bodies at runtime; it only reads the
# external constant (weights), the host_shared_lib code object (model .so), and
# the module attrs (model_name, vocab_uri, runner_library).  The func/buffer
# entries exist only so rax-pack can parse a well-formed module.
#
# Usage:
#   python gen_whisper_manifest.py --spec specs/base.json -o whisper.mlir
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys


def gen_manifest(spec: dict, runner_library: str) -> str:
    model_id = spec.get("model_id", f"{spec['model_family']}_{spec['variant']}")
    params_size = spec["params_size"]
    vocab_size = spec["vocab_size"]
    max_token_len = spec["max_token_len"]
    mel_bins = spec["mel_bins"]
    audio_frames = spec["audio_frames"]
    so_name = spec.get("so_name", "whisper_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    vocab_file = spec.get("vocab_file", "vocab.txt")

    lines = []
    p = lines.append

    # -- Module header ---------------------------------------------------------
    p("rhal.module @whisper attributes {")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{vocab_file}",')
    p(f'    runner_library = "{runner_library}"}} {{')
    p("")

    # -- External constant (weight blob) ---------------------------------------
    p('  rhal.constant @params {id = 1 : i32, storage = "external",')
    p(f"                         type = tensor<{params_size}xf32>,")
    p(f'                         uri = "file:{weight_file}"}}')
    p("")

    # -- Code object (the compiled MLIR kernels) -------------------------------
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')
    p("")

    # -- Buffer descriptors (informational; not read by buddy-cli) -------------
    p(
        f'  rhal.buffer @audio_features {{space = "host", '
        f"type = tensor<1x{mel_bins}x{audio_frames}xf32>}}"
    )
    p(
        f'  rhal.buffer @decoder_tokens {{space = "host", '
        f"type = tensor<1x{max_token_len}xi64>}}"
    )
    p(
        f'  rhal.buffer @logits {{space = "host", '
        f"type = tensor<1x{max_token_len}x{vocab_size}xf32>}}"
    )
    p("")

    # -- forward entrypoint ----------------------------------------------------
    p("  rhal.func @forward {")
    p('    inputs   = ["audio_features", "decoder_tokens"],')
    p('    outputs  = ["logits"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["audio_features", "decoder_tokens", "logits"]}')
    p("}")

    return "\n".join(lines) + "\n"


def _normalize_uri(raw: str) -> str:
    s = raw.strip()
    if ":" in s:
        return s
    return f"file:{s}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate RHAL .mlir manifest for the Whisper model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="whisper_runner.so",
        help="Runner plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    text = gen_manifest(spec, _normalize_uri(args.runner_library))

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[gen_whisper_manifest] Written: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
