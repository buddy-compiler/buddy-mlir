#!/usr/bin/env python3
# ===- gen_kokoro_manifest.py - RHAL manifest for Kokoro-82M ---------------===//
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
# Usage:
#   python gen_kokoro_manifest.py --spec specs/f32.json -o kokoro.mlir
#                                  --runner-library kokoro_runner.so
#
# The buffer/function layout mirrors the AOT forward ABI produced by
# codegen/import-kokoro.py:
#   forward(weights: memref<params_size x f32>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           ref_s: memref<1 x 256 x f32>,
#           speed: memref<1 x f32>)
#     -> (waveform: memref<1 x audio_buffer_size x f32>)
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys


def normalize_uri(raw: str) -> str:
    s = raw.strip()
    if ":" in s:
        return s
    return f"file:{s}"


def gen_manifest(spec: dict, runner_library: str) -> str:
    model_family = spec["model_family"]
    variant = spec.get("variant", "f32")
    model_id = spec.get("model_id", f"{model_family}_{variant}")
    params_size = int(spec["params_size"])
    max_seq_len = int(spec["max_seq_len"])
    hidden_size = int(spec["hidden_size"])
    style_dim = int(spec.get("style_dim", 128))
    max_dur = int(spec.get("max_dur", 50))
    upsample_factor = int(spec.get("upsample_factor", 300))
    # The AOT kernel is fixed at the traced input_ids length (max_seq_len
    # tokens); the maximum output waveform length is
    #   max_seq_len * max_dur * upsample_factor   samples.
    audio_buffer_size = int(spec.get(
        "audio_buffer_size", max_seq_len * max_dur * upsample_factor))
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    # Kokoro's phoneme vocab lives in the `vocab` field of config.json (there is
    # no standalone vocab file), so the staged config doubles as the tokenizer.
    tokenizer_file = spec.get("tokenizer_file", "config.json")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    max_position_embeddings = "{spec.get("max_position_embeddings", 512)}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    style_dim = "{style_dim}",')
    p(f'    audio_buffer_size = "{audio_buffer_size}",')
    p(f'    runner_library = "{normalize_uri(runner_library)}"}} {{')
    p("")
    p('  rhal.constant @params {id = 1 : i32, storage = "external",')
    p(f"                         type = tensor<{params_size}xf32>,")
    p(f'                         uri = "file:{weight_file}"}}')
    p("")
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')
    p("")
    p(f'  rhal.buffer @input_ids {{space = "host", '
      f"type = tensor<1x{max_seq_len}xi64>}}")
    p(f'  rhal.buffer @ref_s {{space = "host", '
      f"type = tensor<1x{2 * style_dim}xf32>}}")
    p(f'  rhal.buffer @speed {{space = "host", type = tensor<1xf32>}}')
    p(f'  rhal.buffer @waveform {{space = "host", '
      f"type = tensor<1x{audio_buffer_size}xf32>}}")
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["input_ids", "ref_s", "speed"],')
    p('    outputs  = ["waveform"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["input_ids", "ref_s", "speed", "waveform"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the Kokoro-82M model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="kokoro_runner.so",
        help="Runner plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    text = gen_manifest(spec, args.runner_library)

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[gen_kokoro_manifest] Written: {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
