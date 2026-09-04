#!/usr/bin/env python3
# ===- gen_kimi_audio_manifest.py - RHAL manifest for Kimi-Audio ----------===//
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
#   python gen_kimi_audio_manifest.py --spec specs/f32.json -o kimi_audio.mlir
#
# The buffer/function layout mirrors the AOT forward ABI produced by
# codegen/import-kimi_audio.py (text-only, whisper features disabled):
#   forward(weights: memref<params_size x f32>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           position_ids: memref<1 x max_seq_len x i64>)
#     -> (audio_logits: memref<1 x max_seq_len x vocab_size x f32>,
#         text_logits : memref<1 x max_seq_len x vocab_size x f32>)
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
    vocab_size = int(spec["vocab_size"])
    num_hidden_layers = int(spec["num_hidden_layers"])
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    tokenizer_file = spec.get("tokenizer_file", "vocab.txt")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    max_position_embeddings = "{spec.get("max_position_embeddings", 8192)}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    vocab_size = "{vocab_size}",')
    p(f'    num_hidden_layers = "{num_hidden_layers}",')
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
    p(f'  rhal.buffer @position_ids {{space = "host", '
      f"type = tensor<1x{max_seq_len}xi64>}}")
    p(f'  rhal.buffer @audio_logits {{space = "host", '
      f"type = tensor<1x{max_seq_len}x{vocab_size}xf32>}}")
    p(f'  rhal.buffer @text_logits {{space = "host", '
      f"type = tensor<1x{max_seq_len}x{vocab_size}xf32>}}")
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["input_ids", "position_ids"],')
    p('    outputs  = ["audio_logits", "text_logits"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["input_ids", "position_ids",')
    p('                "audio_logits", "text_logits"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the Kimi-Audio model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="kimi_audio_runner.so",
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
        print(f"[gen_kimi_audio_manifest] Written: {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
