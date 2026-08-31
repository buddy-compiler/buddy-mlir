#!/usr/bin/env python3
# ===- gen_paligemma_manifest.py - RHAL manifest for PaliGemma -------------===//
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
#   python gen_paligemma_manifest.py --spec specs/f32.json -o paligemma.mlir
#       --runner-library paligemma_runner.so
#
# The buffer/function layout mirrors the AOT forward ABI produced by
# codegen/import-paligemma.py:
#   forward(weights: memref<params_size x f32>,
#           position_ids: memref<num_image_patches x i64>,
#           input_ids: memref<1 x max_seq_len x i64>,
#           pixel_values: memref<1 x 3 x image_size x image_size x f32>,
#           attention_mask: memref<1 x max_seq_len x i64>)
#     -> (image_features: memref<1 x num_image_tokens x hidden_size x f32>,
#         logits: memref<1 x max_seq_len x vocab_size x f32>)
#
# The two results are packed into one C struct, so the runner sees
# `_mlir_ciface_forward(ForwardResults*, weights*, position_ids*,
# input_ids*, pixel_values*, attention_mask*)`.
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
    num_image_tokens = int(spec.get("num_image_tokens", 256))
    num_image_patches = int(spec.get("num_image_patches", 256))
    image_size = int(spec.get("image_size", 224))
    image_token_id = int(spec.get("image_token_id", 257152))
    vision_hidden_size = int(spec.get("vision_hidden_size", 1152))
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    tokenizer_file = spec.get("tokenizer_file", "tokenizer.json")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    vocab_size = "{vocab_size}",')
    p(f'    num_image_tokens = "{num_image_tokens}",')
    p(f'    num_image_patches = "{num_image_patches}",')
    p(f'    image_size = "{image_size}",')
    p(f'    image_token_id = "{image_token_id}",')
    p(f'    vision_hidden_size = "{vision_hidden_size}",')
    p(f'    params_size = "{params_size}",')
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
    p(f'  rhal.buffer @position_ids {{space = "host", '
      f"type = tensor<{num_image_patches}xi64>}}")
    p(f'  rhal.buffer @input_ids {{space = "host", '
      f"type = tensor<1x{max_seq_len}xi64>}}")
    p(f'  rhal.buffer @pixel_values {{space = "host", '
      f"type = tensor<1x3x{image_size}x{image_size}xf32>}}")
    p(f'  rhal.buffer @attention_mask {{space = "host", '
      f"type = tensor<1x{max_seq_len}xi64>}}")
    p(f'  rhal.buffer @image_features {{space = "host", '
      f"type = tensor<1x{num_image_tokens}x{hidden_size}xf32>}}")
    p(f'  rhal.buffer @logits {{space = "host", '
      f"type = tensor<1x{max_seq_len}x{vocab_size}xf32>}}")
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["pixel_values", "input_ids", "attention_mask"],')
    p('    outputs  = ["image_features", "logits"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["pixel_values", "input_ids", "attention_mask",')
    p('                "position_ids", "image_features", "logits"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the PaliGemma model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="paligemma_runner.so",
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
        print(f"[gen_paligemma_manifest] Written: {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
