#!/usr/bin/env python3
# ===- gen_sam2_manifest.py - RHAL manifest for SAM2-hiera-tiny ------------===//
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
#   python gen_sam2_manifest.py --spec specs/f32.json -o sam2.mlir
#                                 --runner-library sam2_runner.so
#
# The buffer/function layout mirrors the AOT forward ABI produced by
# codegen/import-sam2.py (the Sam2VisionModel image encoder):
#   forward(weights: memref<params_size x f32>,
#           pixel_values: memref<1 x 3 x image_size x image_size x f32>)
#     -> (last_hidden_state: memref<1 x H x W x hidden_size x f32>,
#         fpn_0: memref<1 x 256 x 16 x 16 x f32>,
#         fpn_1: memref<1 x 256 x 16 x 16 x f32>,
#         fpn_2: memref<1 x 256 x 32 x 32 x f32>,
#         fpn_3: memref<1 x 256 x 32 x 32 x f32>,
#         fpn_4: memref<1 x 256 x 64 x 64 x f32>,
#         fpn_5: memref<1 x 256 x 64 x 64 x f32>)
#
# SAM2 is a vision model: it has no tokenizer / vocab, so no vocab_uri and no
# text-token buffers are emitted.
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys

# FPN feature-map shapes produced by the traced Sam2VisionModel forward
# (two maps per resolution: 16x16, 32x32, 64x64). Must stay in lock-step with
# codegen/import-sam2.py output; the runner mirrors the same list.
FPN_SHAPES = [(16, 16), (16, 16), (32, 32), (32, 32), (64, 64), (64, 64)]
FPN_CHANNELS = 256


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
    hidden_size = int(spec["hidden_size"])
    image_size = int(spec.get("image_size", 256))
    out_h = int(spec.get("output_height", 8))
    out_w = int(spec.get("output_width", 8))
    max_seq_len = int(spec.get("max_seq_len", out_h * out_w))
    num_hidden_layers = int(spec.get("num_hidden_layers", 12))
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    image_size = "{image_size}",')
    p(f'    output_height = "{out_h}",')
    p(f'    output_width = "{out_w}",')
    p(f'    fpn_channels = "{FPN_CHANNELS}",')
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
    p(f'  rhal.buffer @pixel_values {{space = "host", '
      f"type = tensor<1x3x{image_size}x{image_size}xf32>}}")
    p(f'  rhal.buffer @last_hidden_state {{space = "host", '
      f"type = tensor<1x{out_h}x{out_w}x{hidden_size}xf32>}}")
    for i, (fh, fw) in enumerate(FPN_SHAPES):
        p(f'  rhal.buffer @fpn_{i} {{space = "host", '
          f"type = tensor<1x{FPN_CHANNELS}x{fh}x{fw}xf32>}}")
    p("")
    def str_list(names):
        return "[" + ", ".join(f'"{n}"' for n in names) + "]"

    outputs = ["last_hidden_state"] + [f"fpn_{i}" for i in range(len(FPN_SHAPES))]
    p("  rhal.func @forward {")
    p(f'    inputs   = {str_list(["pixel_values"])},')
    p(f'    outputs  = {str_list(outputs)},')
    p('    dispatch = "model_kernels",')
    p(f'    args     = {str_list(["pixel_values"] + outputs)}}}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the SAM2-hiera-tiny model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="sam2_runner.so",
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
        print(f"[gen_sam2_manifest] Written: {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
