#!/usr/bin/env python3
# ===- gen_timesfm_manifest.py - RHAL manifest for TimesFM 2.5 -------------===//
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
#   python gen_timesfm_manifest.py --spec specs/f32.json -o timesfm.mlir \
#       --runner-library timesfm_runner.so
#
# The buffer/function layout mirrors the AOT forward ABI produced by
# codegen/import-timesfm.py:
#   forward(weights: memref<params_size x f32>,
#           inputs: memref<1 x num_patches x patch_length x f32>,
#           masks:  memref<1 x num_patches x patch_length x f32>)
#     -> (point_forecast: memref<1 x num_patches x (output_patch_len*quantile_len) x f32>)
#
# The forward has a single result, so the C ABI wrapper `_mlir_ciface_forward`
# takes the result memref first, then the three input memrefs in declaration
# order (weights, inputs, masks).
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
    num_patches = int(spec.get("num_patches", 16))
    patch_length = int(spec.get("patch_length", 32))
    # Fixed input context window in time points (num_patches * patch_length).
    max_seq_len = int(spec.get("max_seq_len", num_patches * patch_length))
    hidden_size = int(spec["hidden_size"])
    output_patch_len = int(spec.get("output_patch_len",
                                    spec.get("horizon_length", 128)))
    quantile_len = int(spec.get("quantile_len", 10))
    forecast_features = output_patch_len * quantile_len
    decode_index = int(spec.get("decode_index", 5))
    num_threads = int(spec.get("num_threads", 48))
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    tokenizer_file = spec.get("tokenizer_file", "N/A")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    if tokenizer_file and tokenizer_file.lower() != "n/a":
        p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    num_patches = "{num_patches}",')
    p(f'    patch_length = "{patch_length}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    params_size = "{params_size}",')
    p(f'    output_patch_len = "{output_patch_len}",')
    p(f'    quantile_len = "{quantile_len}",')
    p(f'    decode_index = "{decode_index}",')
    p(f'    num_threads = "{num_threads}",')
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
    p(f'  rhal.buffer @inputs {{space = "host", '
      f"type = tensor<1x{num_patches}x{patch_length}xf32>}}")
    p(f'  rhal.buffer @masks {{space = "host", '
      f"type = tensor<1x{num_patches}x{patch_length}xf32>}}")
    p(f'  rhal.buffer @point_forecast {{space = "host", '
      f"type = tensor<1x{num_patches}x{forecast_features}xf32>}}")
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["inputs", "masks"],')
    p('    outputs  = ["point_forecast"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["inputs", "masks", "point_forecast"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the TimesFM 2.5 model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="timesfm_runner.so",
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
        print(f"[gen_timesfm_manifest] Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
