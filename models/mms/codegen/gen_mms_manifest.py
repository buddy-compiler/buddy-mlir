#!/usr/bin/env python3
# ===- gen_mms_manifest.py - RHAL manifest for MMS -------------------------===//
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
#   python gen_mms_manifest.py --spec specs/f32.json -o mms.mlir \
#       --runner-library mms_runner.so
#
# Mirrors the AOT forward ABI produced by codegen/import-mms.py (vocoder stage):
#   forward(weights: memref<params_size x f32>,
#           latents:  memref<1 x flow_size x max_seq_len x f32>)
#     -> waveform:   memref<1 x audio_buffer_size x f32>
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
    flow_size = int(spec.get("flow_size", 192))
    upsample_rates = [int(r) for r in spec.get("upsample_rates", [8, 8, 2, 2])]
    audio_rate = 1
    for r in upsample_rates:
        audio_rate *= r
    audio_buffer_size = max_seq_len * audio_rate
    so_name = spec.get("so_name", f"{model_family}_model.so")
    weight_file = spec.get("weight_file", "arg0.data")

    lines = []
    p = lines.append
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    max_seq_len = "{max_seq_len}",')
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
    p(f'  rhal.buffer @latents {{space = "host", '
      f"type = tensor<1x{flow_size}x{max_seq_len}xf32>}}")
    p(f'  rhal.buffer @waveform {{space = "host", '
      f"type = tensor<1x{audio_buffer_size}xf32>}}")
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["latents"],')
    p('    outputs  = ["waveform"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["latents", "waveform"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the MMS vocoder model."
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runner-library", default="mms_runner.so")
    parser.add_argument("-o", "--output", default="-")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    text = gen_manifest(spec, args.runner_library)

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[gen_mms_manifest] Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
