#!/usr/bin/env python3
# ===- gen_bge_m3_manifest.py - RHAL manifest for BGE-M3 ------------------===//
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
#   python gen_bge_m3_manifest.py --spec specs/base.json -o bge_m3.mlir
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
    model_id = spec.get("model_id", f"{spec['model_family']}_{spec['variant']}")
    params_size = int(spec["params_size"])
    max_seq_len = int(spec["max_seq_len"])
    max_position_embeddings = int(spec.get("max_position_embeddings", 8194))
    hidden_size = int(spec["hidden_size"])
    so_name = spec.get("so_name", "bge_m3_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    tokenizer_file = spec.get("tokenizer_file", "tokenizer.json")
    tokenizer_helper = spec.get("tokenizer_helper", "bge_m3_tokenize.py")

    lines = []
    p = lines.append
    p("rhal.module @bge_m3 attributes {")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    max_position_embeddings = "{max_position_embeddings}",')
    p(f'    hidden_size = "{hidden_size}",')
    p(f'    runner_library = "{runner_library}"}} {{')
    p("")
    p('  rhal.constant @params {id = 1 : i32, storage = "external",')
    p(f"                         type = tensor<{params_size}xf32>,")
    p(f'                         uri = "file:{weight_file}"}}')
    p('  rhal.constant @tokenizer_helper {id = 2 : i32, storage = "external",')
    p("                                  type = tensor<1xi8>,")
    p(f'                                  uri = "file:{tokenizer_helper}"}}')
    p('  rhal.constant @tokenizer_config {id = 3 : i32, storage = "external",')
    p("                                  type = tensor<1xi8>,")
    p('                                  uri = "file:tokenizer_config.json"}')
    p('  rhal.constant @special_tokens {id = 4 : i32, storage = "external",')
    p("                                type = tensor<1xi8>,")
    p('                                uri = "file:special_tokens_map.json"}')
    p(
        '  rhal.constant @sentencepiece_model {id = 5 : i32, storage = "external",'
    )
    p("                                      type = tensor<1xi8>,")
    p(
        '                                      uri = "file:sentencepiece.bpe.model"}'
    )
    p("")
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')
    p("")
    p(
        f'  rhal.buffer @input_ids {{space = "host", '
        f"type = tensor<1x{max_seq_len}xi64>}}"
    )
    p(
        f'  rhal.buffer @token_type_ids {{space = "host", '
        f"type = tensor<{max_position_embeddings}xi64>}}"
    )
    p(
        f'  rhal.buffer @attention_mask {{space = "host", '
        f"type = tensor<1x{max_seq_len}xi64>}}"
    )
    p(
        f'  rhal.buffer @last_hidden_state {{space = "host", '
        f"type = tensor<1x{max_seq_len}x{hidden_size}xf32>}}"
    )
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["input_ids", "attention_mask"],')
    p('    outputs  = ["last_hidden_state"],')
    p('    dispatch = "model_kernels",')
    p('    args     = ["input_ids", "attention_mask", "last_hidden_state"]}')
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL manifest for the BGE-M3 model."
    )
    parser.add_argument(
        "--spec", required=True, help="Path to the variant spec JSON"
    )
    parser.add_argument(
        "--runner-library",
        default="bge_m3_runner.so",
        help="Runner plugin library URI/name for module attrs.",
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    text = gen_manifest(spec, normalize_uri(args.runner_library))

    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(text)
        print(f"[gen_bge_m3_manifest] Written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
