#!/usr/bin/env python3
# ===- gen_proteinglm_manifest.py - RHAL manifest for ProteinGLM ----------===//

import argparse
import json
import os
import sys


def normalize_uri(raw: str) -> str:
    s = raw.strip()
    if ":" in s:
        return s
    return f"file:{s}"


def gen_manifest(spec: dict, params_size: int, runner_library: str) -> str:
    model_id = spec.get("model_id", f"{spec['model_family']}_{spec['variant']}")
    max_seq_len = int(spec["max_seq_len"])
    vocab_size = int(spec["vocab_size"])
    so_name = spec.get("so_name", "proteinglm_model.so")
    weight_file = spec.get("weight_file", "arg0.data")
    tokenizer_file = spec.get("tokenizer_file", "tokenizer.model")
    top_k = int(spec.get("top_k", 5))

    lines = []
    p = lines.append
    p("rhal.module @proteinglm attributes {")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{tokenizer_file}",')
    p(f'    max_seq_len = "{max_seq_len}",')
    p(f'    vocab_size = "{vocab_size}",')
    p(f'    top_k = "{top_k}",')
    p(f'    runner_library = "{runner_library}"}} {{')
    p("")
    p('  rhal.constant @params {id = 1 : i32, storage = "external",')
    p(f"                         type = tensor<{params_size}xf32>,")
    p(f'                         uri = "file:{weight_file}"}}')
    p('  rhal.constant @tokenizer_config {id = 2 : i32, storage = "external",')
    p("                                  type = tensor<1xi8>,")
    p('                                  uri = "file:tokenizer_config.json"}')
    p('  rhal.constant @special_tokens {id = 3 : i32, storage = "external",')
    p("                                type = tensor<1xi8>,")
    p('                                uri = "file:special_tokens_map.json"}')
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
        f'  rhal.buffer @attention_mask {{space = "host", '
        f"type = tensor<1x{max_seq_len}xi64>}}"
    )
    p(
        f'  rhal.buffer @position_ids {{space = "host", '
        f"type = tensor<1x{max_seq_len}xi64>}}"
    )
    p(
        f'  rhal.buffer @logits {{space = "host", '
        f"type = tensor<1x{max_seq_len}x{vocab_size}xf32>}}"
    )
    p("")
    p("  rhal.func @forward {")
    p('    inputs   = ["input_ids", "attention_mask", "position_ids"],')
    p('    outputs  = ["logits"],')
    p('    dispatch = "model_kernels",')
    p(
        '    args     = ["input_ids", "attention_mask", "position_ids", "logits"]}'
    )
    p("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RHAL .mlir manifest for ProteinGLM"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runner-library", default="proteinglm_runner.so")
    parser.add_argument("-o", "--output", default="-")
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    weight_file = spec.get("weight_file", "arg0.data")
    params_file = os.path.join(output_dir, weight_file)
    if not os.path.exists(params_file):
        params_file = os.path.join(os.path.dirname(output_dir), weight_file)
    params_bytes = os.path.getsize(params_file)
    if params_bytes % 4 != 0:
        raise RuntimeError(f"weight file is not f32-aligned: {params_file}")

    text = gen_manifest(
        spec, params_bytes // 4, normalize_uri(args.runner_library)
    )
    if args.output == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(text)
        print(
            f"[gen_proteinglm_manifest] Written: {args.output}", file=sys.stderr
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
