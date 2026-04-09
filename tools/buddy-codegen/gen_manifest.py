#!/usr/bin/env python3
# ===- gen_manifest.py - Generate RHAL .mlir manifest from config ----------===//
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
# Generates the RHAL dialect .mlir manifest that rax-pack consumes to produce
# a .rax binary.  All model-specific constants (KV layers, shapes, types,
# weight URIs) come from the full config JSON produced by gen_config.py.
#
# Usage:
#   python gen_manifest.py --config deepseek_r1_f32.json -o deepseek_r1.mlir
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys
from io import StringIO


def gen_manifest(config: dict) -> str:
    """Generate the complete RHAL .mlir manifest text."""
    out = StringIO()

    def _p(*a, **kw):
        print(*a, file=out, **kw)

    p = _p

    model_id = config["model_id"]
    model_family = config["model_family"]
    shape = config["shape"]
    tokens = config["tokens"]
    weights = config["weights"]
    compilation = config["compilation"]
    mlir_types = config["mlir_types"]

    head_num = shape["head_num"]
    max_token_len = shape["max_token_len"]
    hidden_size = shape["hidden_size"]
    vocab_size = shape["vocab_size"]
    kv_layers = shape["kv_layers"]
    kv_mlir = mlir_types["kv"]
    logits_mlir = mlir_types["logits"]

    so_name = compilation["so_name"]
    vocab_file = tokens["vocab_file"]

    # ── Module header ────────────────────────────────────────────────────────
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{vocab_file}"}} {{')
    p()

    # ── External constants (weight blobs) ────────────────────────────────────
    for w in weights:
        tag = w["tag"]
        mlir_t = w["mlir_type"]
        num = w["num_elements"]
        fname = w["file"]
        p(f'  rhal.constant @{tag} {{id = 1 : i32, storage = "external",')
        p(f"                         type = tensor<{num}x{mlir_t}>,")
        p(f'                         uri = "file:{fname}"}}')
    p()

    # ── Code object ──────────────────────────────────────────────────────────
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')
    p()

    # ── Buffer descriptors ───────────────────────────────────────────────────
    p(
        f'  rhal.buffer @prefill_tokens {{space = "host", type = tensor<1x{max_token_len}xi64>}}'
    )
    p('  rhal.buffer @decode_token   {space = "host", type = tensor<1x1xi64>}')
    p('  rhal.buffer @cache_position {space = "host", type = tensor<1xi64>}')
    p()

    kv_tensor = f"tensor<1x{head_num}x{max_token_len}x{hidden_size}x{kv_mlir}>"
    for i in range(kv_layers):
        pad = " " * (1 if i < 10 else 0)
        p(f'  rhal.buffer @kv{i}{pad} {{space = "dram", type = {kv_tensor}}}')
    p()

    logits_pfx = f"tensor<1x{max_token_len}x{vocab_size}x{logits_mlir}>"
    logits_dec = f"tensor<1x1x{vocab_size}x{logits_mlir}>"
    p(f'  rhal.buffer @logits_prefill {{space = "host", type = {logits_pfx}}}')
    p(f'  rhal.buffer @logits_decode  {{space = "host", type = {logits_dec}}}')
    p()

    # ── Helper: build argument list ──────────────────────────────────────────
    def _kv_names():
        return [f'"kv{i}"' for i in range(kv_layers)]

    def _format_args(args: list[str], indent: int = 16) -> str:
        """Format a list of quoted names into wrapped lines."""
        lines = []
        cur = ""
        for a in args:
            candidate = f"{cur}, {a}" if cur else a
            if len(candidate) > 72:
                lines.append(cur + ",")
                cur = " " * indent + a
            else:
                cur = candidate
        if cur:
            lines.append(cur)
        return ("\n" + " " * indent).join(lines)

    # rhal.func args only reference rhal.buffer names (not rhal.constant).
    # Weights are bound via rhal.constant and resolved separately by rax-pack.

    # ── forward_prefill ──────────────────────────────────────────────────────
    prefill_args = ['"prefill_tokens"'] + _kv_names() + ['"logits_prefill"']
    p("  rhal.func @forward_prefill {")
    p('    inputs   = ["prefill_tokens"],')
    p('    outputs  = ["logits_prefill"],')
    p('    dispatch = "model_kernels",')
    p(f"    args     = [{_format_args(prefill_args)}]}}")
    p()

    # ── forward_decode ───────────────────────────────────────────────────────
    decode_args = (
        ['"decode_token"', '"cache_position"']
        + _kv_names()
        + ['"logits_decode"']
    )
    p("  rhal.func @forward_decode {")
    p('    inputs   = ["decode_token", "cache_position"],')
    p('    outputs  = ["logits_decode"],')
    p('    dispatch = "model_kernels",')
    p(f"    args     = [{_format_args(decode_args)}]}}")

    p("}")

    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Generate RHAL .mlir manifest from a full model config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to full config JSON"
    )
    parser.add_argument(
        "-o", "--output", default="-", help="Output path (- for stdout)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    mlir_text = gen_manifest(config)

    if args.output == "-":
        sys.stdout.write(mlir_text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"[gen_manifest] Written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
