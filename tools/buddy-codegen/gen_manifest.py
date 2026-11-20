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


def _normalize_dep_uri(raw: str) -> str:
    s = raw.strip()
    if not s:
        raise ValueError("empty dependency URI")
    if ":" in s:
        return s
    return f"file:{s}"


def gen_manifest(
    config: dict,
    dep_shared_libs: list[str] | None = None,
    runner_library: str | None = None,
    model_library: str | None = None,
    version: str = "0.1.0",
) -> str:
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

    stage = config.get("pipeline_stage")
    stage_attrs = {}
    first_stage = last_stage = True
    if stage is not None:
        stage_attrs["pp_stage_version"] = "1"
        for field in (
            "start_layer", "end_layer", "total_layers",
            "activation_type", "activation_width",
        ):
            if field not in stage:
                raise ValueError(f"missing pipeline_stage.{field}")
            stage_attrs[f"pp_{field}"] = str(stage[field])
        first_stage = stage["start_layer"] == 0
        last_stage = stage["end_layer"] == stage["total_layers"]
        if (len(weights) != 1 or weights[0]["mlir_type"] != "f16"
                or weights[0].get("decode_file")):
            raise ValueError("pipeline_stage requires one external f16 weight")
        if int(config.get("prefill_chunk", 0)):
            raise ValueError("pipeline_stage requires prefill_chunk = 0")

    so_name = "stage.so" if stage is not None else compilation["so_name"]
    runner_suffix = "_pipeline_runner.so" if stage is not None else "_runner.so"
    runner_uri = _normalize_dep_uri(
        runner_library or f"{model_family}{runner_suffix}"
    )
    model_uri = _normalize_dep_uri(model_library) if model_library else None

    dep_uris: list[str] = []
    for item in dep_shared_libs or []:
        dep_uris.append(_normalize_dep_uri(item))

    # -- Module header ---------------------------------------------------------
    p(f"rhal.module @{model_family} attributes {{")
    p(f'    version = "{version}",')
    p(f'    model_name = "{model_id}",')
    if first_stage:
        p(f'    vocab_uri = "file:{tokens["vocab_file"]}",')
    for key, value in stage_attrs.items():
        p(f"    {key} = {json.dumps(value)},")
    p(f'    runner_library = "{runner_uri}"', end="")
    if model_uri:
        p(",")
        p(f'    model_library = "{model_uri}"', end="")
    p("} {")
    p()

    # -- External constants (weight blobs) -------------------------------------
    for w in weights:
        tag = w["tag"]
        mlir_t = w["mlir_type"]
        num = w["num_elements"]
        fname = w["file"]
        p(f'  rhal.constant @{tag} {{id = 1 : i32, storage = "external",')
        p(f"                         type = tensor<{num}x{mlir_t}>,")
        p(f'                         uri = "file:{fname}"}}')
    # One logical constant, two layouts; it needs its own entry only because
    # the two have to resolve to two files.
    for w in weights:
        if not w.get("decode_file"):
            continue
        p(
            f"  rhal.constant @{w['tag']}_decode {{id = 1 : i32, "
            'storage = "external",'
        )
        p(
            f"                         type = tensor<{w['num_elements']}x"
            f"{w['mlir_type']}>,"
        )
        p(f'                         uri = "file:{w["decode_file"]}"}}')
    p()

    # -- Code object -----------------------------------------------------------
    p('  rhal.codeobj @model_kernels {id = 1 : i32, kind = "host_shared_lib",')
    p('                                backend = "cpu",')
    p(f'                                uri = "file:{so_name}"}}')

    # Additional runtime dependencies (kept as host_shared_lib entries).
    for idx, dep_uri in enumerate(dep_uris, start=2):
        dep_sym = f"runtime_dep_{idx - 1}"
        p(
            f'  rhal.codeobj @{dep_sym} {{id = {idx} : i32, kind = "host_shared_lib",'
        )
        p('                                backend = "cpu",')
        p(f'                                uri = "{dep_uri}"}}')
    p()

    # -- Buffer descriptors ----------------------------------------------------
    # Chunked prefill (prefill_chunk > 0): forward_prefill takes `chunk` tokens
    # and their start position, like forward_decode.
    chunk = int(config.get("prefill_chunk", 0))
    prefill_len = chunk or max_token_len
    if first_stage:
        prefill_input, decode_input = "prefill_tokens", "decode_token"
        prefill_type = f"tensor<1x{prefill_len}xi64>"
        decode_type = "tensor<1x1xi64>"
    else:
        prefill_input, decode_input = "hidden_prefill_in", "hidden_decode_in"
        width = stage["activation_width"]
        prefill_type = f"tensor<1x{prefill_len}x{width}xf16>"
        decode_type = f"tensor<1x1x{width}xf16>"
    p(f'  rhal.buffer @{prefill_input} {{space = "host", type = {prefill_type}}}')
    p(f'  rhal.buffer @{decode_input} {{space = "host", type = {decode_type}}}')
    p('  rhal.buffer @cache_position {space = "host", type = tensor<1xi64>}')
    p()

    # Stage implementations own their KV buffers behind the Stage interface.
    kv_names = []
    if stage is None:
        kv_tensor = f"tensor<1x{head_num}x{max_token_len}x{hidden_size}x{kv_mlir}>"
        for i in range(kv_layers):
            pad = " " * (1 if i < 10 else 0)
            p(f'  rhal.buffer @kv{i}{pad} {{space = "dram", type = {kv_tensor}}}')
            kv_names.append(f'"kv{i}"')
        p()

    if last_stage:
        prefill_output, decode_output = "logits_prefill", "logits_decode"
        output_width = vocab_size
        output_type = logits_mlir if stage is None else "f16"
    else:
        prefill_output, decode_output = "hidden_prefill_out", "hidden_decode_out"
        output_width = stage["activation_width"]
        output_type = "f16"
    # Chunked prefill returns the logits of the last row only.
    output_rows = 1 if chunk else max_token_len
    prefill_result = f"tensor<1x{output_rows}x{output_width}x{output_type}>"
    decode_result = f"tensor<1x1x{output_width}x{output_type}>"
    p(f'  rhal.buffer @{prefill_output} {{space = "host", type = {prefill_result}}}')
    p(f'  rhal.buffer @{decode_output} {{space = "host", type = {decode_result}}}')
    p()

    # -- Helper: build argument list -------------------------------------------
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

    # -- forward_prefill -------------------------------------------------------
    prefill_inputs = [f'"{prefill_input}"']
    if chunk:
        prefill_inputs.append('"cache_position"')
    prefill_args = prefill_inputs + kv_names + [f'"{prefill_output}"']
    p("  rhal.func @forward_prefill {")
    p(f"    inputs   = [{', '.join(prefill_inputs)}],")
    p(f'    outputs  = ["{prefill_output}"],')
    p('    dispatch = "model_kernels",')
    p(f"    args     = [{_format_args(prefill_args)}]}}")
    p()

    # -- forward_decode --------------------------------------------------------
    decode_inputs = [f'"{decode_input}"', '"cache_position"']
    decode_args = decode_inputs + kv_names + [f'"{decode_output}"']
    p("  rhal.func @forward_decode {")
    p(f"    inputs   = [{', '.join(decode_inputs)}],")
    p(f'    outputs  = ["{decode_output}"],')
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
    parser.add_argument(
        "--dep-shared-lib",
        action="append",
        default=[],
        metavar="URI_OR_NAME",
        help=(
            "Additional host shared library dependency URI/name to place into "
            "rhal.codeobj (repeatable). If no scheme is given, file: is assumed."
        ),
    )
    parser.add_argument(
        "--runner-library",
        default=None,
        metavar="URI_OR_NAME",
        help=(
            "Runner plugin library URI/name to place into module attrs. "
            "If no scheme is given, file: is assumed."
        ),
    )
    parser.add_argument(
        "--model-library",
        default=None,
        metavar="URI_OR_NAME",
        help=("Server model plugin URI/name to place into module attrs."),
    )
    parser.add_argument(
        "--version",
        default="0.1.0",
        help="RAX module version string (usually the CLI release version).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    try:
        mlir_text = gen_manifest(
            config,
            dep_shared_libs=args.dep_shared_lib,
            runner_library=args.runner_library,
            model_library=args.model_library,
            version=args.version,
        )
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output == "-":
        sys.stdout.write(mlir_text)
    else:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"[gen_manifest] Written: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
