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
    serving_library: str | None = None,
    embedding_library: str | None = None,
    masked_lm_library: str | None = None,
    transcription_library: str | None = None,
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

    so_name = compilation["so_name"]
    vocab_file = tokens["vocab_file"]
    runner_uri = _normalize_dep_uri(
        runner_library or f"{model_family}_runner.so"
    )
    serving_uri = (
        _normalize_dep_uri(serving_library) if serving_library else None
    )
    embedding_uri = (
        _normalize_dep_uri(embedding_library) if embedding_library else None
    )
    masked_lm_uri = (
        _normalize_dep_uri(masked_lm_library) if masked_lm_library else None
    )
    transcription_uri = (
        _normalize_dep_uri(transcription_library)
        if transcription_library
        else None
    )

    dep_uris: list[str] = []
    for item in dep_shared_libs or []:
        dep_uris.append(_normalize_dep_uri(item))

    # -- Module header ---------------------------------------------------------
    p(f"rhal.module @{model_family} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{vocab_file}",')
    p(f'    runner_library = "{runner_uri}"', end="")
    if serving_uri:
        p(",")
        p(f'    serving_library = "{serving_uri}"', end="")
        if embedding_uri:
            p(",")
            p(f'    embedding_library = "{embedding_uri}"', end="")
        else:
            pass
    elif embedding_uri:
        p(",")
        p(f'    embedding_library = "{embedding_uri}"', end="")
    if masked_lm_uri:
        p(",")
        p(f'    masked_lm_library = "{masked_lm_uri}"', end="")
    if transcription_uri:
        p(",")
        p(f'    transcription_library = "{transcription_uri}"', end="")
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

    # -- Helper: build argument list -------------------------------------------
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

    # -- forward_prefill -------------------------------------------------------
    prefill_args = ['"prefill_tokens"'] + _kv_names() + ['"logits_prefill"']
    p("  rhal.func @forward_prefill {")
    p('    inputs   = ["prefill_tokens"],')
    p('    outputs  = ["logits_prefill"],')
    p('    dispatch = "model_kernels",')
    p(f"    args     = [{_format_args(prefill_args)}]}}")
    p()

    # -- forward_decode --------------------------------------------------------
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


_RUNTIME_DTYPE_TO_MLIR = {
    "int8": "i8",
    "int32": "i32",
    "int64": "i64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float64": "f64",
    "bool": "i1",
    "complex64": "complex<f32>",
    "complex128": "complex<f64>",
}


def _runtime_tensor_type(resource: str, metadata: dict) -> str:
    dtype = metadata["dtype"]
    try:
        element_type = _RUNTIME_DTYPE_TO_MLIR[dtype]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dtype!r} for resource {resource!r}"
        ) from None
    dimensions = "x".join(str(dimension) for dimension in metadata["shape"])
    if dimensions:
        dimensions += "x"
    return f"tensor<{dimensions}{element_type}>"


def gen_parallel_manifest(
    config: dict,
    runtime_plans: dict[str, dict],
    dep_shared_libs: list[str] | None = None,
    runner_library: str | None = None,
    kernel_library: str | None = None,
) -> str:
    """Generate one rank-local body-form RHAL manifest from Stage 4A plans."""
    if not runtime_plans:
        raise ValueError("at least one runtime plan is required")

    plans = list(runtime_plans.items())
    rank = plans[0][1]["rank"]
    world_size = plans[0][1]["world_size"]
    for function_name, plan in plans[1:]:
        if plan["rank"] != rank:
            raise ValueError(
                f"runtime plan {function_name!r} has rank {plan['rank']}; "
                f"expected {rank}"
            )
        if plan["world_size"] != world_size:
            raise ValueError(
                f"runtime plan {function_name!r} has world_size "
                f"{plan['world_size']}; expected {world_size}"
            )

    parameter_packs = {}
    parameter_metadata = {}
    for function_name, plan in plans:
        resources = plan["resources"]
        for pack in plan["parameter_packs"]:
            resource = pack["resource"]
            if resource not in resources:
                raise ValueError(
                    f"parameter pack {resource!r} in {function_name!r} has "
                    "no resource metadata"
                )
            metadata = resources[resource]
            if metadata["role"] != "parameter":
                raise ValueError(
                    f"parameter pack {resource!r} in {function_name!r} "
                    "does not reference a parameter resource"
                )
            signature = (pack["dtype"], pack["numel"])
            resource_signature = (metadata["dtype"], tuple(metadata["shape"]))
            if resource in parameter_packs:
                if parameter_packs[resource] != signature:
                    raise ValueError(
                        f"parameter pack {resource!r} has inconsistent "
                        "dtype/size metadata"
                    )
                if parameter_metadata[resource] != resource_signature:
                    raise ValueError(
                        f"parameter resource {resource!r} has inconsistent "
                        "dtype/size metadata"
                    )
                continue
            if signature != (metadata["dtype"], metadata["shape"][0]):
                raise ValueError(
                    f"parameter pack {resource!r} metadata does not match "
                    "its resource"
                )
            parameter_packs[resource] = signature
            parameter_metadata[resource] = resource_signature

    wrappers = []
    seen_wrappers = set()
    for function_name, plan in plans:
        for operation in plan["operations"]:
            if operation["kind"] == "dispatch":
                wrapper = operation["wrapper"]
                if wrapper not in seen_wrappers:
                    seen_wrappers.add(wrapper)
                    wrappers.append(wrapper)
            elif operation["kind"] != "collective":
                raise ValueError(
                    f"unsupported operation kind {operation['kind']!r} "
                    f"in {function_name!r}"
                )

    model_id = config["model_id"]
    model_family = config["model_family"]
    vocab_file = config["tokens"]["vocab_file"]
    runner_uri = _normalize_dep_uri(
        runner_library or f"{model_family}_runner.so"
    )
    kernel_uri = _normalize_dep_uri(
        kernel_library or config["compilation"]["so_name"]
    )
    dep_uris = [_normalize_dep_uri(item) for item in dep_shared_libs or []]

    def buffer_symbol(function_name: str, resource: str) -> str:
        return f"{function_name}__{resource}"

    def operand_symbol(function_name: str, plan: dict, resource: str) -> str:
        metadata = plan["resources"].get(resource)
        if metadata is None:
            raise ValueError(
                f"operation in {function_name!r} references unknown "
                f"resource {resource!r}"
            )
        if metadata["role"] == "parameter":
            if resource not in parameter_packs:
                raise ValueError(
                    f"parameter resource {resource!r} in {function_name!r} "
                    "has no parameter-pack metadata"
                )
            return resource
        return buffer_symbol(function_name, resource)

    def quoted_resources(function_name: str, plan: dict, resources) -> str:
        return ", ".join(
            f'"{operand_symbol(function_name, plan, resource)}"'
            for resource in resources
        )

    def at_resources(function_name: str, plan: dict, resources) -> str:
        return ", ".join(
            f"@{operand_symbol(function_name, plan, resource)}"
            for resource in resources
        )

    out = StringIO()

    def p(*args, **kwargs):
        print(*args, file=out, **kwargs)

    p(f"rhal.module @{model_family}_rank{rank} attributes {{")
    p('    version = "0.1.0",')
    p(f'    model_name = "{model_id}",')
    p(f'    vocab_uri = "file:{vocab_file}",')
    p(f'    runner_library = "{runner_uri}"}} {{')
    p()

    for constant_id, resource in enumerate(parameter_packs, start=1):
        metadata = None
        for _, plan in plans:
            if resource in plan["resources"]:
                metadata = plan["resources"][resource]
                break
        tensor_type = _runtime_tensor_type(resource, metadata)
        p(
            f"  rhal.constant @{resource} {{id = {constant_id} : i32, "
            'storage = "external",'
        )
        p(f"                         type = {tensor_type},")
        p(f'                         uri = "file:{resource}.data"}}')
    if parameter_packs:
        p()

    code_object_symbols = {}
    for code_object_id, wrapper in enumerate(wrappers, start=1):
        symbol = f"scheduled_codeobj__{wrapper}"
        code_object_symbols[wrapper] = symbol
        p(
            f"  rhal.codeobj @{symbol} {{id = {code_object_id} : i32, "
            'kind = "host_shared_lib",'
        )
        p('                                backend = "cpu",')
        p(f'                                uri = "{kernel_uri}",')
        p(f'                                entry_symbol = "rax_{wrapper}"}}')
    for dependency_index, dep_uri in enumerate(dep_uris, start=1):
        code_object_id = len(wrappers) + dependency_index
        symbol = f"scheduled_runtime_dep_{dependency_index}"
        p(
            f"  rhal.codeobj @{symbol} {{id = {code_object_id} : i32, "
            'kind = "host_shared_lib",'
        )
        p('                                backend = "cpu",')
        p(f'                                uri = "{dep_uri}"}}')
    if wrappers or dep_uris:
        p()

    for function_name, plan in plans:
        for resource, metadata in plan["resources"].items():
            if metadata["role"] == "parameter":
                continue
            symbol = buffer_symbol(function_name, resource)
            tensor_type = _runtime_tensor_type(resource, metadata)
            p(
                f'  rhal.buffer @{symbol} {{space = "host", '
                f"type = {tensor_type}}}"
            )
    p()

    for function_index, (function_name, plan) in enumerate(plans):
        inputs = quoted_resources(function_name, plan, plan["runtime_inputs"])
        outputs = quoted_resources(function_name, plan, plan["runtime_outputs"])
        p(
            f"  rhal.func @{function_name} {{inputs = [{inputs}], "
            f"outputs = [{outputs}]}} body {{"
        )
        for operation in plan["operations"]:
            if operation["kind"] == "dispatch":
                arguments = at_resources(
                    function_name, plan, operation["arguments"]
                )
                code_object = code_object_symbols[operation["wrapper"]]
                p(f"    rhal.dispatch @{code_object} [{arguments}]")
                continue

            collective = operation["collective"]
            input_symbol = at_resources(
                function_name, plan, (operation["input"],)
            )
            output_symbol = at_resources(
                function_name, plan, (operation["output"],)
            )
            if collective == "all_reduce":
                if operation["input"] != operation["output"]:
                    raise ValueError(
                        f"all_reduce in {function_name!r} must be in-place"
                    )
                if operation.get("reduction") != "sum":
                    raise ValueError(
                        f"all_reduce in {function_name!r} requires sum "
                        "reduction"
                    )
                p(f"    rhal.collective [{input_symbol}] {{")
                p('      kind = "all_reduce",')
                p('      reduction = "sum"')
            elif collective == "all_gatherv":
                if operation["input"] == operation["output"]:
                    raise ValueError(
                        f"all_gatherv in {function_name!r} must be out-of-place"
                    )
                counts = ", ".join(
                    str(item) for item in operation["recv_counts"]
                )
                displacements = ", ".join(
                    str(item) for item in operation["displacements"]
                )
                p(f"    rhal.collective [{input_symbol}] {{")
                p('      kind = "all_gatherv",')
                p(f"      output_buffers = [{output_symbol}],")
                p(f"      recv_counts = array<i64: {counts}>,")
                p(f"      displacements = array<i64: {displacements}>")
            elif collective == "reduce_scatter":
                if operation["input"] == operation["output"]:
                    raise ValueError(
                        f"reduce_scatter in {function_name!r} must be "
                        "out-of-place"
                    )
                if operation.get("reduction") != "sum":
                    raise ValueError(
                        f"reduce_scatter in {function_name!r} requires sum "
                        "reduction"
                    )
                counts = ", ".join(
                    str(item) for item in operation["recv_counts"]
                )
                p(f"    rhal.collective [{input_symbol}] {{")
                p('      kind = "reduce_scatter",')
                p(f"      output_buffers = [{output_symbol}],")
                p(f"      recv_counts = array<i64: {counts}>,")
                p('      reduction = "sum"')
            else:
                raise ValueError(
                    f"unsupported collective {collective!r} in "
                    f"{function_name!r}"
                )
            p("    }")
        p("  }")
        if function_index + 1 != len(plans):
            p()

    p("}")
    return out.getvalue()


def _load_runtime_plans(items: list[str]) -> dict[str, dict]:
    runtime_plans = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"malformed --runtime-plan {item!r}; expected FUNCTION=PATH"
            )
        function_name, path = item.split("=", 1)
        if not function_name or not path:
            raise ValueError(
                f"malformed --runtime-plan {item!r}; expected FUNCTION=PATH"
            )
        if function_name in runtime_plans:
            raise ValueError(
                f"duplicate --runtime-plan function {function_name!r}"
            )
        with open(path) as file:
            runtime_plans[function_name] = json.load(file)
    return runtime_plans


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
        "--serving-library",
        default=None,
        metavar="URI_OR_NAME",
        help=(
            "Resident serving plugin library URI/name to place into module "
            "attrs. If no scheme is given, file: is assumed."
        ),
    )
    parser.add_argument(
        "--embedding-library",
        default=None,
        metavar="URI_OR_NAME",
        help=(
            "Embedding plugin library URI/name to place into module attrs. "
            "If no scheme is given, file: is assumed."
        ),
    )
    parser.add_argument(
        "--masked-lm-library",
        default=None,
        metavar="URI_OR_NAME",
        help=("Masked-LM plugin library URI/name to place into module attrs."),
    )
    parser.add_argument(
        "--transcription-library",
        default=None,
        metavar="URI_OR_NAME",
        help=(
            "Audio transcription plugin URI/name to place into module attrs."
        ),
    )
    parser.add_argument(
        "--runtime-plan",
        action="append",
        default=[],
        metavar="FUNCTION=PATH",
        help=(
            "Stage 4A rank-local runtime plan JSON for a body-form function "
            "(repeatable)"
        ),
    )
    parser.add_argument(
        "--kernel-library",
        default=None,
        metavar="URI_OR_NAME",
        help="Shared library used by scheduled dispatch code objects",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    try:
        if args.runtime_plan:
            runtime_plans = _load_runtime_plans(args.runtime_plan)
            mlir_text = gen_parallel_manifest(
                config,
                runtime_plans,
                dep_shared_libs=args.dep_shared_lib,
                runner_library=args.runner_library,
                kernel_library=args.kernel_library,
            )
        else:
            mlir_text = gen_manifest(
                config,
                dep_shared_libs=args.dep_shared_lib,
                runner_library=args.runner_library,
                serving_library=args.serving_library,
                embedding_library=args.embedding_library,
                masked_lm_library=args.masked_lm_library,
                transcription_library=args.transcription_library,
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
