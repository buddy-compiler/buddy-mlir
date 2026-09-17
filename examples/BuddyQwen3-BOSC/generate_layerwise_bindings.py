#!/usr/bin/env python3
"""Generate direct C-wrapper bindings for the Qwen3 layer-wise scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DTYPE_SIZE = {"int8": 1, "float32": 4, "int64": 8}
PARAM_FILES = {"params_i8.data": 0, "params_f32.data": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def c_shape(shape: list[int]) -> str:
    padded = list(shape) + [1] * (4 - len(shape))
    return "{" + ", ".join(str(value) for value in padded) + "}"


def parameter_spec(entry: dict, expected: dict) -> tuple[int, int, int, list[int]]:
    if entry["dtype"] != expected["dtype"]:
        raise ValueError(
            f"dtype mismatch for {entry['model_parameter']}: "
            f"{entry['dtype']} != {expected['dtype']}"
        )
    if entry["shape"] != expected["shape"]:
        raise ValueError(
            f"shape mismatch for {entry['model_parameter']}: "
            f"{entry['shape']} != {expected['shape']}"
        )
    if entry["file"] not in PARAM_FILES:
        raise ValueError(f"unsupported parameter file: {entry['file']}")
    rank = len(entry["shape"])
    if rank < 1 or rank > 4:
        raise ValueError(f"unsupported parameter rank {rank}")
    return (
        PARAM_FILES[entry["file"]],
        int(entry["element_offset"]) * DTYPE_SIZE[entry["dtype"]],
        rank,
        entry["shape"],
    )


def emit_spec(spec: tuple[int, int, int, list[int]]) -> str:
    file_kind, offset, rank, shape = spec
    return f"{{{file_kind}, {rank}, {offset}ULL, {c_shape(shape)}}}"


def wrapper_declaration(name: str, argument_count: int) -> str:
    arguments = ", ".join(["void *"] * (argument_count + 1))
    return f"extern void {name}({arguments});"


def runtime_arguments(call: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for argument in call["arguments"]:
        if argument["source"] != "runtime":
            continue
        label = argument["runtime_tensor"]
        previous = result.get(label)
        signature = {"shape": argument["shape"], "dtype": argument["dtype"]}
        if previous is not None and previous != signature:
            raise ValueError(f"runtime tensor {label} has inconsistent shapes")
        result[label] = signature
    return result


def member_for(argument: dict, expression: str) -> str:
    rank = len(argument["shape"])
    return f"&{expression}.r{rank}"


def call_arguments(call: dict, runtime_map: dict[str, str]) -> list[str]:
    result = []
    for argument in call["arguments"]:
        if argument["source"] == "parameter":
            index = int(argument["graph_parameter_index"])
            result.append(member_for(argument, f"parameter_descriptors[{index}]"))
        else:
            label = argument["runtime_tensor"]
            if label not in runtime_map:
                raise ValueError(f"no C runtime pointer for {label}")
            result.append(member_for(argument, runtime_map[label]))
    return result


def emit_runtime_initializers(call: dict, runtime_map: dict[str, str]) -> list[str]:
    lines = []
    for label, signature in runtime_arguments(call).items():
        variable, pointer = runtime_map[label].split(":", maxsplit=1)
        rank = len(signature["shape"])
        lines.append(
            f"  init_memref(&{variable}, (void *){pointer}, {rank}, "
            f"(const int64_t[4]){c_shape(signature['shape'])});"
        )
    return lines


def parameter_table_for_bound_call(
    call: dict, manifest: dict, bindings: list[dict]
) -> list[tuple[int, int, int, list[int]]]:
    by_graph_index = {
        int(binding["graph_parameter_index"]): binding for binding in bindings
    }
    parameter_arguments = [
        argument for argument in call["arguments"] if argument["source"] == "parameter"
    ]
    count = max(int(argument["graph_parameter_index"]) for argument in parameter_arguments) + 1
    result = [None] * count
    for argument in parameter_arguments:
        graph_index = int(argument["graph_parameter_index"])
        binding = by_graph_index[graph_index]
        entry = manifest["parameters"][int(binding["pack_manifest_index"])]
        expected = dict(argument)
        expected["shape"] = binding["shape"]
        expected["dtype"] = binding["dtype"]
        result[graph_index] = parameter_spec(entry, expected)
    if any(spec is None for spec in result):
        raise ValueError("parameter graph indices are not dense")
    return result


def parameter_table_for_call(call: dict, manifest: dict) -> list:
    arguments = [
        argument for argument in call["arguments"] if argument["source"] == "parameter"
    ]
    count = max(int(argument["graph_parameter_index"]) for argument in arguments) + 1
    result = [None] * count
    for argument in arguments:
        graph_index = int(argument["graph_parameter_index"])
        entry = manifest["parameters"][int(argument["pack_manifest_index"])]
        result[graph_index] = parameter_spec(entry, argument)
    if any(spec is None for spec in result):
        raise ValueError("parameter graph indices are not dense")
    return result


def emit_table(name: str, rows: list[list]) -> list[str]:
    width = len(rows[0])
    lines = [f"static const ParamSpec {name}[{len(rows)}][{width}] = {{"]
    for row in rows:
        lines.append("  {")
        lines.extend(f"    {emit_spec(spec)}," for spec in row)
        lines.append("  },")
    lines.append("};")
    return lines


def emit_adapter(
    name: str,
    call: dict,
    table_name: str,
    table_row: str,
    runtime_declarations: list[str],
    runtime_map_with_pointers: dict[str, str],
    signature: str,
) -> list[str]:
    parameter_count = 1 + max(
        int(argument["graph_parameter_index"])
        for argument in call["arguments"]
        if argument["source"] == "parameter"
    )
    runtime_map = {
        label: value.split(":", maxsplit=1)[0]
        for label, value in runtime_map_with_pointers.items()
    }
    arguments = call_arguments(call, runtime_map)
    lines = [f"static int {name}({signature}) {{"]
    lines.extend(f"  AnyMemRef {declaration};" for declaration in runtime_declarations)
    lines.append(f"  AnyMemRef parameter_descriptors[{parameter_count}];")
    lines.append("  if (!params || !params->params_i8 || !params->params_f32)")
    lines.append("    return BUDDY_QWEN3_ERR_ARGUMENT;")
    lines.append(
        f"  init_parameters(parameter_descriptors, {table_name}[{table_row}], "
        f"{parameter_count}, params);"
    )
    lines.extend(emit_runtime_initializers(call, runtime_map_with_pointers))
    call_text = ",\n      ".join(["result"] + arguments)
    lines.append(f"  {call['c_wrapper']}(\n      {call_text});")
    lines.append("  return BUDDY_QWEN3_OK;")
    lines.append("}")
    return lines


def generate_bindings(abi_path: Path, manifest_path: Path, output_path: Path) -> None:
    abi = json.loads(abi_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if abi.get("format") != "buddy-qwen3-layerwise-abi-v1":
        raise ValueError("unsupported layer-wise ABI")

    calls = abi["call_abi"]
    embedding_prefill = parameter_table_for_call(calls["embedding_prefill"], manifest)
    embedding_decode = parameter_table_for_call(calls["embedding_decode"], manifest)
    final_head = parameter_table_for_call(calls["final_head"], manifest)
    if embedding_prefill != embedding_decode:
        raise ValueError("prefill/decode embedding parameter bindings differ")

    prefill_rows = []
    decode_rows = []
    for layer in abi["layers"]:
        prefill_rows.append(
            parameter_table_for_bound_call(
                calls["decoder_layer_prefill"],
                manifest,
                layer["prefill_parameter_bindings"],
            )
        )
        decode_rows.append(
            parameter_table_for_bound_call(
                calls["decoder_layer_decode"],
                manifest,
                layer["decode_parameter_bindings"],
            )
        )
    if prefill_rows != decode_rows:
        raise ValueError("prefill/decode layer parameter bindings differ")

    hidden_size = calls["embedding_prefill"]["outputs"][0]["shape"][-1]
    decode_runtime = runtime_arguments(calls["decoder_layer_decode"])
    kv_heads = decode_runtime["old_key"]["shape"][1]
    head_dim = decode_runtime["old_key"]["shape"][3]
    vocab_size = calls["final_head"]["outputs"][0]["shape"][-1]

    lines = [
        "/* Generated by generate_layerwise_bindings.py. Do not edit. */",
        '#include "layerwise_scheduler.h"',
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "extern void *malloc(size_t size);",
        "",
        "typedef union {",
        "  BuddyQwen3MemRef1 r1;",
        "  BuddyQwen3MemRef2 r2;",
        "  BuddyQwen3MemRef3 r3;",
        "  BuddyQwen3MemRef4 r4;",
        "} AnyMemRef;",
        "",
        "typedef struct {",
        "  uint8_t file_kind;",
        "  uint8_t rank;",
        "  uint64_t byte_offset;",
        "  int64_t sizes[4];",
        "} ParamSpec;",
        "",
        "static void init_memref(AnyMemRef *memref, void *data, int rank,",
        "                        const int64_t sizes[4]) {",
        "  int64_t stride = 1;",
        "  memref->r4.allocated = data;",
        "  memref->r4.aligned = data;",
        "  memref->r4.offset = 0;",
        "  if (rank == 1) {",
        "    memref->r1.sizes[0] = sizes[0];",
        "    memref->r1.strides[0] = 1;",
        "  } else if (rank == 2) {",
        "    for (int dim = 1; dim >= 0; --dim) {",
        "      memref->r2.sizes[dim] = sizes[dim];",
        "      memref->r2.strides[dim] = stride;",
        "      stride *= sizes[dim];",
        "    }",
        "  } else if (rank == 3) {",
        "    for (int dim = 2; dim >= 0; --dim) {",
        "      memref->r3.sizes[dim] = sizes[dim];",
        "      memref->r3.strides[dim] = stride;",
        "      stride *= sizes[dim];",
        "    }",
        "  } else {",
        "    for (int dim = 3; dim >= 0; --dim) {",
        "      memref->r4.sizes[dim] = sizes[dim];",
        "      memref->r4.strides[dim] = stride;",
        "      stride *= sizes[dim];",
        "    }",
        "  }",
        "}",
        "",
        "static void init_parameters(AnyMemRef *descriptors,",
        "                            const ParamSpec *specs, size_t count,",
        "                            const BuddyQwen3PackedParams *params) {",
        "  for (size_t index = 0; index < count; ++index) {",
        "    const uint8_t *base = specs[index].file_kind == 0",
        "                              ? params->params_i8",
        "                              : (const uint8_t *)params->params_f32;",
        "    init_memref(&descriptors[index], (void *)(base + specs[index].byte_offset),",
        "                specs[index].rank, specs[index].sizes);",
        "  }",
        "}",
        "",
    ]
    for call in calls.values():
        lines.append(wrapper_declaration(call["c_wrapper"], len(call["arguments"])))
    lines.append("")
    lines.extend(emit_table("embedding_params", [embedding_prefill]))
    lines.append("")
    lines.extend(emit_table("layer_params", prefill_rows))
    lines.append("")
    lines.extend(emit_table("final_head_params", [final_head]))
    lines.append("")

    lines.extend(
        [
            "#ifndef BUDDY_QWEN3_USE_MLIR_EMBEDDING",
            "#define BUDDY_QWEN3_USE_MLIR_EMBEDDING 0",
            "#endif",
            "",
            "#if BUDDY_QWEN3_USE_MLIR_EMBEDDING",
        ]
    )
    lines.extend(
        emit_adapter(
            "generated_embedding_prefill",
            calls["embedding_prefill"],
            "embedding_params",
            "0",
            ["input_ids_desc"],
            {"input_ids": "input_ids_desc:input_ids"},
            "const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result, "
            "const int64_t *input_ids",
        )
    )
    lines.append("")
    lines.extend(
        emit_adapter(
            "generated_embedding_decode",
            calls["embedding_decode"],
            "embedding_params",
            "0",
            ["input_ids_desc"],
            {"input_ids": "input_ids_desc:input_id"},
            "const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result, "
            "const int64_t *input_id",
        )
    )
    lines.append("")
    lines.append("#else")
    lines.extend(
        [
            "/*",
            " * Keep the imported MLIR embedding adapters above as an A/B fallback.",
            " * The scalar MLIR gather currently stalls on FPGA for high vocabulary",
            " * rows.  This equivalent leaf keeps the packed embedding ABI and emits",
            " * the same int8 * per-256-column f32 scale computation.",
            " */",
            "static int generated_quantized_embedding(",
            "    const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result,",
            "    const int64_t *input_ids, int64_t tokens) {",
            "  const ParamSpec *weight_spec = &embedding_params[0][0];",
            "  const ParamSpec *scale_spec = &embedding_params[0][1];",
            "  const int64_t vocab_size = weight_spec->sizes[0];",
            "  const int64_t hidden_size = weight_spec->sizes[1];",
            "  const int64_t scale_groups = scale_spec->sizes[1];",
            "  const int64_t group_size = hidden_size / scale_groups;",
            "  const int8_t *weight;",
            "  const float *scales;",
            "  uint8_t *allocated;",
            "  float *output;",
            "  size_t elements;",
            "",
            "  if (!params || !params->params_i8 || !params->params_f32 || !result ||",
            "      !input_ids || tokens <= 0 || hidden_size <= 0 || scale_groups <= 0 ||",
            "      hidden_size % scale_groups != 0)",
            "    return BUDDY_QWEN3_ERR_ARGUMENT;",
            "",
            "  weight = (const int8_t *)(params->params_i8 + weight_spec->byte_offset);",
            "  scales = (const float *)((const uint8_t *)params->params_f32 +",
            "                           scale_spec->byte_offset);",
            "  elements = (size_t)tokens * (size_t)hidden_size;",
            "  allocated = (uint8_t *)malloc(elements * sizeof(float) + 63u);",
            "  if (!allocated)",
            "    return BUDDY_QWEN3_ERR_KERNEL;",
            "  output = (float *)(((uintptr_t)allocated + 63u) & ~(uintptr_t)63u);",
            "",
            "  for (int64_t row = 0; row < tokens; ++row) {",
            "    int64_t token = input_ids[row];",
            "    if (token < 0 || token >= vocab_size)",
            "      return BUDDY_QWEN3_ERR_ARGUMENT;",
            "    for (int64_t column = 0; column < hidden_size; ++column) {",
            "      output[row * hidden_size + column] =",
            "          (float)weight[token * hidden_size + column] *",
            "          scales[token * scale_groups + column / group_size];",
            "    }",
            "  }",
            "",
            "  result->allocated = allocated;",
            "  result->aligned = output;",
            "  result->offset = 0;",
            "  result->sizes[0] = 1;",
            "  result->sizes[1] = tokens;",
            "  result->sizes[2] = hidden_size;",
            "  result->strides[0] = tokens * hidden_size;",
            "  result->strides[1] = hidden_size;",
            "  result->strides[2] = 1;",
            "  return BUDDY_QWEN3_OK;",
            "}",
            "",
            "static int generated_embedding_prefill(",
            "    const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result,",
            "    const int64_t *input_ids) {",
            f"  return generated_quantized_embedding(params, result, input_ids, {abi['prefill_len']});",
            "}",
            "",
            "static int generated_embedding_decode(",
            "    const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result,",
            "    const int64_t *input_id) {",
            "  return generated_quantized_embedding(params, result, input_id, 1);",
            "}",
            "",
            "#endif",
            "",
        ]
    )
    lines.extend(
        emit_adapter(
            "generated_decoder_layer_prefill",
            calls["decoder_layer_prefill"],
            "layer_params",
            "layer",
            [
                "hidden_desc",
                "rope_cos_desc",
                "rope_sin_desc",
                "attention_mask_desc",
            ],
            {
                "hidden": "hidden_desc:hidden_data",
                "rope_cos": "rope_cos_desc:rope_cos_data",
                "rope_sin": "rope_sin_desc:rope_sin_data",
                "attention_mask": "attention_mask_desc:attention_mask_data",
            },
            "const BuddyQwen3PackedParams *params, int32_t layer, "
            "BuddyQwen3LayerResult *result, const float *hidden_data, "
            "const float *rope_cos_data, const float *rope_sin_data, "
            "const float *attention_mask_data",
        )
    )
    lines.append("")

    decode_lines = emit_adapter(
        "generated_decoder_layer_decode",
        calls["decoder_layer_decode"],
        "layer_params",
        "layer",
        [
            "hidden_desc",
            "rope_cos_desc",
            "rope_sin_desc",
            "attention_mask_desc",
            "cache_write_mask_desc",
            "old_key_desc",
            "old_value_desc",
        ],
        {
            "hidden": "hidden_desc:hidden_data",
            "rope_cos": "rope_cos_desc:rope_cos_data",
            "rope_sin": "rope_sin_desc:rope_sin_data",
            "attention_mask": "attention_mask_desc:attention_mask_data",
            "cache_write_mask": "cache_write_mask_desc:cache_write_mask_data",
            "old_key": "old_key_desc:old_key_data",
            "old_value": "old_value_desc:old_value_data",
        },
        "const BuddyQwen3PackedParams *params, int32_t layer, "
        "BuddyQwen3LayerResult *result, const float *hidden_data, "
        "const float *rope_cos_data, const float *rope_sin_data, "
        "const float *attention_mask_data, const float *cache_write_mask_data, "
        "const float *old_key_data, const float *old_value_data",
    )
    lines.extend(decode_lines)
    lines.append("")
    final_lines = emit_adapter(
        "generated_final_head",
        calls["final_head"],
        "final_head_params",
        "0",
        ["hidden_desc"],
        {"hidden": "hidden_desc:last_hidden"},
        "const BuddyQwen3PackedParams *params, BuddyQwen3MemRef3 *result, "
        "const float *last_hidden",
    )
    lines.extend(final_lines)
    lines.append("")
    lines.extend(
        [
            "const BuddyQwen3ModelSpec buddy_qwen3_generated_spec = {",
            f"  {abi['prefill_len']}, {abi['max_cache_len']}, {abi['num_layers']},",
            f"  {hidden_size}, {kv_heads}, {head_dim}, {vocab_size},",
            "};",
            "",
            "void buddy_qwen3_initialize_generated_ops(BuddyQwen3LayerwiseOps *ops) {",
            "  if (!ops)",
            "    return;",
            "#if !BUDDY_QWEN3_USE_MLIR_EMBEDDING",
            "  /* Retain the linkable MLIR A/B fallback under --gc-sections. */",
            "  __asm__ volatile (\"\" : : \"r\"(_mlir_ciface_subgraph0_embedding_prefill));",
            "  __asm__ volatile (\"\" : : \"r\"(_mlir_ciface_subgraph0_embedding_decode));",
            "#endif",
            "  ops->embedding_prefill = generated_embedding_prefill;",
            "  ops->embedding_decode = generated_embedding_decode;",
            "  ops->decoder_layer_prefill = generated_decoder_layer_prefill;",
            "  ops->decoder_layer_decode = generated_decoder_layer_decode;",
            "  ops->final_head = generated_final_head;",
            "}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    output = args.output or args.build_dir / "layerwise_bindings.c"
    generate_bindings(
        args.build_dir / "layerwise_abi.json",
        args.build_dir / "params_manifest.json",
        output,
    )
    print(f"generated layer-wise C bindings: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
