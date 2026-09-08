#!/usr/bin/env python3
"""Check the Qwen3 layer-wise ABI, parameter bindings, and lowering output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COMPONENTS = (
    "embedding_prefill",
    "embedding_decode",
    "decoder_layer_prefill",
    "decoder_layer_decode",
    "final_head",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument(
        "--require-lowering",
        action="store_true",
        help="also require the five LLVM lowering reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.build_dir
    abi = json.loads((root / "layerwise_abi.json").read_text())
    manifest = json.loads((root / "params_manifest.json").read_text())

    assert abi["format"] == "buddy-qwen3-layerwise-abi-v1"
    assert abi["num_layers"] == 28
    assert len(abi["layers"]) == 28
    assert len(abi["cache_shape"]) == 4
    assert abi["cache_shape"][0] == 28
    assert abi["cache_shape"][1:] == [8, abi["max_cache_len"], 128]
    assert abi["decode_rope_positions"] == [
        abi["prefill_len"],
        abi["max_cache_len"],
    ]
    assert abi["prefill_padding"] == {
        "side": "left",
        "pad_token_id": 151643,
        "actual_length_file": "prompt_length_i32.data",
    }
    config = json.loads((root / "model_compile_config.json").read_text())
    actual_prompt_len = int(config["actual_prompt_len"])
    assert 0 < actual_prompt_len <= abi["prefill_len"]
    prompt_bytes = (root / "prompt_length_i32.data").read_bytes()
    assert len(prompt_bytes) == 4
    prompt_length = int.from_bytes(prompt_bytes, "little", signed=True)
    assert prompt_length == actual_prompt_len
    assert (root / "prefill_input_ids_i64.data").stat().st_size == (
        abi["prefill_len"] * 8
    )
    decode_positions = abi["max_cache_len"] - abi["prefill_len"]
    expected_rope_bytes = decode_positions * 128 * 4
    for name in ("decode_rope_cos_f32.data", "decode_rope_sin_f32.data"):
        path = root / name
        assert path.stat().st_size == expected_rope_bytes, (
            f"{name} must contain every decode position: "
            f"{path.stat().st_size} != {expected_rope_bytes}"
        )

    packed = manifest["parameters"]
    for index, entry in enumerate(packed):
        assert entry.get("model_parameter"), f"unnamed pack entry {index}"
    for layer in abi["layers"]:
        assert layer["layer"] in range(28)
        for key in ("prefill_parameter_bindings", "decode_parameter_bindings"):
            bindings = layer[key]
            assert len(bindings) == 18
            for binding in bindings:
                pack_index = binding["pack_manifest_index"]
                assert 0 <= pack_index < len(packed)
                packed_entry = packed[pack_index]
                assert packed_entry["model_parameter"] == binding["model_parameter"]
                assert packed_entry["kind"] == binding["kind"]

    for component in COMPONENTS:
        call = abi["call_abi"][component]
        assert call["result_pointer_index"] == 0
        assert call["c_wrapper"] == f"_mlir_ciface_subgraph0_{component}"
        assert [arg["position"] for arg in call["arguments"]] == list(
            range(len(call["arguments"]))
        )
        assert [arg["c_wrapper_pointer_index"] for arg in call["arguments"]] == list(
            range(1, len(call["arguments"]) + 1)
        )
        for argument in call["arguments"]:
            if argument["source"] == "parameter":
                pack_index = argument["pack_manifest_index"]
                assert packed[pack_index]["kind"] == argument["kind"]
            else:
                assert argument["source"] == "runtime"
                assert argument["runtime_tensor"]

        source = root / f"subgraph0_{component}.mlir"
        assert source.is_file(), source
        text = source.read_text()
        if component.startswith("decoder_layer"):
            assert text.count("bosc_ame.w8a8_linear") == 7
            assert text.count("bosc_ame.quantize_per_group") == 7
            assert "linalg.matmul" not in text
        elif component == "final_head":
            assert text.count("bosc_ame.w8a8_linear") == 1
            assert text.count("bosc_ame.quantize_per_group") == 1

        if args.require_lowering:
            lowered_dir = root / f"lowering-{component}"
            report = json.loads(
                (lowered_dir / f"subgraph0_{component}.lowering.json").read_text()
            )
            simplified = (
                lowered_dir / f"subgraph0_{component}.01-simplified.mlir"
            ).read_text()
            fusion_enabled = report.get(
                "w8a8_silu_mul_quantize_fusion", False
            )
            gate_up_pair_enabled = report.get(
                "w8a8_gate_up_linear_fusion", False
            )
            if component.startswith("decoder_layer"):
                pair_expected = (
                    gate_up_pair_enabled
                    and component == "decoder_layer_decode"
                )
                expected_regular_linear = 5 if pair_expected else 7
                expected_linear_pairs = 1 if pair_expected else 0
                assert simplified.count(
                    '"bosc_ame.w8a8_linear"'
                ) == expected_regular_linear
                assert simplified.count(
                    '"bosc_ame.w8a8_linear_pair"'
                ) == expected_linear_pairs
                expected_regular_quantize = 3 if fusion_enabled else 4
                expected_fused_quantize = 1 if fusion_enabled else 0
                assert simplified.count(
                    '"bosc_ame.quantize_per_group"'
                ) == expected_regular_quantize, (
                    f"{component} must share activation quantization across "
                    "Q/K/V and Gate/Up and preserve the configured Down "
                    "quantization path before bufferization"
                )
                assert simplified.count(
                    '"bosc_ame.silu_mul_quantize_per_group"'
                ) == expected_fused_quantize, (
                    f"{component} Down SiLU/quantize fusion count does not "
                    "match its lowering report"
                )
            elif component == "final_head":
                assert simplified.count('"bosc_ame.w8a8_linear"') == 1
                assert '"bosc_ame.w8a8_linear_pair"' not in simplified
                assert simplified.count(
                    '"bosc_ame.quantize_per_group"'
                ) == 1
                assert (
                    '"bosc_ame.silu_mul_quantize_per_group"'
                    not in simplified
                )
            assert report["remaining_tosa_ops"] == 0
            assert report["remaining_linalg_ops"] == 0
            boscame = (
                lowered_dir / f"subgraph0_{component}.04-boscame.mlir"
            ).read_text()
            assert "bosc_ame.w8a8_linear" not in boscame
            assert "bosc_ame.w8a8_linear_pair" not in boscame
            assert "bosc_ame.quantize_per_group" not in boscame
            assert "bosc_ame.silu_mul_quantize_per_group" not in boscame
            if component == "decoder_layer_decode":
                bufferized = (
                    lowered_dir / f"subgraph0_{component}.03-bufferized.mlir"
                ).read_text()
                full_cache = (
                    f"1x8x{abi['max_cache_len']}x128xf32"
                )
                full_cache_allocs = re.findall(
                    rf"memref\.alloc\(\)[^\n]*: memref<{full_cache}>",
                    bufferized,
                )
                assert not full_cache_allocs, (
                    "decode lowering must read the persistent K/V cache in "
                    f"place, not allocate a functional {full_cache} copy"
                )

    decode_runtime_shapes = {
        argument["runtime_tensor"]: argument["shape"]
        for argument in abi["call_abi"]["decoder_layer_decode"]["arguments"]
        if argument["source"] == "runtime"
    }
    assert decode_runtime_shapes["cache_write_mask"] == [
        1,
        1,
        abi["max_cache_len"],
        1,
    ]
    assert decode_runtime_shapes["old_key"] == [1, 8, abi["max_cache_len"], 128]
    assert decode_runtime_shapes["old_value"] == [1, 8, abi["max_cache_len"], 128]
    assert decode_runtime_shapes["attention_mask"] == [1, 1, 1, abi["max_cache_len"]]
    expected_layer_outputs = ["new_value", "new_key", "hidden"]
    for component in ("decoder_layer_prefill", "decoder_layer_decode"):
        assert [
            output["runtime_tensor"]
            for output in abi["call_abi"][component]["outputs"]
        ] == expected_layer_outputs, (
            f"{component} C ABI must preserve the generated "
            "(value, key, hidden) result order"
        )
    assert abi["decode_cache_update"] == {
        "mode": "single_position",
        "output_shape": [1, 8, 1, 128],
        "position_source": "cache_write_mask",
    }
    decode_outputs = {
        output["runtime_tensor"]: output["shape"]
        for output in abi["call_abi"]["decoder_layer_decode"]["outputs"]
    }
    assert decode_outputs["new_key"] == [1, 8, 1, 128]
    assert decode_outputs["new_value"] == [1, 8, 1, 128]
    bindings = root / "layerwise_bindings.c"
    assert bindings.is_file(), bindings
    binding_text = bindings.read_text()
    for component in COMPONENTS:
        assert abi["call_abi"][component]["c_wrapper"] in binding_text
    assert (
        "void buddy_qwen3_initialize_generated_ops(" in binding_text
    ), "generated runtime dispatch initializer is missing"
    assert (
        "const BuddyQwen3LayerwiseOps buddy_qwen3_generated_ops" not in binding_text
    ), "link-time function pointer table is unsafe under the FPGA runtime alias"

    print("Qwen3 layer-wise artifacts verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
