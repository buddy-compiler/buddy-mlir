#!/usr/bin/env python3
"""Check that a Buddy Qwen3 layer-wise ELF/bin is safe to load on FPGA."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


RAM_ORIGIN = 0x86400000
RAM_END = 0xC0000000
STACK_SIZE = 0x100000
ALIGNMENT = 64

PAYLOAD_FILES = (
    "prefill_input_ids_i64.data",
    "prompt_length_i32.data",
    "prefill_rope_cos_f32.data",
    "prefill_rope_sin_f32.data",
    "prefill_mask_f32.data",
    "decode_rope_cos_f32.data",
    "decode_rope_sin_f32.data",
    "decode_mask_f32.data",
    "cache_write_mask_f32.data",
    "params_f32.data",
    "params_i8.data",
)

REQUIRED_SYMBOLS = (
    "_start",
    "main",
    "buddy_qwen3_layerwise_prefill",
    "buddy_qwen3_layerwise_decode",
    "buddy_qwen3_layerwise_generate_greedy",
    "buddy_qwen3_initialize_generated_ops",
    "_mlir_ciface_subgraph0_embedding_prefill",
    "_mlir_ciface_subgraph0_embedding_decode",
    "_mlir_ciface_subgraph0_decoder_layer_prefill",
    "_mlir_ciface_subgraph0_decoder_layer_decode",
    "_mlir_ciface_subgraph0_final_head",
    "_buddy_qwen3_params_f32",
    "_buddy_qwen3_params_i8",
    "_buddy_qwen3_prompt_length",
    "__image_start",
    "__state_start",
    "__state_end",
    "__model_start",
    "__model_end",
    "__stack_top",
)

AME_COMPONENTS = (
    "decoder_layer_prefill",
    "decoder_layer_decode",
    "final_head",
)


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def run(*command: str) -> str:
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def read_symbols(elf: Path) -> tuple[dict[str, int], dict[str, int]]:
    output = run("nm", "-n", "-S", str(elf))
    addresses: dict[str, int] = {}
    sizes: dict[str, int] = {}
    pattern = re.compile(
        r"^([0-9a-fA-F]+)(?:\s+([0-9a-fA-F]+))?\s+[A-Za-z]\s+(\S+)$"
    )
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        address, size, name = match.groups()
        addresses[name] = int(address, 16)
        if size is not None:
            sizes[name] = int(size, 16)
    return addresses, sizes


def model_section(elf: Path) -> tuple[int, int]:
    output = run("readelf", "-W", "-S", str(elf))
    pattern = re.compile(
        r"\[\s*\d+\]\s+\.model_blob\s+\S+\s+"
        r"([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)"
    )
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            return int(match.group(1), 16), int(match.group(2), 16)
    raise AssertionError("ELF has no .model_blob section")


def payload_size(artifacts: Path) -> int:
    total = 0
    for name in PAYLOAD_FILES:
        path = artifacts / name
        assert path.is_file(), f"missing payload: {path}"
        total = align(total)
        total += path.stat().st_size
    return align(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--heap-size", type=int, required=True)
    parser.add_argument("--heap-reserve", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--extra-model-blob", type=Path, action="append",
                        default=[])
    parser.add_argument("--require-tokenizer", action="store_true")
    args = parser.parse_args()

    assert args.elf.is_file(), f"missing ELF: {args.elf}"
    assert args.bin.is_file(), f"missing binary: {args.bin}"
    config = json.loads((args.artifacts / "model_compile_config.json").read_text())
    abi = json.loads((args.artifacts / "layerwise_abi.json").read_text())
    remaining_positions = abi["max_cache_len"] - abi["prefill_len"]
    assert 0 < args.decode_steps <= remaining_positions, (
        f"decode steps must be in [1, {remaining_positions}], got "
        f"{args.decode_steps}"
    )
    head_dim = abi["cache_shape"][-1]
    expected_rope_bytes = remaining_positions * head_dim * 4
    for name in ("decode_rope_cos_f32.data", "decode_rope_sin_f32.data"):
        actual_bytes = (args.artifacts / name).stat().st_size
        assert actual_bytes == expected_rope_bytes, (
            f"{name} is {actual_bytes} bytes, expected {expected_rope_bytes} "
            "for every available decode position"
        )
    arena_requirement = max(
        json.loads(
            (
                args.artifacts
                / f"lowering-{component}"
                / f"subgraph0_{component}.lowering.json"
            ).read_text()
        )["static_malloc_bytes"]
        for component in (
            "embedding_prefill",
            "embedding_decode",
            "decoder_layer_prefill",
            "decoder_layer_decode",
            "final_head",
        )
    )
    assert args.heap_size >= args.heap_reserve + arena_requirement, (
        f"heap is {args.heap_size} bytes but persistent runtime state plus one "
        f"component require {args.heap_reserve + arena_requirement} bytes"
    )
    packed_parameters = (
        (args.artifacts / "params_f32.data").stat().st_size
        + (args.artifacts / "params_i8.data").stat().st_size
    )
    assert packed_parameters == config["packed_parameter_bytes"]

    undefined = [line for line in run("nm", "-u", str(args.elf)).splitlines()
                 if line.strip()]
    assert not undefined, "undefined ELF symbols:\n" + "\n".join(undefined)

    symbols, symbol_sizes = read_symbols(args.elf)
    missing = [name for name in REQUIRED_SYMBOLS if name not in symbols]
    assert not missing, "missing required symbols: " + ", ".join(missing)
    if args.require_tokenizer:
        assert "_tokenizer_blob_start" in symbols
        assert "_tokenizer_blob_end" in symbols
        assert symbols["_tokenizer_blob_end"] > symbols["_tokenizer_blob_start"]
    assert "buddy_qwen3_generated_ops" not in symbols, (
        "absolute function-pointer table is unsafe under the FPGA address alias"
    )
    assert symbols["__image_start"] == RAM_ORIGIN
    assert symbols["__state_start"] <= symbols["__state_end"]
    assert symbols["__state_end"] <= symbols["__model_start"]
    assert symbols["__stack_top"] == symbols["__model_end"] + STACK_SIZE
    assert symbols["__stack_top"] <= RAM_END
    assert symbol_sizes.get("heap_buf") == args.heap_size, (
        f"heap_buf is {symbol_sizes.get('heap_buf')} bytes, expected "
        f"{args.heap_size}"
    )

    expected_blob_size = payload_size(args.artifacts)
    for extra_blob in args.extra_model_blob:
        assert extra_blob.is_file(), f"missing extra model blob: {extra_blob}"
        expected_blob_size = align(expected_blob_size)
        expected_blob_size += extra_blob.stat().st_size
    expected_blob_size = align(expected_blob_size)
    section_address, section_size = model_section(args.elf)
    assert section_address == symbols["__model_start"]
    assert section_size == expected_blob_size, (
        f".model_blob is {section_size} bytes, expected {expected_blob_size}"
    )
    assert symbols["__model_end"] - symbols["__model_start"] == section_size

    image_size = args.bin.stat().st_size
    expected_image_size = symbols["__model_end"] - RAM_ORIGIN
    assert image_size == expected_image_size, (
        f"flat image is {image_size} bytes, expected {expected_image_size}"
    )
    assert image_size % ALIGNMENT == 0
    ddr_high_word = image_size // ALIGNMENT - 1

    ame_instruction_count = 0
    for component in AME_COMPONENTS:
        assembly = args.elf.parent / f"{component}_bare.s"
        assert assembly.is_file(), f"missing encoded AME assembly: {assembly}"
        text = assembly.read_text()
        component_mma_count = text.count("# mqma.b.mm")
        assert component_mma_count > 0, f"{component} has no encoded AME MMA"
        assert "# mlae8.m" in text, f"{component} has no encoded AME A load"
        assert "# mlbe8.m" in text, f"{component} has no encoded AME B load"
        assert "# msce32.m" in text, f"{component} has no encoded AME C store"
        ame_instruction_count += component_mma_count

    print("Buddy Qwen3 layer-wise FPGA image verification: PASS")
    print(f"  ELF: {args.elf} ({args.elf.stat().st_size} bytes)")
    print(f"  model blob: {section_size} bytes")
    print(f"  encoded AME MMA instructions: {ame_instruction_count}")
    print(f"  static state + arena: "
          f"{symbols['__state_end'] - symbols['__state_start']} bytes")
    if args.heap_reserve:
        print(f"  persistent heap reserve: {args.heap_reserve} bytes")
    print(f"  flat binary: {image_size} bytes ({image_size / (1024**2):.2f} MiB)")
    print(f"  DDR remaining after 1 MiB stack: "
          f"{RAM_END - symbols['__stack_top']} bytes")
    print(f"  writemem range: ariane_xilinx.i_ddr[{ddr_high_word}:0]")


if __name__ == "__main__":
    main()
