#!/usr/bin/env python3
"""Lower Buddy Frontend TOSA/Linalg to the BOSCAME/VIR LLVM backend."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUDDY_ROOT = HERE.parents[1]
BUDDY_OPT = BUDDY_ROOT / "build/bin/buddy-opt"
BUDDY_TRANSLATE = BUDDY_ROOT / "build/bin/buddy-translate"
MLIR_OPT = BUDDY_ROOT / "llvm/build/bin/mlir-opt"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def make_w8a8_pass(args: argparse.Namespace) -> str:
    """Build the W8A8 lowering option string, including A/B fallbacks."""
    option_list = []
    if args.w8a8_scalar_fallback:
        option_list.append("scalar-fallback")
    if args.w8a8_profile_phases:
        option_list.append("profile-phases")
    if args.w8a8_disable_decode_n128:
        option_list.append("experimental-decode-n128=false")
    if args.w8a8_disable_quantize_unroll:
        option_list.append("quantize-unroll=false")
    if args.w8a8_quantize_reciprocal:
        option_list.append("quantize-reciprocal=true")
    if args.w8a8_quantize_one_ahead:
        option_list.append("quantize-one-ahead=true")
    result = "--lower-qwen-w8a8-to-boscame"
    if option_list:
        result += f"={' '.join(option_list)}"
    return result


def make_simplify_passes(args: argparse.Namespace) -> list[str]:
    """Build the pre-bufferization Qwen graph simplification pipeline."""
    passes = ["--simplify-tosa-reshape", "--cse"]
    if args.w8a8_gate_up_linear_fusion:
        passes.append("--fuse-qwen-gate-up-w8a8-linear")
    if args.w8a8_silu_mul_quantize_fusion:
        passes.extend(["--fuse-qwen-silu-mul-quantize", "--cse"])
    return passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Buddy Qwen3 TOSA/Linalg -> BOSCAME/VIR -> LLVM IR"
    )
    parser.add_argument("input", type=Path, help="Buddy Frontend MLIR module")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-width", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=("vir", "scalar"),
        default="vir",
        help="Use VIR vectors or the scalar affine-loop fallback",
    )
    parser.add_argument(
        "--emit-c-wrapper",
        action="store_true",
        help="Emit _mlir_ciface_* wrappers for public functions",
    )
    parser.add_argument(
        "--cycle-trace",
        action="store_true",
        help=(
            "Lower buddy_trace.start/end to the bare-metal cycle trace "
            "runtime hooks"
        ),
    )
    parser.add_argument(
        "--optimize-attention-bmm",
        action="store_true",
        help=(
            "Use Buddy's N-vectorized, M-unrolled batch-matmul lowering for "
            "the QK/PV attention matmuls. This is intended for static "
            "prefill sequence lengths divisible by the vector width."
        ),
    )
    parser.add_argument(
        "--w8a8-scalar-fallback",
        action="store_true",
        help=(
            "Use the original N64 AME schedule and scalar fp32 accumulation "
            "for W8A8 FPGA A/B diagnosis"
        ),
    )
    parser.add_argument(
        "--w8a8-profile-phases",
        action="store_true",
        help=(
            "Instrument W8A8 phase boundaries with the bare-metal cycle "
            "trace runtime (IDs 249 through 255)"
        ),
    )
    parser.add_argument(
        "--w8a8-experimental-decode-n128",
        action="store_true",
        help=(
            "Compatibility no-op: paired OUTBLK64 acc0..acc7 decode N128 "
            "is enabled by default"
        ),
    )
    parser.add_argument(
        "--w8a8-disable-decode-n128",
        action="store_true",
        help=(
            "Disable paired decode N128 and retain the reliable N64 fallback"
        ),
    )
    parser.add_argument(
        "--w8a8-disable-quantize-unroll",
        action="store_true",
        help=(
            "Disable the GS512/1024 scalar eight-way quantizer unroll and "
            "retain the original element loop for FPGA A/B"
        ),
    )
    quantize_reciprocal = parser.add_mutually_exclusive_group()
    quantize_reciprocal.add_argument(
        "--w8a8-enable-quantize-reciprocal",
        dest="w8a8_quantize_reciprocal",
        action="store_true",
        help=(
            "Experimentally use the handwritten reciprocal/multiply "
            "quantizer; FPGA checkpoints can differ at rounding boundaries"
        ),
    )
    quantize_reciprocal.add_argument(
        "--w8a8-disable-quantize-reciprocal",
        dest="w8a8_quantize_reciprocal",
        action="store_false",
        help=(
            "Retain bit-stable per-element division (default; compatibility "
            "spelling for explicit A/B commands)"
        ),
    )
    parser.set_defaults(w8a8_quantize_reciprocal=False)
    quantize_one_ahead = parser.add_mutually_exclusive_group()
    quantize_one_ahead.add_argument(
        "--w8a8-enable-quantize-one-ahead",
        dest="w8a8_quantize_one_ahead",
        action="store_true",
        help=(
            "Use the FPGA-validated exact fdiv.s one-ahead helper for "
            "GS512/1024 quantize writeback (default)"
        ),
    )
    quantize_one_ahead.add_argument(
        "--w8a8-disable-quantize-one-ahead",
        dest="w8a8_quantize_one_ahead",
        action="store_false",
        help="Retain the inlined scalar quantize-write loop for FPGA A/B",
    )
    # Resolve the default after parsing reciprocal: the exact one-ahead path
    # is the measured production default, while explicitly selecting the
    # non-bit-exact reciprocal experiment selects its own alternative path.
    parser.set_defaults(w8a8_quantize_one_ahead=None)
    silu_quant_fusion = parser.add_mutually_exclusive_group()
    silu_quant_fusion.add_argument(
        "--w8a8-enable-silu-mul-quantize-fusion",
        dest="w8a8_silu_mul_quantize_fusion",
        action="store_true",
        help=(
            "Experimentally fuse exact SiLU-times-Up production with the "
            "GS512 Down activation quantizer; disabled by default after "
            "FPGA phase profiling showed a prefill regression"
        ),
    )
    silu_quant_fusion.add_argument(
        "--w8a8-disable-silu-mul-quantize-fusion",
        dest="w8a8_silu_mul_quantize_fusion",
        action="store_false",
        help="Retain the standalone TOSA SiLU/mul and quantize fallback",
    )
    # The outer-product fusion is FPGA checkpoint-exact, but its measured
    # S=22 phase profile regresses prefill and decode.  Keep it opt-in and
    # retain the independently validated unfused one-ahead path by default.
    parser.set_defaults(w8a8_silu_mul_quantize_fusion=False)
    gate_up_linear_fusion = parser.add_mutually_exclusive_group()
    gate_up_linear_fusion.add_argument(
        "--w8a8-enable-gate-up-linear-fusion",
        dest="w8a8_gate_up_linear_fusion",
        action="store_true",
        help=(
            "Pair the T=1 Qwen Gate/Up W8A8 projections and "
            "share their output-zero/AME-resynchronization preamble; "
            "prefill retains two independent projections"
        ),
    )
    gate_up_linear_fusion.add_argument(
        "--w8a8-disable-gate-up-linear-fusion",
        dest="w8a8_gate_up_linear_fusion",
        action="store_false",
        help="Retain two independent Gate/Up W8A8 Linear operations",
    )
    # FPGA checkpoints are byte-exact and repeated same-board profiling shows
    # a stable decode win.  Static T>1 prefill is rejected by the matcher and
    # retains the independently validated two-Linear schedule.
    parser.set_defaults(w8a8_gate_up_linear_fusion=True)
    parser.add_argument(
        "--fpga-assume-dynamic-inner-stride-one",
        dest="fpga_assume_dynamic_inner_stride_one",
        action="store_true",
        default=True,
        help=(
            "Use direct vector loads for dynamic-layout Qwen tensors whose "
            "runtime descriptors are dense (default)"
        ),
    )
    parser.add_argument(
        "--fpga-preserve-dynamic-transfer-reads",
        dest="fpga_assume_dynamic_inner_stride_one",
        action="store_false",
        help=(
            "Keep generic vector.transfer lowering when the runtime "
            "innermost stride is not guaranteed to be one. This is the "
            "semantic fallback for non-Qwen descriptors and FPGA A/B."
        ),
    )
    parser.add_argument("--name", default=None, help="Output basename")
    args = parser.parse_args()
    if args.w8a8_quantize_one_ahead is None:
        args.w8a8_quantize_one_ahead = not args.w8a8_quantize_reciprocal
    elif args.w8a8_quantize_reciprocal and args.w8a8_quantize_one_ahead:
        parser.error(
            "--w8a8-enable-quantize-reciprocal and "
            "--w8a8-enable-quantize-one-ahead are mutually exclusive"
        )
    return args


def require_tools() -> None:
    missing = [
        str(path) for path in (BUDDY_OPT, BUDDY_TRANSLATE, MLIR_OPT)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing compiler tools: " + ", ".join(missing))


def count_ops(text: str, prefix: str) -> int:
    return len(re.findall(rf"\b{re.escape(prefix)}[A-Za-z0-9_.]*", text))


def scalarize_strided_transfer_stores(path: Path) -> int:
    """Scalarize full-tile non-unit-stride transfer copies.

    LLVM lowers a vector.transfer_read with a static non-unit innermost
    stride by inserting every scalar lane with vslideup.  The current BOSC
    FPGA vector implementation does not reliably execute that sequence.
    These transfers are immediately copied to a contiguous vector.store and
    already have a separate scalar tail, so spelling the full tile as scalar
    loads/stores is both exact and avoids unsupported slide instructions.
    """
    lines = path.read_text().splitlines()
    rewritten: list[str] = []
    scalarized_count = 0
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        marker = " = vector.transfer_read "
        if (
            marker not in line
            or "strided<[" not in line
            or line_index + 1 >= len(lines)
        ):
            rewritten.append(line)
            line_index += 1
            continue

        indentation = line[: len(line) - len(line.lstrip())]
        result, operation = line.strip().split(marker, 1)
        operands, type_list = operation.split(" : ", 1)
        source, _padding = operands.rsplit(", ", 1)
        source_type, vector_type = type_list.rsplit(", ", 1)
        strides = re.search(r"strided<\[([^]]+)\]", source_type)
        vector_shape = re.fullmatch(r"vector<(\d+)x([^>]+)>", vector_type)
        if strides is None or vector_shape is None:
            rewritten.append(line)
            line_index += 1
            continue
        innermost_stride = strides.group(1).split(",")[-1].strip()
        if innermost_stride in ("0", "1", "?"):
            rewritten.append(line)
            line_index += 1
            continue

        store_line = lines[line_index + 1]
        store_prefix = f"vector.store {result}, "
        if not store_line.strip().startswith(store_prefix):
            rewritten.append(line)
            line_index += 1
            continue
        store_operation = store_line.strip()[len(store_prefix):]
        destination, store_type_list = store_operation.split(" : ", 1)
        destination_type, store_vector_type = store_type_list.rsplit(", ", 1)
        if store_vector_type != vector_type:
            rewritten.append(line)
            line_index += 1
            continue

        source_memref, source_indices_text = source.split("[", 1)
        destination_memref, destination_indices_text = destination.split("[", 1)
        source_indices = [
            value.strip() for value in source_indices_text[:-1].split(",")
        ]
        destination_indices = [
            value.strip() for value in destination_indices_text[:-1].split(",")
        ]
        if not source_indices or len(source_indices) != len(destination_indices):
            rewritten.append(line)
            line_index += 1
            continue

        vector_length, element_type = vector_shape.groups()
        stem = f"buddy_strided_transfer_{scalarized_count}"
        for lane in range(int(vector_length)):
            source_lane_indices = list(source_indices)
            destination_lane_indices = list(destination_indices)
            if lane:
                constant = f"%{stem}_c{lane}"
                rewritten.append(
                    f"{indentation}{constant} = arith.constant {lane} : index"
                )
                source_index = f"%{stem}_source_index_{lane}"
                rewritten.append(
                    f"{indentation}{source_index} = arith.addi "
                    f"{source_indices[-1]}, {constant} : index"
                )
                source_lane_indices[-1] = source_index
                if destination_indices[-1] == source_indices[-1]:
                    destination_lane_indices[-1] = source_index
                else:
                    destination_index = f"%{stem}_destination_index_{lane}"
                    rewritten.append(
                        f"{indentation}{destination_index} = arith.addi "
                        f"{destination_indices[-1]}, {constant} : index"
                    )
                    destination_lane_indices[-1] = destination_index
            value = f"%{stem}_value_{lane}"
            rewritten.append(
                f"{indentation}{value} = memref.load {source_memref}["
                + ", ".join(source_lane_indices)
                + f"] : {source_type}"
            )
            rewritten.append(
                f"{indentation}memref.store {value}, {destination_memref}["
                + ", ".join(destination_lane_indices)
                + f"] : {destination_type}"
            )
        scalarized_count += 1
        line_index += 2

    if scalarized_count:
        path.write_text("\n".join(rewritten) + "\n")
    return scalarized_count


def scalarize_vector_math_chains(path: Path) -> int:
    """Scalarize vector exp/sqrt chains that LLVM implements with slides.

    The scalar libm calls themselves work on BOSC FPGA, but LLVM's fixed
    vector legalization extracts and reinserts their lanes with vslidedown
    and vfslide1down.  Keep ordinary arithmetic vectorized and only expand a
    straight load/elementwise-math/store chain containing exp or sqrt.
    """
    lines = path.read_text().splitlines()
    rewritten: list[str] = []
    scalarized_count = 0
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        load_marker = " = vector.load "
        if load_marker not in line or "vector<16xf32>" not in line:
            rewritten.append(line)
            line_index += 1
            continue

        indentation = line[: len(line) - len(line.lstrip())]
        end_index = line_index
        has_vector_math = False
        while end_index < min(len(lines), line_index + 16):
            candidate = lines[end_index]
            if candidate[: len(candidate) - len(candidate.lstrip())] != indentation:
                break
            stripped = candidate.strip()
            has_vector_math |= (
                ("math.exp " in stripped or "math.sqrt " in stripped)
                and "vector<16xf32>" in stripped
            )
            if stripped.startswith("vector.store "):
                break
            if not any(
                operation in stripped
                for operation in (
                    "vector.load ",
                    "vector.broadcast ",
                    "math.exp ",
                    "math.sqrt ",
                    "arith.negf ",
                    "arith.addf ",
                    "arith.subf ",
                    "arith.mulf ",
                    "arith.divf ",
                )
            ):
                break
            end_index += 1
        if (
            end_index >= len(lines)
            or not lines[end_index].strip().startswith("vector.store ")
            or not has_vector_math
        ):
            rewritten.append(line)
            line_index += 1
            continue

        chain = lines[line_index:end_index + 1]
        stem = f"buddy_vector_math_{scalarized_count}"
        for lane in range(16):
            value_map: dict[str, str] = {}
            index_map: dict[str, str] = {}
            lane_constant = f"%{stem}_c{lane}"
            if lane:
                rewritten.append(
                    f"{indentation}{lane_constant} = arith.constant {lane} : index"
                )

            def lane_indices(text: str) -> tuple[str, list[str]]:
                memref, indices_text = text.split("[", 1)
                indices = [
                    value.strip() for value in indices_text[:-1].split(",")
                ]
                if lane:
                    original = indices[-1]
                    if original not in index_map:
                        scalar_index = (
                            f"%{stem}_index_{lane}_{len(index_map)}"
                        )
                        rewritten.append(
                            f"{indentation}{scalar_index} = arith.addi "
                            f"{original}, {lane_constant} : index"
                        )
                        index_map[original] = scalar_index
                    indices[-1] = index_map[original]
                return memref, indices

            for operation_line in chain:
                stripped = operation_line.strip()
                if " = vector.load " in operation_line:
                    result, operation = stripped.split(" = vector.load ", 1)
                    source, type_list = operation.split(" : ", 1)
                    memref_type, vector_type = type_list.rsplit(", ", 1)
                    if vector_type != "vector<16xf32>":
                        raise ValueError(f"unexpected vector math load: {stripped}")
                    source_memref, indices = lane_indices(source)
                    scalar_result = (
                        f"%{stem}_lane{lane}_value_"
                        + re.sub(r"[^A-Za-z0-9_$.-]", "_", result[1:])
                    )
                    rewritten.append(
                        f"{indentation}{scalar_result} = memref.load "
                        f"{source_memref}[" + ", ".join(indices)
                        + f"] : {memref_type}"
                    )
                    value_map[result] = scalar_result
                    continue
                if stripped.startswith("vector.store "):
                    operation = stripped.removeprefix("vector.store ")
                    operands, type_list = operation.split(" : ", 1)
                    value, destination = operands.split(", ", 1)
                    memref_type, vector_type = type_list.rsplit(", ", 1)
                    if vector_type != "vector<16xf32>" or value not in value_map:
                        raise ValueError(f"unexpected vector math store: {stripped}")
                    destination_memref, indices = lane_indices(destination)
                    rewritten.append(
                        f"{indentation}memref.store {value_map[value]}, "
                        f"{destination_memref}[" + ", ".join(indices)
                        + f"] : {memref_type}"
                    )
                    continue

                result, scalar_operation = stripped.split(" = ", 1)
                if scalar_operation.startswith("vector.broadcast "):
                    source = scalar_operation.removeprefix(
                        "vector.broadcast "
                    ).split(" : ", 1)[0]
                    value_map[result] = value_map.get(source, source)
                    continue
                scalar_result = (
                    f"%{stem}_lane{lane}_value_"
                    + re.sub(r"[^A-Za-z0-9_$.-]", "_", result[1:])
                )
                for vector_value, scalar_value in sorted(
                    value_map.items(), key=lambda item: -len(item[0])
                ):
                    scalar_operation = re.sub(
                        rf"(?<![A-Za-z0-9_.$]){re.escape(vector_value)}"
                        rf"(?![A-Za-z0-9_.$])",
                        scalar_value,
                        scalar_operation,
                    )
                scalar_operation = scalar_operation.replace(
                    "vector<16xf32>", "f32"
                )
                rewritten.append(
                    f"{indentation}{scalar_result} = {scalar_operation}"
                )
                value_map[result] = scalar_result

        scalarized_count += 1
        line_index = end_index + 1

    if scalarized_count:
        path.write_text("\n".join(rewritten) + "\n")
    return scalarized_count


def scalarize_vector_reductions(path: Path) -> int:
    """Expand fixed-width fp32 reductions into ordered scalar loads.

    The current BOSC FPGA does not reliably execute ``vfredosum`` and
    ``vfredmax``.  VIR emits these reductions as a four-operation
    load/seed/reduce/store chain, so replace that chain with sixteen scalar
    loads and ordered arithmetic while leaving ordinary W8A8 RVV arithmetic
    untouched.
    """
    lines = path.read_text().splitlines()
    rewritten: list[str] = []
    scalarized_count = 0
    line_index = 0
    reduction_re = re.compile(
        r"^(%[^ ]+) = vector\.reduction <(add|maximumf)>, "
        r"(%[^,]+), (%[^ ]+) : vector<16xf32> into f32$"
    )
    while line_index < len(lines):
        if line_index + 3 >= len(lines):
            rewritten.extend(lines[line_index:])
            break
        load_line = lines[line_index]
        seed_line = lines[line_index + 1]
        reduction_line = lines[line_index + 2]
        store_line = lines[line_index + 3]
        indentation = load_line[: len(load_line) - len(load_line.lstrip())]
        load_marker = " = vector.load "
        reduction = reduction_re.fullmatch(reduction_line.strip())
        if (
            load_marker not in load_line
            or " = memref.load " not in seed_line
            or reduction is None
            or not store_line.strip().startswith("memref.store ")
            or any(
                candidate[: len(candidate) - len(candidate.lstrip())]
                != indentation
                for candidate in (
                    seed_line,
                    reduction_line,
                    store_line,
                )
            )
        ):
            rewritten.append(load_line)
            line_index += 1
            continue

        vector_result, load_operation = load_line.strip().split(load_marker, 1)
        source, load_types = load_operation.split(" : ", 1)
        memref_type, vector_type = load_types.rsplit(", ", 1)
        reduction_result, kind, reduction_vector, seed = reduction.groups()
        if vector_type != "vector<16xf32>" or reduction_vector != vector_result:
            rewritten.append(load_line)
            line_index += 1
            continue
        source_memref, source_indices_text = source.split("[", 1)
        source_indices = [
            value.strip() for value in source_indices_text[:-1].split(",")
        ]
        if not source_indices:
            rewritten.append(load_line)
            line_index += 1
            continue

        rewritten.append(seed_line)
        stem = f"buddy_vector_reduction_{scalarized_count}"
        accumulator = seed
        scalar_op = "arith.addf" if kind == "add" else "arith.maximumf"
        for lane in range(16):
            indices = list(source_indices)
            if lane:
                lane_constant = f"%{stem}_c{lane}"
                lane_index = f"%{stem}_index_{lane}"
                rewritten.append(
                    f"{indentation}{lane_constant} = arith.constant {lane} : index"
                )
                rewritten.append(
                    f"{indentation}{lane_index} = arith.addi "
                    f"{source_indices[-1]}, {lane_constant} : index"
                )
                indices[-1] = lane_index
            value = f"%{stem}_value_{lane}"
            rewritten.append(
                f"{indentation}{value} = memref.load {source_memref}["
                + ", ".join(indices)
                + f"] : {memref_type}"
            )
            next_accumulator = (
                reduction_result
                if lane == 15
                else f"%{stem}_acc_{lane}"
            )
            rewritten.append(
                f"{indentation}{next_accumulator} = {scalar_op} "
                f"{accumulator}, {value} : f32"
            )
            accumulator = next_accumulator
        rewritten.append(store_line)
        scalarized_count += 1
        line_index += 4

    if scalarized_count:
        path.write_text("\n".join(rewritten) + "\n")
    return scalarized_count


def specialize_vector_transfer_reads(
    path: Path, *, assume_dynamic_inner_stride_one: bool = False
) -> tuple[int, int]:
    """Specialize transfers for the fixed contiguous Qwen tensor ABI.

    Generic vector.transfer lowering builds these values one lane at a time
    with vslideup.  A zero innermost stride is a scalar broadcast.  Broadcast
    its integer bit pattern and bitcast the vector back to fp32 so LLVM emits
    ``vmv.v.x`` plus fp32 vector-vector arithmetic; the current FPGA's
    float-scalar ``vfmul.vf`` path is not reliable.  A dynamic stride is not
    proof of a unit runtime stride: only specialize it behind the explicit
    FPGA A/B option.
    """
    rewritten: list[str] = []
    broadcast_count = 0
    contiguous_count = 0
    for line in path.read_text().splitlines():
        marker = " = vector.transfer_read "
        if marker not in line or "strided<[" not in line:
            rewritten.append(line)
            continue
        indentation = line[: len(line) - len(line.lstrip())]
        result, operation = line.strip().split(marker, 1)
        operands, type_list = operation.split(" : ", 1)
        source, _padding = operands.rsplit(", ", 1)
        memref_type, vector_type = type_list.rsplit(", ", 1)
        strides = re.search(r"strided<\[([^]]+)\]", memref_type)
        vector_element = re.fullmatch(r"vector<\d+x([^>]+)>", vector_type)
        if strides is None or vector_element is None:
            rewritten.append(line)
            continue
        innermost_stride = strides.group(1).split(",")[-1].strip()
        if innermost_stride == "?":
            if not assume_dynamic_inner_stride_one:
                rewritten.append(line)
                continue
            source_memref, indices = source.split("[", 1)
            index_values = [value.strip() for value in indices[:-1].split(",")]
            rank = len(index_values)
            vector_shape = re.fullmatch(r"vector<(\d+)x([^>]+)>", vector_type)
            assert vector_shape is not None
            vector_length, element_type = vector_shape.groups()
            stem = f"buddy_contiguous_transfer_{contiguous_count}"
            base = f"%{stem}_base"
            offset = f"%{stem}_offset"
            sizes = f"%{stem}_sizes"
            dynamic_strides = f"%{stem}_strides"
            rewritten.append(
                f"{indentation}{base}, {offset}, {sizes}:{rank}, "
                f"{dynamic_strides}:{rank} = memref.extract_strided_metadata "
                f"{source_memref} : {memref_type} -> memref<{element_type}>, "
                + ", ".join(["index"] * (1 + 2 * rank))
            )
            linear_offset = offset
            for dimension, index_value in enumerate(index_values):
                product = f"%{stem}_product_{dimension}"
                next_offset = f"%{stem}_offset_{dimension}"
                rewritten.append(
                    f"{indentation}{product} = arith.muli {index_value}, "
                    f"{dynamic_strides}#{dimension} : index"
                )
                rewritten.append(
                    f"{indentation}{next_offset} = arith.addi {linear_offset}, "
                    f"{product} : index"
                )
                linear_offset = next_offset
            contiguous = f"%{stem}_view"
            contiguous_type = (
                f"memref<{vector_length}x{element_type}, "
                "strided<[1], offset: ?>>"
            )
            zero = f"%{stem}_zero"
            rewritten.append(f"{indentation}{zero} = arith.constant 0 : index")
            rewritten.append(
                f"{indentation}{contiguous} = memref.reinterpret_cast {base} "
                f"to offset: [{linear_offset}], sizes: [{vector_length}], "
                f"strides: [1] : memref<{element_type}> to {contiguous_type}"
            )
            rewritten.append(
                f"{indentation}{result} = vector.load {contiguous}[{zero}] : "
                f"{contiguous_type}, {vector_type}"
            )
            contiguous_count += 1
            continue
        if innermost_stride != "0":
            rewritten.append(line)
            continue
        scalar = f"%buddy_stride0_broadcast_{broadcast_count}"
        scalar_bits = f"{scalar}_bits"
        vector_bits = f"{scalar}_vector_bits"
        element_type = vector_element.group(1)
        if element_type != "f32":
            rewritten.append(line)
            continue
        rewritten.append(
            f"{indentation}{scalar} = memref.load {source} : {memref_type}"
        )
        rewritten.append(
            f"{indentation}{scalar_bits} = arith.bitcast {scalar} : "
            "f32 to i32"
        )
        vector_length = re.fullmatch(r"vector<(\d+)xf32>", vector_type)
        if vector_length is None:
            rewritten.append(line)
            continue
        integer_vector_type = f"vector<{vector_length.group(1)}xi32>"
        rewritten.append(
            f"{indentation}{vector_bits} = vector.broadcast {scalar_bits} : "
            f"i32 to {integer_vector_type}"
        )
        rewritten.append(
            f"{indentation}{result} = vector.bitcast {vector_bits} : "
            f"{integer_vector_type} to {vector_type}"
        )
        broadcast_count += 1
    if broadcast_count or contiguous_count:
        path.write_text("\n".join(rewritten) + "\n")
    return broadcast_count, contiguous_count


def rewrite_float_vector_broadcasts(path: Path) -> int:
    """Broadcast fp32 values through their integer bit pattern.

    LLVM normally combines a floating-point ``vector.broadcast`` with its
    consumer into RVV ``*.vf`` instructions.  That float-scalar path is not
    reliable on the current BOSC FPGA.  The validated hand-written kernels
    use ``vmv.v.x`` followed by vector-vector fp32 operations, which this
    integer broadcast plus vector bitcast expresses without changing bits.
    """
    broadcast_re = re.compile(
        r"^(\s*)(%[^ ]+) = vector\.broadcast (%[^ ]+) : "
        r"f32 to vector<(\d+)xf32>$"
    )
    rewritten: list[str] = []
    rewrite_count = 0
    for line in path.read_text().splitlines():
        match = broadcast_re.fullmatch(line)
        if match is None:
            rewritten.append(line)
            continue
        indentation, result, source, vector_length = match.groups()
        stem = f"%buddy_float_broadcast_{rewrite_count}"
        integer_vector_type = f"vector<{vector_length}xi32>"
        rewritten.append(
            f"{indentation}{stem}_bits = arith.bitcast {source} : f32 to i32"
        )
        rewritten.append(
            f"{indentation}{stem}_vector_bits = vector.broadcast "
            f"{stem}_bits : i32 to {integer_vector_type}"
        )
        rewritten.append(
            f"{indentation}{result} = vector.bitcast {stem}_vector_bits : "
            f"{integer_vector_type} to vector<{vector_length}xf32>"
        )
        rewrite_count += 1
    if rewrite_count:
        path.write_text("\n".join(rewritten) + "\n")
    return rewrite_count


def main() -> int:
    args = parse_args()
    require_tools()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.vector_width <= 0:
        raise ValueError("--vector-width must be positive")
    cycle_trace = args.cycle_trace or "buddy_trace." in args.input.read_text()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.input.stem
    simplified = output_dir / f"{name}.01-simplified.mlir"
    linalg = output_dir / f"{name}.02-linalg.mlir"
    bufferized = output_dir / f"{name}.03-bufferized.mlir"
    boscame = output_dir / f"{name}.04-boscame.mlir"
    vector = output_dir / f"{name}.05-vector.mlir"
    llvm_mlir = output_dir / f"{name}.06-llvm.mlir"
    llvm_ir = output_dir / f"{name}.ll"

    run([
        str(BUDDY_OPT), str(args.input.resolve()),
        # The Qwen graph contains separate but identical reshape/quantize
        # chains for Q/K/V and Gate/Up.  QuantizePerGroupOp is effect-free in
        # tensor form, so CSE shares those activation quantizations before
        # bufferization while preserving effectful memref-form calls.
        *make_simplify_passes(args), "-o", str(simplified),
    ])
    run([
        # Use buddy-opt here because native Qwen3 semantic operations are
        # registered in the BOSCAME dialect and must survive TOSA conversion.
        str(BUDDY_OPT), str(simplified),
        "-pass-pipeline=builtin.module(func.func(tosa-to-linalg-named),"
        "func.func(tosa-to-linalg),func.func(tosa-to-tensor),"
        "func.func(tosa-to-arith))",
        "-o", str(linalg),
    ])
    run([
        str(BUDDY_OPT), str(linalg),
        "--eliminate-empty-tensors",
        "--empty-tensor-to-alloc-tensor",
        "--convert-elementwise-to-linalg",
        "--one-shot-bufferize=allow-return-allocs-from-loops=true "
        "bufferize-function-boundaries",
        "--expand-strided-metadata",
        "-o", str(bufferized),
    ])
    w8a8_pass = make_w8a8_pass(args)
    boscame_passes = [
        str(BUDDY_OPT), str(bufferized),
        w8a8_pass,
        "--lower-linalg-to-boscame",
        "--lower-bosc-ame",
    ]
    if args.optimize_attention_bmm:
        boscame_passes.append(
            f"--batchmatmul-optimize=vector-size={args.vector_width}"
        )
    boscame_passes.extend(["--cse", "-o", str(boscame)])
    run(boscame_passes)

    if args.mode == "vir":
        run([
            str(BUDDY_OPT), str(boscame),
            "--lower-linalg-to-vir",
            f"--lower-vir-to-vector=vector-width={args.vector_width}",
            "--cse",
            "-o", str(vector),
        ])
    else:
        vector.write_bytes(boscame.read_bytes())

    scalarized_strided_transfers = (
        scalarize_strided_transfer_stores(vector) if args.mode == "vir" else 0
    )
    scalarized_vector_math_chains = (
        scalarize_vector_math_chains(vector) if args.mode == "vir" else 0
    )
    scalarized_vector_reductions = (
        scalarize_vector_reductions(vector) if args.mode == "vir" else 0
    )
    zero_stride_broadcasts, contiguous_transfer_loads = (
        specialize_vector_transfer_reads(
            vector,
            assume_dynamic_inner_stride_one=(
                args.fpga_assume_dynamic_inner_stride_one
            ),
        )
        if args.mode == "vir"
        else (0, 0)
    )
    integer_bitpattern_vector_broadcasts = (
        rewrite_float_vector_broadcasts(vector) if args.mode == "vir" else 0
    )

    llvm_passes = [
        str(BUDDY_OPT), str(vector),
        "--convert-linalg-to-affine-loops",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-math-to-llvm",
        "--convert-math-to-libm",
        "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
        "--convert-vector-to-scf",
        "--convert-vector-to-llvm=vector-transpose-lowering=eltwise",
        "--convert-ub-to-llvm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-complex-to-llvm",
        "--convert-index-to-llvm",
        "--memref-expand",
        "--finalize-memref-to-llvm",
    ]
    if cycle_trace:
        # Trace lowering emits func.call operations, so run it before the
        # normal Func-to-LLVM conversion below.  The runtime hooks themselves
        # are implemented by examples/tools/bare_runtime.c using rdcycle.
        llvm_passes.append("--convert-trace-to-llvm=cycle-trace")
    if args.emit_c_wrapper:
        llvm_passes.append("--llvm-request-c-wrappers")
    llvm_passes.extend([
        "--convert-func-to-llvm",
        "--lower-affine",
        "--convert-arith-to-llvm",
        "--reconcile-unrealized-casts",
        "--mlir-print-debuginfo",
        "-o", str(llvm_mlir),
    ])
    run(llvm_passes)
    run([
        str(BUDDY_TRANSLATE), "--buddy-to-llvmir", str(llvm_mlir),
        "-o", str(llvm_ir),
    ])

    simplified_text = simplified.read_text()
    vector_text = vector.read_text()
    # In scalar mode the .05 file intentionally still contains Linalg ops;
    # they are eliminated by --convert-linalg-to-affine-loops in the next
    # stage.  Validate the final LLVM-dialect MLIR in that mode instead of
    # rejecting the expected scalar fallback input.
    dialect_check_text = (
        vector_text if args.mode == "vir" else llvm_mlir.read_text()
    )
    llvm_text = llvm_ir.read_text()
    static_malloc_sizes = [
        int(value)
        for value in re.findall(r"call ptr @malloc\(i64 ([0-9]+)\)", llvm_text)
    ]
    report = {
        "input": str(args.input.resolve()),
        "mode": args.mode,
        "vector_width": args.vector_width,
        "cycle_trace": cycle_trace,
        "optimize_attention_bmm": args.optimize_attention_bmm,
        "w8a8_scalar_fallback": args.w8a8_scalar_fallback,
        "w8a8_profile_phases": args.w8a8_profile_phases,
        "w8a8_decode_n128": not args.w8a8_disable_decode_n128,
        "w8a8_quantize_unroll": not args.w8a8_disable_quantize_unroll,
        "w8a8_quantize_reciprocal": args.w8a8_quantize_reciprocal,
        "w8a8_quantize_one_ahead": args.w8a8_quantize_one_ahead,
        "w8a8_silu_mul_quantize_fusion": (
            args.w8a8_silu_mul_quantize_fusion
        ),
        "w8a8_gate_up_linear_fusion": args.w8a8_gate_up_linear_fusion,
        "simplified_quantize_per_group_ops": count_ops(
            simplified_text, "bosc_ame.quantize_per_group"
        ),
        "simplified_silu_mul_quantize_ops": count_ops(
            simplified_text, "bosc_ame.silu_mul_quantize_per_group"
        ),
        "simplified_w8a8_linear_pair_ops": count_ops(
            simplified_text, "bosc_ame.w8a8_linear_pair"
        ),
        "fpga_assume_dynamic_inner_stride_one": (
            args.fpga_assume_dynamic_inner_stride_one
        ),
        "llvm_ir": str(llvm_ir),
        "remaining_tosa_ops": count_ops(dialect_check_text, "tosa."),
        "remaining_linalg_ops": count_ops(dialect_check_text, "linalg."),
        "vector_reductions": llvm_text.count("llvm.vector.reduce."),
        "vector_fma": llvm_text.count("llvm.fmuladd."),
        "zero_stride_vector_broadcasts": zero_stride_broadcasts,
        "contiguous_vector_transfer_loads": contiguous_transfer_loads,
        "integer_bitpattern_vector_broadcasts": (
            integer_bitpattern_vector_broadcasts
        ),
        "scalarized_strided_vector_transfers": scalarized_strided_transfers,
        "scalarized_vector_math_chains": scalarized_vector_math_chains,
        "scalarized_vector_reductions": scalarized_vector_reductions,
        "malloc_calls": llvm_text.count("call ptr @malloc("),
        "free_calls": llvm_text.count("call void @free("),
        "static_malloc_bytes": sum(static_malloc_sizes),
        "largest_static_malloc_bytes": max(static_malloc_sizes, default=0),
        "trace_cycle_regions": llvm_text.count(
            "call void @buddyTraceCycleStartPath("
        ),
        "trace_cycle_region_ends": llvm_text.count(
            "call void @buddyTraceCycleEndPath("
        ),
    }
    if report["remaining_tosa_ops"] or report["remaining_linalg_ops"]:
        raise RuntimeError(
            "frontend dialects remain after VIR lowering: " + json.dumps(report)
        )
    (output_dir / f"{name}.lowering.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
