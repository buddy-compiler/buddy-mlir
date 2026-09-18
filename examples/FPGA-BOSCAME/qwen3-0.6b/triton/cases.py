"""Static specializations of genuine Triton kernels, matched to parent cases.

This module deliberately imports no Triton package so the inventory can be read
and validated before the compiler environment has been built.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent

# Extra case roots, searched after PARENT. The end-to-end model keeps its own
# kernel specializations under examples/FPGA-BOSCAME/qwen3-0.6b/model/ instead of
# copying the shared 72-case set, while the operator kernels themselves stay in
# this directory. Unset means the previous behaviour exactly.
EXTRA_CASE_ROOTS = tuple(
    Path(entry) for entry in os.environ.get("QWEN_CASE_ROOTS", "").split(os.pathsep)
    if entry
)


def case_search_roots():
    return (PARENT,) + EXTRA_CASE_ROOTS


def case_directory(name):
    """Locate a case directory, PARENT first so existing cases win."""
    for root in case_search_roots():
        candidate = root / name
        if (candidate / "metadata.json").is_file():
            return candidate
    raise ValueError("unknown case: " + name)


def power_of_two(value):
    return 1 << (value - 1).bit_length()


def cdiv(value, divisor):
    return (value + divisor - 1) // divisor


def describe(name):
    directory = case_directory(name)
    metadata = json.loads((directory / "metadata.json").read_text())
    kind = metadata["kind"]
    constants = {}
    grid = (1, 1, 1)
    arguments = []
    kernel_module = "kernels"

    def arg(name, rank, dtype="f32"):
        arguments.append({"name": name, "rank": rank, "dtype": dtype})

    if kind in ("matmul_i8", "matmul_f32"):
        m, n, k = metadata["shape"]
        integer = kind == "matmul_i8"
        kernel = "linear"
        arg("A", 2, "i8" if integer else "f32")
        arg("B", 2, "i8" if integer else "f32")
        arg("C", 2, "i32" if integer else "f32")
        bm, bn, bk = power_of_two(min(m, 16)), 16, 64
        if integer:
            bn = int(os.environ.get("QWEN_TRITON_AME_N", "16"))
            if bn not in (16, 32, 64):
                raise ValueError("QWEN_TRITON_AME_N must be 16, 32, or 64")
        if "nr_tail_regression" not in metadata.get("roles", []):
            # Preserve pointer views for full tiles. NR itself splits K into
            # 64-byte hardware blocks, avoiding Triton padding/copy per block.
            bk = k & -k
        constants = dict(M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, INTEGER=integer)
        grid = (cdiv(m, bm), cdiv(n, bn), 1)
    elif kind in ("attention_qk", "attention_pv"):
        heads, m, n, k = metadata["shape"]
        kernel = "attention_dot"
        arg("A", 3)
        arg("B", 3)
        if metadata.get("runtime_position"):
            kernel = "attention_dot_position"
            kernel_module = "kernels_position"
            if metadata.get("native_key_layout"):
                if kind != "attention_qk":
                    raise ValueError("native key layout is valid only for QK")
                kernel = "attention_qk_position_native"
                kernel_module = "kernels_position_native"
            arg("Position", 1, "i32")
        arg("C", 3)
        bm, bn, bk = power_of_two(min(m, 16)), 16, power_of_two(min(k, 64))
        constants = dict(M=m, N=n, K=k, BM=bm, BN=bn, BK=bk)
        if metadata.get("runtime_position"):
            constants["QK"] = kind == "attention_qk"
        grid = (cdiv(m, bm), cdiv(n, bn), heads)
    elif kind in ("per_token_quantization", "dequantization"):
        m, width = metadata["shape"]
        if kind == "per_token_quantization":
            kernel = "quantize"
            arg("X", 2)
            arg("Q", 2, "i8")
            arg("Scale", 1)
            constants = dict(WIDTH=width, BLOCK=power_of_two(width))
            grid = (m, 1, 1)
        else:
            kernel = "dequantize"
            arg("X", 2, "i32")
            arg("Row", 1)
            arg("Column", 1)
            arg("Out", 2)
            constants = dict(ROWS=m, COLS=width, BLOCK=128)
            grid = (cdiv(m * width, 128), 1, 1)
            dequant_mode = os.environ.get("QWEN_TRITON_DEQUANT", "baseline")
            if dequant_mode == "rvv":
                kernel = "dequantize_rows"
                kernel_module = "kernels_dequant"
                grid = (cdiv(width, 128), m, 1)
            elif dequant_mode != "baseline":
                raise ValueError("QWEN_TRITON_DEQUANT must be baseline or rvv")
    else:
        buffers = metadata["buffers"]
        shapes = [buffer["shape"] for buffer in buffers]
        labels = {"add": ("X", "Y", "Out"), "mul": ("X", "Y", "Out"),
                  "silu": ("X", "Out"), "rmsnorm": ("X", "Weight", "Sums", "Out"),
                  "rope": ("X", "Cosine", "Sine", "Out"),
                  "softmax": ("X", "Maxima", "Sums", "Out"),
                  "attention_scale_mask": ("X", "Out"),
                  "attention_scale_mask_position": ("X", "Position", "Out"),
                  "embedding": ("Ids", "Out", "Weight"),
                  "embedding_w8a8": ("Ids", "Out", "Weight", "Scale"),
                  "kv_cache_update": ("X", "Cache"),
                  "kv_cache_update_position": ("X", "Position", "Cache"),
                  "gqa_repeat": ("X", "Out"), "layout_transpose": ("X", "Out")}[kind]
        for label, buffer in zip(labels, buffers):
            arg(label, len(buffer["shape"]), buffer["dtype"])
        kernel = kind
        if kind in ("add", "mul", "silu"):
            count = shapes[0][0] * shapes[0][1]
            constants = dict(COUNT=count, BLOCK=128)
            if kind in ("add", "mul"):
                kernel = "binary"
                constants["MULTIPLY"] = kind == "mul"
            grid = (cdiv(count, 128), 1, 1)
        elif kind == "rmsnorm":
            rows, width = shapes[0]
            constants = dict(WIDTH=width, BLOCK=power_of_two(width))
            grid = (rows, 1, 1)
        elif kind == "rope":
            sequence, heads, _, _ = shapes[0]
            constants = dict(HEADS=heads)
            grid = (sequence, heads, 1)
        elif kind == "attention_scale_mask_position":
            heads, sequence, total = shapes[0]
            constants = dict(SEQUENCE=sequence, TOTAL=total, BLOCK=128)
            grid = (cdiv(heads * sequence * total, 128), 1, 1)
        elif kind in ("softmax", "attention_scale_mask"):
            heads, sequence, total = shapes[0]
            if kind == "softmax":
                constants = dict(LENGTH=total, BLOCK=power_of_two(total))
                grid = (heads * sequence, 1, 1)
            else:
                constants = dict(SEQUENCE=sequence, TOTAL=total,
                                 PAST=total-sequence, BLOCK=128)
                grid = (cdiv(heads * sequence * total, 128), 1, 1)
        elif kind in ("embedding", "embedding_w8a8"):
            sequence, width = shapes[1]
            constants = dict(WIDTH=width, BLOCK=power_of_two(width))
            grid = (sequence, 1, 1)
        elif kind == "kv_cache_update_position":
            sequence = shapes[0][0]
            constants = dict(CAPACITY=shapes[2][1])
            grid = (sequence, 8, 1)
        elif kind == "kv_cache_update":
            sequence = shapes[0][0]
            constants = dict(PAST=0 if sequence == 16 else 16, CAPACITY=shapes[1][1])
            grid = (sequence, 8, 1)
        elif kind == "gqa_repeat":
            total = shapes[0][1]
            constants = dict(TOTAL=total)
            grid = (16, total, 1)
        elif kind == "layout_transpose":
            kernel = "transpose"
            permutation = (0, 2, 1) if name.startswith("layout_k_") else (1, 0, 2)
            d0, d1, d2 = shapes[0]
            constants = dict(D0=d0, D1=d1, D2=d2, P0=permutation[0],
                             P1=permutation[1], P2=permutation[2], BLOCK=128)
            grid = (cdiv(d0*d1*d2, 128), 1, 1)
    pointer_type = {"f32": "*fp32", "i8": "*i8", "i32": "*i32", "i64": "*i64"}
    quant_mode = os.environ.get("QWEN_TRITON_QUANT", "baseline")
    if quant_mode not in ("baseline", "rvv"):
        raise ValueError("QWEN_TRITON_QUANT must be baseline or rvv")
    return {"name": name, "family": kind, "kernel": kernel,
            **({"quantization_lowering": quant_mode}
               if kind == "per_token_quantization" else {}),
            "kernel_module": kernel_module,
            "symbol": "triton_" + name, "arguments": arguments,
            "signature": {argument["name"]: pointer_type[argument["dtype"]] for argument in arguments},
            "constexprs": constants, "grid": list(grid),
            "directory": str(directory),
            "launch_source": str(directory / "launch.c"),
            "validation": "Unmodified parent launch.c and its independent numerical oracle"}


def inventory():
    found = {}
    for root in case_search_roots():
        for path in sorted(root.glob("*/metadata.json")):
            found.setdefault(path.parent.name, path)
    return [describe(name) for name in sorted(found)]


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
