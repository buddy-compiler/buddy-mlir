#!/usr/bin/env python3
"""Softmax固定profile的导出契约；导出、头文件和RAX共享，运行期不读取。"""

import hashlib
from pathlib import Path

# These are build-time constraints, not a second runtime dispatch registry.
EXPORT_PROFILE = {
    "entry": "softmax_kernel",
    "query_entry": "buddy_external_kernel_query_v1",
    "kernel_abi": "buddy.external.v1",
    "operation": "softmax",
    "target": "riscv64-unknown-linux-gnu",
    "profile": "softmax.f32.2x4.rvv1.vlen256.v1",
    "dtype": "f32",
    "layout": "row_major",
    "shape": [2, 4],
    "strides": [4, 1],
    "axis": 1,
    "block_size": 4,
    "grid": [2, 1, 1],
    "features": "+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb,+zvfbfmin",
    "cpu_isa": "i,m,a,f,d,c,v,zicsr,zifencei,zfh,zvfh,zba,zbb,zvfbfmin",
    "required_rvv_version": "1.0",
    "rvv_vlen_bits": 256,
    "required_matrix_features": "none",
    "requires_single_cpu": False,
    "llvm_parameter_types": ["i64", "ptr", "i64", "ptr"] + ["i32"] * 9,
}
EXPORT_PROFILE["required_isa"] = (
    EXPORT_PROFILE["cpu_isa"] + ",rvv1p0,vlen256,linux-lp64d"
)


def same_value(actual, expected):
    """JSON bool/int must not compare equal for ABI/launch dimensions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_value(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def validate_export(metadata, object_path=None):
    for key, expected in EXPORT_PROFILE.items():
        if key not in metadata or not same_value(metadata[key], expected):
            raise ValueError(f"导出profile字段不一致: {key}")
    digest = metadata.get("object_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("导出记录缺少合法object_sha256")
    if object_path is not None and (
        hashlib.sha256(Path(object_path).read_bytes()).hexdigest() != digest
    ):
        raise ValueError("导出记录与对象内容不一致")


def kernel_attributes(metadata):
    validate_export(metadata)
    keys = [
        "kernel_abi",
        "operation",
        "dtype",
        "layout",
        "profile",
        "target",
        "required_isa",
        "required_rvv_version",
        "required_matrix_features",
    ]
    attrs = {key: metadata[key] for key in keys}
    attrs.update(
        optional="true",
        shape=",".join(map(str, metadata["shape"])),
        strides=",".join(map(str, metadata["strides"])),
        axis=str(metadata["axis"]),
        rvv_vlen_bits=str(metadata["rvv_vlen_bits"]),
        object_sha256=metadata["object_sha256"],
    )
    return attrs


def target_header(metadata):
    validate_export(metadata)
    return (
        "// 由 export.py 生成，必须与 softmax.obj 一起重建。\n"
        f'#define TRITON_SOFTMAX_CPU_ISA "{metadata["cpu_isa"]}"\n'
        f'#define TRITON_SOFTMAX_REQUIRED_FEATURES "{metadata["required_isa"]}"\n'
        f"#define TRITON_SOFTMAX_VLEN_BITS {metadata['rvv_vlen_bits']}\n"
    )
