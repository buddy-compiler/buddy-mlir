#!/usr/bin/env python3
# ===- gen_rax_shims.py - Generate RAX-to-MLIR ABI shims -----------------===//
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

"""Generate raw-payload RAX entry points for scheduled TP wrappers."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DTYPES = {
    "bool": ("i1", "bool", "I1"),
    "f32": ("f32", "float", "F32"),
    "float32": ("f32", "float", "F32"),
    "i64": ("i64", "int64_t", "I64"),
    "int64": ("i64", "int64_t", "I64"),
}


@dataclass(frozen=True)
class MemRef:
    dtype: str
    shape: tuple[int, ...]


def _fail(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def load_wrapper_abis(paths: list[Path]) -> dict[str, tuple[MemRef, ...]]:
    """Load and validate wrapper ABIs from Stage 4 runtime plans."""
    if not paths:
        raise ValueError("at least one runtime plan is required")

    wrappers: dict[str, tuple[MemRef, ...]] = {}
    for path in paths:
        try:
            plan = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise _fail(path, f"cannot load runtime plan: {error}") from error

        if not isinstance(plan, dict):
            raise _fail(path, "runtime plan must be a JSON object")
        resources = plan.get("resources")
        operations = plan.get("operations")
        if not isinstance(resources, dict):
            raise _fail(path, "'resources' must be an object")
        if not isinstance(operations, list):
            raise _fail(path, "'operations' must be an array")

        for operation_index, operation in enumerate(operations):
            where = f"operation {operation_index}"
            if not isinstance(operation, dict):
                raise _fail(path, f"{where} must be an object")
            if operation.get("kind") != "dispatch":
                continue

            wrapper = operation.get("wrapper")
            if not isinstance(wrapper, str) or not _SYMBOL_RE.fullmatch(
                wrapper
            ):
                raise _fail(
                    path, f"{where} has invalid wrapper symbol {wrapper!r}"
                )

            arguments = operation.get("arguments")
            if not isinstance(arguments, list):
                raise _fail(path, f"dispatch {wrapper!r} has no argument array")
            argument_groups = []
            for field in ("parameter_packs", "inputs", "outputs"):
                group = operation.get(field)
                if not isinstance(group, list):
                    raise _fail(
                        path,
                        f"dispatch {wrapper!r} field {field!r} must be an array",
                    )
                argument_groups.extend(group)
            expected_arguments = argument_groups
            if arguments != expected_arguments:
                raise _fail(
                    path,
                    f"dispatch {wrapper!r} arguments do not equal "
                    "parameter_packs + inputs + outputs",
                )

            signature = []
            for argument_index, resource in enumerate(arguments):
                if not isinstance(resource, str) or resource not in resources:
                    raise _fail(
                        path,
                        f"dispatch {wrapper!r} argument {argument_index} "
                        f"references unknown resource {resource!r}",
                    )
                metadata = resources[resource]
                if not isinstance(metadata, dict):
                    raise _fail(
                        path, f"resource {resource!r} must be an object"
                    )
                dtype = metadata.get("dtype")
                if dtype not in _DTYPES:
                    raise _fail(
                        path,
                        f"resource {resource!r} has unsupported dtype {dtype!r}; "
                        "the TP shim path supports f32 plus the i64/i1 control "
                        "tensors present in Stage 4 plans",
                    )
                shape = metadata.get("shape")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or any(
                        type(dimension) is not int or dimension <= 0
                        for dimension in shape
                    )
                ):
                    raise _fail(
                        path,
                        f"resource {resource!r} must have a non-empty static shape",
                    )
                signature.append(MemRef(_DTYPES[dtype][0], tuple(shape)))

            signature_tuple = tuple(signature)
            previous = wrappers.setdefault(wrapper, signature_tuple)
            if previous != signature_tuple:
                raise _fail(
                    path,
                    f"wrapper {wrapper!r} has inconsistent ABI across runtime plans",
                )

    if not wrappers:
        raise ValueError("runtime plans contain no dispatch wrappers")
    return wrappers


def required_symbols(wrappers: dict[str, tuple[MemRef, ...]]) -> list[str]:
    return [f"rax_{wrapper}" for wrapper in sorted(wrappers)]


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    running = 1
    for dimension in reversed(shape):
        strides.append(running)
        running *= dimension
    return tuple(reversed(strides))


def generate_cpp(wrappers: dict[str, tuple[MemRef, ...]]) -> str:
    out = [
        "// Generated by tools/buddy-codegen/gen_rax_shims.py. Do not edit.",
        "#include <cstdint>",
        "",
        "template <typename T, int Rank> struct MemRefDescriptor {",
        "  T *allocated;",
        "  T *aligned;",
        "  int64_t offset;",
        "  int64_t sizes[Rank];",
        "  int64_t strides[Rank];",
        "};",
        "",
    ]
    descriptor_types = sorted(
        {
            (memref.dtype, len(memref.shape))
            for abi in wrappers.values()
            for memref in abi
        }
    )
    dtype_info = {value[0]: value[1:] for value in _DTYPES.values()}
    for dtype, rank in descriptor_types:
        c_type, suffix = dtype_info[dtype]
        out.append(
            f"using MemRef{rank}D{suffix} = MemRefDescriptor<{c_type}, {rank}>;"
        )
    out.append("")

    for wrapper in sorted(wrappers):
        abi = wrappers[wrapper]
        parameters = ", ".join(
            f"MemRef{len(memref.shape)}D{dtype_info[memref.dtype][1]} *arg{index}"
            for index, memref in enumerate(abi)
        )
        out.append(f'extern "C" void _mlir_ciface_{wrapper}({parameters});')
    out.append("")

    for wrapper in sorted(wrappers):
        abi = wrappers[wrapper]
        out.append(f'extern "C" void rax_{wrapper}(void **args) {{')
        for index, memref in enumerate(abi):
            rank = len(memref.shape)
            c_type, suffix = dtype_info[memref.dtype]
            sizes = ", ".join(str(dimension) for dimension in memref.shape)
            strides = ", ".join(
                str(stride) for stride in _strides(memref.shape)
            )
            out.extend(
                [
                    f"  auto *data{index} = static_cast<{c_type} *>(args[{index}]);",
                    f"  MemRef{rank}D{suffix} arg{index} = "
                    f"{{data{index}, data{index}, 0, {{{sizes}}}, {{{strides}}}}};",
                ]
            )
        arguments = ", ".join(f"&arg{index}" for index in range(len(abi)))
        out.append(f"  _mlir_ciface_{wrapper}({arguments});")
        out.append("}")
        out.append("")
    return "\n".join(out)


def exported_symbols(library: Path, nm: str = "nm") -> set[str]:
    try:
        result = subprocess.run(
            [nm, "-D", "--defined-only", str(library)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"failed to inspect {library} with {nm}: {error}"
        ) from error
    return {
        line.split()[-1] for line in result.stdout.splitlines() if line.split()
    }


def validate_library(
    wrappers: dict[str, tuple[MemRef, ...]], library: Path, nm: str = "nm"
) -> None:
    required = set(required_symbols(wrappers))
    missing = sorted(required - exported_symbols(library, nm))
    if missing:
        raise ValueError(
            f"{library}: missing {len(missing)} required RAX symbols: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RAX-to-MLIR shims from Stage 4 runtime plans"
    )
    parser.add_argument(
        "--runtime-plan",
        action="append",
        required=True,
        type=Path,
        help="Stage 4 rank runtime-plan JSON (repeatable)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Generated RaxShims.cpp"
    )
    parser.add_argument(
        "--validate-library",
        type=Path,
        help="Check that the shared library exports every required rax_* symbol",
    )
    parser.add_argument(
        "--nm", default="nm", help="nm executable for validation"
    )
    args = parser.parse_args()
    if args.output is None and args.validate_library is None:
        parser.error(
            "at least one of --output or --validate-library is required"
        )

    try:
        wrappers = load_wrapper_abis(args.runtime_plan)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(generate_cpp(wrappers))
            print(
                f"[gen_rax_shims] Written: {args.output} "
                f"({len(wrappers)} wrappers)",
                file=sys.stderr,
            )
        if args.validate_library is not None:
            validate_library(wrappers, args.validate_library, args.nm)
            print(
                f"[gen_rax_shims] Validated {len(wrappers)} RAX symbols in "
                f"{args.validate_library} (missing=0)",
                file=sys.stderr,
            )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
