#!/usr/bin/env python3
"""Compile real JIT functions to TTIR and linalg without a GPU or FPGA."""
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.compiler import CPUBackend, _ttir_to_ttsharedir
from triton.compiler import ASTSource


@triton.jit
def integer_dot(A, B, C):
    m = tl.arange(0, 1)
    n = tl.arange(0, 16)
    k = tl.arange(0, 64)
    a = tl.load(A + m[:, None] * 64 + k[None, :])
    b = tl.load(B + n[None, :] * 64 + k[:, None])
    c = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + m[:, None] * 16 + n[None, :], c)


@triton.jit
def normalized(X, W, Y):
    i = tl.arange(0, 128)
    x = tl.load(X + i)
    w = tl.load(W + i)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 128 + 1.0e-6)
    tl.store(Y + i, x * inv * w)


def main():
    target = GPUTarget("cpu", 0, 0)
    backend = CPUBackend(target)
    options = backend.parse_options({})
    for function, signature, expected in (
        (integer_dot, {"A": "*i8", "B": "*i8", "C": "*i32"}, "linalg.matmul"),
        (normalized, {"X": "*fp32", "W": "*fp32", "Y": "*fp32"}, "linalg.reduce"),
    ):
        source = ASTSource(fn=function, signature=signature, constexprs={})
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                                backend.get_module_map(), context)
        module = backend.make_ttir(module, {}, options)
        linalg = _ttir_to_ttsharedir(module)
        if expected not in linalg:
            raise RuntimeError(f"{function.__name__}: missing {expected}")
        print(f"PASS: @triton.jit {function.__name__} -> TTIR -> {expected}")
    print(f"Triton {triton.__version__}")


if __name__ == "__main__":
    main()
