"""Benchmark native Cumsum with output allocation and release included."""

import argparse
import ctypes
import inspect
import json
import statistics
import time

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy_mlir import runtime as rt


class Scan(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x, self.dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=("f32", "i64"), default="f32")
    parser.add_argument("--dim", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if min(args.rows, args.width) < 1:
        parser.error("rows and width must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.randint(-8, 9, (args.rows, args.width)).to(
        torch.float32 if args.dtype == "f32" else torch.int64
    )
    model = Scan(args.dim)
    ep = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(ep.graph_module, [x])
    run = compiler.dynamo_run()
    env = inspect.getclosurevars(run).nonlocals
    graph, engine = env["graph"], env["ee"]
    array = x.numpy().copy()
    desc = rt.get_ranked_memref_descriptor(array)
    ins = ctypes.pointer(ctypes.pointer(desc))
    output = graph._output_descriptor()
    outs = ctypes.pointer(ctypes.pointer(output))
    packed = (ctypes.c_void_p * 2)(
        ctypes.cast(outs, ctypes.c_void_p), ctypes.cast(ins, ctypes.c_void_p)
    )
    fn = engine.lookup(graph._func_name)
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]

    def release():
        libc.free(ctypes.cast(getattr(output, "0").allocated, ctypes.c_void_p))

    fn(packed)
    try:
        actual = rt.ranked_memref_to_numpy(
            ctypes.pointer(getattr(output, "0"))
        ).copy()
        assert actual.tobytes() == model(x).numpy().tobytes()
    finally:
        release()

    def invoke():
        fn(packed)
        release()

    for _ in range(5):
        invoke()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(10):
            invoke()
        samples.append((time.perf_counter_ns() - start) / 10000)
    print(
        json.dumps(
            {
                "shape": list(x.shape),
                "dtype": args.dtype,
                "dim": args.dim,
                "seed": 0,
                "torch_version": torch.__version__,
                "threads": 1,
                "samples_us": samples,
                "median_us": statistics.median(samples),
                "bitwise_equal": True,
            }
        )
    )


if __name__ == "__main__":
    main()
