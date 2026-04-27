#!/usr/bin/env python3
# ===- import_matmul_model.py --------------------------------------------------
#
# Import a tiny PyTorch model to MLIR with TORQ-Tile external call.
#
# ===---------------------------------------------------------------------------

import argparse
import os

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    replace_matmul_with_buddy_runtime,
    simply_fuse,
)
from buddy.compiler.ops import func, tosa
from torch._inductor.decomposition import decompositions as inductor_decomp


class MatmulOnlyModel(torch.nn.Module):
    """y = matmul(x, weight) — replaced with buddy_matmul_f32 (runtime ABI)."""

    def forward(self, x, weight):
        return torch.matmul(x, weight)


def main():
    parser = argparse.ArgumentParser(
        description="Import MatmulOnlyModel with TORQ-Tile integration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=64,
        help="Rows of A / result (M).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=64,
        help="Inner dimension (K).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=64,
        help="Columns of B / result (N).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = MatmulOnlyModel()
    x = torch.randn(args.m, args.k)
    weight = torch.randn(args.k, args.n)

    combined_registry = {**tosa.ops_registry, **func.ops_registry}
    dynamo_compiler = DynamoCompiler(
        primary_registry=combined_registry,
        aot_autograd_decomposition=inductor_decomp,
        enable_external_calls=True,
    )

    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, x=x, weight=weight)

    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]

    pattern_list = [
        replace_matmul_with_buddy_runtime,
        simply_fuse,
    ]
    graphs[0].fuse_ops(pattern_list)

    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()

    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as f:
        print(driver.construct_main_graph(True), file=f)

    arg0_path = os.path.join(output_dir, "arg0.data")
    if params:
        all_param = numpy.concatenate(
            [param.detach().numpy().reshape([-1]) for param in params]
        )
        all_param.tofile(arg0_path)
    else:
        with open(arg0_path, "wb") as f:
            pass

    matmul_count = sum(
        1 for node in graph.body if type(node).__name__ == "MatmulOp"
    )
    call_count = sum(
        1
        for node in graph.body
        if type(node).__name__ in ("CallOp", "CallExternalOp")
    )
    print(f"MatmulOp count: {matmul_count} (expect 0)")
    print(f"CallExternalOp count: {call_count} (expect >= 1)")
    print(f"Wrote MLIR to {output_dir}")


if __name__ == "__main__":
    main()
