#!/usr/bin/env python3
# ===- import_simple_model.py --------------------------------------------------
#
# Import script to convert SimpleModel from PyTorch to MLIR with oneDNN calls.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, replace_matmul_with_onednn


class SimpleModel(torch.nn.Module):
    """
    Simple model for testing oneDNN integration.
    
    Forward pass:
        y = matmul(x, weight)  <- Will be replaced with oneDNN
        y = y + bias           <- Will use TOSA
        y = relu(y)            <- Will use TOSA
    """
    def forward(self, x, weight, bias):
        y = torch.matmul(x, weight)
        y = y + bias
        y = torch.relu(y)
        return y


def main():
    parser = argparse.ArgumentParser(
        description="Import SimpleModel with oneDNN integration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for input tensors.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4,
        help="Sequence length for input tensors.",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=8,
        help="Input feature dimension.",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=6,
        help="Output feature dimension.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = SimpleModel()

    # Create sample inputs
    x = torch.randn(args.batch_size, args.seq_len, args.in_features)
    weight = torch.randn(args.in_features, args.out_features)
    bias = torch.randn(1, 1, args.out_features)

    print("=" * 80)
    print("SimpleModel with oneDNN Integration")
    print("=" * 80)
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias: {bias.shape}")

    # Initialize Dynamo Compiler
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import the model
    print("\nImporting model to Buddy Graph...")
    with torch.no_grad():
        graphs = dynamo_compiler.importer(
            model,
            x=x,
            weight=weight,
            bias=bias,
        )

    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]

    # Save Buddy Graph before transform
    print("Saving Buddy Graph representation...")
    with open(os.path.join(output_dir, "graph.log"), "w") as graph_file:
        print("=" * 80, file=graph_file)
        print("BUDDY GRAPH BEFORE TRANSFORM", file=graph_file)
        print("=" * 80, file=graph_file)
        print(f"Number of nodes: {len(graph.body)}", file=graph_file)
        print(f"Number of parameters: {len(params)}", file=graph_file)
        print("=" * 80, file=graph_file)
        print("GRAPH NODES:", file=graph_file)
        print("=" * 80, file=graph_file)

        for i, node in enumerate(graph.body):
            node_type = type(node).__name__.replace('Op', '')
            print(f"Node {i+1}: {node.name} | {node_type}", file=graph_file)

            if hasattr(node, 'tensor_meta') and node.tensor_meta:
                meta = node.tensor_meta
                if isinstance(meta, dict) and 'shape' in meta:
                    shape = meta['shape']
                    print(f"  Output shape: {list(shape) if hasattr(shape, '__iter__') else shape}", file=graph_file)

            print("-" * 40, file=graph_file)

    # Apply graph transformations
    # IMPORTANT: replace_matmul_with_onednn must be called BEFORE simply_fuse
    # because simply_fuse creates op_groups which will capture the current graph.body
    pattern_list = [
        replace_matmul_with_onednn,  # Replace MatmulOp with oneDNN calls
        simply_fuse,                  # Then create op_groups
    ]
    
    for pattern in pattern_list:
        print(f"  - {pattern.__name__}")
    
    graphs[0].fuse_ops(pattern_list)

    # Save graph after transform
    with open(os.path.join(output_dir, "graph_transformed.log"), "w") as fused_file:
        print("=" * 80, file=fused_file)
        print("BUDDY GRAPH AFTER TRANSFORM", file=fused_file)
        print("=" * 80, file=fused_file)
        print(f"Number of nodes: {len(graph.body)}", file=fused_file)
        print("=" * 80, file=fused_file)
        print("TRANSFORMED GRAPH NODES:", file=fused_file)
        print("=" * 80, file=fused_file)

        for i, node in enumerate(graph.body):
            node_type = type(node).__name__.replace('Op', '')
            print(f"Node {i+1}: {node.name} | {node_type}", file=fused_file)
            
            # Print CallOp details
            if node_type == "Call" and hasattr(node, 'call_func_name'):
                print(f"  Calls: {node.call_func_name}", file=fused_file)

            if hasattr(node, 'tensor_meta') and node.tensor_meta:
                meta = node.tensor_meta
                if isinstance(meta, dict) and 'shape' in meta:
                    shape = meta['shape']
                    print(f"  Output shape: {list(shape) if hasattr(shape, '__iter__') else shape}", file=fused_file)

            print("-" * 40, file=fused_file)

    # Lower to MLIR
    print("\nLowering to MLIR...")
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()

    # Save MLIR files
    print("Saving MLIR files...")
    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

    # Save parameters
    print("Saving parameters...")
    if params:
        all_param = numpy.concatenate(
            [param.detach().numpy().reshape([-1]) for param in params]
        )
        all_param.tofile(os.path.join(output_dir, "arg0.data"))
    else:
        print("  No parameters to save")

    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    print(f"  - graph.log (before transform)")
    print(f"  - graph_transformed.log (after transform)")
    print(f"  - forward.mlir (main graph)")
    print(f"  - subgraph0.mlir (subgraph)")
    print(f"  - arg0.data (parameters)")
    print(f"\nModel parameters: {sum(p.numel() for p in params)} elements")
    print(f"Input shape: {x.shape}")
    
    # Verify transform
    print("\n" + "=" * 80)
    print("Verification:")
    print("=" * 80)
    
    matmul_count = sum(1 for node in graph.body if type(node).__name__ == "MatmulOp")
    call_count = sum(1 for node in graph.body if type(node).__name__ == "CallOp")
    
    print(f"MatmulOp count: {matmul_count} (should be 0)")
    print(f"CallOp count: {call_count} (should be 1)")
    
    if matmul_count == 0 and call_count >= 1:
        print("Transform successful! MatmulOp replaced with CallOp")
    else:
        print("Transform may have issues")



if __name__ == "__main__":
    main()

