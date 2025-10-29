# ===- import-transformer.py -----------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# DeepSeek R1 Transformer Block AOT Importer
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
from buddy.compiler.graph.transform import simply_fuse

from transformer_model import create_transformer_model, create_sample_inputs


def main():
    # Add argument parser to allow custom output directory
    parser = argparse.ArgumentParser(
        description="DeepSeek R1 Transformer Block AOT Importer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory to save output files.",
    )
    # F32 precision only as requested
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for input tensors.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=40,
        help="Sequence length for input tensors.",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create the transformer block model (f32 precision)
    model = create_transformer_model()

    # Create sample inputs
    hidden_states, attention_mask = create_sample_inputs(
        args.batch_size, args.seq_len
    )

    # Initialize Dynamo Compiler with specific configurations as an importer
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import the model into MLIR module and parameters
    with torch.no_grad():
        graphs = dynamo_compiler.importer(
            model,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]

    # Save Buddy Graph representation before lowering to TOSA
    print("Saving Buddy Graph representation...")
    with open(os.path.join(output_dir, "graph.log"), "w") as graph_file:
        print("=" * 80, file=graph_file)
        print("BUDDY GRAPH REPRESENTATION", file=graph_file)
        print("=" * 80, file=graph_file)
        print(f"Number of nodes: {len(graph.body)}", file=graph_file)
        print(f"Number of parameters: {len(params)}", file=graph_file)
        print("=" * 80, file=graph_file)
        print("GRAPH NODES:", file=graph_file)
        print("=" * 80, file=graph_file)

        for i, node in enumerate(graph.body):
            node_type = type(node).__name__.replace("Op", "")
            print(f"Node {i+1}: {node.name} | {node_type}", file=graph_file)

            # Extract output shape if available
            if hasattr(node, "tensor_meta") and node.tensor_meta:
                meta = node.tensor_meta
                if isinstance(meta, dict) and "shape" in meta:
                    shape = meta["shape"]
                    print(
                        f"  Output shape: {list(shape) if hasattr(shape, '__iter__') else shape}",
                        file=graph_file,
                    )

            print("-" * 40, file=graph_file)

        print("=" * 80, file=graph_file)
        print("PARAMETER SHAPES:", file=graph_file)
        print("=" * 80, file=graph_file)
        for i, param in enumerate(params):
            print(
                f"Parameter {i}: {param.shape} ({param.dtype})", file=graph_file
            )

    # Apply graph transformations
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)

    # Save graph after fusion
    with open(os.path.join(output_dir, "graph_fused.log"), "w") as fused_file:
        print("=" * 80, file=fused_file)
        print("BUDDY GRAPH AFTER FUSION", file=fused_file)
        print("=" * 80, file=fused_file)
        print(
            f"Number of nodes after fusion: {len(graph.body)}", file=fused_file
        )
        print("=" * 80, file=fused_file)
        print("FUSED GRAPH NODES:", file=fused_file)
        print("=" * 80, file=fused_file)

        for i, node in enumerate(graph.body):
            node_type = type(node).__name__.replace("Op", "")
            print(f"Node {i+1}: {node.name} | {node_type}", file=fused_file)

            # Extract output shape if available
            if hasattr(node, "tensor_meta") and node.tensor_meta:
                meta = node.tensor_meta
                if isinstance(meta, dict) and "shape" in meta:
                    shape = meta["shape"]
                    print(
                        f"  Output shape: {list(shape) if hasattr(shape, '__iter__') else shape}",
                        file=fused_file,
                    )

            print("-" * 40, file=fused_file)

    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()

    # Save the generated files to the specified output directory
    # Save MLIR files
    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

    # Save parameters
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))

    print(f"Generated f32 MLIR files and parameters in {output_dir}")
    print(f"Generated Buddy Graph logs: graph.log and graph_fused.log")

    # Print summary information
    print(f"Model parameters: {sum(p.numel() for p in params)} elements")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")


if __name__ == "__main__":
    main()
