# ===- print_graph_nodes.py --------------------------------------------------
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
# This script loads a DeepSeekR1 model, imports it into a Buddy Graph,
# applies operator fusion, and prints all the nodes in the graph.
#
# ===---------------------------------------------------------------------------

import sys
import os

# Add the parent directory for 'buddy' packages
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "build", "python_packages"
        )
    )
)

# Create a symbolic link or alias for buddy_mlir as mlir
buddy_mlir_path = "/home/jjji/personal_projects/OSPP/buddy-mlir/build/python_packages"
sys.path.append(buddy_mlir_path)

# Import buddy_mlir and alias it as mlir in sys.modules
import buddy_mlir
sys.modules['mlir'] = buddy_mlir

import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion

# 1. Set the model path
# Note: Ensure you have access to this path.
model_path = "/home/jjji/LLM_Models"

# 2. Initialize the model from the specified model path.
print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torchscript=True
).eval()
model.config.use_cache = False
print("Model loaded successfully.")

# 3. Initialize Dynamo Compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# 4. Import the model into MLIR module and parameters.
print("Importing model to Buddy Graph...")
with torch.no_grad():
    data = {
        "input_ids": torch.zeros((1, 40), dtype=torch.int64),
        "attention_mask": torch.zeros((1, 40), dtype=torch.int64),
    }
    graphs = dynamo_compiler.importer(
        model,
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
    )
print("Graph imported successfully.")

# 5. Print the nodes of the graph before and after fusion
assert len(graphs) == 1
graph = graphs[0]

def print_graph_nodes(graph, title):
    """Print graph node information with detailed input/output information"""
    content = []
    content.append("=" * 100)
    content.append(title)
    content.append("=" * 100)
    content.append(f"Total nodes: {len(graph.body)}")

    # Count different node types
    node_types = {}
    for node in graph.body:
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    content.append("Node type statistics:")
    for node_type, count in sorted(node_types.items()):
        content.append(f"  {node_type}: {count}")
    content.append("")

    for i, node in enumerate(graph.body, 1):
        # Get operation type name
        op_type = type(node).__name__.replace('Op', '')

        # Get output shape information
        output_shape = _extract_shape_info(node)

        # Build basic information line
        basic_info = f"{i:3d}. {node.name} | {op_type} | Out:{output_shape}"

        # Add extra information for special nodes
        if type(node).__name__ == 'QKVFusedOp':
            q_dim = getattr(node, 'q_dim', '?')
            k_dim = getattr(node, 'k_dim', '?')
            v_dim = getattr(node, 'v_dim', '?')
            basic_info += f" | Q:{q_dim},K:{k_dim},V:{v_dim}"

        content.append(basic_info)

        # Add parent-child relationship information
        parent_info = _get_parent_info(node)
        if parent_info:
            content.append(f"     Parents: {parent_info}")

        children_info = _get_children_info(node)
        if children_info:
            content.append(f"     Children: {children_info}")

        content.append("")  # Empty line separator

    return content


def _extract_shape_info(node):
    """Extract shape information from node"""
    if hasattr(node, 'tensor_meta') and node.tensor_meta:
        meta = node.tensor_meta
        if isinstance(meta, dict) and 'shape' in meta:
            shape = meta['shape']
            if hasattr(shape, 'size'):
                return list(shape.size())
            elif isinstance(shape, (list, tuple)):
                return list(shape)
    return "Unknown"


def _get_parent_info(node):
    """Get parent node information"""
    if hasattr(node, '_parents') and node._parents:
        return ", ".join(node._parents)
    return None


def _get_children_info(node):
    """Get children node information"""
    if hasattr(node, '_children') and node._children:
        return ", ".join(node._children)
    return None

def analyze_fusable_patterns(graph):
    """Analyze fusable patterns in the graph"""
    patterns = []

    # Check Permute + AddMM pattern
    for i, node in enumerate(graph.body):
        if hasattr(node, '__class__') and node.__class__.__name__ == 'AddMMOp':
            # Check if there's a PermuteOp as input
            for arg in node.args:
                if isinstance(arg, str) and arg in graph.node_table:
                    parent_node = graph.node_table[arg]
                    if hasattr(parent_node, '__class__') and parent_node.__class__.__name__ == 'PermuteOp':
                        if hasattr(parent_node, 'args') and len(parent_node.args) > 1:
                            perm = parent_node.args[1]
                            if perm == [1, 0]:  # transpose pattern
                                patterns.append(f"Transpose+MatMul: {parent_node.name} -> {node.name}")

    # Check RMSNorm pattern (Pow + Mean + Add + Rsqrt)
    for i, node in enumerate(graph.body):
        if hasattr(node, '__class__') and node.__class__.__name__ == 'PowOp':
            if i + 3 < len(graph.body):
                next_nodes = graph.body[i:i+4]
                node_types = [n.__class__.__name__ for n in next_nodes]
                if node_types == ['PowOp', 'MeanOp', 'AddOp', 'RsqrtOp']:
                    patterns.append(f"RMSNorm pattern: {' -> '.join([n.name for n in next_nodes])}")

    return patterns

# 5.1 Analyze fusable patterns before fusion
print("Analyzing fusable patterns...")
fusable_patterns = analyze_fusable_patterns(graph)

# 5.2 Print graph before fusion
print("Analyzing graph BEFORE fusion...")
before_fusion_content = print_graph_nodes(graph, "DeepSeekR1 Graph Nodes - BEFORE Fusion")

# 5.3 Apply fusion and print graph after fusion
print("Applying operator fusion...")
apply_classic_fusion(graph)
print("Analyzing graph AFTER fusion...")
after_fusion_content = print_graph_nodes(graph, "DeepSeekR1 Graph Nodes - AFTER Fusion")

# 5.4 Show fusion results
print(f"\nFusion Results:")
# Extract node count information safely
before_count = "N/A"
after_count = "N/A"
for line in before_fusion_content:
    if line.startswith("Total nodes:"):
        before_count = line.split(": ")[1]
        break
for line in after_fusion_content:
    if line.startswith("Total nodes:"):
        after_count = line.split(": ")[1]
        break
print(f"  Nodes before fusion: {before_count}")
print(f"  Nodes after fusion:  {after_count}")

# 5.5 Save to separate files for comparison
# Save before fusion
before_fusion_file = "graph_before_fusion.log"
before_fusion_text = "\n".join(before_fusion_content + ["=" * 60, "Analysis completed."])
with open(before_fusion_file, "w", encoding="utf-8") as f:
    f.write(before_fusion_text)

# Save after fusion
after_fusion_file = "graph_after_fusion.log"
after_fusion_text = "\n".join(after_fusion_content + ["=" * 60, "Analysis completed."])
with open(after_fusion_file, "w", encoding="utf-8") as f:
    f.write(after_fusion_text)

# Print summary to console
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Fusable patterns found: {len(fusable_patterns)}")
print(f"Nodes before fusion: {before_count}")
print(f"Nodes after fusion:  {after_count}")
print(f"Files generated:")
print(f"  - Before fusion: {before_fusion_file}")
print(f"  - After fusion:  {after_fusion_file}")
print("=" * 60)
print("Script finished successfully.")
