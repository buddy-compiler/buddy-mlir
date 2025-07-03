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
# and prints all the nodes in the graph.
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
buddy_mlir_path = "/home/jjji/personal_projects/OSPP2025/buddy-mlir/build/python_packages"
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
model_path = "/mnt/sdb/llm_models/DeepSeek-R1-Distill-Qwen-1.5B"

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
    """打印图节点信息的通用函数"""
    content = []
    content.append("=" * 60)
    content.append(title)
    content.append("=" * 60)
    content.append(f"Total nodes: {len(graph.body)}")
    content.append("")
    
    for i, node in enumerate(graph.body, 1):
        content.append(f"{i:3d}. Node Name: {node.name}")
        content.append(f"     Op Type: {type(node).__name__}")
        if hasattr(node, 'args') and node.args:
            content.append(f"     Args: {node.args}")
        if hasattr(node, 'tensor_meta') and node.tensor_meta:
            content.append(f"     Tensor Meta: {node.tensor_meta}")
        content.append("")
    
    return content

def analyze_fusable_patterns(graph):
    """分析图中可融合的模式"""
    patterns = []
    
    # 检查 Permute + AddMM 模式
    for i, node in enumerate(graph.body):
        if hasattr(node, '__class__') and node.__class__.__name__ == 'AddMMOp':
            # 检查是否有 PermuteOp 作为输入
            for arg in node.args:
                if isinstance(arg, str) and arg in graph.node_table:
                    parent_node = graph.node_table[arg]
                    if hasattr(parent_node, '__class__') and parent_node.__class__.__name__ == 'PermuteOp':
                        if hasattr(parent_node, 'args') and len(parent_node.args) > 1:
                            perm = parent_node.args[1]
                            if perm == [1, 0]:  # transpose pattern
                                patterns.append(f"Transpose+MatMul: {parent_node.name} -> {node.name}")
    
    # 检查 RMSNorm 模式 (Pow + Mean + Add + Rsqrt)
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
print(f"Found {len(fusable_patterns)} fusable patterns:")
for pattern in fusable_patterns:
    print(f"  - {pattern}")

# 5.2 Print graph before fusion
print("\nAnalyzing graph BEFORE fusion...")
before_fusion_content = print_graph_nodes(graph, "DeepSeekR1 Graph Nodes - BEFORE Fusion")

# 5.3 Apply fusion and print graph after fusion
print("Applying operator fusion...")
# 使用真正的算子融合，而不是简单的子图组织
apply_classic_fusion(graph)
print("Analyzing graph AFTER fusion...")
after_fusion_content = print_graph_nodes(graph, "DeepSeekR1 Graph Nodes - AFTER Fusion")

# 5.4 Show fusion results
print(f"\nFusion Results:")
# 安全地提取节点数量信息
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
# Save fusable patterns to separate file
patterns_file = "fusable_patterns.log"
pattern_content = []
pattern_content.append("=" * 60)
pattern_content.append("DeepSeekR1 Fusable Patterns Analysis")
pattern_content.append("=" * 60)
pattern_content.append(f"Total fusable patterns found: {len(fusable_patterns)}")
pattern_content.append("")
if fusable_patterns:
    pattern_content.append("DETECTED PATTERNS:")
    pattern_content.append("-" * 40)
    for i, pattern in enumerate(fusable_patterns, 1):
        pattern_content.append(f"{i:2d}. {pattern}")
else:
    pattern_content.append("No fusable patterns detected.")
pattern_content.append("")
pattern_content.append("=" * 60)
pattern_content.append("Analysis completed.")

with open(patterns_file, "w", encoding="utf-8") as f:
    f.write("\n".join(pattern_content))

# Save before fusion (without pattern info now)
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
print(f"  - Fusable patterns: {patterns_file}")
print(f"  - Before fusion:    {before_fusion_file}")
print(f"  - After fusion:     {after_fusion_file}")
print("=" * 60)
print("Script finished successfully. Compare the two files manually.")
