#!/usr/bin/env python3

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "build", "python_packages"
        )
    )
)

buddy_mlir_path = "/home/jjji/personal_projects/OSPP2025/buddy-mlir/build/python_packages"
sys.path.append(buddy_mlir_path)

import buddy_mlir
sys.modules['mlir'] = buddy_mlir

import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform import apply_classic_fusion

def debug_qkv_order():
    """Debug the order of operations after QKV fusion"""
    
    model_path = "/mnt/sdb/llm_models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torchscript=True
    ).eval()
    model.config.use_cache = False
    
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    
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
    
    graph = graphs[0]
    
    print("\nApplying QKV fusion...")
    apply_classic_fusion(graph)
    
    # Find view_6 and analyze its dependencies
    view_6_node = None
    for node in graph.body:
        if hasattr(node, 'name') and node.name == 'view_6':
            view_6_node = node
            break
    
    if view_6_node:
        print(f"\nFound view_6:")
        print(f"  Name: {view_6_node.name}")
        print(f"  Args: {view_6_node.args}")
        print(f"  Parents: {getattr(view_6_node, '_parents', 'No parents')}")
        
        # Check if the input tensor exists in graph
        input_tensor = view_6_node.args[0]
        print(f"  Input tensor: {input_tensor}")
        
        # Find the input tensor in the graph
        input_node = None
        for i, node in enumerate(graph.body):
            if hasattr(node, 'name') and node.name == input_tensor:
                input_node = node
                print(f"  Found input node at index {i}: {node.__class__.__name__}")
                break
        
        if not input_node:
            print(f"  ❌ Input tensor {input_tensor} NOT found in graph.body")
            
            # Check if it's in node_table
            if input_tensor in graph.node_table:
                print(f"  ✅ But found in graph.node_table")
                input_node = graph.node_table[input_tensor]
                
                # Find its position in graph.body
                if input_node in graph.body:
                    idx = graph.body.index(input_node)
                    print(f"  Input node is at index {idx} in graph.body")
                else:
                    print(f"  ❌ Input node NOT in graph.body despite being in node_table")
            else:
                print(f"  ❌ Input tensor {input_tensor} NOT found in graph.node_table either")
        
        # Find view_6's position
        view_6_index = graph.body.index(view_6_node)
        print(f"  view_6 is at index {view_6_index}")
        
        if input_node and input_node in graph.body:
            input_index = graph.body.index(input_node)
            print(f"  Input node is at index {input_index}")
            if input_index > view_6_index:
                print(f"  ❌ PROBLEM: Input node comes AFTER view_6 (dependency violation)")
            else:
                print(f"  ✅ Input node comes before view_6")
    else:
        print("\n❌ view_6 not found in graph")
    
    # Print some context around where QKV operations are
    print(f"\nGraph has {len(graph.body)} operations")
    
    # Find QKV related operations
    qkv_ops = []
    for i, node in enumerate(graph.body):
        if hasattr(node, 'name') and 'qkv_fused' in node.name:
            qkv_ops.append((i, node.name, node.__class__.__name__))
    
    print(f"\nFound {len(qkv_ops)} QKV-related operations:")
    for idx, name, op_type in qkv_ops:
        print(f"  {idx}: {name} ({op_type})")
    
    # Check around view_6 area
    if view_6_node:
        view_6_index = graph.body.index(view_6_node)
        start = max(0, view_6_index - 10)
        end = min(len(graph.body), view_6_index + 10)
        
        print(f"\nOperations around view_6 (indices {start}-{end}):")
        for i in range(start, end):
            node = graph.body[i]
            mark = ">>> " if i == view_6_index else "    "
            print(f"{mark}{i}: {node.name} ({node.__class__.__name__})")

if __name__ == "__main__":
    debug_qkv_order()
