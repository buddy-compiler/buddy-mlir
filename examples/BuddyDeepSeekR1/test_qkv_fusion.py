#!/usr/bin/env python3
# ===- test_qkv_fusion.py ------------------------------------------------
#
# Test script for QKV fusion implementation
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

def test_qkv_fusion():
    """Test QKV fusion on DeepSeekR1 model"""
    
    # 1. Set the model path
    model_path = "/mnt/sdb/llm_models/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torchscript=True
    ).eval()
    model.config.use_cache = False
    print("Model loaded successfully.")
    
    # 2. Initialize Dynamo Compiler
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    
    # 3. Import the model into MLIR module and parameters
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
    
    assert len(graphs) == 1
    graph = graphs[0]
    
    # 4. Print graph statistics before fusion
    print(f"\nBefore QKV fusion:")
    print(f"Total nodes: {len(graph.body)}")
    
    # Count AddMMOp nodes
    addmm_count_before = sum(1 for node in graph.body if hasattr(node, '__class__') and node.__class__.__name__ == 'AddMMOp')
    print(f"AddMMOp nodes: {addmm_count_before}")
    
    # 5. Apply QKV fusion
    print("\nApplying QKV fusion...")
    apply_classic_fusion(graph)
    
    # 6. Print graph statistics after fusion
    print(f"\nAfter QKV fusion:")
    print(f"Total nodes: {len(graph.body)}")
    
    # Count AddMMOp and QKVFusedOp nodes
    addmm_count_after = sum(1 for node in graph.body if hasattr(node, '__class__') and node.__class__.__name__ == 'AddMMOp')
    qkv_fused_count = sum(1 for node in graph.body if hasattr(node, '__class__') and node.__class__.__name__ == 'QKVFusedOp')
    
    print(f"AddMMOp nodes: {addmm_count_after}")
    print(f"QKVFusedOp nodes: {qkv_fused_count}")
    
    # 7. Verify fusion results
    if qkv_fused_count > 0:
        print(f"\n✅ QKV fusion successful! Created {qkv_fused_count} fused operations.")
        print(f"Reduced AddMMOp count from {addmm_count_before} to {addmm_count_after}")
        
        # Print details of fused operations
        for node in graph.body:
            if hasattr(node, '__class__') and node.__class__.__name__ == 'QKVFusedOp':
                print(f"  - {node.name}: Q_dim={node.q_dim}, K_dim={node.k_dim}, V_dim={node.v_dim}")
    else:
        print(f"\n❌ QKV fusion failed. No QKVFusedOp nodes created.")
        
        # Debug: Print some AddMMOp nodes to understand the pattern
        print("\nDebugging - First few AddMMOp nodes:")
        addmm_nodes = [node for node in graph.body if hasattr(node, '__class__') and node.__class__.__name__ == 'AddMMOp']
        for i, node in enumerate(addmm_nodes[:6]):  # Print first 6 AddMMOp nodes
            shape = node.tensor_meta.get('shape', 'Unknown') if hasattr(node, 'tensor_meta') else 'No meta'
            print(f"  {i}: {node.name} - shape: {shape}")

if __name__ == "__main__":
    test_qkv_fusion()
