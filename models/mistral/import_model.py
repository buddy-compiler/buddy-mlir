#!/usr/bin/env python3
# ===- import_model.py ---------------------------------------------------
#
# Mistral-7B-Instruct-v0.2 Official Model Importer (buddy-mlir Pipeline)
#
# Architecture: MistralForCausalLM (standard decoder-only LLM with GQA)
#   - 32 layers, hidden=4096, heads=32, kv_heads=8, head_dim=128
#   - vocab=32000, sliding window attention
#
# ===---------------------------------------------------------------------------

import argparse, os, types
import numpy, torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    simply_fuse, apply_classic_fusion, eliminate_transpose,
    eliminate_matmul_transpose_reshape, flash_attention_prefill, gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModelForCausalLM, StaticCache

parser = argparse.ArgumentParser(description="Mistral-7B Model AOT Importer")
parser.add_argument("--output-dir", type=str, default="./")
parser.add_argument("--precision", type=str, default="f32", choices=["f32"])
args = parser.parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_path = os.environ.get("MISTRAL_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.2")
print("[Mistral-Import] Loading Mistral-7B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
model.config.use_cache = False

for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

print(f"   hidden={model.config.hidden_size}, layers={model.config.num_hidden_layers}, kv_heads={model.config.num_key_value_heads}")

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)
dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

max_seq_len = 1024

past_kv_prefill = StaticCache(config=model.config, max_cache_len=max_seq_len)
past_kv_decode = StaticCache(config=model.config, max_cache_len=max_seq_len)

print("[Mistral-Import] Tracing prefill...")
with torch.no_grad():
    graphs_prefill = dynamo_compiler_prefill.importer(
        model, input_ids=torch.zeros((1, max_seq_len), dtype=torch.int64),
        use_cache=True, past_key_values=past_kv_prefill,
        cache_position=torch.arange(max_seq_len, dtype=torch.int64),
        cache_implementation="static",
    )
    model(input_ids=torch.zeros((1, 1), dtype=torch.int64), past_key_values=past_kv_decode,
          use_cache=True, cache_implementation="static")
    graphs_decode = dynamo_compiler_decode.importer(
        model, input_ids=torch.zeros((1, 1), dtype=torch.int64),
        use_cache=True, cache_position=torch.tensor([200], dtype=torch.int64),
        past_key_values=past_kv_decode, cache_implementation="static",
    )

assert len(graphs_prefill) == len(graphs_decode) == 1
graph_prefill, graph_decode = graphs_prefill[0], graphs_decode[0]
params = dynamo_compiler_prefill.imported_params[graph_prefill]
print(f"[Mistral-Import] {len(params)} params")

for g in [graph_prefill, graph_decode]:
    g.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])

graph_prefill.fuse_ops([simply_fuse, apply_classic_fusion, flash_attention_prefill])
graph_decode.fuse_ops([simply_fuse, apply_classic_fusion, gqa_attention_fusion])

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop("subgraph0")
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU
graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop("subgraph0")
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graph_prefill); driver_prefill.subgraphs[0].lower_to_top_level_ir()
driver_decode = GraphDriver(graph_decode); driver_decode.subgraphs[0].lower_to_top_level_ir()

layer_dir = os.path.join(output_dir, "layer_partitioned")
os.makedirs(layer_dir, exist_ok=True)

with open(os.path.join(layer_dir, "subgraph0_prefill.mlir"), "w") as f: print(driver_prefill.subgraphs[0]._imported_module, file=f)
with open(os.path.join(layer_dir, "forward_prefill.mlir"), "w") as f: print(driver_prefill.construct_main_graph(True), file=f)
with open(os.path.join(layer_dir, "subgraph0_decode.mlir"), "w") as f: print(driver_decode.subgraphs[0]._imported_module, file=f)
with open(os.path.join(layer_dir, "forward_decode.mlir"), "w") as f: print(driver_decode.construct_main_graph(True), file=f)

all_param = numpy.concatenate([p.detach().cpu().numpy().reshape([-1]) for p in params])
all_param.tofile(os.path.join(output_dir, "arg0.data"))
print("[Mistral-Import] Done!")
