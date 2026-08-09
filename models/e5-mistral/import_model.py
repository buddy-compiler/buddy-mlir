#!/usr/bin/env python3
import argparse, os, numpy, torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import simply_fuse,apply_classic_fusion,eliminate_transpose,eliminate_matmul_transpose_reshape
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel

p=argparse.ArgumentParser();p.add_argument("--output-dir",default="./");a=p.parse_args()
os.makedirs(a.output_dir,exist_ok=True)

print("[E5-Import] Loading e5-mistral-7b-instruct...")
m=AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct",dtype=torch.float32).eval()

dc=DynamoCompiler(primary_registry=tosa.ops_registry,aot_autograd_decomposition=inductor_decomp,func_name="forward")

dummy=torch.zeros((1,128),dtype=torch.int64)
mask=torch.ones((1,128),dtype=torch.int64)

with torch.no_grad():
    g=dc.importer(m,input_ids=dummy,attention_mask=mask)
assert len(g)==1;graph=g[0];params=dc.imported_params[graph]
print(f"[E5-Import] 1 graph, {len(params)} params")

graph.perform([eliminate_transpose,eliminate_matmul_transpose_reshape])
graph.fuse_ops([simply_fuse,apply_classic_fusion])
graph.op_groups["subgraph0"]=graph.op_groups.pop("subgraph0");graph.group_map_device["subgraph0"]=DeviceType.CPU
dr=GraphDriver(graph);dr.subgraphs[0].lower_to_top_level_ir()
ld=os.path.join(a.output_dir,"layer_partitioned");os.makedirs(ld,exist_ok=True)
with open(os.path.join(ld,"subgraph0.mlir"),"w") as f:print(dr.subgraphs[0]._imported_module,file=f)
with open(os.path.join(ld,"forward.mlir"),"w") as f:print(dr.construct_main_graph(True),file=f)
numpy.concatenate([p.detach().cpu().numpy().reshape([-1]) for p in params]).tofile(os.path.join(a.output_dir,"arg0.data"))
print("[E5-Import] Done!")
