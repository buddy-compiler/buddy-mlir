#!/usr/bin/env python3
import argparse,os,numpy,torch
import transformers.masking_utils as mask_utils
def universal_bidirectional_mask(*args,**kwargs):
    if args:
        first_arg=args[0]
        shape=first_arg.shape if isinstance(first_arg,torch.Tensor)else first_arg
    else:
        shape=kwargs.get("input_shape",torch.zeros(1,512).shape)
    batch_size,seq_len=shape[0],shape[1]
    return torch.ones((batch_size,1,1,seq_len),device=kwargs.get("device","cpu"))
mask_utils.create_bidirectional_mask=universal_bidirectional_mask
import torch._dynamo;torch._dynamo.config.suppress_errors=True
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import*
from buddy.compiler.graph.transform import simply_fuse,apply_classic_fusion,eliminate_transpose,eliminate_matmul_transpose_reshape
from buddy.compiler.graph.type import DeviceType;from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel
p=argparse.ArgumentParser();p.add_argument("--output-dir",default="./");a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True)
print("[MoLFormer-Import] Loading ibm/MoLFormer-XL-both-10pct...")
m=AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",trust_remote_code=True,dtype=torch.float32).eval()
print(f"  params: {sum(p.numel() for p in m.parameters()):,}")
dc=DynamoCompiler(primary_registry=tosa.ops_registry,aot_autograd_decomposition=inductor_decomp,func_name="forward")
dummy=torch.zeros((1,128),dtype=torch.int64);mask=torch.ones((1,128),dtype=torch.int64)
with torch.no_grad():g=dc.importer(m,input_ids=dummy,attention_mask=mask)
print(f"[MoLFormer-Import] {len(g)} graphs");graph=g[0];params=dc.imported_params.get(graph,[])
print(f"[MoLFormer-Import] {len(params)} params")
graph.perform([eliminate_transpose,eliminate_matmul_transpose_reshape]);graph.fuse_ops([simply_fuse,apply_classic_fusion])
graph.op_groups["subgraph0"]=graph.op_groups.pop("subgraph0");graph.group_map_device["subgraph0"]=DeviceType.CPU
dr=GraphDriver(graph);dr.subgraphs[0].lower_to_top_level_ir()
ld=os.path.join(a.output_dir,"layer_partitioned");os.makedirs(ld,exist_ok=True)
with open(os.path.join(ld,"subgraph0.mlir"),"w")as f:print(dr.subgraphs[0]._imported_module,file=f)
with open(os.path.join(ld,"forward.mlir"),"w")as f:print(dr.construct_main_graph(True),file=f)
numpy.concatenate([p.detach().cpu().numpy().reshape([-1])for p in m.parameters()]).tofile(os.path.join(a.output_dir,"arg0.data"))
print("[MoLFormer-Import] Done!")
