#!/usr/bin/env python3
"""Validate Qwen3 metadata and budget unique storage; never infer DDR capacity.

Remote mode reads only the safetensors header using HTTP Range. A header proves
shape/storage metadata, not tensor contents or equality of tied weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.parse
import urllib.request

EXPECTED = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
            "num_hidden_layers": 28, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "tie_word_embeddings": True, "attention_bias": False, "hidden_act": "silu"}


def tensor_shapes(config):
    shapes = {"model.embed_tokens.weight": [config["vocab_size"], 1024],
              "model.norm.weight": [1024]}
    projections = {"self_attn.q_proj": [2048,1024], "self_attn.k_proj": [1024,1024],
                   "self_attn.v_proj": [1024,1024], "self_attn.o_proj": [1024,2048],
                   "mlp.gate_proj": [3072,1024], "mlp.up_proj": [3072,1024],
                   "mlp.down_proj": [1024,3072], "self_attn.q_norm": [128],
                   "self_attn.k_norm": [128], "input_layernorm": [1024],
                   "post_attention_layernorm": [1024]}
    for layer in range(config["num_hidden_layers"]):
        for name, shape in projections.items():
            shapes[f"model.layers.{layer}.{name}.weight"] = shape
    return shapes


def remote_header(repository, revision):
    if len(revision)!=40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("immutable checkpoint commit required")
    base=f"https://huggingface.co/{urllib.parse.quote(repository,safe='/')}/resolve/{revision}/model.safetensors"
    def read(start,end):
        # Range-specific URL avoids proxy caches returning the first range twice.
        url=base+f"?metadata_range={start}-{end}"
        req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}"})
        with urllib.request.urlopen(req,timeout=60) as response:
            content_range=response.headers.get("Content-Range","")
            if response.status!=206 or not content_range.startswith(f"bytes {start}-{end}/"):
                raise ValueError("server ignored Range; refusing to download the full checkpoint")
            data=response.read(end-start+2)
            if len(data)!=end-start+1:raise ValueError("incorrect Range response length")
            return data,int(content_range.split("/")[1])
    length,total=read(0,7)
    count=struct.unpack("<Q",length)[0]
    if not 2<=count<=16*1024*1024:raise ValueError("invalid safetensors header size")
    raw,total2=read(8,count+7)
    if total!=total2:raise ValueError("checkpoint size changed")
    return json.loads(raw),{"file_bytes":total,"header_bytes":count,
                            "header_sha256":hashlib.sha256(raw).hexdigest(),
                            "revision":revision,"repository":repository,
                            "tensor_contents_verified":False,"tied_values_verified":False}


def analyze(config, header, context, tokenizer_bytes, workspace_bytes, runtime_bytes):
    for key,value in EXPECTED.items():
        if config.get(key)!=value:raise ValueError(f"unexpected {key}: {config.get(key)!r}")
    if context<24 or context>config["max_position_embeddings"]:
        raise ValueError("context must fit 16-token prefill + 8 decode writes and model positions")
    shapes=tensor_shapes(config)
    header_tensors={k:v for k,v in header.items() if k!="__metadata__"}
    unexpected=set(header_tensors)-set(shapes)-{"lm_head.weight"}
    if unexpected:raise ValueError("unknown checkpoint tensors: "+str(sorted(unexpected)))
    for name,shape in shapes.items():
        tensor=header_tensors.get(name)
        if tensor is None or tensor["shape"]!=shape or tensor["dtype"] not in ("BF16","F16","F32"):
            raise ValueError("missing/incompatible tensor: "+name)
    if "lm_head.weight" in header_tensors and header_tensors["lm_head.weight"]["shape"]!=[151936,1024]:
        raise ValueError("lm_head shape mismatch")
    unique=sum(math.prod(shape) for shape in shapes.values())
    matrix=sum(math.prod(shape) for shape in shapes.values() if len(shape)==2)
    norms=unique-matrix
    rows=sum(shape[0] for shape in shapes.values() if len(shape)==2)
    # Matrix rows use one signed symmetric int8 scale each. Embedding and head
    # share the SAME quantized matrix and scales; gather dequantizes its row.
    kv=2*28*8*context*128*4
    activation=16*(2*1024+2048+2*1024+3*3072)*4
    scores=16*16*context*4
    logits=151936*4
    base={"kv_cache_f32":kv,"activation_conservative_separate_buffers":activation,
          "attention_scores_and_probabilities":2*scores,"last_token_logits_f32":logits,
          "tokenizer_blob":tokenizer_bytes,"reusable_workspace_reserve":workspace_bytes,
          "runtime_code_bss_uart_reserve":runtime_bytes,"nh_and_ra_stacks":2*1024*1024}
    plans={}
    for name,weights,scales in (("f32_tied",unique*4,0),
                                ("bf16_storage_requires_kernel_support",unique*2,0),
                                ("w8a8_shared_embedding",matrix+norms*4,rows*4)):
        components=dict(base,weights=weights,weight_scales_f32=scales)
        plans[name]={"components_bytes":components,"total_bytes":sum(components.values()),
                     "total_mib":sum(components.values())/2**20}
    return {"config_verified":True,"unique_model_parameters":unique,"matrix_parameters":matrix,
            "norm_parameters":norms,"stored_checkpoint_parameters":sum(math.prod(x["shape"]) for x in header_tensors.values()),
            "checkpoint_tensor_count":len(header_tensors),"duplicated_lm_head_in_file":"lm_head.weight" in header_tensors,
            "tie_policy":"must verify full tensor equality before deduplicating; header does not prove equality",
            "context_capacity":context,"kv_layout":"[layer, K_or_V, kv_head, context, head_dim]",
            "plans":plans,"ddr_capacity_verified":False,
            "linker_declared_regions":[{"start":"0x80000000","end_exclusive":"0xb0000000","bytes":0x30000000},
                                       {"start":"0xb8000000","end_exclusive":"0x100000000","bytes":0x48000000}],
            "linker_regions_are_not_hardware_capacity_proof":True,
            "planning_status":"estimates before graph liveness / measured DDR probe; no addresses assigned",
            "expected_checkpoint_shapes":shapes}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--header",type=Path,help="previously saved safetensors-header.json; otherwise range-fetch official header")
    parser.add_argument("--tokenizer-blob",type=Path,required=True)
    parser.add_argument("--context",type=int,default=512)
    parser.add_argument("--workspace-mib",type=int,default=64)
    parser.add_argument("--runtime-mib",type=int,default=8)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.workspace_mib<0 or args.runtime_mib<0:parser.error("negative memory reserve")
    config=json.loads((args.assets/"config.json").read_text())
    source=json.loads((args.assets/"assets-manifest.json").read_text())
    if args.header:
        header=json.loads(args.header.read_text()); provenance={"tensor_contents_verified":False,"local_header_sha256":hashlib.sha256(args.header.read_bytes()).hexdigest()}
    else:header,provenance=remote_header(source["repository"],source["revision"])
    report=analyze(config,header,args.context,args.tokenizer_blob.stat().st_size,args.workspace_mib*2**20,args.runtime_mib*2**20)
    report["checkpoint_provenance"]=provenance
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"safetensors-header.json").write_text(json.dumps(header,indent=2)+"\n")
    (args.output/"memory-budget.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v["total_mib"] for k,v in report["plans"].items()},indent=2))
