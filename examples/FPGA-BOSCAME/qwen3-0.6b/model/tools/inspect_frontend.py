#!/usr/bin/env python3
"""Record the existing Buddy importer/tool environment without rebuilding it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def cmake_values(path,names):
    if not path.exists():return {}
    result={}
    for line in path.read_text().splitlines():
        if line.startswith(("#","//")) or "=" not in line:continue
        variable,value=line.split("=",1)
        if variable.split(":",1)[0] in names:result[variable]=value
    return result


def inspect(repo,assets,meta):
    source=repo/"examples/BuddyQwen3"
    report={"python_version":sys.version,"packages":{},"source_sha256":{},
            "actual_buddy_model_ir_inspected":False,
            "model_graph_mapping_status":"blocked until importer Python bindings are available; source review only"}
    for package in ("torch","transformers","tokenizers","buddy","mlir"):
        spec=importlib.util.find_spec(package)
        report["packages"][package]="available" if spec else "missing"
    for file in ("import-qwen3.py","CMakeLists.txt","buddy-qwen3-0.6b-main.cpp"):
        report["source_sha256"][str((source/file).relative_to(repo))]=hashlib.sha256((source/file).read_bytes()).hexdigest()
    report["buddy_cmake"]=cmake_values(repo/"build-migrate/CMakeCache.txt",{"BUDDY_MLIR_ENABLE_PYTHON_PACKAGES","Python3_EXECUTABLE","_Python3_EXECUTABLE"})
    report["llvm_cmake"]=cmake_values(repo/"llvm/build-2d26/CMakeCache.txt",{"MLIR_ENABLE_BINDINGS_PYTHON","Python3_EXECUTABLE","_Python3_EXECUTABLE","_Python_EXECUTABLE"})
    # ABI version of built extensions matters; directory names alone do not.
    libraries=repo/"llvm/build-2d26/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
    report["mlir_python_extension_names"]=sorted(p.name for p in libraries.glob("_mlir.cpython-*.so"))
    if meta:
        import torch
        import transformers
        config=transformers.AutoConfig.from_pretrained(assets,local_files_only=True)
        with torch.device("meta"):
            model=transformers.AutoModelForCausalLM.from_config(config,dtype=torch.float32)
        report["transformers_meta_check"]={"torch":torch.__version__,"transformers":transformers.__version__,
            "checkpoint_loaded":False,"inference_executed":False,
            "unique_parameters":sum(p.numel() for p in model.parameters()),
            "buffers":{name:{"shape":list(value.shape),"dtype":str(value.dtype)} for name,value in model.named_buffers()},
            "embedding_lm_head_same_parameter":model.model.embed_tokens.weight is model.lm_head.weight}
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository",type=Path,default=Path(__file__).resolve().parents[5])
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--meta-check",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=inspect(args.repository,args.assets,args.meta_check)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
