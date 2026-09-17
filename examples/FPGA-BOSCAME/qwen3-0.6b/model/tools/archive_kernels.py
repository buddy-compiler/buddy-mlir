#!/usr/bin/env python3
"""Snapshot existing Triton/Buddy objects into a deterministic model static lib.

Read-only with respect to the shared Triton build. This tool does not rebuild
kernels, take over FPGA sessions, or include test launch/runtime objects.
Graph replacement must still check every descriptor/layout precondition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
import subprocess
import tempfile


def sha(data):return hashlib.sha256(data).hexdigest()


def run(tool,*args):
    return subprocess.check_output([*shlex.split(tool),*map(str,args)],text=True)


def symbols(nm,path,defined):
    flags="--defined-only" if defined else "--undefined-only"
    output=run(nm,"--format=posix","--extern-only",flags,path)
    names={line.split()[0] for line in output.splitlines() if len(line.split())>=2}
    return names,output


def archive(root,cases,output,ar,nm):
    if output.exists():raise ValueError("output exists; use a fresh directory for an immutable archive snapshot")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-",dir=output.parent) as directory:
        staging=Path(directory); objects=[]; records=[]; definitions=set(); header=[]
        for name in sorted(set(cases)):
            if not re.fullmatch(r"[a-z][a-z0-9_]*",name):raise ValueError("invalid case name")
            source=root/name; nr=source/"nr"; destination=staging/"evidence"/name
            destination.mkdir(parents=True)
            manifest_bytes=(nr/"manifest.json").read_bytes()
            manifest=json.loads(manifest_bytes)
            front_bytes=(source/"frontend.json").read_bytes();front=json.loads(front_bytes)
            if manifest["case"]["name"]!=name or front["name"]!=name:
                raise ValueError("manifest case mismatch")
            if manifest.get("elf_audit",{}).get("status")!="PASS":
                raise ValueError("case has no successful full linked ELF audit: "+name)
            if manifest.get("heap_allocations")!=0:raise ValueError("kernel heap allocation remains")
            checks={"ttir_sha256":source/"kernel.ttir","linalg_sha256":source/"kernel.linalg.mlir",
                    "llvm_sha256":nr/"kernel.ll","adapter_sha256":source/"adapter.c",
                    "frontend_manifest_sha256":source/"frontend.json"}
            file_hashes={}
            for key,path in checks.items():
                data=path.read_bytes()
                if manifest.get(key)!=sha(data):raise ValueError("stale or changing build: "+str(path))
                (destination/path.name).write_bytes(data);file_hashes[path.name]=sha(data)
            expected={front["symbol"],"_mlir_ciface_kernel_"+name}
            undefined=set(); case_defs=set()
            for basename in ("kernel.o","adapter.o"):
                path=nr/basename;data=path.read_bytes()
                if data[:6]!=b"\x7fELF\x02\x01" or struct.unpack_from("<H",data,18)[0]!=243:
                    raise ValueError("not a RISC-V ELF64 object: "+str(path))
                copied=staging/(name+"."+basename);copied.write_bytes(data)
                defs,dump=symbols(nm,copied,True); undefs,udump=symbols(nm,copied,False)
                if not defs<=expected or defs&definitions:
                    raise ValueError("unexpected/duplicate global symbols: "+str(defs))
                if undefs & {"malloc","calloc","realloc","free","printf","main","launch"}:
                    raise ValueError("unexpected heap/test dependency: "+str(undefs))
                definitions.update(defs);case_defs.update(defs);undefined.update(undefs)
                (destination/(basename+".symbols.txt")).write_text(dump+udump)
                if data!=path.read_bytes():raise ValueError("object changed while snapshotting")
                file_hashes[basename]=sha(data);objects.append(copied)
            if case_defs!=expected:raise ValueError("missing kernel/adapter entry")
            if manifest_bytes!=(nr/"manifest.json").read_bytes() or front_bytes!=(source/"frontend.json").read_bytes():
                raise ValueError("build manifest changed while snapshotting")
            (destination/"manifest.json").write_bytes(manifest_bytes)
            for filename in ("kernel.s","kernel.nr.S"):
                data=(nr/filename).read_bytes();(destination/filename).write_bytes(data);file_hashes[filename]=sha(data)
            args=", ".join("QwenMemRef"+str(a["rank"])+" *" for a in front["arguments"])
            header.append(f"void _mlir_ciface_kernel_{name}({args});")
            records.append({"case":name,"symbols":sorted(case_defs),"undefined_runtime_symbols":sorted(undefined-expected),
                            "files_sha256":file_hashes,"frontend":front,
                            "model_graph_binding":"pending semantic/layout/alias/accumulation matching",
                            "adapter_precondition":"static shapes/grid/physical layout from frontend; offsets supported; strides not dynamically checked",
                            "hardware_validation":"consult parent final verification; archive creation does not run hardware"})
        lib=staging/"libqwen_triton.a"
        run(ar,"rcsD",lib,*objects)
        # Archive member names are unique; runtime/launch/test data cannot enter.
        members=run(ar,"t",lib).splitlines()
        if sorted(members)!=sorted(p.name for p in objects):raise ValueError("archive members differ")
        # Archive-wide nm includes headings, retained as raw evidence instead.
        (staging/"symbols.txt").write_text(run(nm,"--extern-only",lib))
        declarations=["#ifndef QWEN_TRITON_STATIC_ABI_H","#define QWEN_TRITON_STATIC_ABI_H","#include <stdint.h>",
                      "/* Generated static-shape ABI. Graph lowering must validate size/stride/alias first. */",
                      "#ifdef __cplusplus",'extern "C" {',"#endif"]
        for rank in range(1,5):
            declarations.append(f"typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[{rank}], strides[{rank}]; }} QwenMemRef{rank};")
        declarations.extend(header+["#ifdef __cplusplus","}","#endif","#endif",""])
        (staging/"qwen_triton_static_abi.h").write_text("\n".join(declarations))
        report={"format":1,"archive":"libqwen_triton.a","sha256":sha(lib.read_bytes()),
                "members":members,"case_count":len(records),"cases":records,
                "contains_test_launch":False,"contains_runtime":False,"graph_calls_verified":False}
        (staging/"archive.json").write_text(json.dumps(report,indent=2)+"\n")
        # Move the complete snapshot only after all validations succeeded.
        staging.rename(output)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-build",type=Path,required=True)
    parser.add_argument("--case",action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ar",default="llvm-ar")
    parser.add_argument("--nm",default="llvm-nm")
    args=parser.parse_args()
    result=archive(args.triton_build,args.case,args.output,args.ar,args.nm)
    print(f"{result['case_count']} kernels, {len(result['members'])} objects; {args.output/'libqwen_triton.a'}")
