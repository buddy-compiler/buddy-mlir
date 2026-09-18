#!/usr/bin/env python3
"""Lower the (replaced) model graph toward an NR binary and report what blocks it.

This runs the same chain the handwritten operator examples use, but starting from
the imported model graph instead of a single kernel:

    Buddy graph -> LLVM dialect -> buddy-translate -> LLVM IR -> llc -> RISC-V .o

It deliberately stops at the object file. Linking a full bare-metal image needs
three things this step is meant to *measure* rather than assume:

  * every undefined symbol the object still needs, grouped by provider;
  * whether the graph pulled in an OpenMP runtime (the graph pipeline lowers
    ``scf`` to OpenMP, and bare-metal NR has no ``libomp``);
  * how much heap traffic bufferization left in, since the NR examples require
    zero heap allocations in a kernel object.

Those readings, not a guess, are what decide the next fix, so the report is the
deliverable here.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys

OPENMP_PREFIX = "__kmpc_"
HEAP = {"malloc", "free", "calloc", "realloc", "aligned_alloc", "posix_memalign"}
LIBC_LIBM = {"memcpy", "memset", "memmove", "expf", "logf", "sinf", "cosf",
             "sqrtf", "tanhf", "memrefCopy", "fmodf", "powf", "floorf"}


def run(command, **kwargs):
    result = subprocess.run([str(c) for c in command], capture_output=True,
                            text=True, **kwargs)
    return result


def classify(symbols):
    groups = {"openmp_runtime": [], "heap": [], "libc_libm": [],
              "qwen_graph_kernels": [], "other": []}
    for name in sorted(symbols):
        if name.startswith(OPENMP_PREFIX):
            groups["openmp_runtime"].append(name)
        elif name in HEAP:
            groups["heap"].append(name)
        elif name.startswith("_mlir_ciface_qwen_graph_"):
            groups["qwen_graph_kernels"].append(name)
        elif name in LIBC_LIBM:
            groups["libc_libm"].append(name)
        else:
            groups["other"].append(name)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument("--kind", default="prefill", choices=["prefill", "decode"])
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument("--attention", action="store_true",
                        help="replace the fused attention with the layout/QK/mask/"
                             "softmax/PV kernels")
    parser.add_argument("--attention-position", action="store_true",
                        help="opt in to position-bounded QK/PV (requires --attention)")
    parser.add_argument("--attention-native-key", action="store_true",
                        help="read original K cache (requires --attention-position)")
    parser.add_argument("--w8a8", action="store_true",
                        help="express the linears and the embedding as "
                             "quantize/int8-matmul/dequantize and an int8 gather")
    parser.add_argument("--openmp", action="store_true",
                        help="keep OpenMP lowering (default off: bare-metal NR has "
                             "no libomp, and the operator pipeline uses cf)")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--share-activation-quantization", action="store_true",
                        help="reuse quantized inputs of proven identical RMSNorm views")
    args = parser.parse_args()
    if args.share_activation_quantization and (not args.w8a8 or args.no_replace):
        parser.error("--share-activation-quantization requires W8A8 replacement")
    if args.attention_position and (not args.attention or args.no_replace):
        parser.error("--attention-position requires --attention and enabled replacement")
    if args.attention_native_key and not args.attention_position:
        parser.error("--attention-native-key requires --attention-position")

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im
    import triton_call_replace as tcr
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp
    from transformers import StaticCache

    repo = args.repo_root or Path(__file__).resolve().parents[5]
    llvm_bin = repo / "llvm/build-2d26/bin"
    buddy_bin = repo / "build-migrate/bin"
    args.output.mkdir(parents=True, exist_ok=True)

    model, config, _ = im.build_model(args.assets, args.checkpoint, torch.float32,
                                      args.layers)
    cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len,
                        batch_size=1)
    with torch.no_grad():
        model(input_ids=torch.zeros((1, 1), dtype=torch.int64),
              past_key_values=cache, use_cache=True,
              cache_implementation="static",
              cache_position=torch.tensor([0], dtype=torch.int64))
    for tensor in im.all_cache_tensors(cache):
        tensor.zero_()

    index = tcr.load_kernel_index(args.triton_build)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decomp,
        func_name=f"forward_{args.kind}",
        enable_external_calls=not args.no_replace,
    )
    kwargs = dict(input_ids=torch.zeros(
                      (1, args.prefill_len if args.kind == "prefill" else 1),
                      dtype=torch.int64),
                  past_key_values=cache, use_cache=True,
                  cache_implementation="static")
    if args.kind == "decode":
        kwargs["cache_position"] = torch.tensor([args.prefill_len],
                                                dtype=torch.int64)
    with torch.no_grad():
        graphs = compiler.importer(model, **kwargs)
    graph = graphs[0]

    replacement = None
    if not args.no_replace:
        graph._enable_external_calls = True
        local = {"replaced": [], "uncovered": [], "kept_interior": []}
        # Must run first: it turns the prefill lm_head into an ordinary M=1
        # linear, which is what removes the transposed-weight copy.
        local["lm_head_last_token_rewrites"] = \
            tcr.rewrite_lm_head_last_token(graph, local)
        counts = {}
        shared_attention = {}
        if args.attention:
            counts["attention"] = tcr.rewrite_attention_to_triton(
                graph, index, {}, local, shared_attention,
                runtime_length=args.attention_position, native_key=args.attention_native_key)
            counts["kv_cache"] = tcr.rewrite_kv_cache_to_triton(
                graph, index, {}, local, shared_attention)
        if args.w8a8:
            # The W8A8 rewrite consumes the same ops as the f32 linear and
            # embedding matchers, so it replaces them rather than adding to them.
            tied = {}
            counts["w8a8_embedding"] = tcr.rewrite_embedding_to_w8a8(
                graph, index, {}, local, tied)
            counts["w8a8"] = tcr.rewrite_linear_to_w8a8(
                graph, index, {}, local, tied,
                    share_activation_quantization=args.share_activation_quantization)
        selected = [n for n, _ in (("linear", tcr.match_linear),
                                   ("rmsnorm", tcr.match_rmsnorm),
                                   ("silu", tcr.match_silu),
                                   ("embedding", tcr.match_embedding))]
        if args.w8a8:
            selected = [n for n in selected if n not in ("linear", "embedding")]
        for name, matcher in (("linear", tcr.match_linear),
                              ("rmsnorm", tcr.match_rmsnorm),
                              ("silu", tcr.match_silu),
                              ("embedding", tcr.match_embedding)):
            if name not in selected:
                continue
            counts[name] = tcr.replace(graph, matcher, index, {}, local)
        if args.w8a8 or args.attention:
            params_now, inputs_now = list(graph.params), list(graph.inputs)
            pruned = tcr.prune_dead_nodes(graph, local)
            tcr._reindex(graph, params_now, inputs_now)
            local["pruned_nodes"] = len(pruned)
            local["parameters"] = [
                {"name": p.name, "shape": list(tcr.shape_of(p) or []),
                 "dtype": str(p.tensor_meta.get("dtype"))}
                for p in graph.params]
            local["workspace_inputs"] = local.get("w8a8_workspace_inputs") or []
            local["inputs"] = [
                {"name": p.name, "shape": list(tcr.shape_of(p) or []),
                 "dtype": str(p.tensor_meta.get("dtype"))}
                for p in graph.inputs]
        replacement = {"pattern_matches": counts,
                       "lm_head_last_token_rewrites":
                           local["lm_head_last_token_rewrites"],
                       "uncovered": local["uncovered"],
                       "external_calls": sum(
                           1 for op in graph.body
                           if type(op).__name__ == "CallExternalOp")}
        # The W8A8 path records the parameter and workspace lists so the segment
        # builder and the image builder can be driven from this same report.
        for key in ("pruned_nodes", "parameters", "workspace_inputs", "inputs"):
            if key in local:
                replacement[key] = local[key]

    graph.op_groups = {"sg": [op for op in graph.body
                              if not isinstance(op, PlaceholderOp)]}
    graph.group_map_device = {"sg": DeviceType.CPU}
    # A bare-metal target must not pull in libomp; the handwritten NR operator
    # pipeline lowers scf to cf instead.
    from graph_contract import describe_entry_abi, remaining_dense_work, llvm_entry_evidence
    dense_coverage = remaining_dense_work(graph)
    graph.lower_to_top_level_ir()
    entry_abi = describe_entry_abi(graph)
    (args.output / "entry-abi.json").write_text(json.dumps(entry_abi, indent=2) + "\n")
    graph.lower_to_llvm_ir(enable_openmp=args.openmp)

    name = f"forward_{args.kind}"
    module_path = args.output / f"{name}.llvm.mlir"
    module_path.write_text(str(graph._imported_module))

    steps = {}
    ll_path = args.output / f"{name}.ll"
    result = run([buddy_bin / "buddy-translate", "--buddy-to-llvmir",
                  module_path, "-o", ll_path])
    steps["buddy_translate"] = {"returncode": result.returncode,
                                "stderr": result.stderr[-2000:]}
    if result.returncode != 0 or not ll_path.is_file():
        report = {"stage": "model graph -> RISC-V object", "kind": args.kind,
                  "replacement": replacement,
        "openmp_enabled": args.openmp, "steps": steps,
                  "status": "FAILED at buddy-translate"}
        (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    native_key_abi = None
    if args.attention_native_key:
        from audit_attention_native_ir import audit
        native_key_abi = {
            "status": "PASS", "scope": "actual post-bufferization key size/stride descriptors",
            "callsites": audit(ll_path.read_text(), config.num_hidden_layers),
        }
        (args.output / "native-key-abi.json").write_text(json.dumps(native_key_abi, indent=2) + "\n")

    object_path = args.output / f"{name}.o"
    result = run([llvm_bin / "llc", ll_path, "-O2", "-filetype=obj",
                  "-mtriple=riscv64", "-target-abi=lp64d",
                  "-mattr=+m,+a,+f,+d,+c,+v,+zvl512b,+xboscame",
                  "-riscv-v-vector-bits-min=512", "-code-model=medium",
                  "-o", object_path])
    steps["llc"] = {"returncode": result.returncode, "stderr": result.stderr[-2000:]}
    if result.returncode != 0 or not object_path.is_file():
        report = {"stage": "model graph -> RISC-V object", "kind": args.kind,
                  "replacement": replacement,
        "openmp_enabled": args.openmp, "steps": steps,
                  "status": "FAILED at llc"}
        (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    undefined = run([llvm_bin / "llvm-nm", "--undefined-only", object_path])
    defined = run([llvm_bin / "llvm-nm", "--defined-only",
                   "--extern-only", object_path])
    undef = sorted({line.split()[-1] for line in undefined.stdout.splitlines()
                    if line.strip()})
    defs = sorted({line.split()[-1] for line in defined.stdout.splitlines()
                   if line.strip()})
    groups = classify(undef)

    report = {
        "stage": "model graph -> RISC-V object (stops before linking a board image)",
        "kind": args.kind,
        "layers": config.num_hidden_layers,
        "replacement": replacement,
        "entry_abi": entry_abi,
        "dense_coverage": dense_coverage,
        "llvm_entry_evidence": llvm_entry_evidence(ll_path.read_text(), name),
        "native_key_abi": native_key_abi,
        "openmp_enabled": args.openmp,
        "artifacts": {
            "llvm_dialect_mlir": {"path": str(module_path),
                                  "bytes": module_path.stat().st_size},
            "llvm_ir": {"path": str(ll_path), "bytes": ll_path.stat().st_size},
            "riscv_object": {"path": str(object_path),
                             "bytes": object_path.stat().st_size},
        },
        "steps": steps,
        "symbols": {
            "defined": len(defs),
            "undefined_total": len(undef),
            "undefined_by_provider": groups,
        },
        "blockers": [],
        "status": "object built; not linkable as a bare-metal image yet",
    }
    if groups["openmp_runtime"]:
        report["blockers"].append({
            "blocker": "OpenMP runtime",
            "evidence": f"{len(groups['openmp_runtime'])} undefined __kmpc_* symbols: "
                        + ", ".join(groups["openmp_runtime"]),
            "why": "graph pipeline lowers scf with convert-scf-to-openmp; bare-metal "
                   "NR has no libomp and the handwritten operator pipeline uses "
                   "convert-scf-to-cf instead",
        })
    if groups["heap"]:
        report["blockers"].append({
            "blocker": "heap allocation",
            "evidence": f"undefined {', '.join(groups['heap'])}",
            "why": "one-shot-bufferize leaves memref.alloc/dealloc in the graph; the "
                   "NR examples require zero heap allocations in a kernel object",
        })
    if groups["other"]:
        report["blockers"].append({
            "blocker": "unresolved provider",
            "evidence": ", ".join(groups["other"]),
            "why": "not supplied by the NR runtime, libm or the Triton archive; "
                   "must be identified before linking",
        })
    (args.output / "nr-lower.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "symbols"}, indent=2))
    print(json.dumps(report["symbols"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
