#!/usr/bin/env python3
"""Execute the Buddy-compiled Qwen3 graph on the host and score it.

This runs the *compiled graph* -- the same `forward_prefill` / `forward_decode`
functions whose MLIR is saved as build evidence -- through Buddy's MLIR
execution engine. It is the bridge between "the graph imported" and "the graph
computes the right thing", and it is what stage B/C/D are scored against.

The weight buffer is built from `validation/weight-layout.json`, i.e. from the
offsets the compiler itself resolved, and each section is filled from the raw
safetensors tensor that content-matching identified. Nothing here infers a
layout by hand.

The KV cache is passed in and returned by both entry points, so prefill and the
8 decode steps share one set of persistent buffers -- exactly the dataflow the
board will use. `cache_position` stays a runtime tensor, so one decode graph
serves every step.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def load_layout(path):
    layout = json.loads(Path(path).read_text())
    if layout["status"] != "PASS":
        raise SystemExit("weight layout is not PASS; refusing to build weights")
    return layout


def build_weight_buffer(layout, checkpoint, dtype=np.float32):
    """Materialise the flat parameter buffer the compiled graph expects."""
    import struct
    with open(checkpoint, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    base = 8 + header_len

    total = layout["weight_buffer_elements"]
    buffer = np.zeros(total, dtype=dtype)
    filled = 0
    computed = []
    with open(checkpoint, "rb") as handle:
        for section in layout["sections"]:
            name = section["checkpoint_tensor"]
            offset = section["offset_elements"]
            count = section["elements"]
            if name is None:
                computed.append(section)
                continue
            entry = header[name]
            begin = base + entry["data_offsets"][0]
            handle.seek(begin)
            if entry["dtype"] == "BF16":
                raw = np.frombuffer(handle.read(count * 2), dtype=np.uint16)
                values = (raw.astype(np.uint32) << 16).view(np.float32)
            elif entry["dtype"] == "F32":
                values = np.frombuffer(handle.read(count * 4), dtype=np.float32)
            else:
                raise ValueError(f"unsupported checkpoint dtype {entry['dtype']}")
            buffer[offset:offset + count] = values.astype(dtype, copy=False)
            filled += 1
    return buffer, filled, computed


def fill_computed(buffer, computed, config, get_inv_freq):
    """Fill sections the checkpoint does not contain (rotary inv_freq)."""
    records = []
    for section in computed:
        if section["shape"] == [64]:
            values = get_inv_freq(config)
            buffer[section["offset_elements"]:
                   section["offset_elements"] + section["elements"]] = values
            records.append({"index": section["index"], "kind": "inv_freq",
                            "elements": section["elements"]})
        else:
            raise ValueError(f"no rule to compute section {section}")
    return records


def prepare_llvm_libs(repo_root, work_dir):
    """Assemble the runtime libraries the MLIR JIT needs into one directory.

    The frontend expects ``libmlir_runner_utils``, ``libmlir_c_runner_utils`` and
    ``libomp`` side by side, but this repository's LLVM tree keeps libomp under
    the runtimes build directory rather than ``lib/``. Symlinking them into a
    single directory is what makes ``LLVM_LIBS_DIR`` usable, and it avoids
    patching the shared frontend.
    """
    import glob
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = ["libmlir_runner_utils.so", "libmlir_c_runner_utils.so", "libomp.so"]
    found = {}
    search = [
        repo_root / "llvm/build-2d26/lib",
        repo_root / "llvm/build/lib",
    ]
    runtimes = sorted(glob.glob(str(
        repo_root / "llvm/build*/runtimes/runtimes-bins/openmp/runtime/src/libomp.so")))
    for name in needed:
        for directory in search:
            candidate = directory / name
            if candidate.is_file():
                found[name] = candidate
                break
        if name not in found:
            for candidate in glob.glob(str(repo_root / f"llvm/**/{name}"),
                                       recursive=True):
                found[name] = Path(candidate)
                break
    for candidate in runtimes:
        found.setdefault("libomp.so", Path(candidate))
    for name, source in found.items():
        link = work_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source)
    missing = [n for n in needed if n not in found]
    return str(work_dir), {n: str(p) for n, p in found.items()}, missing


class GraphRunner:
    """Compile one Buddy graph and expose a plain numpy call.

    The frontend's own helper hardcodes ``opt_level=3``. On a 3217-op graph that
    spends tens of minutes in LLVM and spawns hundreds of OpenMP threads, which
    makes the stage C/D check impractical rather than wrong. The MLIR-level
    lowering is identical at every opt level, so this builds the execution
    engine directly with a configurable level and a bounded thread count.
    """

    def __init__(self, graph, compiler, llvm_libs_dir=None, opt_level=3,
                 threads=None, external_libs=()):
        from buddy_mlir.execution_engine import ExecutionEngine
        from buddy_mlir import runtime as rt
        import ctypes
        import os
        import platform

        self._rt = rt
        self._ctypes = ctypes
        if llvm_libs_dir:
            os.environ["LLVM_LIBS_DIR"] = llvm_libs_dir
        if threads:
            # Set before libomp initialises; otherwise the EE spawns one thread
            # per core per parallel region and thrashes.
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["OMP_DYNAMIC"] = "FALSE"

        def get_lib_extension():
            return ".dylib" if platform.system() == "Darwin" else ".so"

        from graph_contract import describe_entry_abi, remaining_dense_work
        graph.lower_to_top_level_ir()
        self.entry_abi = describe_entry_abi(graph)
        self.dense_coverage = remaining_dense_work(graph)
        graph.lower_to_llvm_ir()
        extension = get_lib_extension()
        base = Path(llvm_libs_dir) if llvm_libs_dir else (
            Path(__file__).resolve().parents[5] / "llvm/build-2d26/lib")
        shared_libs = [
            str(base / ("libmlir_runner_utils" + extension)),
            str(base / ("libmlir_c_runner_utils" + extension)),
        ]
        openmp = base / ("libomp" + extension)
        if openmp.is_file():
            shared_libs.append(str(openmp))
        # Libraries providing the external call targets the graph now calls
        # (the Triton kernels, via the generated ABI wrappers).
        shared_libs.extend(str(lib) for lib in external_libs)
        self._engine = ExecutionEngine(
            graph._imported_module, opt_level=opt_level,
            shared_libs=shared_libs,
            enable_pic=platform.machine().startswith("riscv"),
        )
        self._graph = graph
        self.opt_level = opt_level
        self.func_name = graph._func_name
        self.output_count = len(graph._output_memref)
        self.output_shapes = [item["shape"] for item in self.entry_abi["outputs"]]

    def __call__(self, *arrays):
        """Invoke the compiled entry with plain numpy inputs, in graph order."""
        rt = self._rt
        ctypes = self._ctypes
        input_slots = []
        # The engine reads the descriptors during the call, so the arrays must
        # stay alive for its duration; holding them in one list is enough.
        self._keepalive = [np.ascontiguousarray(a) for a in arrays]
        for array in self._keepalive:
            descriptor = rt.get_ranked_memref_descriptor(array)
            input_slots.append(ctypes.pointer(ctypes.pointer(descriptor)))
        output_struct = self._graph._output_descriptor()
        output_slot = ctypes.pointer(ctypes.pointer(output_struct))
        self._engine.invoke(self.func_name, output_slot, *input_slots)
        outputs = []
        for index in range(len(self._graph._output_memref)):
            descriptor = getattr(output_struct, str(index))
            value = rt.ranked_memref_to_numpy(ctypes.pointer(descriptor))
            outputs.append(np.array(value))
        return outputs


def cache_tensors(cache_object):
    """Return (keys, values) per decoder layer for a transformers Cache.

    transformers 5.x keeps per-layer cache objects rather than a flat
    ``key_cache`` list, and the tensors only exist after the cache has been
    initialised by a forward pass.
    """
    keys = list(getattr(cache_object, "key_cache", None) or [])
    values = list(getattr(cache_object, "value_cache", None) or [])
    if keys:
        return keys, values
    for layer in getattr(cache_object, "layers", None) or []:
        keys.append(getattr(layer, "keys", None))
        values.append(getattr(layer, "values", None))
    if any(t is None for t in keys):
        return [], []
    return keys, values


def rotary_inv_freq(config):
    """inv_freq[k] = theta^(-2k/head_dim), the standard Qwen3 rotary table."""
    head_dim = getattr(config, "head_dim", None) or \
        config.hidden_size // config.num_attention_heads
    parameters = getattr(config, "rope_parameters", None) or {}
    theta = parameters.get("rope_theta") or getattr(config, "rope_theta", 10000.0)
    index = np.arange(0, head_dim, 2, dtype=np.float64)
    return (1.0 / (theta ** (index / head_dim))).astype(np.float32)


def error_stats(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"nonfinite": True}
    reference, candidate = reference.reshape(-1), candidate.reshape(-1)
    diff = np.abs(reference - candidate)
    cosine = float(np.dot(reference, candidate) /
                   (np.linalg.norm(reference) * np.linalg.norm(candidate) + 1e-30))
    return {"max_abs_error": float(diff.max()) if diff.size else 0.0,
            "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
            "cosine_similarity": cosine,
            "elements": int(diff.size)}


def compare_reference(arrays, report, reference_dir, *, quantized=False):
    """Keep compiler error separate from quantization error and token drift."""
    reference = np.load(reference_dir / "arrays.npz")
    filename = "quant-reference.json" if quantized else "reference.json"
    metadata = json.loads((reference_dir / filename).read_text())
    actual_prefill = arrays["prefill_logits"]
    reference_prefill = reference["prefill_logits"]
    if reference_prefill.ndim == 2:
        reference_prefill = reference_prefill[None, ...]
    if actual_prefill.shape[1] == 1:
        reference_prefill = reference_prefill[:, -1:, :]
    comparisons = {"prefill_logits": error_stats(reference_prefill, actual_prefill)}
    ref_seed = int(np.argmax(reference_prefill[0, -1]))
    ref_steps = metadata.get("decode_steps_recorded", [])
    contexts_equal = ref_seed == report["prefill"]["argmax_last"]
    step_contexts = []
    for step, actual_step in enumerate(report["decode_steps_recorded"]):
        ref_step = ref_steps[step] if step < len(ref_steps) else {}
        contexts_equal = (contexts_equal and ref_step.get("input_token") == actual_step["input_token"]
                          and ref_step.get("cache_position") == actual_step["cache_position"])
        step_contexts.append(contexts_equal)
        comparisons[f"decode_logits_{step}"] = error_stats(
            reference[f"decode_logits_{step}"], arrays[f"decode_logits_{step}"])
    cache_keys = ["kv_key_used", "kv_value_used"]
    cache_keys += [f"decode_kv_{kind}_{step}" for step in range(len(ref_steps))
                   for kind in ("key", "value")]
    for key in cache_keys:
        if key in reference and key in arrays:
            expected = reference[key]
            # quant_model_reference.py has always stored L,T,H,D; older
            # reports predate the explicit layout field.
            if metadata.get("kv_layout", "layer,position,kv_head,head_dim" if quantized else None) == "layer,position,kv_head,head_dim":
                expected = expected.transpose(0, 2, 1, 3)
            comparisons[key] = error_stats(expected, arrays[key])
    return {"reference_dir": str(reference_dir), "comparisons": comparisons,
            "arithmetic_profile": metadata.get("arithmetic_profile", "unspecified"),
            "cache_snapshots_compared": [key for key in cache_keys if key in comparisons],
            "prefill_argmax_match": ref_seed == report["prefill"]["argmax_last"],
            "token_trajectory_match": (ref_seed == report["prefill"]["argmax_last"]
                                       and metadata["generated_ids"] == report["generated_ids"]),
            "reference_generated_ids": metadata["generated_ids"],
            "decode_same_input_context": step_contexts,
            "interpretation": "After token/context divergence, free-running logits include trajectory error and are not an isolated kernel check"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--llvm-libs", type=Path, default=None,
                        help="directory holding the MLIR runner libs; prepared "
                             "automatically when omitted")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--opt-level", type=int, default=1,
                        help="LLVM opt level for the JIT (frontend default is 3, "
                             "which is impractically slow on the 28-layer graph)")
    parser.add_argument("--threads", type=int, default=8,
                        help="bound OpenMP threads used by the execution engine")
    parser.add_argument("--replace-pattern", action="append", default=None,
                        choices=["linear", "rmsnorm", "silu", "embedding"],
                        help="restrict --replace to these patterns (default both)")
    parser.add_argument("--replace", action="store_true",
                        help="apply the graph -> Triton external call transform first")
    parser.add_argument("--triton-build", type=Path, default=None,
                        help="Triton build tree used to resolve kernel symbols")
    parser.add_argument("--external-lib", type=Path, action="append", default=None,
                        help="shared library providing the external call targets")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="directory holding the FP32 reference arrays.npz")
    parser.add_argument("--quant-reference-dir", type=Path,
                        help="independent W8A8 reference; scored separately from FP32")
    parser.add_argument("--max-abs-error", type=float, default=1e-3)
    parser.add_argument("--mean-abs-error", type=float, default=1e-4)
    parser.add_argument("--capture-intermediates", action="store_true",
                        help="save caller-owned workspace after prefill and each decode")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--attention", action="store_true",
                        help="replace the fused attention with the four attention "
                             "kernels")
    parser.add_argument("--attention-position", action="store_true",
                        help="opt in to position-bounded QK/PV (requires --attention)")
    parser.add_argument("--attention-native-key", action="store_true",
                        help="read original K cache (requires --attention-position)")
    parser.add_argument("--w8a8", action="store_true",
                        help="replace each linear with quantize / int8 matmul / "
                             "dequantize; the parameters are quantised offline "
                             "with the same contract the Triton kernels implement")
    args = parser.parse_args()
    if args.attention_position and not args.attention:
        parser.error("--attention-position requires --attention")
    if args.attention_native_key and not args.attention_position:
        parser.error("--attention-native-key requires --attention-position")

    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp
    from transformers import StaticCache

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im

    prompt_ids = [int(v) for v in args.prompt_ids.split(",") if v.strip()]
    if not prompt_ids or args.decode_steps < 0 or len(prompt_ids) + args.decode_steps > args.max_cache_len:
        raise SystemExit("prompt plus decode must fit the positive cache capacity")
    if (args.attention or args.w8a8) and not args.replace:
        raise SystemExit("--attention and --w8a8 require --replace")
    model, config, _weight_report = im.build_model(
        args.assets, args.checkpoint, torch.float32, args.layers)

    # Shared persistent KV buffers: both entry points read and return them, so
    # the board keeps one set of cache arrays.
    cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len,
                        batch_size=1)
    with torch.no_grad():
        model(input_ids=torch.zeros((1, 1), dtype=torch.int64),
              past_key_values=cache, use_cache=True,
              cache_implementation="static",
              cache_position=torch.tensor([0], dtype=torch.int64))
    key_cache, value_cache = cache_tensors(cache)
    if not key_cache:
        raise SystemExit("could not locate cache tensors on StaticCache")
    for tensor in key_cache + value_cache:
        tensor.zero_()
    layer_count = len(key_cache)

    if args.llvm_libs:
        lib_dir, lib_sources, missing = str(args.llvm_libs), {}, []
    else:
        repo_root = args.repo_root or Path(__file__).resolve().parents[5]
        lib_dir, lib_sources, missing = prepare_llvm_libs(
            repo_root, args.output.parent / "llvm-libs")

    layout = load_layout(args.layout)
    weight_buffer, filled, computed = build_weight_buffer(
        layout, args.checkpoint / "model.safetensors")
    computed_records = fill_computed(weight_buffer, computed, config,
                                     rotary_inv_freq)

    external_libs = [lib for lib in (args.external_lib or [])]
    w8a8_reports = []
    workspace = {"prefill": [], "decode": []}
    workspace_index = {"prefill": [], "decode": []}
    report = {
        "stage": "host execution of the Buddy-compiled graph",
        "prompt_ids": prompt_ids,
        "decode_steps": args.decode_steps,
        "max_cache_len": args.max_cache_len,
        "layers": layer_count,
        "weight_buffer_elements": int(weight_buffer.size),
        "weight_buffer_used_for_crosscheck_only": True,
        "weight_sections_filled": filled,
        "computed_sections": computed_records,
        "layout_sha256": layout["layout_sha256"],
        "external_calls_enabled": bool(args.replace),
        "external_libs": [str(lib) for lib in external_libs],
        "jit_opt_level": args.opt_level,
        "jit_threads": args.threads,
        "mlir_runtime_libs": lib_sources,
        "mlir_runtime_libs_missing": missing,
    }

    replacement_report = None
    kernel_index = {}
    if args.replace:
        if not args.triton_build:
            raise SystemExit("--replace needs --triton-build")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import triton_call_replace as tcr
        kernel_index = tcr.load_kernel_index(args.triton_build)

    def build_graph(kind, ids):
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=decomp,
            func_name=f"forward_{kind}",
            enable_external_calls=bool(args.replace),
        )
        kwargs = dict(input_ids=ids, past_key_values=cache, use_cache=True,
                      cache_implementation="static")
        if kind == "decode":
            kwargs["cache_position"] = torch.tensor([args.prefill_len],
                                                    dtype=torch.int64)
        with torch.no_grad():
            graphs = compiler.importer(model, **kwargs)
        graph = graphs[0]
        if args.replace:
            graph._enable_external_calls = True
            local_report = {"replaced": [], "uncovered": [], "kept_interior": []}
            local_report["lm_head_last_token_rewrites"] = \
                tcr.rewrite_lm_head_last_token(graph, local_report)
            counts = {}
            shared_attention = {}
            if getattr(args, "attention", False):
                counts["attention"] = tcr.rewrite_attention_to_triton(
                    graph, kernel_index, {}, local_report, shared_attention,
                    runtime_length=args.attention_position, native_key=args.attention_native_key)
                counts["kv_cache"] = tcr.rewrite_kv_cache_to_triton(
                    graph, kernel_index, {}, local_report, shared_attention)
            if getattr(args, "w8a8", False):
                # Before the f32 matchers: it consumes the same MatmulOps, and the
                # embedding goes first so the tied int8 matrix exists for lm_head.
                tied = {}
                counts["w8a8_embedding"] = tcr.rewrite_embedding_to_w8a8(
                    graph, kernel_index, {}, local_report, tied)
                counts["w8a8"] = tcr.rewrite_linear_to_w8a8(
                    graph, kernel_index, {}, local_report, tied)
            selected = args.replace_pattern or ["linear", "rmsnorm", "silu",
                                                "embedding"]
            if getattr(args, "w8a8", False):
                selected = [n for n in selected
                            if n not in ("linear", "embedding")]
            for name, matcher in (("linear", tcr.match_linear),
                                  ("rmsnorm", tcr.match_rmsnorm),
                                  ("silu", tcr.match_silu),
                                  ("embedding", tcr.match_embedding)):
                if name not in selected:
                    continue
                counts[name] = tcr.replace(graph, matcher, kernel_index, {},
                                           local_report)
            if getattr(args, "w8a8", False) or getattr(args, "attention", False):
                # Pruning must come after *every* matcher, exactly as
                # tools/triton_call_replace.py does it: the other matchers still
                # walk the views and aliases that pruning would remove.
                params_snapshot = list(graph.params)
                inputs_snapshot = list(graph.inputs)
                pruned = tcr.prune_dead_nodes(graph, local_report)
                tcr._reindex(graph, params_snapshot, inputs_snapshot)
                local_report["pruned_nodes"] = len(pruned)
                w8a8_reports.append(local_report)
                # The W8A8 workspace buffers are graph inputs, so the caller owns
                # them. The accumulator must be zero because the int8 matmul
                # accumulates into C; the others are written before being read,
                # but zeroing everything is simpler and equally correct.
                entries = local_report.get("w8a8_workspace_inputs") or []
                workspace[kind] = [
                    np.zeros(entry["shape"], dtype={
                        "TensorDType.Int8": np.int8,
                        "TensorDType.Int32": np.int32,
                    }.get(entry["dtype"], np.float32))
                    for entry in entries]
                workspace_index[kind] = [
                    {"name": entry["name"], "position": i, "role": entry.get("role")}
                    for i, entry in enumerate(entries)]
            nonlocal replacement_report
            if replacement_report is None:
                replacement_report = local_report
            else:
                for key in ("replaced", "uncovered", "kept_interior"):
                    replacement_report[key].extend(local_report[key])
            report.setdefault("replacement", {})[kind] = {
                "pattern_matches": counts,
                "lm_head_last_token_rewrites":
                    local_report["lm_head_last_token_rewrites"],
                "external_calls": sum(1 for op in graph.body
                                      if type(op).__name__ == "CallExternalOp"),
            }
        graph.op_groups = {"sg": [op for op in graph.body
                                  if not isinstance(op, PlaceholderOp)]}
        graph.group_map_device = {"sg": DeviceType.CPU}
        return graph, compiler

    args.prefill_len = len(prompt_ids)
    prefill_started = time.time()
    prefill_graph, prefill_compiler = build_graph(
        "prefill", torch.tensor([prompt_ids], dtype=torch.int64))
    prefill_runner = GraphRunner(prefill_graph, prefill_compiler, lib_dir,
                                 opt_level=args.opt_level, threads=args.threads,
                                 external_libs=external_libs)
    report["prefill"] = {"func": prefill_runner.func_name,
                         "outputs": prefill_runner.output_shapes,
                         "entry_abi": prefill_runner.entry_abi,
                         "dense_coverage": prefill_runner.dense_coverage,
                         "compile_seconds": round(time.time() - prefill_started, 3)}

    decode_graph, decode_compiler = build_graph(
        "decode", torch.zeros((1, 1), dtype=torch.int64))
    decode_runner = GraphRunner(decode_graph, decode_compiler, lib_dir,
                                opt_level=args.opt_level, threads=args.threads,
                                external_libs=external_libs)
    report["decode"] = {"func": decode_runner.func_name,
                        "entry_abi": decode_runner.entry_abi,
                        "dense_coverage": decode_runner.dense_coverage,
                        "outputs": decode_runner.output_shapes}

    # The compiled entry point takes the parameters first (in the same order the
    # flat deployment buffer uses) and then the graph inputs. Verifying that the
    # two orders agree is what ties this host run to the deployment layout.
    def param_arrays(compiler, graph):
        return [p.detach().cpu().numpy()
                for p in compiler.imported_params[graph]]

    def w8a8_param_arrays(graph, report):
        """Parameters for a W8A8 graph: kept f32 tensors plus quantised weights.

        The rewrite replaced each f32 matrix with an int8 matrix and a
        per-output-channel scale, so those buffers have to be produced the same
        way the deployment image will produce them -- by quantising the
        checkpoint with the contract the Triton kernels implement (see
        tools/quant_reference.py).
        """
        from quant_reference import quantize_rows
        sections = layout["sections"]
        originals = report["original_parameters"]
        if len(originals) != len(sections):
            raise SystemExit(
                f"the rewrite reported {len(originals)} original parameters but "
                f"the layout has {len(sections)} sections")
        # original parameter name -> the flat buffer slice holding its value
        source = {}
        for name, section in zip(originals, sections):
            start = section["offset_elements"]
            source[name] = weight_buffer[start:start + section["elements"]]
        by_new_name = {}
        for entry in report.get("w8a8_linears") or []:
            values = source[entry["weight_param"]].reshape(entry["n"], entry["k"])
            quantised, scale = quantize_rows(values)
            by_new_name[entry["weight_param_name"]] = quantised
            by_new_name[entry["scale_param_name"]] = scale.astype(np.float32)
        # The embedding's int8 matrix is the same tied tensor the lm_head uses;
        # the rewrite shares one placeholder between them, so quantising it here
        # by its own record is what fills that shared parameter.
        for entry in report.get("w8a8_embeddings") or []:
            values = source[entry["source_param"]]
            quantised, scale = quantize_rows(
                values.reshape(len(values) // entry["width"], entry["width"]))
            by_new_name[entry["weights_param"]] = quantised
            by_new_name[entry["scale_param"]] = scale.astype(np.float32)
        arrays = []
        for parameter in graph.params:
            name = parameter.name
            if name in by_new_name:
                arrays.append(np.ascontiguousarray(by_new_name[name]))
            elif name in source:
                arrays.append(np.ascontiguousarray(source[name]))
            else:
                raise SystemExit(f"no value for parameter {name}")
        return arrays

    # The attention rewrite adds workspace inputs but leaves the parameters alone,
    # so only W8A8 changes how the parameters are built.
    uses_workspace = (getattr(args, "w8a8", False)
                      or getattr(args, "attention", False))
    if uses_workspace:
        merged = {"w8a8_linears": [], "w8a8_embeddings": [],
                  "original_parameters": None}
        for local in w8a8_reports:
            merged["w8a8_linears"].extend(local.get("w8a8_linears") or [])
            merged["w8a8_embeddings"].extend(local.get("w8a8_embeddings") or [])
            merged["original_parameters"] = (merged["original_parameters"]
                                             or local.get("original_parameters"))
        report["w8a8"] = merged
    if getattr(args, "w8a8", False):
        prefill_params = w8a8_param_arrays(prefill_graph, report["w8a8"])
        decode_params = w8a8_param_arrays(decode_graph, report["w8a8"])
    else:
        prefill_params = param_arrays(prefill_compiler, prefill_graph)
        decode_params = param_arrays(decode_compiler, decode_graph)
    report["parameter_count"] = len(prefill_params)
    report["parameter_shapes"] = [list(p.shape) for p in prefill_params]
    report["param_order_matches_layout"] = (
        [list(p.shape) for p in prefill_params] ==
        [s["shape"] for s in layout["sections"]])
    if getattr(args, "w8a8", False):
        # In W8A8 the parameter list is deliberately different from the f32
        # layout, so the layout-order check does not apply; the shape/order
        # agreement that matters is that every parameter got a value, which
        # w8a8_param_arrays enforces by raising.
        report["param_order_matches_layout"] = "n/a for W8A8"
    elif not report["param_order_matches_layout"]:
        # The compiled entry takes its parameters positionally, so a layout built
        # for a different layer count would still "run" while filling nothing.
        # Failing here is what stops a mislabelled run being recorded as a result.
        raise SystemExit(
            f"parameter list ({len(prefill_params)} tensors) does not match the "
            f"weight layout ({len(layout['sections'])} sections); rebuild the "
            f"layout from this graph or pass the matching --layers")

    def entry_arguments(params, ids, position):
        arguments = list(params)
        arguments.append(ids.astype(np.int64))
        for layer in range(layer_count):
            arguments.append(np.array([position], dtype=np.int64))
            arguments.append(key_cache[layer].numpy())
            arguments.append(value_cache[layer].numpy())
        return arguments

    def w8a8_entry_arguments(params, ids, position, kind):
        arguments = entry_arguments(params, ids, position)
        # The int8 matmul accumulates into C (`tt.load %C` then `tt.dot(a, b,
        # previous)`), so every accumulator must be zero *before each graph
        # call*. Reusing the buffers without clearing them makes the logits grow
        # by a fixed step per call -- measured as a linear ramp 25, 36, 47, 58 ...
        # against a reference that stays at 22.
        for entry in workspace_index[kind]:
            if entry["position"] >= len(workspace[kind]):
                continue
            array = workspace[kind][entry["position"]]
            if entry["role"] == "accumulator":
                array[...] = 0
            elif entry["role"] == "cache_positions" and array.dtype == np.int32:
                # one boundary per query row: start + arange(S), the same
                # expression the graph uses for its rotary and cache positions
                array[...] = position + np.arange(array.size, dtype=np.int32)
        arguments.extend(workspace[kind])
        return arguments

    arrays = {}
    def capture_workspace(kind, label):
        if args.capture_intermediates:
            for entry in workspace_index[kind]:
                arrays[f"{label}_workspace_{entry['name']}"] = workspace[kind][entry["position"]].copy()
    # --- prefill -------------------------------------------------------
    ids = np.array([prompt_ids], dtype=np.int64)
    prefill_call = (w8a8_entry_arguments if uses_workspace
                    else lambda p, i, pos, k: entry_arguments(p, i, pos))
    decode_call = prefill_call
    outputs = prefill_runner(*prefill_call(prefill_params, ids, 0, "prefill"))
    capture_workspace("prefill", "prefill")
    logits = outputs[-1]
    arrays["prefill_logits"] = logits.astype(np.float32)
    report["prefill"]["logits_shape"] = list(logits.shape)
    report["prefill"]["argmax_all"] = [int(v) for v in np.argmax(logits[0], axis=-1)]
    report["prefill"]["argmax_last"] = int(np.argmax(logits[0, -1]))

    # Persist the returned caches into the shared buffers for decode.
    cursor = 0
    kv_k = np.zeros((layer_count, 1, 8, args.max_cache_len, 128), dtype=np.float32)
    kv_v = np.zeros_like(kv_k)
    for layer in range(layer_count):
        returned_k = outputs[cursor + 1]
        returned_v = outputs[cursor + 2]
        cursor += 3
        key_cache[layer].copy_(torch.from_numpy(returned_k))
        value_cache[layer].copy_(torch.from_numpy(returned_v))
        kv_k[layer] = returned_k
        kv_v[layer] = returned_v
    arrays["prefill_kv_key"] = kv_k
    arrays["prefill_kv_value"] = kv_v

    # --- decode --------------------------------------------------------
    steps = []
    next_token = report["prefill"]["argmax_last"]
    for step in range(args.decode_steps):
        position = args.prefill_len + step
        outputs = decode_runner(*prefill_call(
            decode_params, np.array([[next_token]], dtype=np.int64), position,
            "decode"))
        capture_workspace("decode", f"decode_{step}")
        step_logits = outputs[-1][0, -1].astype(np.float32)
        arrays[f"decode_logits_{step}"] = step_logits
        chosen = int(np.argmax(step_logits))
        cursor = 0
        for layer in range(layer_count):
            key_cache[layer].copy_(torch.from_numpy(outputs[cursor + 1]))
            value_cache[layer].copy_(torch.from_numpy(outputs[cursor + 2]))
            cursor += 3
        arrays[f"decode_kv_key_{step}"] = np.stack(
            [tensor.numpy()[0, :, :position + 1, :].copy() for tensor in key_cache])
        arrays[f"decode_kv_value_{step}"] = np.stack(
            [tensor.numpy()[0, :, :position + 1, :].copy() for tensor in value_cache])
        order = np.argsort(-step_logits)
        steps.append({
            "step": step,
            "cache_position": position,
            "input_token": next_token,
            "generated_token": chosen,
            "top_k": [[int(i), float(step_logits[i])] for i in order[:args.top_k]],
        })
        next_token = chosen
    report["decode_steps_recorded"] = steps
    report["generated_ids"] = [s["generated_token"] for s in steps]
    used = args.prefill_len + args.decode_steps
    arrays["kv_key_used"] = np.stack([tensor.numpy()[0, :, :used, :].copy()
                                      for tensor in key_cache])
    arrays["kv_value_used"] = np.stack([tensor.numpy()[0, :, :used, :].copy()
                                        for tensor in value_cache])
    report["kv_layout"] = "layer,kv_head,position,head_dim"

    # FP32 comparison measures total deployment error, including quantization.
    if args.reference_dir:
        comparison = compare_reference(arrays, report, args.reference_dir)
        report["graph_vs_fp32"] = comparison
        report["comparisons"] = comparison["comparisons"]
        report["token_trajectory_matches_reference"] = comparison["token_trajectory_match"]
        report["reference_generated_ids"] = comparison["reference_generated_ids"]
    # Independent quantized reference is the compiler/kernel correctness check.
    validation = None
    if args.quant_reference_dir:
        validation = compare_reference(arrays, report, args.quant_reference_dir,
                                       quantized=True)
        report["graph_vs_quantized_reference"] = validation
    elif not args.w8a8 and args.reference_dir:
        validation = report["graph_vs_fp32"]
    if validation:
        metrics = validation["comparisons"].values()
        passed = (validation["token_trajectory_match"]
                  and all(validation["decode_same_input_context"])
                  and all("max_abs_error" in item
                          and item["max_abs_error"] <= args.max_abs_error
                          and item["mean_abs_error"] <= args.mean_abs_error
                          for item in metrics))
        report["numerical_validation"] = {
            "status": "PASS" if passed else "FAIL",
            "max_abs_error_limit": args.max_abs_error,
            "mean_abs_error_limit": args.mean_abs_error,
            "reference": "quantized" if args.quant_reference_dir else "FP32"}
    else:
        report["numerical_validation"] = {
            "status": "UNVERIFIED",
            "reason": "No matching independent numerical reference supplied"}

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "host-run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:6000])
    return 1 if report["numerical_validation"]["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
