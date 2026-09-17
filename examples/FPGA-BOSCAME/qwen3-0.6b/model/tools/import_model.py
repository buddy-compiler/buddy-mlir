#!/usr/bin/env python3
"""Import the real Qwen3-0.6B model into a Buddy graph and save the IR.

This is the entry point for every later stage: it is the only place that turns
PyTorch Qwen3 into a Buddy computation graph, and it must be reproducible from
the pinned checkpoint alone.

Design decisions that matter for the FPGA target:

  * ``cache_position`` is passed as a *tensor argument* to the decode import, so
    the traced graph keeps it as a data input rather than baking one position
    into the constants. Whether the position really stays dynamic is checked by
    diffing the captured graphs, not assumed.
  * The cache object is created explicitly with ``StaticCache(max_cache_len=N)``
    and handed to the model, so the cache capacity is our choice and not
    ``max_position_embeddings`` (40960), which would never fit on the board.
  * Weights are loaded from the pinned safetensors checkpoint; the framework is
    not allowed to silently re-tie ``lm_head`` because the raw file already
    proved the two tensors are bit-identical.

Everything is parameterised (checkpoint, prefill length, cache capacity, layer
count, output directory) and all stages write evidence files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256_file(path, limit=None):
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
            if limit and total >= limit:
                break
    return digest.hexdigest()


def build_model(config_path, checkpoint_dir, dtype, layers, device="cpu"):
    """Build the model from the pinned config and raw safetensors.

    Deliberately avoids ``from_pretrained`` on the checkpoint directory: that
    path also pulls ``generation_config.json`` and applies framework defaults we
    do not want during graph import. Loading the state dict ourselves keeps the
    parameter set exactly what the file contains and lets us report key
    mismatches instead of silently re-tying weights.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(config_path)
    if layers is not None:
        if not 1 <= layers <= config.num_hidden_layers:
            raise ValueError(f"layers must be in [1, {config.num_hidden_layers}]")
        config.num_hidden_layers = layers
        # layer_types drives the sliding/full attention mix; keep it consistent
        # with the truncated layer count or the model indexes out of range.
        if hasattr(config, "layer_types") and config.layer_types:
            config.layer_types = config.layer_types[:layers]
    model = AutoModelForCausalLM.from_config(config, dtype=dtype).eval()
    model.requires_grad_(False)

    weight_file = Path(checkpoint_dir) / "model.safetensors"
    state = load_file(str(weight_file))
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    if missing:
        raise ValueError("checkpoint is missing required model tensors: " + ", ".join(missing))
    import re
    unexplained = [name for name in unexpected
                   if not (layers is not None and (match := re.match(r"model\.layers\.(\d+)\.", name))
                           and int(match[1]) >= layers)]
    if unexplained:
        raise ValueError("checkpoint has unexpected tensors outside truncated layers: " + ", ".join(unexplained))
    model.load_state_dict(state, strict=False)
    model.config.use_cache = False
    report = {"weight_file": str(weight_file),
              "checkpoint_tensor_count": len(state),
              "model_tensor_count": len(own),
              "missing_keys": missing,
              "unexpected_key_count": len(unexpected),
              "unexpected_key_sample": sorted(unexpected)[:5]}
    return model, config, report


def _json_default(value):
    """Frontend uses enum-like TensorDType/DeviceType; record them as text."""
    return str(value)


def _shape_of(op):
    meta = getattr(op, "_tensor_meta", None)
    if not isinstance(meta, dict):
        return None
    shape = meta.get("shape")
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        return [list(s) for s in shape]
    try:
        return list(shape)
    except TypeError:
        return str(shape)


def all_cache_tensors(cache_object):
    """Return every K/V cache tensor, for both Cache layouts.

    transformers 5.x keeps per-layer cache objects whose tensors only exist
    after a forward pass; older layouts expose flat ``key_cache``/``value_cache``
    lists. Callers need the tensors themselves, not the container.
    """
    tensors = []
    key_cache = getattr(cache_object, "key_cache", None) or []
    value_cache = getattr(cache_object, "value_cache", None) or []
    tensors.extend(key_cache)
    tensors.extend(value_cache)
    if not tensors:
        layers = getattr(cache_object, "layers", None) or []
        for layer in layers:
            for attr in ("keys", "values"):
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    tensors.append(tensor)
    return tensors


def describe_graph(graph):
    """Return a plain-data description of the Buddy graph body.

    Parent *shapes* are recorded next to parent names because the op type alone
    does not determine the ABI: the same MatmulOp is a different kernel
    depending on its operand shapes and on whether the producer was a view.
    """
    ops = []
    by_name = {op.name: op for op in graph.body}
    for index, op in enumerate(graph.body):
        entry = {
            "index": index,
            "name": op.name,
            "op_type": type(op).__name__,
            "arguments": list(getattr(op, "args", [])),
            "keyword_arguments": dict(getattr(op, "kwargs", {})),
        }
        if getattr(op, "_parents", None):
            entry["parents"] = list(op._parents)
            entry["parent_shapes"] = [_shape_of(by_name[p])
                                      for p in op._parents if p in by_name]
            entry["parent_types"] = [type(by_name[p]).__name__
                                     for p in op._parents if p in by_name]
        if getattr(op, "_children", None):
            entry["children"] = list(op._children)
        if hasattr(op, "_tensor_meta") and op._tensor_meta is not None:
            meta = op._tensor_meta
            if isinstance(meta, dict):
                entry["tensor_meta"] = {
                    k: (list(v) if isinstance(v, (list, tuple)) else str(v))
                    for k, v in meta.items()
                    if k in ("shape", "dtype", "stride")
                }
            else:
                try:
                    entry["tensor_meta"] = {
                        "shape": list(meta.shape),
                        "dtype": str(meta.dtype),
                    }
                except TypeError:
                    entry["tensor_meta"] = str(meta)
        for attr in ("input_shape", "output_shape", "input_dtype", "output_dtype",
                     "transpose_dim", "keepdim", "axis"):
            if hasattr(op, attr):
                value = getattr(op, attr)
                try:
                    json.dumps(value)
                    entry[attr] = value
                except (TypeError, ValueError):
                    entry[attr] = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value)
        ops.append(entry)
    return ops


def op_histogram(graph):
    counts = {}
    for op in graph.body:
        name = type(op).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="directory with the official config.json")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="directory containing model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=None,
                        help="truncate layer count (diagnostic only)")
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--fuse", default="none",
                        choices=["none", "classic", "prefill", "decode"],
                        help="which fusion set to apply before saving")
    parser.add_argument("--no-fuse", action="store_true",
                        help="alias for --fuse none (kept explicit in logs)")
    parser.add_argument("--save-mlir", action="store_true")
    parser.add_argument("--init-cache", action="store_true",
                        help="warm-up the StaticCache so prefill takes it as an input")
    parser.add_argument("--no-param-dump", action="store_true",
                        help="skip SHA256/fingerprint dump of graph parameters")
    args = parser.parse_args()

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.graph.transform import (
        apply_classic_fusion,
        eliminate_matmul_transpose_reshape,
        eliminate_transpose,
        flash_attention_prefill,
        gqa_attention_fusion,
        simply_fuse,
    )
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as inductor_decomp

    import transformers
    dtype = getattr(torch, args.dtype)
    model, config, weight_report = build_model(
        args.assets, args.checkpoint, dtype, args.layers)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "buddy_python_packages": os.environ.get("PYTHONPATH", ""),
        "config_source": str(args.assets),
        "checkpoint_source": str(args.checkpoint),
        "layers": model.config.num_hidden_layers,
        "prefill_len": args.prefill_len,
        "max_cache_len": args.max_cache_len,
        "decode_steps": args.decode_steps,
        "dtype": args.dtype,
        "fuse": args.fuse,
        "head_dim": head_dim,
    }

    results = {"environment": environment, "weights": weight_report, "graphs": {}}

    # StaticCache must be constructed before tracing and must have the capacity
    # we intend to deploy; the default would be max_position_embeddings.
    from transformers import StaticCache
    cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len,
                        batch_size=1)

    # A *freshly constructed* StaticCache is not yet an input to the traced
    # graph: the model allocates its own zero-filled cache inside the graph and
    # returns it. That would force the deployment to copy 28x2 cache tensors out
    # of prefill and back into decode. One warm-up forward pass makes the cache
    # tensors real graph inputs, so prefill and decode share one ABI and can both
    # write into the board's persistent KV buffers.
    #
    # The warm-up leaves real values in the cache; they are zeroed afterwards so
    # a stale-cache read during later numeric work is obvious rather than
    # plausible-looking.
    if args.init_cache:
        with torch.no_grad():
            model(
                input_ids=torch.zeros((1, 1), dtype=torch.int64),
                past_key_values=cache,
                use_cache=True,
                cache_implementation="static",
                cache_position=torch.tensor([0], dtype=torch.int64),
            )
        tensors = all_cache_tensors(cache)
        for tensor in tensors:
            tensor.zero_()
        results["cache_initialized"] = {
            "method": "warm-up forward, then zero",
            "tensor_count": len(tensors),
            "shapes": [list(t.shape) for t in tensors[:2]],
        }

    # --- prefill import -------------------------------------------------
    prefill_ids = torch.zeros((1, args.prefill_len), dtype=torch.int64)
    prefill_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_prefill",
    )
    prefill_started = time.time()
    with torch.no_grad():
        graphs = prefill_compiler.importer(
            model,
            input_ids=prefill_ids,
            past_key_values=cache,
            use_cache=True,
            cache_implementation="static",
        )
    results["graphs"]["prefill"] = {
        "import_seconds": round(time.time() - prefill_started, 3),
        "graph_count": len(graphs),
    }
    prefill_graph = graphs[0]
    results["graphs"]["prefill"]["raw_op_histogram"] = op_histogram(prefill_graph)
    results["graphs"]["prefill"]["raw_op_count"] = len(prefill_graph.body)
    (args.output / "prefill-raw-ops.json").write_text(
        json.dumps(describe_graph(prefill_graph), indent=2,
                   default=_json_default) + "\n")

    params = prefill_compiler.imported_params[prefill_graph]
    results["graphs"]["prefill"]["param_count"] = len(params)
    results["graphs"]["prefill"]["param_shapes"] = [
        list(p.shape) for p in params]

    # Parameter identity: the graph only gives an ordered tensor list, and the
    # deployment image must place the *right* checkpoint tensor at each offset.
    # Names are therefore resolved by content, not by position guessing: each
    # parameter gets a full SHA256 plus a cheap fingerprint, which the weight
    # layout step matches against the safetensors tensors.
    if not args.no_param_dump:
        import hashlib
        import numpy as np
        entries = []
        for index, param in enumerate(params):
            array = param.detach().cpu().numpy()
            raw = np.ascontiguousarray(array).tobytes()
            flat = array.reshape(-1)
            step = max(1, flat.size // 64)
            sample = flat[::step][:64]
            entries.append({
                "index": index,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "elements": int(array.size),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "head": [float(v) for v in flat[:8]],
                "tail": [float(v) for v in flat[-8:]],
                "sample": [float(v) for v in sample],
                "sample_step": step,
            })
        (args.output / "params.json").write_text(
            json.dumps(entries, indent=2) + "\n")
        results["graphs"]["prefill"]["param_dump"] = "params.json"
        results["graphs"]["prefill"]["param_total_bytes"] = sum(
            e["bytes"] for e in entries)

    # --- decode import --------------------------------------------------
    decode_ids = torch.zeros((1, 1), dtype=torch.int64)
    cache_position = torch.tensor([args.prefill_len], dtype=torch.int64)
    decode_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name="forward_decode",
    )
    decode_started = time.time()
    with torch.no_grad():
        decode_graphs = decode_compiler.importer(
            model,
            input_ids=decode_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            cache_implementation="static",
        )
    results["graphs"]["decode"] = {
        "import_seconds": round(time.time() - decode_started, 3),
        "graph_count": len(decode_graphs),
    }
    decode_graph = decode_graphs[0]
    results["graphs"]["decode"]["raw_op_histogram"] = op_histogram(decode_graph)
    results["graphs"]["decode"]["raw_op_count"] = len(decode_graph.body)
    (args.output / "decode-raw-ops.json").write_text(
        json.dumps(describe_graph(decode_graph), indent=2,
                   default=_json_default) + "\n")

    # Is cache_position actually a graph input, or did tracing bake it in?
    placeholders = [op.name for op in decode_graph.body
                    if type(op).__name__ == "PlaceholderOp"]
    results["graphs"]["decode"]["placeholders"] = placeholders
    results["graphs"]["prefill"]["placeholders"] = [
        op.name for op in prefill_graph.body
        if type(op).__name__ == "PlaceholderOp"]

    # --- optional fusion ------------------------------------------------
    # "none" must really mean none: the unfused graph is the reference the
    # kernel-matching work reads, so it may not be silently rewritten.
    if args.fuse != "none" and not args.no_fuse:
        for target, graph in (("prefill", prefill_graph), ("decode", decode_graph)):
            graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
            if args.fuse == "classic":
                graph.fuse_ops([simply_fuse, apply_classic_fusion])
            elif args.fuse == "prefill" and target == "prefill":
                graph.fuse_ops([simply_fuse, apply_classic_fusion,
                                flash_attention_prefill])
            elif args.fuse == "decode" and target == "decode":
                graph.fuse_ops([simply_fuse, apply_classic_fusion,
                                gqa_attention_fusion])
            results["graphs"][target]["fused_op_histogram"] = op_histogram(graph)
            results["graphs"][target]["fused_op_count"] = len(graph.body)
            (args.output / f"{target}-fused-ops.json").write_text(
                json.dumps(describe_graph(graph), indent=2,
                           default=_json_default) + "\n")

    # --- IR ------------------------------------------------------------
    if args.save_mlir:
        from buddy.compiler.graph.operation import PlaceholderOp
        for label, graph, group in (("prefill", prefill_graph, "subgraph0_prefill"),
                                    ("decode", decode_graph, "subgraph0_decode")):
            # The importer leaves op_groups empty; grouping is normally done by a
            # fusion pass. Lowering needs exactly one legal group, so build it
            # here without applying any fusion.
            graph.op_groups = {
                group: [op for op in graph.body
                        if not isinstance(op, PlaceholderOp)]
            }
            graph.group_map_device = {group: DeviceType.CPU}
            driver = GraphDriver(graph)
            driver.subgraphs[0].lower_to_top_level_ir()
            (args.output / f"subgraph0_{label}.mlir").write_text(
                str(driver.subgraphs[0]._imported_module))
            (args.output / f"forward_{label}.mlir").write_text(
                str(driver.construct_main_graph(True)))
            results["graphs"][label]["typed_ir_saved"] = True
            text = str(driver.construct_main_graph(True))
            cache_args = text.count("memref<1x8x512x128xf32>")
            results["graphs"][label]["main_graph_cache_memref_mentions"] = cache_args

    results["elapsed_seconds"] = round(time.time() - started, 3)
    results["stage"] = "frontend-import-only; no lowering to FPGA, no inference"
    (args.output / "import.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "graphs"},
                     indent=2))
    for name, info in results["graphs"].items():
        print(f"[{name}] ops={info.get('raw_op_count')} "
              f"import_s={info.get('import_seconds')}")
        hist = info.get("raw_op_histogram") or {}
        top = list(hist.items())[:18]
        for op_name, count in top:
            print(f"    {count:6d}  {op_name}")


if __name__ == "__main__":
    main()
