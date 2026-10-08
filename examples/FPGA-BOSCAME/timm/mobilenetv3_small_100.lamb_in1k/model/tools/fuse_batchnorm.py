#!/usr/bin/env python3
"""Fold inference Conv2d -> BatchNorm2d and validate the complete MobileNetV3."""

import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import urllib.request

import timm
from timm.layers import BatchNormAct2d
import torch
from torch import fx, nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors.torch import load_file, save_file


MODEL_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"
ARCHITECTURE = "mobilenetv3_small_100"
REVISION = "1824797e7887cbec1990e4adbd6675960a36c589"
CHECKSUMS = {
    "config.json": "07194b4b5f5140b0d1d1b80c49b6568b726c6e2f88858340cb7618061816b6e8",
    "model.safetensors": "46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697",
}
INPUT_SHAPE = (1, 3, 224, 224)
EXPECTED_BN = 34
ATOL = 1e-4
RTOL = 1e-4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FusionTracer(fx.Tracer):
    """Keep timm's BN+activation intact when finding producer/consumer edges."""

    def is_leaf_module(self, module, name):
        return isinstance(module, nn.BatchNorm2d) or super().is_leaf_module(module, name)


def inference_only(model):
    require(not any(m.training for m in model.modules()),
            "BN folding requires model.eval() for every module")


def fusion_inventory(model):
    """Find every BN's exclusive Conv producer; refuse unsafe graph rewrites."""
    inference_only(model)
    graph = FusionTracer().trace(model)
    calls = Counter(n.target for n in graph.nodes if n.op == "call_module")
    pairs = []
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        bn = model.get_submodule(node.target)
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        require(type(bn) in (nn.BatchNorm2d, BatchNormAct2d),
                f"Unsupported BatchNorm subclass: {node.target}: {type(bn)}")
        require(len(node.args) == 1 and not node.kwargs,
                f"Expected a single positional BN input: {node.target}")
        producer = node.args[0]
        require(isinstance(producer, fx.Node) and producer.op == "call_module",
                f"BN does not directly follow a Conv2d: {node.target}")
        conv = model.get_submodule(producer.target)
        require(type(conv) is nn.Conv2d,
                f"BN producer is not a plain Conv2d: {node.target}")
        require(len(producer.users) == 1,
                f"Conv output has other consumers: {producer.target}")
        require(calls[producer.target] == calls[node.target] == 1,
                f"Shared Conv/BN invocation cannot be folded: {node.target}")
        require(bn.track_running_stats and bn.running_mean is not None
                and bn.running_var is not None,
                f"Inference running statistics required: {node.target}")
        require(conv.out_channels == bn.num_features,
                f"Conv/BN channel mismatch: {node.target}")
        pairs.append({
            "conv": producer.target, "bn": node.target,
            "weight_shape": list(conv.weight.shape),
            "original_bias": conv.bias is not None,
            "fused_bias_shape": [conv.out_channels],
            "groups": conv.groups, "stride": list(conv.stride),
            "padding": list(conv.padding), "dilation": list(conv.dilation),
            "eps": bn.eps, "affine": bn.affine,
            "preserved_children": {name: type(child).__name__
                                   for name, child in bn.named_children()},
        })
    bn_names = {name for name, module in model.named_modules()
                if isinstance(module, nn.BatchNorm2d)}
    require({p["bn"] for p in pairs} == bn_names,
            "Not every BatchNorm2d is covered by a safe Conv2d -> BN edge")
    return pairs


def fold_batchnorm(model):
    """Return a deep-copied model with BN folded into OIHW Conv parameters.

    scale[o] = gamma[o] / sqrt(running_var[o] + eps)
    W'[o,i,h,w] = W[o,i,h,w] * scale[o]
    b'[o] = beta[o] + (b[o] - running_mean[o]) * scale[o]

    A missing Conv bias means b=0. All spatial attributes and groups are kept.
    BatchNormAct2d's drop/activation children run after the fused Conv unchanged.
    """
    pairs = fusion_inventory(model)
    fused = deepcopy(model)
    with torch.no_grad():
        for pair in pairs:
            conv = fused.get_submodule(pair["conv"])
            bn = fused.get_submodule(pair["bn"])
            gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
            beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
            bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            scale = gamma * torch.rsqrt(bn.running_var + bn.eps)
            weight = conv.weight * scale.reshape(-1, 1, 1, 1)
            bias = beta + (bias - bn.running_mean) * scale
            require(torch.isfinite(weight).all().item() and torch.isfinite(bias).all().item(),
                    f"Non-finite folded parameters: {pair['conv']}")
            conv.weight = nn.Parameter(weight, requires_grad=False)
            conv.bias = nn.Parameter(bias, requires_grad=False)
            if isinstance(bn, BatchNormAct2d):
                replacement = nn.Sequential(OrderedDict([
                    ("drop", bn.drop), ("act", bn.act),
                ]))
            else:
                replacement = nn.Identity()
            fused.set_submodule(pair["bn"], replacement.eval())
    require(not any(isinstance(m, nn.BatchNorm2d) for m in fused.modules()),
            "BatchNorm2d remains after folding")
    return fused.eval(), pairs


class OperationCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def execute(model, x):
    counter = OperationCounter()
    with torch.no_grad(), counter:
        logits = model(x)
    require(list(logits.shape) == [1, 1000], "Unexpected model output shape")
    require(torch.isfinite(logits).all().item(), "Non-finite model logits")
    return logits, dict(sorted(counter.counts.items()))


def inspect_graph(model):
    # Default tracing expands BatchNormAct2d, exposing F.batch_norm itself.
    graph = fx.symbolic_trace(model)
    bn_nodes = []
    for node in graph.graph.nodes:
        if (node.op == "call_function" and node.target is F.batch_norm
                or node.op == "call_module" and isinstance(
                    graph.get_submodule(node.target), nn.BatchNorm2d)):
            bn_nodes.append(node.name)
    return str(graph.graph), bn_nodes


def prepare_checkpoint(directory, endpoint):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, expected in CHECKSUMS.items():
        path = directory / name
        url = f"{endpoint.rstrip('/')}/{MODEL_ID}/resolve/{REVISION}/{name}"
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            urllib.request.urlretrieve(url, temporary)
            require(sha256(temporary) == expected, f"Downloaded checksum mismatch: {name}")
            temporary.replace(path)
        require(sha256(path) == expected, f"Checkpoint checksum mismatch: {path}")
        sources[name] = {"url": url, "sha256": expected}
    return sources


def create_model(config):
    require(config["architecture"] == ARCHITECTURE and config["num_classes"] == 1000,
            "Expected the MobileNetV3 Small 100 ImageNet-1k architecture")
    return timm.create_model(ARCHITECTURE, pretrained=False, num_classes=1000).float().cpu().eval()


def load_fused_model(directory):
    """Reconstruct the folded module structure and strictly load saved tensors."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model, pairs = fold_batchnorm(create_model(config))
    require(len(pairs) == EXPECTED_BN, "Unexpected folded model structure")
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=MODEL_ROOT.parent / "build/checkpoint")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "build/bn-fusion")
    parser.add_argument("--validation-dir", type=Path, default=MODEL_ROOT / "validation/bn-fusion")
    args = parser.parse_args()
    require(args.output_dir.resolve() != args.checkpoint_dir.resolve(),
            "Fused output must not overwrite the original checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sources = prepare_checkpoint(args.checkpoint_dir, args.endpoint)
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    original = create_model(config)
    original.load_state_dict(load_file(str(args.checkpoint_dir / "model.safetensors")), strict=True)
    # A dedicated generator makes the fixed input independent of model initialization.
    x = torch.randn(INPUT_SHAPE, generator=torch.Generator().manual_seed(0))
    graph_before, nodes_before = inspect_graph(original)
    reference, ops_before = execute(original, x)
    fused, pairs = fold_batchnorm(original)
    result, ops_after = execute(fused, x)
    graph_after, nodes_after = inspect_graph(fused)
    difference = (result - reference).abs()
    numerical_pass = torch.allclose(result, reference, atol=ATOL, rtol=RTOL)
    top1_before, top1_after = reference.argmax(1).tolist(), result.argmax(1).tolist()
    module_bn_before = sum(isinstance(m, nn.BatchNorm2d) for m in original.modules())
    module_bn_after = sum(isinstance(m, nn.BatchNorm2d) for m in fused.modules())
    runtime_bn_before = sum(n for op, n in ops_before.items() if "batch_norm" in op)
    runtime_bn_after = sum(n for op, n in ops_after.items() if "batch_norm" in op)
    structure_pass = (
        len(pairs) == len(nodes_before) == module_bn_before == runtime_bn_before == EXPECTED_BN
        and len(nodes_after) == module_bn_after == runtime_bn_after == 0
    )
    # The only expected ATen changes are removed BN and its temporary allocations.
    semantic_before = {op: count for op, count in ops_before.items()
                       if "batch_norm" not in op and not op.startswith("aten.empty.")}
    semantic_after = {op: count for op, count in ops_after.items()
                      if "batch_norm" not in op and not op.startswith("aten.empty.")}
    preserved_ops = semantic_before == semantic_after
    report = {
        "schema_version": 1, "model": MODEL_ID, "revision": REVISION,
        "scope": "Offline inference BatchNorm folding only; no Triton kernels or FPGA execution",
        "profile": {"input_shape": list(INPUT_SHAPE), "output_shape": [1, 1000],
                    "layout": "NCHW", "weight_layout": "OIHW", "dtype": "float32",
                    "mode": "eval", "device": "cpu", "seed": 0, "num_threads": 1,
                    "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()},
        "sources": sources,
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("torch", "timm", "safetensors")}},
        "tool_sha256": sha256(__file__),
        "fusion_count": len(pairs), "fusions": pairs,
        "comparison": {"max_abs_error": difference.max().item(),
                       "mean_abs_error": difference.mean().item(),
                       "atol": ATOL, "rtol": RTOL, "allclose": numerical_pass,
                       "top1_before": top1_before, "top1_after": top1_after,
                       "top1_equal": top1_before == top1_after},
        "graph_check": {"bn_modules_before": module_bn_before, "bn_modules_after": module_bn_after,
                        "bn_graph_nodes_before": len(nodes_before), "bn_graph_nodes_after": len(nodes_after),
                        "bn_runtime_calls_before": runtime_bn_before, "bn_runtime_calls_after": runtime_bn_after,
                        "bn_nodes_before": nodes_before, "bn_nodes_after": nodes_after,
                        "non_bn_operations_preserved": preserved_ops, "structure_pass": structure_pass},
        "aten_before": ops_before, "aten_after": ops_after,
        "fpga_result": "NOT EXECUTED", "compiler_artifacts": "NOT APPLICABLE: offline weight transform",
    }
    passed = numerical_pass and top1_before == top1_after and structure_pass and preserved_ops
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    (args.validation_dir / "before.fx.txt").write_text(graph_before + "\n")
    (args.validation_dir / "after.fx.txt").write_text(graph_after + "\n")
    report["graph_sha256"] = {
        name: sha256(args.validation_dir / name) for name in ("before.fx.txt", "after.fx.txt")}
    report["status"] = "INCOMPLETE"
    report_path = args.validation_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    require(passed, f"BN fusion validation failed; see {report_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({k: v.detach().contiguous() for k, v in fused.state_dict().items()},
              str(args.output_dir / "model.safetensors"),
              metadata={"format": "pt", "model": MODEL_ID, "transform": "inference_conv_bn_fold"})
    # Verify the persistent artifact, not just the in-memory rewrite.
    restored = load_fused_model(args.output_dir)
    restored_logits, restored_ops = execute(restored, x)
    reload_pass = restored_logits.numpy().tobytes() == result.numpy().tobytes()
    reload_pass = reload_pass and restored_ops == ops_after
    require(reload_pass, "Saved fused checkpoint does not reproduce the validated model")
    save_file({"input": x, "original_logits": reference, "fused_logits": result},
              str(args.output_dir / "validation_tensors.safetensors"))
    report["artifacts"] = {
        name: {"path": str((args.output_dir / name).resolve()), "sha256": sha256(args.output_dir / name)}
        for name in ("config.json", "model.safetensors", "validation_tensors.safetensors")}
    report["saved_checkpoint_reload_bitwise_equal"] = reload_pass
    report["status"] = "COMPLETE"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"BN_FUSION_HOST_PASS: {len(pairs)}/{EXPECTED_BN} BatchNorm folded")
    print(json.dumps(report["comparison"], indent=2))
    print(f"BN modules / graph nodes / runtime calls: {EXPECTED_BN} -> 0")
    print(f"Saved checkpoint reload: PASS; report: {report_path}")
    print("FPGA: NOT EXECUTED (offline BN folding only)")
    print("COMPLETE")


if __name__ == "__main__":
    main()
