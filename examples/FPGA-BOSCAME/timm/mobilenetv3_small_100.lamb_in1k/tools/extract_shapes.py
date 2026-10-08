#!/usr/bin/env python3
"""Record actual pretrained timm inference shapes, including functional ops."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import urllib.request

import timm
import torch
from safetensors.torch import load_file
from torch.utils._python_dispatch import TorchDispatchMode


MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"
REVISION = "1824797e7887cbec1990e4adbd6675960a36c589"
EXPECTED_SHA256 = {
    "config.json": "07194b4b5f5140b0d1d1b80c49b6568b726c6e2f88858340cb7618061816b6e8",
    "model.safetensors": "46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697",
}
ROOT = Path(__file__).resolve().parents[1]


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "stride": list(value.stride()),
            "numel": value.numel(),
        }
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def tensor_shapes(value):
    if isinstance(value, dict):
        if "shape" in value:
            return [value["shape"]]
        return [s for v in value.values() for s in tensor_shapes(v)]
    if isinstance(value, list):
        return [s for v in value for s in tensor_shapes(v)]
    return []


def shape_text(value):
    return ", ".join(str(s) for s in tensor_shapes(value)) or "—"


def attributes(module):
    names = (
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
        "dilation", "groups", "padding_mode", "in_features", "out_features",
        "eps", "momentum", "affine", "track_running_stats", "inplace",
        "p", "start_dim", "end_dim", "output_size", "has_skip",
    )
    result = {name: describe(getattr(module, name)) for name in names
              if hasattr(module, name)}
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        result["bias"] = module.bias is not None
    return result


class Recorder(TorchDispatchMode):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        arguments = {}
        for i, spec in enumerate(func._schema.arguments):
            value = args[i] if i < len(args) else kwargs.get(spec.name, spec.default_value)
            arguments[spec.name] = describe(value)
        result = func(*args, **kwargs)
        self.operations.append({
            "index": len(self.operations),
            "op": str(func),
            "module": self.stack[-1]["name"] if self.stack else "<root>",
            "arguments": arguments,
            "outputs": describe(result),
        })
        return result


def matrix_cases(modules):
    cases = []
    for m in modules:
        a = m["attributes"]
        if m["type"] == "Conv2d":
            batch, cout, hout, wout = tensor_shapes(m["outputs"])[0]
            kh, kw = a["kernel_size"]
            groups = a["groups"]
            kind = "depthwise" if groups == a["in_channels"] and groups > 1 else "conv2d"
            mnk = [batch * hout * wout, cout // groups,
                   (a["in_channels"] // groups) * kh * kw]
        elif m["type"] == "Linear":
            output = tensor_shapes(m["outputs"])[0]
            groups, kind = 1, "linear"
            mnk = [math.prod(output[:-1]), a["out_features"], a["in_features"]]
        else:
            continue
        cases.append({
            "module": m["name"], "kind": kind,
            "input": tensor_shapes(m["inputs"])[0],
            "weight": m["parameters"]["weight"]["shape"],
            "bias": m["parameters"].get("bias", {}).get("shape"),
            "output": tensor_shapes(m["outputs"])[0],
            "attributes": a, "groups": groups, "mnk_per_group": mnk,
            "macs": math.prod(mnk) * groups,
        })
    return cases


def unique_operations(operations):
    groups = {}
    for op in operations:
        signature = {k: op[k] for k in ("op", "arguments", "outputs")}
        key = json.dumps(signature, sort_keys=True)
        if key not in groups:
            groups[key] = {**signature, "count": 0, "occurrences": []}
        groups[key]["count"] += 1
        groups[key]["occurrences"].append({"index": op["index"], "module": op["module"]})
    return list(groups.values())


def write_report(data, output):
    s = data["summary"]
    lines = [
        "# MobileNetV3 Small 100 实际 shape 清单", "",
        "由 `tools/extract_shapes.py` 从指定 timm checkpoint 实跑生成。",
        f"输入 `{data['profile']['input_shape']}`，NCHW、FP32、CPU、eval；输出 `{data['profile']['output_shape']}`。",
        "本清单覆盖此静态 profile；不同 batch/分辨率需重新提取。预处理和输出 softmax 不属于 model.forward。", "",
        f"参数 {s['parameter_numel']:,}；checkpoint 张量 {s['checkpoint_tensors']}；运行时 state_dict 张量 {s['state_tensors']}。",
        f"模块调用 {s['module_calls']}；ATen 调用 {s['aten_calls']}；去重后 {s['unique_aten_signatures']} 个签名。",
        f"卷积 {s['conv2d_calls']}（含 depthwise {s['depthwise_calls']}），Linear {s['linear_calls']}；卷积及 Linear MAC 总计 {s['conv_linear_macs']:,}。", "",
        "## 主干与分类头", "",
        "| 模块 | 类型 | 输入 | 输出 |", "| --- | --- | --- | --- |",
    ]
    for m in data["modules"]:
        name = m["name"]
        if name == "<root>" or "." not in name or (name.startswith("blocks.") and name.count(".") == 2):
            lines.append(f"| {name} | {m['type']} | {shape_text(m['inputs'])} | {shape_text(m['outputs'])} |")
    lines += ["", "## 全部卷积及全连接", "",
              "权重为 OIHW；Linear 权重为 [out,in]。M,N,K 采用与 Qwen3 相同的数学约定：A[M,K] × B[K,N] → C[M,N]。",
              "卷积为逻辑展开：M=B×Hout×Wout，N=Cout/groups，K=(Cin/groups)×Kh×Kw；每组执行一次。",
              "depthwise 必须保留 groups，不能当作跨通道的 dense GEMM。这里只描述数学映射，尚未选择 im2col、布局变换或硬件实现。", "",
              "| 模块 | 类型 | 输入 | 权重 / bias | 输出 | kernel / stride / padding | groups | M,N,K（每组） |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in data["matrix_cases"]:
        a = c["attributes"]
        spatial = " / ".join(str(a.get(k, "—")) for k in ("kernel_size", "stride", "padding"))
        lines.append(f"| {c['module']} | {c['kind']} | {c['input']} | {c['weight']} / {c['bias']} | {c['output']} | {spatial} | {c['groups']} | {c['mnk_per_group']} |")
    lines += ["", "## ATen 算子统计", "",
              "包括 SE 的 mean / 广播乘法、残差 add、激活、BN、池化、flatten，以及实际出现的辅助操作。",
              "BN 的空输出张量、empty、视图等按运行时原样记录，不代表均需独立 FPGA 内核。", "",
              "| ATen | 调用次数 |", "| --- | ---: |"]
    for op, count in s["aten_counts"].items():
        lines.append(f"| {op} | {count} |")
    lines += ["", "## 完整 ATen 执行序列", "",
              "表中只展示张量 shape；标量参数、归约轴、keepdim、广播输入、dtype、stride 和 BN epsilon 详见 model.json。", "",
              "| 序号 | 所属模块 | ATen | 输入张量 | 输出张量 |",
              "| ---: | --- | --- | --- | --- |"]
    for op in data["operations"]:
        lines.append(f"| {op['index']} | {op['module']} | {op['op']} | {shape_text(op['arguments'])} | {shape_text(op['outputs'])} |")
    lines += ["", "## 全部参数及 buffer", "",
              "BN num_batches_tracked 是整数标量 buffer，不参与 eval 计算；表中区分参数、buffer 及 checkpoint 来源。", "",
              "| 名称 | 类型 | shape | dtype | checkpoint 中存在 |",
              "| --- | --- | --- | --- | --- |"]
    for p in data["state"]:
        lines.append(f"| {p['name']} | {p['kind']} | {p['shape']} | {p['dtype']} | {p['in_checkpoint']} |")
    (output / "MODEL.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "build/checkpoint")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for filename in ("config.json", "model.safetensors"):
        path = args.checkpoint_dir / filename
        url = f"{args.endpoint.rstrip('/')}/{MODEL_ID}/resolve/{REVISION}/{filename}"
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256[filename]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        sources[filename] = {"url": url, "sha256": digest}
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
    weights = load_file(str(args.checkpoint_dir / "model.safetensors"))
    load_result = model.load_state_dict(weights, strict=True)
    model.eval()
    channels, height, width = config["pretrained_cfg"]["input_size"]
    input_shape = [args.batch_size, channels,
                   args.height if args.height is not None else height,
                   args.width if args.width is not None else width]
    if any(d <= 0 for d in input_shape):
        parser.error("batch-size, height and width must be positive")
    x = torch.randn(input_shape)
    with torch.no_grad():
        reference = model(x)
    stack, modules, handles = [], [], []

    def pre_hook(name):
        def hook(module, inputs):
            record = {
                "index": len(modules), "name": name or "<root>",
                "type": type(module).__name__, "inputs": describe(inputs),
                "attributes": attributes(module),
                "parameters": {n: describe(p) for n, p in module.named_parameters(recurse=False)},
                "buffers": {n: describe(b) for n, b in module.named_buffers(recurse=False)},
            }
            modules.append(record)
            stack.append(record)
        return hook

    def post_hook(module, inputs, output):
        stack.pop()["outputs"] = describe(output)

    for name, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook))
    recorder = Recorder(stack)
    try:
        with torch.no_grad(), recorder:
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert output.numpy().tobytes() == reference.numpy().tobytes()
    assert not stack
    assert list(output.shape) == [args.batch_size, config["num_classes"]]
    assert torch.isfinite(output).all()
    parameters = dict(model.named_parameters())
    state = [{"name": name, "kind": "parameter" if name in parameters else "buffer",
              "in_checkpoint": name in weights, **describe(tensor)}
             for name, tensor in model.state_dict().items()]
    matrices = matrix_cases(modules)
    # Independently check convolution output arithmetic and hook/ATen coverage.
    conv_ops = [o for o in recorder.operations if o["op"] == "aten.convolution.default"]
    conv_cases = [c for c in matrices if c["kind"] != "linear"]
    assert len(conv_ops) == len(conv_cases)
    for case, op in zip(conv_cases, conv_ops, strict=True):
        a = case["attributes"]
        expected_hw = [
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                case["input"][2:], a["padding"], a["dilation"],
                a["kernel_size"], a["stride"], strict=True)
        ]
        assert case["output"] == [input_shape[0], a["out_channels"], *expected_hw]
        assert case["weight"] == [a["out_channels"], a["in_channels"] // a["groups"], *a["kernel_size"]]
        assert case["module"] == op["module"]
        assert case["input"] == op["arguments"]["input"]["shape"]
        assert case["weight"] == op["arguments"]["weight"]["shape"]
        assert case["output"] == op["outputs"]["shape"]
    unique = unique_operations(recorder.operations)
    data = {
        "schema_version": 1, "model": MODEL_ID,
        "sources": {"revision": REVISION, "files": sources},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "timm", "safetensors")},
        "official_config": config,
        "profile": {"input_shape": input_shape, "output_shape": list(output.shape),
                    "layout": "NCHW", "dtype": "float32", "device": "cpu", "mode": "eval",
                    "seed": 0, "input": "torch.randn; no image preprocessing", "num_threads": 1},
        "validation": {"strict_checkpoint_load": True,
                       "checkpoint_sha256_verified": True,
                       "missing_keys": load_result.missing_keys,
                       "unexpected_keys": load_result.unexpected_keys,
                       "instrumented_equals_eager_bitwise": True,
                       "conv_shapes_formula_and_aten_verified": True,
                       "output_finite": True,
                       "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()},
        "summary": {
            "parameter_numel": sum(p.numel() for p in parameters.values()),
            "checkpoint_tensors": len(weights), "state_tensors": len(state),
            "checkpoint_numel": sum(t.numel() for t in weights.values()),
            "module_calls": len(modules), "aten_calls": len(recorder.operations),
            "unique_aten_signatures": len(unique),
            "conv2d_calls": sum(c["kind"] != "linear" for c in matrices),
            "depthwise_calls": sum(c["kind"] == "depthwise" for c in matrices),
            "linear_calls": sum(c["kind"] == "linear" for c in matrices),
            "conv_linear_macs": sum(c["macs"] for c in matrices),
            "aten_counts": dict(sorted(Counter(o["op"] for o in recorder.operations).items())),
        },
        "matrix_convention": "Per group: A[M,K] @ B[K,N] -> C[M,N]; conv logical im2col only, no lowering selected",
        "modules": modules, "operations": recorder.operations,
        "matrix_cases": matrices, "state": state,
        "checkpoint": [{"name": n, **describe(t)} for n, t in weights.items()],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "operator_shapes.json").write_text(json.dumps(unique, indent=2) + "\n")
    write_report(data, args.output_dir)
    print(json.dumps(data["summary"], indent=2))
    print(f"Shapes written to {args.output_dir}")


if __name__ == "__main__":
    main()
