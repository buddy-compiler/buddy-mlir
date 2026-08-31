#!/usr/bin/env python3
# ===- import-sam2.py - SAM2-hiera-tiny AOT importer ---------------------===//
# Adapted from the original PR import_model.py to the buddy-codegen
# single_forward interface: `--spec` + `--output-dir`, MLIR+weights at root.
#
# Like the original PR, we trace the *vision encoder* sub-module
# (m.vision_encoder, a Sam2VisionModel) rather than the full
# Sam2VideoModel: the full video model consumes prompt points/boxes and a
# memory state that is only meaningful at inference time, whereas the vision
# encoder is a pure fixed-shape image -> feature-map graph.  The traced graph
# is:
#
#   forward(pixel_values: 1 x 3 x image_size x image_size x f32)
#     -> last_hidden_state: 1 x H x W x hidden_size x f32
#
# Fusions are intentionally restricted to `graph.fuse_ops([simply_fuse])` --
# no eliminate_transpose / apply_classic_fusion / etc.
# ===----------------------------------------------------------------------===//
import argparse, os, json, re, numpy, torch
import torch._dynamo; torch._dynamo.config.suppress_errors = True
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel


def _repair_exported(module_text):
    """Repair element-count-mismatched tosa.reshape ops.

    The traced RoPE stack emits ``tosa.reshape`` of a 5-D tensor
    ``[1, W, H, C, 2]`` to 4-D ``[1, W, H, C]``, which drops the trailing
    channel (element count halves: C*2 -> C).  The only element-count
    preserving 4-D target is ``[1, W, H, 2*C]``; downstream the result is
    consumed by a ``tensor.insert_slice`` that previously inserted a 128-wide
    half, so the consuming slice is widened to the full source instead.
    """
    reshape_pat = re.compile(
        r'(%\w+) = "tosa\.reshape"\((%\w+), (%\w+)\) : '
        r'\(tensor<([0-9x]+xf32)>, !tosa\.shape<(\d+)>\) -> '
        r'tensor<([0-9x]+xf32)>')

    def _elems(s):
        n = 1
        for d in s[:-4].split('x'):
            n *= int(d)
        return n

    fixes = []

    def _repl(m):
        result, op, const, ins, _, outs = (m.group(1), m.group(2), m.group(3),
                                           m.group(4), m.group(5), m.group(6))
        if _elems(ins) == _elems(outs):
            return m.group(0)
        dims_in = [int(d) for d in ins[:-4].split('x')]
        if len(dims_in) != 5:
            return m.group(0)
        new_dims = dims_in[:4]
        new_dims[3] = dims_in[3] * dims_in[4]
        new_outs = 'tensor<' + 'x'.join(map(str, new_dims)) + 'xf32>'
        result = result.lstrip('%')
        fixes.append((const, new_dims, new_outs, result,
                      'tensor<' + outs + '>'))
        return f'%{result} = "tosa.reshape"({op}, {const}) : ' \
               f'(tensor<{ins}>, !tosa.shape<4>) -> {new_outs}'

    module_text = reshape_pat.sub(_repl, module_text)

    for const, new_dims, new_outs, result, old_outs in fixes:
        # const_shape values: dense<[old]> -> dense<[new_dims]>.
        cs = f'{const} = "tosa.const_shape"() <{{values = dense<['
        cidx = module_text.find(cs)
        if cidx != -1:
            vs = cidx + len(cs)
            vend = module_text.find(']', vs)
            module_text = (module_text[:vs] +
                           ', '.join(map(str, new_dims)) +
                           module_text[vend:])
        # Consumer insert_slice: source is now 2*C wide, so the whole source
        # must be inserted at offset 0 (size 2*C) instead of a C-wide half at
        # offset C.
        needle = f'"tensor.insert_slice"(%{result}'
        start = 0
        while True:
            idx = module_text.find(needle, start)
            if idx == -1:
                break
            aend = module_text.find('}> : (', idx)
            seg = module_text[idx:aend]
            seg = re.sub(r'(static_offsets = array<i64: )([0-9, ]+)',
                         lambda mo: mo.group(1) +
                         re.sub(r'\d+$', '0', mo.group(2).rstrip()), seg)
            seg = re.sub(r'(static_sizes = array<i64: )([0-9, ]+)',
                         lambda mo: mo.group(1) +
                         re.sub(r'\d+$', str(new_dims[-1]),
                                mo.group(2).rstrip()), seg)
            module_text = (module_text[:idx] + seg + module_text[aend:])
            typ = module_text.find(': (', idx)
            seg2 = module_text[typ + 3: typ + 3 + len(old_outs)]
            if seg2 == old_outs:
                module_text = (module_text[:typ + 3] + new_outs +
                               module_text[typ + 3 + len(old_outs):])
            start = idx + len(needle)
    return module_text, fixes

p = argparse.ArgumentParser(description="SAM2-hiera-tiny AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()
with open(a.spec) as f:
    spec = json.load(f)
model_path = (os.environ.get("SAM2_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "facebook/sam2-hiera-tiny"))
image_size = int(spec.get("image_size", 256))
os.makedirs(a.output_dir, exist_ok=True)

print("[import-sam2] Loading facebook/sam2-hiera-tiny (vision_encoder) from:",
      model_path)
m = AutoModel.from_pretrained(model_path, dtype=torch.float32).eval()
ve = m.vision_encoder
print(f"  vision_encoder: {type(ve).__name__}, "
      f"params: {sum(pp.numel() for pp in ve.parameters()):,}")

dc = DynamoCompiler(primary_registry=tosa.ops_registry,
                    aot_autograd_decomposition=inductor_decomp, func_name="forward")
dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
with torch.no_grad():
    g = dc.importer(ve, dummy)
print(f"[import-sam2] {len(g)} graphs")
assert len(g) == 1, "expected a single traced graph"
graph = g[0]
params = dc.imported_params[graph]
print(f"[import-sam2] first graph: {len(params)} params, "
      f"{sum(pp.numel() for pp in params):,} elems")

graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU
dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()
subgraph0_text = str(dr.subgraphs[0]._imported_module)
subgraph0_text, _fixes = _repair_exported(subgraph0_text)
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(subgraph0_text, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)
all_param = numpy.concatenate(
    [pp.detach().cpu().numpy().reshape([-1]) for pp in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))
print(f"[import-sam2] Wrote forward.mlir, subgraph0.mlir, arg0.data to "
      f"{a.output_dir}")
