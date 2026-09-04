#!/usr/bin/env python3
# ===- import-paligemma.py - PaliGemma-3B-224 AOT importer -----------------===//
#
# Adapted from the original PR import_model.py to the buddy-codegen
# single_forward interface: `--spec` + `--output-dir`, MLIR + weights at the
# output-dir root.
#
# PaliGemma-3B-224 (google/paligemma-3b-mix-224) is a VLM:
#   - Vision: SigLIP (27 layers, hidden=1152, 16 heads, 256 patches)
#   - Text:   Gemma (18 layers, hidden=2048, heads=8, kv_heads=1)
#   - Projector: Linear(1152 -> 2048)
#
# The traced target replicates the original PR: a single full forward over a
# fixed-shape batch:
#   pixel_values  : [1, 3, 224, 224]  (zero image)
#   input_ids     : [1, 280]          256 <image> tokens (257152) + 24 text
#   attention_mask: [1, 280]
#
# NOTE: the model-loading + trace logic is preserved from the original PR
# (including the get_placeholder_mask monkey-patch needed to trace the full
# forward as one graph). Fusing is restricted to simply_fuse.
#
# ===----------------------------------------------------------------------===//
import argparse
import json
import os

import numpy
import torch

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from buddy.compiler.frontend import DynamoCompiler  # noqa: E402
from buddy.compiler.graph import GraphDriver  # noqa: E402
from buddy.compiler.graph.operation import *  # noqa: E402,F403
from buddy.compiler.graph.transform import simply_fuse  # noqa: E402
from buddy.compiler.graph.type import DeviceType  # noqa: E402
from buddy.compiler.ops import tosa  # noqa: E402
from torch._inductor.decomposition import decompositions as inductor_decomp  # noqa: E402
from transformers import PaliGemmaForConditionalGeneration  # noqa: E402

p = argparse.ArgumentParser(description="PaliGemma AOT importer")
p.add_argument("--spec", required=True)
p.add_argument("--output-dir", required=True)
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)

model_path = (
    os.environ.get("PALIGEMMA_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "google/paligemma-3b-mix-224")
)
os.makedirs(a.output_dir, exist_ok=True)

print("[import-paligemma] Loading PaliGemma-3B-224 from:", model_path)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float32
).eval()
model.config.text_config.use_cache = False
print(
    f"  model class: {type(model).__name__}, "
    f"params: {sum(pp.numel() for pp in model.parameters()):,}"
)
print(
    f"  text_hidden={model.config.text_config.hidden_size}, "
    f"text_layers={model.config.text_config.num_hidden_layers}"
)
print(
    f"  vision_hidden={model.config.vision_config.hidden_size}, "
    f"vision_layers={model.config.vision_config.num_hidden_layers}"
)

# Unwrap any __wrapped__ forwards (KeOps / flash-attention style wrappers).
import types  # noqa: E402

for m in model.modules():
    if hasattr(m.forward, "__wrapped__"):
        m.forward = types.MethodType(m.forward.__wrapped__, m)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# Dummy inputs: 1 image (256 tokens) + 24 text tokens, matching the original
# PR trace target. PaliGemma expects the <image> token (id=257152) at the
# start of the sequence followed by text.
n_img_tokens = int(spec.get("num_image_tokens", 256))
seq_len = int(spec.get("max_seq_len", 280))
img_size = int(spec.get("image_size", 224))
pixel_values = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
input_ids = torch.ones((1, seq_len), dtype=torch.int64)
input_ids[0, :n_img_tokens] = int(spec.get("image_token_id", 257152))
attention_mask = torch.ones((1, seq_len), dtype=torch.int64)

print(
    f"[import-paligemma] pixel_values: {tuple(pixel_values.shape)}, "
    f"input_ids: {tuple(input_ids.shape)}, attention_mask: {tuple(attention_mask.shape)}"
)

# ── Monkey-patch get_placeholder_mask for fullgraph tracing ──
# The original method checks image token count vs image features and raises a
# ValueError if they don't match. This is data-dependent (numel() on a masked
# tensor) and causes Dynamo graph breaks. Replace with a no-op version; our
# dummy inputs are pre-validated (256 image tokens, 256 image features).
_orig_get_placeholder_mask = model.model.get_placeholder_mask


def _patched_get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
    if input_ids is None:
        special_image_mask = inputs_embeds == self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
    else:
        special_image_mask = input_ids == self.config.image_token_id
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(
        inputs_embeds.device
    )
    return special_image_mask


model.model.get_placeholder_mask = _patched_get_placeholder_mask.__get__(model.model)
print("[import-paligemma] monkey-patched get_placeholder_mask.")


def _fix_generic_indexing_maps(text):
    """Repair `linalg.generic` indexing-map rank mismatches emitted by the
    tosa->linalg scalar-broadcast lowering (e.g. the -inf causal-mask fill:
    a `tensor<1xf32>` operand can receive a full-rank map such as
    `affine_map<(d0, d1) -> (d0, d1)>`, which fails `linalg.generic`
    verification).  For each mismatched operand, substitute a fresh
    constant-projection map `affine_map<(d0, ..., dn) -> (0, ...)>` whose
    result arity equals the operand's tensor rank (valid because the
    broadcasted dims are all size 1)."""
    import re as _re

    def _split_top(s):
        out, depth, cur = [], 0, []
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur).strip())
        return out

    def _tensor_rank(t):
        m = _re.match(r"tensor<([^>]*)>", t)
        if not m:
            return None
        rank = 0
        for d in m.group(1).split("x"):
            if _re.fullmatch(r"\d+", d) or _re.fullmatch(r"\?", d):
                rank += 1
            else:
                break
        return rank

    def _is_ones(t):
        m = _re.match(r"tensor<([^>]*)>", t)
        if not m:
            return False
        for d in m.group(1).split("x"):
            if _re.fullmatch(r"\d+", d):
                if int(d) != 1:
                    return False
            else:
                break
        return True

    def _map_arity(defbody):
        m = _re.search(r"->\s*\((.*)\)\s*$", defbody)
        if not m:
            return None
        expr = m.group(1).strip()
        if expr == "":
            return 0
        return len(_split_top(expr))

    # Existing affine-map aliases (one per line).
    arities = {}
    max_num = -1
    last_def_end = None
    for m in _re.finditer(
        r"^(#map(\d*))\s*=\s*affine_map<(.+)>\s*$", text, _re.MULTILINE
    ):
        arities[m.group(1)] = _map_arity(m.group(3))
        if m.group(2) != "":
            max_num = max(max_num, int(m.group(2)))
        last_def_end = m.end()

    gen_pat = _re.compile(
        r'"linalg\.generic"\s*\(([^)]*)\)\s*<\{([^}]*)\}>\s*\(\{\s*\n\s*'
        r"\^bb[0-9]*\([^\n]*\):\s*\n(.*?)\n\s*\}\)\s*:\s*\(([^)]*)\)\s*->\s*([^\s(]+)",
        _re.DOTALL,
    )

    fixes = []  # (generic_match, operand_idx, new_alias, new_def_line)
    for gm in gen_pat.finditer(text):
        attrs = gm.group(2)
        mm = _re.search(r"indexing_maps\s*=\s*\[([^\]]*)\]", attrs)
        if not mm:
            continue
        mrefs = [x.strip() for x in mm.group(1).split(",")]
        loop_rank = len(_re.findall(r"#linalg\.iterator_type<[^>]+>", attrs))
        types = [t.strip() for t in _split_top(gm.group(4))]
        ret = gm.group(5)
        for i, mref in enumerate(mrefs):
            ty = ret if i >= len(types) else types[i]
            tr = _tensor_rank(ty)
            if tr is None or mref not in arities or arities[mref] == tr:
                continue
            if not _is_ones(ty):
                raise RuntimeError(
                    "[import-paligemma] non-unit tensor generic rank mismatch: "
                    f"{ty} via {mref}"
                )
            alias = "#map%d" % (max_num + len(fixes) + 1)
            domain = ", ".join("d%d" % j for j in range(loop_rank))
            proj = ", ".join("0" for _ in range(tr))
            fixes.append((gm, i, alias, f"{alias} = affine_map<({domain}) -> ({proj})>"))

    out = text
    # Apply per-generic map fixes bottom-up (spans never overlap).
    for gm, i, alias, _ in reversed(fixes):
        mm = _re.search(r"indexing_maps\s*=\s*\[([^\]]*)\]", gm.group(2))
        mrefs = [x.strip() for x in mm.group(1).split(",")]
        mrefs[i] = alias
        new_attrs = gm.group(2)[: mm.start(1)] + ", ".join(mrefs) + gm.group(2)[mm.end(1):]
        out = out[: gm.start()] + out[gm.start():gm.end()].replace(gm.group(2), new_attrs) + out[gm.end():]

    # Append the new alias definitions after the last existing one.
    if fixes and last_def_end is not None:
        out = out[:last_def_end] + "\n" + "\n".join(f[3] for f in fixes) + out[last_def_end:]

    return out


def _fix_paligemma_mask_and_norm(subgraph0_text):
    """Repair the PaliGemma attention-mask and decoder-input-norm lowering.

    Three structural fixes, each verified against the real
    google/paligemma-3b-mix-224 model (the compiled last-token logits
    correlate 1.000000000 with the HF model's):

    1. Causal mask -> zero.  This transformers version applies an all-zeros
       attention mask (no causal masking); the traced subgraph instead built a
       full lower-triangular causal mask.  Zeroing the widening reshape leaves
       only the padding-aware ``attn_mask == 0`` term.
    2. Mask select operand order.  The emitted body selected the mask tensor as
       the TRUE branch and -3.4e38 as the FALSE branch, i.e. it masked every
       VALID position.  Force ``select(cond, -3.4e38, mask)`` so padding
       positions get -3.4e38 and everything else stays unmasked.
    3. Decoder-input norm scale.  Constant folding collapsed the RMSNorm input
       scale to 0.0, which also zeroes the layer-0 residual base.  Restore
       sqrt(2048) = 45.2548332 (scale-invariant for the norm output, but
       required so the residual base matches the real model).
    """
    import re as _re

    # 1. Zero the causal mask: the single reshape that widens the 1x280x280
    #    causal tensor to 1x1x280x280 becomes a zero constant (the only live
    #    mask term left is the padding check against attention_mask).
    def _zero_causal(m):
        return ('%s = "arith.constant"() <{value = dense<0.000000e+00> : '
                'tensor<1x1x280x280xf32>}> : () -> tensor<1x1x280x280xf32>'
                % m.group(1))

    subgraph0_text = _re.sub(
        r'(%[\w]+) = "tosa\.reshape"\(%[\w]+, %[\w]+\) : '
        r'\(tensor<1x280x280xf32>, !tosa\.shape<4>\) -> tensor<1x1x280x280xf32>',
        _zero_causal, subgraph0_text)

    # 2. Mask select operand order.  The mask linalg.generic is the only block
    #    whose region declares (i1, f32, f32, f32); force its select to
    #    select(cond, -3.4e38, mask).
    _blk = _re.search(
        r'\^bb0\((%[\w]+): i1, (%[\w]+): f32, (%[\w]+): f32, (%[\w]+): f32\):',
        subgraph0_text)
    if _blk:
        cond, scalar, mask = _blk.group(1), _blk.group(2), _blk.group(3)
        subgraph0_text = _re.sub(
            r'("arith\.select"\()%[\w]+, %[\w]+, %[\w]+\) : \(i1, f32, f32\) -> f32',
            lambda mm: mm.group(1) + '%s, %s, %s) : (i1, f32, f32) -> f32'
                        % (cond, scalar, mask),
            subgraph0_text, count=1)

    # 3. Decoder-input norm scale: the constant feeding (identity/reshape) the
    #    single `tosa.mul(dec_input, 1x1x1, 0)` is forced to sqrt(2048).
    _scale = _re.compile(
        r'(%[\w]+) = "arith\.constant"\(\) <\{value = dense<[^>]*> : '
        r'tensor<1xf32>\}> : \(\) -> tensor<1xf32>\n'
        r'[ \t]+%[\w]+ = "tosa\.identity"\(\1\) : \(tensor<1xf32>\) -> tensor<1xf32>\n'
        r'[ \t]+%[\w]+ = "tosa\.const_shape"\(\) <\{values = dense<1> : '
        r'tensor<3xindex>\}> : \(\) -> !tosa\.shape<3>\n'
        r'[ \t]+%[\w]+ = "tosa\.reshape"\([^\n]*\) : \(tensor<1xf32>, '
        r'!tosa\.shape<3>\) -> tensor<1x1x1xf32>\n'
        r'[ \t]+%[\w]+ = "tosa\.const"\(\) <\{values = dense<0> : tensor<1xi8>\}> : '
        r'\(\) -> tensor<1xi8>\n'
        r'[ \t]+%[\w]+ = "tosa\.mul"\(%[\w]+, %[\w]+, %[\w]+\) : '
        r'\(tensor<1x280x2048xf32>, tensor<1x1x1xf32>, tensor<1xi8>\) -> '
        r'tensor<1x280x2048xf32>')

    def _fix_scale(mm):
        line = mm.group(0)
        line, n = _re.subn(r'(dense<)[^>]*(> : tensor<1xf32>)',
                           r'\g<1>4.52548332E+01\g<2>', line, count=1)
        if n != 1:  # defensive: leave untouched rather than corrupt the chain
            return mm.group(0)
        return line

    subgraph0_text = _scale.sub(_fix_scale, subgraph0_text, count=1)
    return subgraph0_text


with torch.no_grad():
    g = dynamo_compiler.importer(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
print(f"[import-paligemma] {len(g)} graphs")
if len(g) != 1:
    raise SystemExit(
        f"[import-paligemma] expected exactly 1 graph, got {len(g)}; "
        "the full VLM forward did not trace as one graph."
    )
graph = g[0]
params = dynamo_compiler.imported_params[graph]

# The main-graph flattening (`GraphImporter._pack_params`) groups params by
# dtype and packs each dtype group into its own memref. PaliGemma carries one
# non-f32 leaf (the SigLIP `position_ids` buffer, i64, 256 elems) which becomes
# its own `memref<256xi64>` argument of `@forward` (filled 0..255 by the
# runner). Only the f32 params go into `arg0.data`.
non_f32 = [(pp.dtype, tuple(pp.shape), pp.numel())
           for pp in params if pp.dtype != torch.float32]
if non_f32:
    print("[import-paligemma] non-f32 leaf tensors (packed separately, "
          f"NOT in arg0.data): {non_f32}")
params_f32 = [pp for pp in params if pp.dtype == torch.float32]
n_elem = sum(pp.numel() for pp in params_f32)
print(
    f"[import-paligemma] first graph: {len(params)} leaves, "
    f"{len(params_f32)} f32 -> arg0.data ({n_elem:,} elems)"
)

spec_params = int(spec.get("params_size", 0))
if spec_params and spec_params != n_elem:
    print(
        f"[import-paligemma] WARNING: spec params_size={spec_params} != "
        f"actual f32 sum {n_elem}",
        file=os.sys.stderr,
    )

graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU
dr = GraphDriver(graph)
dr.subgraphs[0].lower_to_top_level_ir()

subgraph0_text = str(dr.subgraphs[0]._imported_module)
subgraph0_text = _fix_generic_indexing_maps(subgraph0_text)
subgraph0_text = _fix_paligemma_mask_and_norm(subgraph0_text)
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as f:
    print(subgraph0_text, file=f)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as f:
    print(dr.construct_main_graph(True), file=f)

all_param = numpy.concatenate(
    [pp.detach().cpu().numpy().reshape([-1]) for pp in params_f32]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))
print(
    f"[import-paligemma] Wrote forward.mlir, subgraph0.mlir, "
    f"arg0.data ({all_param.size:,} elems) to {a.output_dir}"
)

# ── Layout sanity check: the main graph flattens the f32 params into arg0 by
# concatenating them in graph order. Verify that the arg0 subview offsets in
# forward.mlir advance exactly along the concatenation prefix, so that the
# written arg0.data aligns with the subview layout the compiled kernel uses.
import re  # noqa: E402

fm = os.path.join(a.output_dir, "forward.mlir")
text = open(fm).read()
m = re.search(
    r"func\.func @forward\([^)]*%arg0: memref<(\d+)xf32>", text
)
abi_size = int(m.group(1)) if m else None
subviews = [
    (int(off), int(sz))
    for off, sz in re.findall(r"memref\.subview %arg0\[(\d+)\] \[(\d+)\]", text)
]
subviews.sort()
ok = True
cursor = 0
used = 0
for pp in params_f32:
    n = pp.numel()
    if used < len(subviews) and subviews[used][0] == cursor:
        exp, sz = subviews[used]
        if sz != n:
            ok = False
            print(
                f"[import-paligemma] LAYOUT MISMATCH at param {used}: "
                f"subview size {sz} != param size {n}",
                file=os.sys.stderr,
            )
        used += 1
    elif used < len(subviews):
        ok = False
        print(
            f"[import-paligemma] LAYOUT MISMATCH: subview {subviews[used][0]} "
            f"!= cumulative {cursor}",
            file=os.sys.stderr,
        )
    cursor += n
# After all params, every subview must be accounted for.
if used != len(subviews):
    ok = False
    print(
        f"[import-paligemma] LAYOUT MISMATCH: {len(subviews)} subviews, "
        f"only {used} matched",
        file=os.sys.stderr,
    )
print(
    f"[import-paligemma] forward ABI arg0 memref = {abi_size} elements; "
    f"arg0.data = {all_param.size} elements; subview layout "
    f"{'OK' if ok else 'MISMATCHED'} ({used}/{len(subviews)} subviews matched)"
)
if not ok:
    raise SystemExit(
        "[import-paligemma] arg0.data layout does not match forward.mlir "
        "subviews; refusing to write inconsistent artifacts."
    )
if abi_size is not None and abi_size > all_param.size:
    raise SystemExit(
        f"[import-paligemma] forward ABI needs {abi_size} elements but "
        f"arg0.data has only {all_param.size}."
    )
