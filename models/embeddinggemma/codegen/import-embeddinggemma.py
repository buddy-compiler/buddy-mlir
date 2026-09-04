#!/usr/bin/env python3
# ===- import-embeddinggemma.py - embeddinggemma AOT importer ------------===//
# Adapted from the original PR import_model.py (buddy-mlir pipeline) to the
# buddy-codegen single_forward interface: `--spec` + `--output-dir`, with
# subgraph0.mlir / forward.mlir / arg0.data written at the output-dir ROOT.
#
# Architecture (unchanged from the original PR):
#   Gemma3TextModel -> Mean Pooling -> Dense(768->3072) ->
#   Dense(3072->768) -> L2 Normalize -> 768-dim embedding
#
# Model source resolution (in order):
#   $EMBEDDINGGEMMA_MODEL_PATH  (set by buddy_add_model LOCAL_MODEL_ENV)
#   $BUDDY_LOCAL_MODEL_PATH     (generic fallback)
#   spec["hf_model_path"]       (e.g. google/embeddinggemma-300m)
#
# ===----------------------------------------------------------------------===//
import argparse
import os
import json
import numpy
import torch
import torch.nn as nn
import torch._dynamo
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from sentence_transformers import SentenceTransformer

torch._dynamo.config.suppress_errors = True


# ==============================================================================
# Wrapper module for Dynamo tracing (same loading/trace logic as the original
# PR's import_model.py). Takes (input_ids, attention_mask) and returns the
# 768-dim L2-normalized embedding.
# ==============================================================================
class EmbeddingGemmaWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        st_model = SentenceTransformer(model_path, device="cpu")
        st_model.eval()

        # Extract sub-modules
        self.transformer = st_model._first_module()
        self.pooling = st_model._modules["1"]
        self.dense1 = st_model._modules["2"]
        self.dense2 = st_model._modules["3"]
        self.normalize = st_model._modules["4"]

        # Force float32
        for p in self.parameters():
            p.data = p.data.float()

        # Single forward pass only: disable KV cache so Dynamo does not
        # materialize past-key/value tensors in the compiled graph.
        try:
            auto_model = getattr(self.transformer, "auto_model", None)
            if auto_model is not None:
                auto_model.config.use_cache = False
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # 1. Transformer (Gemma3TextModel)
        trans_out = self.transformer(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        token_emb = trans_out["token_embeddings"]

        # 2. Mean Pooling (traceable: all tensor ops, no control flow)
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_emb / sum_mask

        # 3. Dense 768 -> 3072 (Linear + Identity activation)
        pooled = {"sentence_embedding": pooled}
        d1_out = self.dense1(pooled)["sentence_embedding"]

        # 4. Dense 3072 -> 768 (Linear + Identity activation)
        d2_out = self.dense2({"sentence_embedding": d1_out})["sentence_embedding"]

        # 5. L2 Normalize
        normed = self.normalize({"sentence_embedding": d2_out})["sentence_embedding"]

        return normed


# ==============================================================================
# 1. Argument parsing
# ==============================================================================
p = argparse.ArgumentParser(description="embeddinggemma-300m AOT importer")
p.add_argument("--spec", required=True, help="Variant spec JSON")
p.add_argument("--output-dir", required=True, help="Directory to save outputs")
a = p.parse_args()

with open(a.spec) as f:
    spec = json.load(f)

model_path = (
    os.environ.get("EMBEDDINGGEMMA_MODEL_PATH")
    or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
    or spec.get("hf_model_path", "google/embeddinggemma-300m")
)
max_seq_len = int(spec.get("max_seq_len", 128))
os.makedirs(a.output_dir, exist_ok=True)

# ==============================================================================
# 2. Load model
# ==============================================================================
print("[import-embeddinggemma] Loading embeddinggemma-300m from:", model_path)
model = EmbeddingGemmaWrapper(model_path)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

# ==============================================================================
# 3. Initialize Dynamo Compiler
# ==============================================================================
dc = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

# ==============================================================================
# 4. Dummy inputs (fixed shape; matches the original PR)
# ==============================================================================
dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.int64)
dummy_attention_mask = torch.ones((1, max_seq_len), dtype=torch.int64)
print(f"[import-embeddinggemma] Dummy inputs:")
print(f"   input_ids:      {tuple(dummy_input_ids.shape)}")
print(f"   attention_mask: {tuple(dummy_attention_mask.shape)}")

# ==============================================================================
# 5. Trace the model
# ==============================================================================
print("\n[import-embeddinggemma] Tracing forward graph...")
with torch.no_grad():
    graphs = dc.importer(
        model,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
    )

assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
graph = graphs[0]
params = dc.imported_params[graph]
print(f"[import-embeddinggemma] {len(graphs)} graph(s), "
      f"{len(params)} params, "
      f"{sum(p.numel() for p in params):,} elems")

# ==============================================================================
# 6. Graph optimization (simply_fuse ONLY)
# ==============================================================================
graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()


def _repair_subgraph0(module_text):
    """Repair two classes of invalid ops the buddy-codegen TOSA lowering emits
    for Gemma3's float attention mask.

    1) `arith.select` with an f32 condition (mask elements used directly as a
       boolean) is not valid MLIR. Rewrite it to `arith.cmpf ne %cond, 0.0`
       followed by an i1-conditioned select (nonzero -> keep).
    2) `tosa.bitwise_or` / `tosa.bitwise_and` on f32 tensors is invalid TOSA
       (bitwise ops require integer types). The operands are always 0.0/1.0
       masks (they come from `tosa.cast` of i1 masks plus 0.0/1.0 constants),
       so OR is elementwise max and AND is elementwise min. Replace with
       `tosa.maximum` / `tosa.minimum`, which are valid f32 TOSA ops with the
       identical type signature and the same result on {0, 1} masks.
    """
    import re

    sel_pat = re.compile(
        r'^(\s*)%(\w+) = "arith\.select"\((%\w+), (%\w+), (%\w+)\) : '
        r'\(f32, f32, f32\) -> f32$',
        re.M)

    def _sel_repl(mo):
        ind, res, cond, t, f = (mo.group(1), mo.group(2), mo.group(3),
                                mo.group(4), mo.group(5))
        return "\n".join([
            f'{ind}%selz_{res} = "arith.constant"() '
            f'<{{value = 0.000000e+00 : f32}}> : () -> f32',
            f'{ind}%selc_{res} = "arith.cmpf"({cond}, %selz_{res}) '
            f'<{{fastmath = #arith.fastmath<none>, predicate = 13 : i64}}> : '
            f'(f32, f32) -> i1',
            f'{ind}%{res} = "arith.select"(%selc_{res}, {t}, {f}) : '
            f'(i1, f32, f32) -> f32',
        ])

    module_text = sel_pat.sub(_sel_repl, module_text)

    f32ty = r'(?:tensor<[^>]*f32>|f32)'
    _bit_pat = re.compile(
        r'"(tosa\.bitwise_or|tosa\.bitwise_and)"\(([^)]*)\) : \('
        + r'(' + f32ty + r'), (' + f32ty + r')\) -> (' + f32ty + r')')

    def _bit_repl(mo):
        op = mo.group(1)
        repl = "tosa.maximum" if op == "tosa.bitwise_or" else "tosa.minimum"
        return f'"{repl}"({mo.group(2)}) : ({mo.group(3)}, {mo.group(4)}) -> ' \
            f'{mo.group(5)}'

    module_text = _bit_pat.sub(_bit_repl, module_text)
    return module_text


# ==============================================================================
# 7. Save outputs at the output-dir ROOT (single_forward build convention)
# ==============================================================================
with open(os.path.join(a.output_dir, "subgraph0.mlir"), "w") as module_file:
    print(_repair_subgraph0(str(driver.subgraphs[0]._imported_module)),
          file=module_file)
with open(os.path.join(a.output_dir, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

all_param = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(a.output_dir, "arg0.data"))

print(f"[import-embeddinggemma] Wrote forward.mlir, subgraph0.mlir, "
      f"arg0.data ({all_param.size} f32 elems) to {a.output_dir}")
