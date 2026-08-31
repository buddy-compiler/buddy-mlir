#!/usr/bin/env python3
# ===- import-kokoro.py - Kokoro-82M TTS AOT importer --------------------===//
#
# Adapted from the original PR models/kokoro/import_model.py to the buddy-codegen
# single_forward interface: `--spec` + `--output-dir`, with subgraph0.mlir /
# forward.mlir / arg0.data written to the output-dir ROOT (no layer_partitioned).
#
# Architecture: KModel — text-to-speech with:
#   - Albert-based phoneme encoder (12 layers, hidden=768, 12 heads)
#   - Duration predictor (LSTM + prosody)
#   - ISTFTNet vocoder (generator + discriminator)
#   - 81.8M params, 178-token phoneme vocabulary
#
# ===---------------------------------------------------------------------------

import argparse, json, os
import numpy, torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    simply_fuse, eliminate_transpose, eliminate_matmul_transpose_reshape,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from kokoro import KModel
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def _fix_degenerate_subgraph0(subgraph0_mlir):
    """Rewrite compile-time constant tensors in a degenerate (constant-output)
    subgraph into runtime `linalg.fill` values.

    `one-shot-bufferize` with `bufferize-function-boundaries` inserts a
    `bufferization.clone` for any function result derived purely from
    compile-time constants, and the shared subgraph0 pipeline
    (tools/buddy-codegen/cmake/buddy_model.cmake) does not lower that op, so
    `mlir-translate` fails with "Dialect `bufferization' not found". Filling an
    allocated tensor at runtime instead of returning the folded constant avoids
    the immutable constant memref and the clone. Only uniform (splat) constants
    are rewritten; anything else is left untouched.
    """
    import re

    def _rewrite_constant(m):
        name = m.group(1)
        value_txt = m.group(2)            # e.g. "30", "0.000000e+00"
        tensor_type = m.group(3)          # e.g. "1xi64", "i64", "8x280x280xf32"
        elt = tensor_type.split("x")[-1]
        vals = [v.strip() for v in value_txt.split(",") if v.strip()]
        if not vals or len(set(vals)) != 1:
            return m.group(0)             # non-uniform constant, leave as-is
        scalar = vals[0]
        # Scalar arith.constant for the fill value.
        if elt == "i1":
            scalar = "true" if scalar in ("true", "1") else "false"
        return (
            "    %%empty = tensor.empty() : tensor<%s>\n"
            "    %%c_fill = arith.constant %s : %s\n"
            "    %s = \"linalg.fill\"(%%c_fill, %%empty) "
            "<{operandSegmentSizes = array<i32: 1, 1>}> ({\n"
            "    ^bb0(%%a: %s, %%b: %s):\n"
            "      \"linalg.yield\"(%%a) : (%s) -> ()\n"
            "    }) : (%s, tensor<%s>) -> tensor<%s>" % (
                tensor_type, scalar, elt, name, elt, elt, elt, elt, tensor_type, tensor_type)
        )

    # Match `%x = arith.constant dense<V> : tensor<T>` lines (splat constants).
    return re.sub(
        r"(%[\w]+) = arith\.constant dense<([^>]*)> : tensor<([^>]*)>",
        _rewrite_constant, subgraph0_mlir)


parser = argparse.ArgumentParser(description="Kokoro-82M TTS Model AOT Importer")
parser.add_argument("--spec", type=str, required=True,
                    help="Variant spec JSON (e.g. models/kokoro/specs/f32.json)")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory for subgraph0.mlir / forward.mlir / arg0.data")
args = parser.parse_args()
output_dir = args.output_dir
with open(args.spec) as f:
    spec = json.load(f)
os.makedirs(output_dir, exist_ok=True)

# The model comes from the local HF snapshot staged through the build
# (BUDDY_KOKORO_MODEL_PATH / KOKORO_MODEL_PATH), falling back to the repo id.
model_path = (os.environ.get("KOKORO_MODEL_PATH")
              or os.environ.get("BUDDY_LOCAL_MODEL_PATH")
              or spec.get("hf_model_path", "hexgrad/Kokoro-82M"))

print(f"[Kokoro-Import] Loading Kokoro-82M TTS model from: {model_path}")
if os.path.isdir(model_path):
    # KModel accepts local checkpoint + config paths, avoiding the HF download.
    model = KModel(
        config=os.path.join(model_path, "config.json"),
        model=os.path.join(model_path, "kokoro-v1_0.pth"),
        disable_complex=True,
    ).to("cpu").eval()
else:
    model = KModel(repo_id=model_path, disable_complex=True).to("cpu").eval()
print(f"   params: {sum(p.numel() for p in model.parameters()):,}")

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

dummy_ids = torch.randint(0, 100, (1, 30), dtype=torch.int64)
dummy_ref = torch.randn(1, 256, dtype=torch.float32)

print(f"[Kokoro-Import] Tracing forward_with_tokens... input_ids={dummy_ids.shape}, ref_s={dummy_ref.shape}")

# The main forward does string→token conversion (untraceable).
# Trace forward_with_tokens which takes tensors directly.
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model.forward_with_tokens, input_ids=dummy_ids, ref_s=dummy_ref, speed=1.0,
    )

graph_count = len(graphs)
print(f"[Kokoro-Import] {graph_count} graph(s) captured")
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
print(f"[Kokoro-Import] {len(params)} params in first graph")

graph.perform([eliminate_transpose, eliminate_matmul_transpose_reshape])
graph.fuse_ops([simply_fuse])
graph.op_groups["subgraph0"] = graph.op_groups.pop("subgraph0")
graph.group_map_device["subgraph0"] = DeviceType.CPU

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

subgraph0_mlir = str(driver.subgraphs[0]._imported_module)
subgraph0_mlir = _fix_degenerate_subgraph0(subgraph0_mlir)
with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as f:
    print(subgraph0_mlir, file=f)
with open(os.path.join(output_dir, "forward.mlir"), "w") as f:
    print(driver.construct_main_graph(True), file=f)

# Export weights from all params (flattened f32, in model.parameters() order).
all_param = numpy.concatenate(
    [p.detach().cpu().numpy().reshape([-1]) for p in model.parameters()]
).astype(numpy.float32, copy=False)
all_param.tofile(os.path.join(output_dir, "arg0.data"))
print(f"[Kokoro-Import] Done! {len(list(model.parameters()))} parameter tensors, "
      f"{graph_count} graph(s), weights -> {os.path.join(output_dir, 'arg0.data')}")
