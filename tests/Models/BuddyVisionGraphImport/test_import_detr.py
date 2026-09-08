# RUN: %PYTHON %s
# ===- test_import_detr.py ---------------------------------------------------

import os
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_transpose,
    simply_fuse,
)
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig

script_dir = Path(__file__).parent
build_dir = os.environ.get("BUDDY_MLIR_BUILD_DIR")
if build_dir:
    output_dir = Path(build_dir) / "tests/Models/BuddyVisionGraphImport/detr"
else:
    repo_root = script_dir.parent.parent.parent
    output_dir = repo_root / "build/tests/Models/BuddyVisionGraphImport/detr"
output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve model path from environment variable
model_path = os.environ.get("DETR_MODEL_PATH")
if model_path is None:
    model_path = "facebook/detr-resnet-50"

print(f"Loading DETR model config from: {model_path}")

backbone_config = ResNetConfig(
    num_channels=3,
    embedding_size=16,
    depths=[1, 1, 1, 1],
    hidden_sizes=[16, 32, 64, 128],
    layer_type="basic",
    out_features=["stage4"],
)
config = DetrConfig(
    num_channels=3,
    d_model=64,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    num_queries=8,
    num_labels=10,
    backbone_config=backbone_config,
)
model = DetrForObjectDetection(config).eval()
pixel_values = torch.zeros((1, 3, 64, 64), dtype=torch.float32)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    capture_scalar_outputs=True,
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, pixel_values=pixel_values)

print(f"Graph import completed: {len(graphs)} graph(s)")
assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
print("✓ No graph break detected")

graph = graphs[0]
graph.perform([eliminate_transpose])
graph.fuse_ops([simply_fuse, apply_classic_fusion])

driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

mlir_str = str(driver.subgraphs[0]._imported_module)
assert "func.func" in mlir_str, "Missing func.func in generated MLIR"
assert "return" in mlir_str, "Missing return in generated MLIR"

with open(output_dir / "subgraph0.mlir", "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)
with open(output_dir / "forward.mlir", "w") as f:
    print(driver.construct_main_graph(True), file=f)

print("✓ DETR graph construction test PASSED")
