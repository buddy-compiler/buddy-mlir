# RUN: %PYTHON %s
# ===- test_import_mobilebert.py --------------------------------------------

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
from transformers import MobileBertConfig, MobileBertModel

script_dir = Path(__file__).parent
build_dir = os.environ.get("BUDDY_MLIR_BUILD_DIR")
if build_dir:
    output_dir = Path(build_dir) / "tests/Models/BuddyLLMGraphImport/mobilebert"
else:
    repo_root = script_dir.parent.parent.parent
    output_dir = repo_root / "build/tests/Models/BuddyLLMGraphImport/mobilebert"
output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve model path from environment variable
model_path = os.environ.get("MOBILEBERT_MODEL_PATH")
if model_path is None:
    model_path = "google/mobilebert-uncased"

print(f"Loading MobileBERT model config from: {model_path}")

config = MobileBertConfig(
    vocab_size=128,
    max_position_embeddings=64,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
    intra_bottleneck_size=64,
    use_bottleneck=True,
)
model = MobileBertModel(config).eval()
input_ids = torch.zeros((1, 16), dtype=torch.int64)
attention_mask = torch.ones((1, 16), dtype=torch.int64)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    capture_scalar_outputs=True,
)

with torch.no_grad():
    graphs = dynamo_compiler.importer(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

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

print("✓ MobileBERT graph construction test PASSED")
