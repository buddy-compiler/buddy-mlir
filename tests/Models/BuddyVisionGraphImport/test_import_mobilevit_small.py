import os
import argparse
from pathlib import Path

# Import buddy/MLIR first to avoid LLVM option conflicts
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
)
from buddy.compiler.ops import tosa

import torch
from transformers import AutoConfig, AutoModel
from torch._inductor.decomposition import decompositions as inductor_decomp

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Mobilevit-Small graph coverage test"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to save output MLIR files (default: build/tests/Models/BuddyVisionGraphImport/mobilevit-small)",
)
args = parser.parse_args()

# Determine output directory
if args.output_dir is None:
    script_dir = Path(__file__).parent
    build_dir = os.environ.get("BUDDY_MLIR_BUILD_DIR")
    if build_dir:
        output_dir = (
            Path(build_dir)
            / "tests/Models/BuddyVisionGraphImport/mobilevit-small"
        )
    else:
        repo_root = script_dir.parent.parent.parent
        output_dir = (
            repo_root
            / "build/tests/Models/BuddyVisionGraphImport/mobilevit-small"
        )
else:
    output_dir = Path(args.output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve model path from environment variable
model_path = os.environ.get("MOBILEVIT_SMALL_MODEL_PATH")
if model_path is None:
    model_path = "apple/mobilevit-small"

print(f"Loading mobilevit-small model from: {model_path}")

config = AutoConfig.from_pretrained(model_path)

print("Model config loaded:")
print(f"  Model type: {config.model_type}")
print(f"  Image size: {config.image_size}")
print(f"  Num channels: {config.num_channels}")
print(f"  Hidden sizes: {config.hidden_sizes}")
print(f"  Num attention heads: {config.num_attention_heads}")

# Create model from config (random weights)
model = AutoModel.from_config(config).eval()

# Create dummy image input
pixel_values = torch.randn(1, 3, 256, 256, dtype=torch.float32)

# Initialize compiler
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    # verbose=True,
)

print("DynamoCompiler initialized")

with torch.no_grad():
    print("Importing model graph...")
    graphs = dynamo_compiler.importer(
        model,
        pixel_values=pixel_values,
    )

print(f"Graph import completed: {len(graphs)} graph(s) generated")

# 1. Verify no graph break
assert (
    len(graphs) == 1
), f"Expected 1 graph (no graph break), but got {len(graphs)}"

print("✓ No graph break detected")

graph = graphs[0]

# Apply transformations and lower to MLIR
print("Applying graph transformations...")
graph.perform([eliminate_transpose])
graph.fuse_ops([simply_fuse, apply_classic_fusion])

print("Lowering to MLIR...")
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()

print("MLIR generation completed")

# 2. Verify MLIR structure
mlir_str = str(driver.subgraphs[0]._imported_module)
assert "func.func @subgraph0" in mlir_str, "Missing func.func @subgraph0"
assert "return" in mlir_str, "Missing return in generated MLIR"

print("✓ MLIR structure verified")

# 3. Save MLIR output files
subgraph_path = output_dir / "subgraph0.mlir"
with open(subgraph_path, "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)
print(f"  Saved subgraph MLIR to: {subgraph_path}")

forward_path = output_dir / "forward.mlir"
with open(forward_path, "w") as f:
    print(driver.construct_main_graph(True), file=f)
print(f"  Saved forward MLIR to: {forward_path}")

print("✓ Clip-ViT-Base graph construction test PASSED")
