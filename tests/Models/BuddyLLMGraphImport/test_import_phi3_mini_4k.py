# RUN: %PYTHON %s
# ===- test_import_phi3_mini_4k_static.py ------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the graph coverage test for Phi-3-mini-4k-instruct model.
#
# ===---------------------------------------------------------------------------

import argparse
import os
from pathlib import Path

# Then import torch and transformers
import torch

# Import buddy/MLIR first to avoid LLVM option conflicts
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_transpose,
    simply_fuse,
)
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Phi-3-mini-4k graph coverage test"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="/build/tests/Models/BuddyLLMGraphImport/phi3_mini_4k",
    help="Directory to save output files (default: %(default)s)",
)
args = parser.parse_args()

# Determine output directory
if args.output_dir is None:
    script_dir = Path(__file__).parent
    build_dir = os.environ.get("BUDDY_MLIR_BUILD_DIR")
    if build_dir:
        output_dir = (
            Path(build_dir) / "tests/Models/BuddyLLMGraphImport/phi3_mini_4k"
        )
    else:
        repo_root = script_dir.parent.parent.parent
        output_dir = (
            repo_root / "build/tests/Models/BuddyLLMGraphImport/phi3_mini_4k"
        )
else:
    output_dir = Path(args.output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

# Retrieve model path from environment variable
model_path = os.environ.get("PHI3_MINI_4K_MODEL_PATH")
if model_path is None:
    model_path = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading Phi-3-mini-4k model config from: {model_path}")

MaxTokenLength = 128
input_seq_len = 32

config = AutoConfig.from_pretrained(model_path)
config.use_cache = False
print(f"Model config loaded: {config.num_hidden_layers} layers")

model = AutoModelForCausalLM.from_config(config).eval()
print("Model created with random weights")

# Pre-initialize StaticCache to avoid Dynamo graph breaks.
print("Pre-initializing StaticCache...")

cache = StaticCache(
    config=config,
    max_cache_len=MaxTokenLength,
    batch_size=1,
)

# 进行一次 dummy 前向，触发缓存内部所有懒加载分配
with torch.no_grad():
    dummy_ids = torch.zeros((1, 1), dtype=torch.int64)
    model(
        input_ids=dummy_ids,
        past_key_values=cache,
        use_cache=True,
        cache_implementation="static",
    )

cache.reset()

# Init DynamoCompiler
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
print("DynamoCompiler initialized")

with torch.no_grad():
    input_ids = torch.zeros((1, input_seq_len), dtype=torch.int64)
    print("Importing model graph (use_cache=True, StaticCache)...")
    graphs = dynamo_compiler.importer(
        model,
        input_ids=input_ids,
        use_cache=True,
        cache_implementation="static",
        past_key_values=cache,
    )

print(f"Graph import completed: {len(graphs)} graph(s) generated")

# 1. Verify no graph break
assert len(graphs) == 1, (
    f"Expected 1 graph (no graph break), but got {len(graphs)}"
)
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

print("✓ Phi-3-mini-4k-instruct fully static graph construction test PASSED")
