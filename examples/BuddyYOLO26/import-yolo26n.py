#!/usr/bin/env python3
# ===- import-yolo26n.py ------------------------------------------------------
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
# This is the yolo26n model AOT importer.
#
# ===---------------------------------------------------------------------------

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from ultralytics import YOLO

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa


parser = argparse.ArgumentParser(description="yolo26n model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=640,
    help="Input image size for model import.",
)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

default_model_path = Path(__file__).resolve().parents[2] / "yolo26n.pt"
model_path = os.environ.get(
    "YOLO26N_MODEL_PATH",
    str(default_model_path if default_model_path.exists() else "yolo26n.pt"),
)
model = YOLO(model_path).model.eval()
detect_head = model.model[-1]
detect_head.end2end = True
detect_head.export = True
detect_head.xyxy = True

input_tensor = torch.randn(
    (1, 3, args.img_size, args.img_size), dtype=torch.float32
)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward",
)

with torch.no_grad():
    # Warm up once to initialize Detect anchors/strides for the fixed input shape.
    model(input_tensor)
    graphs = dynamo_compiler.importer(model, input_tensor)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

np.concatenate(
    [
        param.detach().numpy().reshape([-1])
        for param in params
        if param.dtype == torch.float32
    ]
).tofile(output_dir / "arg0.data")
