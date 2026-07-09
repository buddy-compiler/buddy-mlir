#!/usr/bin/env python3
# ===- import-bge-m3.py - BGE-M3 AOT importer -----------------------------===//
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
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys
import types

import numpy
import torch

if "tomli" not in sys.modules:
    tomli_stub = types.ModuleType("tomli")

    def _tomli_unavailable(*_args, **_kwargs):
        raise RuntimeError("tomli is required only when loading trace configs")

    tomli_stub.load = _tomli_unavailable
    tomli_stub.loads = _tomli_unavailable
    sys.modules["tomli"] = tomli_stub

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform.fuse_ops import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoModel


class BgeM3DenseWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def import_bge_m3(spec: dict, output_dir: str) -> None:
    model_path = (
        os.environ.get("BUDDY_LOCAL_MODEL_PATH")
        or os.environ.get("BGE_M3_MODEL_PATH")
        or spec["hf_model_path"]
    )
    max_seq_len = int(spec.get("max_seq_len", 512))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[import-bge-m3] Loading model from: {model_path}")
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model.config.use_cache = False
    wrapped = BgeM3DenseWrapper(model).eval()

    input_ids = torch.zeros((1, max_seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, max_seq_len), dtype=torch.long)

    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    with torch.no_grad():
        graphs = dynamo_compiler.importer(
            wrapped, input_ids=input_ids, attention_mask=attention_mask
        )

    if len(graphs) != 1:
        raise RuntimeError(f"expected one graph, got {len(graphs)}")

    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()

    with open(os.path.join(output_dir, "subgraph0.mlir"), "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(os.path.join(output_dir, "forward.mlir"), "w") as f:
        print(driver.construct_main_graph(True), file=f)

    all_param = numpy.concatenate(
        [param.detach().cpu().numpy().reshape([-1]) for param in params]
    ).astype(numpy.float32, copy=False)
    all_param.tofile(os.path.join(output_dir, "arg0.data"))
    print(
        "[import-bge-m3] Wrote forward.mlir, subgraph0.mlir, arg0.data "
        f"to {output_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BGE-M3 dense encoder importer"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    import_bge_m3(spec, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
