#!/usr/bin/env python3
# ===- import-deepseek-r1-w8a32.py ---------------------------------------------
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
# This is the W8A32 weight-only int8 quantization (f32 activations) version of the
# DeepSeekR1 model importer. It applies graph-level quantization to convert
# MatmulOp weights to int8 with per-channel scaling factors, then generates
# MLIR files and parameter data for the quantized model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM, StaticCache
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import (
    simply_fuse,
    apply_classic_fusion,
    eliminate_transpose,
    eliminate_matmul_transpose_reshape,
    flash_attention_prefill,
    gqa_attention_fusion,
)
from buddy.compiler.graph.type import DeviceType, TensorDType
from buddy.compiler.graph.operation import *
from buddy.compiler.graph.transform.quantization import weight_only_channel_wise

parser = argparse.ArgumentParser(description="W8A32 AOT Importer")
parser.add_argument("--output-dir", type=str, default="./")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torchscript=True
).eval()
model.config.use_cache = False

dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_prefill",
)
dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

with torch.no_grad():
    past_key_values_prefill = StaticCache(
        config=model.config, max_cache_len=1024
    )
    past_key_values_decode = StaticCache(
        config=model.config, max_cache_len=1024
    )

    data_prefill = {"input_ids": torch.zeros((1, 1024), dtype=torch.int64)}
    data_decode = {"input_ids": torch.zeros((1, 1), dtype=torch.int64)}
    cache_position = torch.tensor([200], dtype=torch.int64)

    graphs_prefill = dynamo_compiler_prefill.importer(
        model,
        input_ids=data_prefill["input_ids"],
        use_cache=True,
        cache_implementation="static",
    )

    model(
        input_ids=data_decode["input_ids"],
        past_key_values=past_key_values_decode,
        use_cache=True,
        cache_implementation="static",
    )

    graphs_decode = dynamo_compiler_decode.importer(
        model,
        input_ids=data_decode["input_ids"],
        use_cache=True,
        cache_position=cache_position,
        past_key_values=past_key_values_decode,
        cache_implementation="static",
    )

assert len(graphs_prefill) == 1
assert len(graphs_decode) == 1
graph_prefill = graphs_prefill[0]
graph_decode = graphs_decode[0]

original_params = dynamo_compiler_prefill.imported_params[graph_prefill]
num_original_params = len(original_params)

# Apply graph transformations
graphs_prefill[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)
graphs_decode[0].perform(
    [eliminate_transpose, eliminate_matmul_transpose_reshape]
)

# Apply weight-only int8 quantization at graph level
weight_only_channel_wise(graph_prefill)
weight_only_channel_wise(graph_decode)

# Compute actual int8 weights and scale values
all_param_tensors = []

for i, param_node in enumerate(graph_prefill.params):
    if i < num_original_params:
        original_tensor = original_params[i]
        if param_node.tensor_meta.get("dtype") == TensorDType.Int8:
            scaler_name = "scaler_" + param_node.name
            scaler_node = graph_prefill.node_table.get(scaler_name)
            assert (
                scaler_node is not None
            ), f"Missing scaler for {param_node.name}"

            scaler_shape = list(scaler_node.tensor_meta["shape"])
            quant_axis = next(i for i, s in enumerate(scaler_shape) if s != 1)

            reduce_dims = [
                d for d in range(original_tensor.dim()) if d != quant_axis
            ]
            if not reduce_dims:
                reduce_dims = [0]
            amax = original_tensor.abs().amax(dim=reduce_dims, keepdim=True)
            scale = (amax / 127.0).clamp(min=1e-10)
            weight_i8 = torch.clamp(
                torch.round(original_tensor / scale), -128, 127
            ).to(torch.int8)

            all_param_tensors.append(("i8", weight_i8, param_node.name))
        else:
            all_param_tensors.append(("f32", original_tensor, param_node.name))
    else:
        assert param_node.name.startswith(
            "scaler_"
        ), f"Expected scaler, got {param_node.name}"
        weight_name = param_node.name[len("scaler_") :]
        weight_idx = next(
            j
            for j, p in enumerate(graph_prefill.params[:num_original_params])
            if p.name == weight_name
        )
        original_tensor = original_params[weight_idx]
        scaler_shape = list(param_node.tensor_meta["shape"])
        quant_axis = next(i for i, s in enumerate(scaler_shape) if s != 1)

        reduce_dims = [
            d for d in range(original_tensor.dim()) if d != quant_axis
        ]
        if not reduce_dims:
            reduce_dims = [0]
        amax = original_tensor.abs().amax(dim=reduce_dims, keepdim=True)
        scale = (amax / 127.0).clamp(min=1e-10)

        all_param_tensors.append(("f32", scale.float(), param_node.name))

# Serialize params by dtype (matching _pack_params sort order:
# str(TensorDType.Float32) < str(TensorDType.Int8))
f32_params_data = []
i8_params_data = []
for dtype_tag, tensor, name in all_param_tensors:
    if dtype_tag == "f32":
        f32_params_data.append(tensor.detach().float().numpy().reshape([-1]))
    else:
        i8_params_data.append(tensor.detach().numpy().reshape([-1]))

# Op grouping and fusion
pattern_list_prefill = [
    simply_fuse,
    apply_classic_fusion,
    flash_attention_prefill,
]
pattern_list_decode = [simply_fuse, apply_classic_fusion, gqa_attention_fusion]

graphs_prefill[0].fuse_ops(pattern_list_prefill)
graphs_decode[0].fuse_ops(pattern_list_decode)

graph_prefill.op_groups["subgraph0_prefill"] = graph_prefill.op_groups.pop(
    "subgraph0"
)
graph_prefill.group_map_device["subgraph0_prefill"] = DeviceType.CPU

graph_decode.op_groups["subgraph0_decode"] = graph_decode.op_groups.pop(
    "subgraph0"
)
graph_decode.group_map_device["subgraph0_decode"] = DeviceType.CPU

driver_prefill = GraphDriver(graphs_prefill[0])
driver_prefill.subgraphs[0].lower_to_top_level_ir()

driver_decode = GraphDriver(graphs_decode[0])
driver_decode.subgraphs[0].lower_to_top_level_ir()

# Save MLIR files
with open(os.path.join(output_dir, "subgraph0_prefill-w8a32.mlir"), "w") as f:
    print(driver_prefill.subgraphs[0]._imported_module, file=f)
with open(os.path.join(output_dir, "forward_prefill-w8a32.mlir"), "w") as f:
    print(driver_prefill.construct_main_graph(True), file=f)

with open(os.path.join(output_dir, "subgraph0_decode-w8a32.mlir"), "w") as f:
    print(driver_decode.subgraphs[0]._imported_module, file=f)
with open(os.path.join(output_dir, "forward_decode-w8a32.mlir"), "w") as f:
    print(driver_decode.construct_main_graph(True), file=f)

# Save param data files
if f32_params_data:
    all_f32 = numpy.concatenate(f32_params_data)
    all_f32.tofile(os.path.join(output_dir, "arg0-w8a32-f32.data"))

if i8_params_data:
    all_i8 = numpy.concatenate(i8_params_data)
    all_i8.tofile(os.path.join(output_dir, "arg0-w8a32-i8.data"))
