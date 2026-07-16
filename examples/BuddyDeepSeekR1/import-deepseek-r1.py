#!/usr/bin/env python3
# ===- import-deepseek-r1.py ---------------------------------------------------
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
# This is the test of DeepSeekR1 model.
#
# ===---------------------------------------------------------------------------

import argparse
import os

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import *  # noqa: F403
from buddy.compiler.graph.transform import (
    apply_classic_fusion,
    eliminate_matmul_transpose_reshape,
    eliminate_transpose,
    flash_attention_prefill,
    gqa_attention_fusion,
    pack_decode_matmul_weights,
    simply_fuse,
)
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import (
    AutoModelForCausalLM,
    StaticCache,
)

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="DeepSeekR1 Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--precision",
    type=str,
    default="f32",
    choices=["f32", "f16", "bf16"],
    help="Precision mode for MLIR/input data. Choose from %(choices)s.",
)
parser.add_argument(
    "--pack-decode-weights",
    action="store_true",
    help="Panel-pack every decode matmul weight (q/k/v/o_proj, "
    "gate/up/down_proj, lm_head) and write decode's parameters to their own "
    "file, arg0-decode.data, instead of sharing prefill's arg0.data. Decode "
    "is a GEMV (m == 1) and reads each weight byte once, so it is bound by "
    "how the weights are laid out; prefill is compute-bound and wants them "
    "plain, so the two phases genuinely need different layouts and therefore "
    "different buffers. Compiling decode then requires "
    "-matmul-vectorization-decode-packed with a matching --pack-vector-size; "
    "the plain decode kernel would read the packed bytes as row-major and "
    "silently produce a wrong model. f32 only for now.",
)
parser.add_argument(
    "--pack-vector-size",
    type=int,
    default=32,
    help="Panel width used by --pack-decode-weights. Must match the "
    "vector-size passed to -matmul-vectorization-decode-packed.",
)
args = parser.parse_args()


# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Initialize the model from the specified model path.
if args.precision == "f16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
        .eval()
        .half()
    )
elif args.precision == "bf16":
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        .eval()
        .bfloat16()
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).eval()
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
prefill_func_name = "forward_prefill"
dynamo_compiler_prefill = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name=prefill_func_name,
)

dynamo_compiler_decode = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_decode",
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    if args.precision == "f16":
        past_key_values_prefill = StaticCache(
            config=model.config, max_cache_len=1024
        )
        past_key_values_decode = StaticCache(
            config=model.config, max_cache_len=1024
        )

        data_prefill = {
            "input_ids": torch.zeros((1, 1024), dtype=torch.int64),
        }
        data_decode = {
            "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        }

        cache_position = torch.tensor([200], dtype=torch.int64)

        graphs_prefill = dynamo_compiler_prefill.importer(
            model,
            input_ids=data_prefill["input_ids"],
            use_cache=True,
            # past_key_values=past_key_values_prefill,
            cache_implementation="static",
        )
        # Initialize past_key_values once during the first forward call
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
    else:
        past_key_values_prefill = StaticCache(
            config=model.config, max_cache_len=1024
        )
        past_key_values_decode = StaticCache(
            config=model.config, max_cache_len=1024
        )

        data_prefill = {
            "input_ids": torch.zeros((1, 1024), dtype=torch.int64),
        }
        data_decode = {
            "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        }

        cache_position = torch.tensor([200], dtype=torch.int64)

        graphs_prefill = dynamo_compiler_prefill.importer(
            model,
            input_ids=data_prefill["input_ids"],
            use_cache=True,
            # past_key_values=past_key_values_prefill,
            cache_implementation="static",
        )
        # Initialize past_key_values once during the first forward call
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

if args.precision == "f16":
    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    graph_prefill = graphs_prefill[0]
    graph_decode = graphs_decode[0]

    params = dynamo_compiler_prefill.imported_params[graph_prefill]
    # Enable verbose mode for debugging eliminate_matmul_transpose_reshape
    graphs_prefill[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    graphs_decode[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    pattern_list_prefill = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_list_decode = [
        simply_fuse,
        apply_classic_fusion,
        gqa_attention_fusion,
    ]

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
else:
    assert len(graphs_prefill) == 1
    assert len(graphs_decode) == 1
    graph_prefill = graphs_prefill[0]
    graph_decode = graphs_decode[0]

    params = dynamo_compiler_prefill.imported_params[graph_prefill]
    params_decode = dynamo_compiler_decode.imported_params[graph_decode]
    # Enable verbose mode for debugging eliminate_matmul_transpose_reshape
    graphs_prefill[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )
    graphs_decode[0].perform(
        [eliminate_transpose, eliminate_matmul_transpose_reshape]
    )

    if args.pack_decode_weights:
        # Decode only, and after eliminate_transpose so the weights are
        # already [K, N]. Prefill's copies stay plain -- see
        # pack_decode_matmul_weights for what makes that hold.
        packed = pack_decode_matmul_weights(
            graph_decode, vecsize=args.pack_vector_size
        )
        print(
            f"[import-deepseek-r1] packed {len(packed)} decode matmul weights "
            f"into panel layout (vector-size={args.pack_vector_size}); "
            "prefill's weights stay plain."
        )

    pattern_list_prefill = [
        simply_fuse,
        apply_classic_fusion,
        flash_attention_prefill,
    ]
    pattern_list_decode = [
        simply_fuse,
        apply_classic_fusion,
        gqa_attention_fusion,
    ]

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

# Save the generated files to the specified output directory.
if args.precision == "f16":
    with open(
        os.path.join(output_dir, "subgraph0_prefill-f16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_prefill-f16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.construct_main_graph(True), file=module_file)
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0-f16.data"))

    with open(
        os.path.join(output_dir, "subgraph0_decode-f16.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode-f16.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
elif args.precision == "bf16":
    with open(
        os.path.join(output_dir, "subgraph0_prefill-bf16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_prefill-bf16.mlir"), "w"
    ) as module_file:
        print(driver_prefill.construct_main_graph(True), file=module_file)
    # Convert BF16 parameters to float32 first, then to numpy
    all_param = numpy.concatenate(
        [param.detach().float().numpy().reshape([-1]) for param in params]
    )
    # Convert float32 to BF16 format (uint16) for storage
    all_param_bf16 = numpy.frombuffer(
        all_param.astype(numpy.float32).tobytes(), dtype=numpy.uint16
    )[1::2]
    all_param_bf16.tofile(os.path.join(output_dir, "arg0-bf16.data"))

    with open(
        os.path.join(output_dir, "subgraph0_decode-bf16.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode-bf16.mlir"), "w"
    ) as module_file:
        print(driver_decode.construct_main_graph(True), file=module_file)
else:
    with open(
        os.path.join(output_dir, "subgraph0_prefill.mlir"), "w"
    ) as module_file:
        print(driver_prefill.subgraphs[0]._imported_module, file=module_file)
    forward_prefill_text = str(driver_prefill.construct_main_graph(True))

    with open(
        os.path.join(output_dir, "subgraph0_decode.mlir"), "w"
    ) as module_file:
        print(driver_decode.subgraphs[0]._imported_module, file=module_file)
    forward_decode_text = str(driver_decode.construct_main_graph(True))

    with open(
        os.path.join(output_dir, "forward_prefill.mlir"), "w"
    ) as module_file:
        print(forward_prefill_text, file=module_file)
    with open(
        os.path.join(output_dir, "forward_decode.mlir"), "w"
    ) as module_file:
        print(forward_decode_text, file=module_file)

    # arg0.data: prefill's parameters, always plain row-major.
    all_param = numpy.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    all_param.tofile(os.path.join(output_dir, "arg0.data"))

    if args.pack_decode_weights:
        # The same parameters, with every matmul weight in panel layout.
        # Identical size and offsets to arg0.data, so the two files are drop-in
        # interchangeable -- which one a phase is handed is the whole of the
        # difference, and decode's MLIR needs no rewriting at all.
        all_param_decode = numpy.concatenate(
            [param.detach().numpy().reshape([-1]) for param in params_decode]
        )
        if all_param_decode.size != all_param.size:
            raise RuntimeError(
                "[import-deepseek-r1] decode params have "
                f"{all_param_decode.size} elements but prefill params have "
                f"{all_param.size}; packing must not change the layout."
            )
        all_param_decode.tofile(os.path.join(output_dir, "arg0-decode.data"))
