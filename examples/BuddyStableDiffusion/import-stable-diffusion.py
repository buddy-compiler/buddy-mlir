import os

from pathlib import Path
import numpy
import torch
import torchvision.models as models
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from diffusers import StableDiffusionPipeline

device = torch.device("cuda")
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
tokenizer = pipe.tokenizer

pipe.text_encoder.eval()
pipe.unet.eval()
pipe.vae.eval()
text_encoder = pipe.text_encoder.forward
unet = pipe.unet.forward
vae = pipe.vae.decode


# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler_text_encoder = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_text_encoder",
)

dynamo_compiler_unet = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_unet",
)

dynamo_compiler_vae = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_vae",
)

data_text_encoder = tokenizer(
    prompt, return_tensors="pt", padding="max_length"
).to(device)
data_unet = {
    "sample": torch.ones((2, 4, 64, 64), dtype=torch.float32).to(device),
    "timestep": torch.tensor([1], dtype=torch.float32).to(device),
    "encoder_hidden_states": torch.ones((2, 77, 1024), dtype=torch.float32).to(
        device
    ),
}
data_vae = torch.ones((1, 4, 64, 64), dtype=torch.float32).to(device)


# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs_text_encoder = dynamo_compiler_text_encoder.importer(
        text_encoder, data_text_encoder["input_ids"].to(device), None
    )
    graphs_unet = dynamo_compiler_unet.importer(unet, **data_unet)
    graphs_vae = dynamo_compiler_vae.importer(vae, data_vae)


assert len(graphs_text_encoder) == 1
assert len(graphs_unet) == 1
assert len(graphs_vae) == 1

graph_text_encoder = graphs_text_encoder[0]
graph_unet = graphs_unet[0]
graph_vae = graphs_vae[0]

params_text_encoder = dynamo_compiler_text_encoder.imported_params[
    graph_text_encoder
]
params_unet = dynamo_compiler_unet.imported_params[graph_unet]
params_vae = dynamo_compiler_vae.imported_params[graph_vae]

pattern_list = [simply_fuse]

graphs_text_encoder[0].fuse_ops(pattern_list)
graphs_unet[0].fuse_ops(pattern_list)
graphs_vae[0].fuse_ops(pattern_list)

driver_text_encoder = GraphDriver(graphs_text_encoder[0])
driver_unet = GraphDriver(graphs_unet[0])
driver_vae = GraphDriver(graphs_vae[0])

driver_text_encoder._subgraphs[
    "subgraph0_text_encoder"
] = driver_text_encoder._subgraphs.pop("subgraph0")
driver_text_encoder._subgraphs_inputs[
    "subgraph0_text_encoder"
] = driver_text_encoder._subgraphs_inputs.pop("subgraph0")
driver_text_encoder._subgraphs_outputs[
    "subgraph0_text_encoder"
] = driver_text_encoder._subgraphs_outputs.pop("subgraph0")
driver_unet._subgraphs["subgraph0_unet"] = driver_unet._subgraphs.pop(
    "subgraph0"
)
driver_unet._subgraphs_inputs[
    "subgraph0_unet"
] = driver_unet._subgraphs_inputs.pop("subgraph0")
driver_unet._subgraphs_outputs[
    "subgraph0_unet"
] = driver_unet._subgraphs_outputs.pop("subgraph0")
driver_vae._subgraphs["subgraph0_vae"] = driver_vae._subgraphs.pop("subgraph0")
driver_vae._subgraphs_inputs[
    "subgraph0_vae"
] = driver_vae._subgraphs_inputs.pop("subgraph0")
driver_vae._subgraphs_outputs[
    "subgraph0_vae"
] = driver_vae._subgraphs_outputs.pop("subgraph0")

driver_text_encoder.subgraphs[0]._func_name = "subgraph0_text_encoder"
driver_unet.subgraphs[0]._func_name = "subgraph0_unet"
driver_vae.subgraphs[0]._func_name = "subgraph0_vae"

driver_text_encoder.subgraphs[0].lower_to_top_level_ir()
driver_unet.subgraphs[0].lower_to_top_level_ir()
driver_vae.subgraphs[0].lower_to_top_level_ir()

path_prefix = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(path_prefix, "subgraph0_text_encoder.mlir"), "w"
) as module_file:
    print(driver_text_encoder.subgraphs[0]._imported_module, file=module_file)
with open(
    os.path.join(path_prefix, "forward_text_encoder.mlir"), "w"
) as module_file:
    print(driver_text_encoder.construct_main_graph(True), file=module_file)

with open(os.path.join(path_prefix, "subgraph0_unet.mlir"), "w") as module_file:
    print(driver_unet.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward_unet.mlir"), "w") as module_file:
    print(driver_unet.construct_main_graph(True), file=module_file)

with open(os.path.join(path_prefix, "subgraph0_vae.mlir"), "w") as module_file:
    print(driver_vae.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward_vae.mlir"), "w") as module_file:
    print(driver_vae.construct_main_graph(True), file=module_file)

float32_param_text_encoder = numpy.concatenate(
    [
        param.detach().cpu().numpy().reshape([-1])
        for param in params_text_encoder[:-1]
    ]
)
float32_param_text_encoder.tofile(
    os.path.join(path_prefix, "arg0_text_encoder.data")
)

int64_param_text_encoder = (
    params_text_encoder[-1].detach().cpu().numpy().reshape([-1])
)
int64_param_text_encoder.tofile(
    os.path.join(path_prefix, "arg1_text_encoder.data")
)

param_unet = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params_unet]
)
param_unet.tofile(os.path.join(path_prefix, "arg0_unet.data"))

param_vae = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params_vae]
)
param_vae.tofile(os.path.join(path_prefix, "arg0_vae.data"))
