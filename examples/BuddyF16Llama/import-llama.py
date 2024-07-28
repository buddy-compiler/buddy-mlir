import os
import torch
import argparse
import torch._dynamo as dynamo
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse



def load_model_llama2(model_path, dtype) -> tuple[LlamaTokenizer, LlamaForCausalLM]:
    print(f'Loading model as data type {dtype}')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if dtype == 'fp32':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float32)
    elif dtype == 'fp16':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float16)
    elif dtype == 'bf16':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    return tokenizer, model

def load_model_tinyllama(model_path, dtype) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f'Loading model as data type {dtype}')
    checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if dtype == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float32, device_map="auto")
    elif dtype == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float16, device_map="auto")
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    return tokenizer, model

def import_model(model, mtype, dtype):
    # Initialize Dynamo Compiler with specific configurations as an importer.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import the model into MLIR module and parameters.
    with torch.no_grad():
        data = torch.tensor([[i for i in range(40)]], dtype=torch.int64)
        expected_output = model(data)[0].reshape(-1).float().cpu().numpy()
        graphs = dynamo_compiler.importer(model, data)

    assert len(graphs) == 1, f'{len(graphs)} graphs imported'
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()

    all_op_names = set()
    for op in driver.subgraphs[0]._body:
        all_op_names.add(op.__class__.__name__)
    print(f'{len(all_op_names)} ops in total')
    for op_name in sorted(all_op_names):
        print(op_name)
    
    # persist
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_prefix, f"{mtype}-{dtype}-subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(path_prefix, f"{mtype}-{dtype}-forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    
    all_param = numpy.concatenate(
        [param.detach().float().numpy().reshape([-1]) for param in params]
    )
    print(f'{mtype}-{dtype}: parameter:', all_param[:100])
    all_param.tofile(os.path.join(path_prefix, f"params.data"))
    expected_output.tofile(os.path.join(path_prefix, f"{mtype}-{dtype}-output.data"))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtype', type=str, required=True, choices=['7b', '1.1b'])
    parser.add_argument('--dtype', type=str, required=True, choices=['fp32', 'fp16', 'bf16'])

    args = parser.parse_args()
    mtype = args.mtype
    dtype = args.dtype

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is None:
        raise EnvironmentError(
            "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
        )
    if mtype == '7b':
        tokenizer, model = load_model_llama2(model_path, dtype)
    elif mtype == '1.1b':
        tokenizer, model = load_model_tinyllama(model_path, dtype)
    import_model(model, mtype, dtype)