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


def load_model_llama(model_path, dtype) -> tuple[LlamaTokenizer, LlamaForCausalLM, dict[str, int]]:
    print(f'Loading model as data type {dtype}')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if dtype == 'fp32':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float32)
    elif dtype == 'fp16':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float16)
    elif dtype == 'bf16':
        model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    return model, tokenizer

def import_model(model, dtype):
    
    config = {}
    
    # Initialize Dynamo Compiler with specific configurations as an importer.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import the model into MLIR module and parameters.
    TOKEN_LENGTH = 40
    with torch.no_grad():
        data = torch.tensor([[i for i in range(TOKEN_LENGTH)]], dtype=torch.int64)
        expected_output = model(data)[0]
        print(model(data))
        graphs = dynamo_compiler.importer(model, data)
    
    assert len(graphs) == 1, f'{len(graphs)} graphs imported'
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()
    
    # persist
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_prefix, f"{dtype}-subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(path_prefix, f"{dtype}-forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    
    all_param = numpy.concatenate(
        [param.detach().float().numpy().reshape([-1]) for param in params]
    )
    print(f'parameter:', all_param[:100])
    all_param.tofile(os.path.join(path_prefix, "params.data"))
    expected_output.reshape(-1).float().cpu().numpy().tofile(os.path.join(path_prefix, f"{dtype}-debug-output.data"))
    
    config["paramSize"] = all_param.shape[0]
    config["maxTokenLength"] = TOKEN_LENGTH
    config["maxVocabSize"] = model.config.vocab_size
    config["hiddenSize"] = model.config.hidden_size
    with open(os.path.join(path_prefix, f"config.txt"), "w") as config_file:
        config_file.write(f'paramSize {config["paramSize"]}\n')
        config_file.write(f'hiddenSize {config["hiddenSize"]}\n')
        config_file.write(f'maxVocabSize {config["maxVocabSize"]}\n')
        config_file.write(f'maxTokenLength {config["maxTokenLength"]}\n')
    return config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, required=True, choices=['fp32', 'fp16', 'bf16'])

    args = parser.parse_args()
    dtype = args.dtype

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is None:
        raise EnvironmentError(
            "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
        )
    model, tokenizer = load_model_llama(model_path, dtype)
    import_model(model, dtype)
