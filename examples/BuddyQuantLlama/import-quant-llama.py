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


def load_model_llama(model_path) -> tuple[LlamaTokenizer, LlamaForCausalLM, dict[str, int]]:
    print(f'Loading model')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True, torch_dtype=torch.float32)
    model.config.use_cache = False
    return model, tokenizer

def import_model(model, example_inputs):
    
    config = {}
    
    # Initialize Dynamo Compiler with specific configurations as an importer.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import the model into MLIR module and parameters.
    
    with torch.no_grad():
        expected_output = model(example_inputs)[0]
        print(model(example_inputs))
        graphs = dynamo_compiler.importer(model, example_inputs)
    
    assert len(graphs) == 1, f'{len(graphs)} graphs imported'
    graph = graphs[0]
    for op in graph._body:
        print(op.name, op.args, op.kwargs, op.tensor_meta)
    
    params = dynamo_compiler.imported_params[graph]
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()
    
    # persist
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path_prefix, "quant-subgraph0.mlir"), "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)
    with open(os.path.join(path_prefix, "quant-forward.mlir"), "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)
    
    # pack params for different dtypes
    params_of_dtype = {}
    for param in params:
        param_np = param.detach().numpy().reshape([-1])
        dtype = param_np.dtype
        params_of_dtype.setdefault(dtype, [])
        params_of_dtype[dtype].append(param_np)
    for dtype, param_list in params_of_dtype.items():
        params = numpy.concatenate(param_list)
        print(f'parameter of type {dtype}:', params[:10])
        params.tofile(os.path.join(path_prefix, f"params-quantized-{dtype}.data"))
        config[f"{dtype}ParamSize"] = params.shape[0]
    expected_output.reshape(-1).float().cpu().numpy().tofile(os.path.join(path_prefix, f"{dtype}-debug-output-quantized.data"))
    
    config["maxTokenLength"] = TOKEN_LENGTH
    config["maxVocabSize"] = model.config.vocab_size
    config["hiddenSize"] = model.config.hidden_size
    with open(os.path.join(path_prefix, f"config.txt"), "w") as config_file:
        for k in ['float32ParamSize', 'int64ParamSize', 'int8ParamSize',
                  'hiddenSize', 'maxVocabSize', 'maxTokenLength']:
            config_file.write(f'{k} {config[k]}\n')
    return config

def get_gm_op_set(gm: torch.fx.GraphModule):
    opset = set()
    for node in gm.graph.nodes:
        if node.target is not None and hasattr(node.target, '__name__'):
            opset.add(str(node.target.__name__))
    return opset

def get_model_size_mb(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10, bias=False)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    import copy
    from torch_quantize import *
    
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path is None:
        raise EnvironmentError(
            "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
        )
    model, tokenizer = load_model_llama(model_path)
    TOKEN_LENGTH = 40
    example_inputs = torch.tensor([[i for i in range(TOKEN_LENGTH)]], dtype=torch.int64)
    
    # for test
    # model = M()
    # example_inputs = torch.rand(8, 5)
    
    # quantization
    float_model = model
    
    original_gm, guards = torchdynamo.export(
        float_model,
        copy.deepcopy(example_inputs),
        aten_graph=True,
    )
    original_opset = get_gm_op_set(original_gm)
    

    quantizer = BackendQuantizer()
    operator_config = get_symmetric_quantization_config()
    quantizer.set_global(operator_config)
    # Note: ``prepare_pt2e_quantizer`` will be updated to ``prepare_pt2e`` soon
    prepared_gm = prepare_pt2e(original_gm, quantizer)
    after_prepare_result = prepared_gm(example_inputs)
    converted_gm = convert_pt2e(prepared_gm)
    
    print('original model is:')
    original_gm.print_readable()
    print("converted module is:")
    converted_gm.print_readable()
    
    converted_opset = get_gm_op_set(converted_gm)
    
    print('\n\noriginal op set:')
    print(original_opset)
    print('\n\nconverted op set:')
    print(converted_opset)
    print('\n\ndifference op set')
    print(converted_opset.difference(original_opset))
    print(original_opset.difference(converted_opset))
    
    print(f'\n\noriginal model size: {get_model_size_mb(original_gm)} MB')
    print(f'\n\nconverted model size: {get_model_size_mb(converted_gm)} MB')
    
    converted_gm.recompile()
    quantized_model = converted_gm
    quantized_model.config = model.config
    import_model(quantized_model, example_inputs)
