import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions
import numpy
import os

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

tokenizer = LlamaTokenizer.from_pretrained('/llama-2-7B-hf')
model = LlamaForCausalLM.from_pretrained('/llama-2-7B-hf', torchscript=True)
prompt = "Hey, please say hello world to me!"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.input_ids

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

gm, params = dynamo_compiler.importer(model, torch.tensor([[1 for i in range(80)]], dtype=torch.int64))
with open(os.path.dirname(os.path.abspath(__file__))+"/llama.mlir", 'w') as module_file:
    print(gm, file=module_file)

all_param = numpy.concatenate([param.detach().numpy().reshape([-1]) for param in params])
all_param.tofile(os.path.dirname(os.path.abspath(__file__))+"/arg0.data")
