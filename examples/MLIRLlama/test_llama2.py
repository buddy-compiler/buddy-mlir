import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

class foo(torch.nn.Module):

    def forward(self, a):
        return torch.nn.functional.silu(a)

tokenizer = LlamaTokenizer.from_pretrained('/path to llama2')
model = LlamaForCausalLM.from_pretrained('/path to llama2', torchscript=True)
prompt = "Hey, please say hello world to me!"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.input_ids

f = foo()
a = torch.ones([1, 13, 11008])

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
)

#gm, params = dynamo_compiler.importer(f, [a])
gm, params = dynamo_compiler.importer(model, [torch.tensor([[1 for i in range(80)]], dtype=torch.int64)])
print(gm)

all_param = numpy.concatenate([param.detach().numpy().reshape([-1]) for param in params])
all_param.tofile("path to save model params")