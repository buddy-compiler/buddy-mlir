import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig
import torch._dynamo as dynamo
import sys
sys.path.append("/buddy-mlir/examples/MLIRLlama")
from buddy.LlamaCompiler import BuddyDynamoCompiler, from_dynamo
from torch._functorch.aot_autograd import aot_autograd_decompositions

class foo(torch.nn.Module):
    def forward(self, a, b):
        return a+b

if __name__ == "__main__":
    #tokenizer = LlamaTokenizer.from_pretrained('/llama-2-7B-hf')
    #model = LlamaForCausalLM.from_pretrained('/llama-2-7B-hf', torchscript=True)
    #prompt = "Hey, please say hello world to me!"
    #inputs = tokenizer(prompt, return_tensors="pt")
    #inputs = inputs.input_ids
    a = torch.tensor([13, 13])
    f = foo()
    gm, params = from_dynamo(f, [a, a])
    print(gm)
    print(params)
