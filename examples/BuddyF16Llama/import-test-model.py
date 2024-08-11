import os
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler

from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse


class AddOne(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    
    def forward(self, data):
        return data.to(torch.float16) + self.param
    
    @staticmethod
    def get_test_data():
        return torch.randn((4, 4), dtype=torch.bfloat16)

class AddMM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    
    def forward(self, data):
        # return torch.addmm(data, self.param, data)
        return torch.mm(data, self.param)
    
    @staticmethod
    def get_test_data():
        return torch.randn((4, 4), dtype=torch.bfloat16)


class CastModel(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        return data.to(torch.bfloat16)
    
    @staticmethod
    def get_test_data():
        return torch.randn((4, 4), dtype=torch.float32)


class OneMinusModel(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        # return torch.ones((4, 4), dtype=torch.float32) - data  # ok
        # return 1.0 - data
        return data + 1.0
    
    @staticmethod
    def get_test_data():
        return torch.randn((4, 4), dtype=torch.bfloat16)
        # return torch.randn((4, 4), dtype=torch.float32)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
# model = AddOne()
# model = AddMM()
model = OneMinusModel()
with torch.no_grad():
    data = model.get_test_data()
    model_opt = dynamo.optimize(dynamo_compiler._compile_fx)(model)
    model_opt(data)
    graphs = dynamo_compiler._imported_graphs
    

assert len(graphs) == 1, f'{len(graphs)} graphs imported'
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
# all_param = numpy.concatenate(
#     [param.detach().numpy().reshape([-1]) for param in params]
# )
# all_param.tofile(os.path.join(path_prefix, "arg0.data"))