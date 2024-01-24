import os
import torch
import torch._dynamo as dynamo
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse

class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1024, 1024)

    def forward(self, b):
        return self.linear(b)

model = TestModule()
a, b = torch.randn((1024, 1024)), torch.randn((1024, 1024))

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp
)

# AOT Mode
# Import PyTorch model to Buddy Graph and MLIR/LLVM IR.
graphs = dynamo_compiler.importer(
    model, a
)

assert len(graphs) == 1
for g in graphs:
    g.lower_to_top_level_ir()
    # print(g._imported_module)
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
# print(driver.construct_main_graph(True))

# Retrieve the LLaMA model path from environment variables.
model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

# Initialize the tokenizer and model from the specified model path.
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True)
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = torch.tensor([[1 for i in range(40)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
for g in graphs:
    g.lower_to_top_level_ir()
    print(g._imported_module)
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
print(driver.construct_main_graph(True))