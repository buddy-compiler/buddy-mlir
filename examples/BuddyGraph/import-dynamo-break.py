import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions
from transformers import BertModel, BertTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.Conv2d(3, 255, (5, 5), 3, bias=False)

    def forward(self, b, c):
        return torch.add(b, c)

# test graph break
model = TestModule()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions
)
a, b = torch.randn((1024, 1024)), torch.randn((1024, 1024))
print(model(a, b))
model_opt = torch.compile(model, backend=dynamo_compiler._compile_fx)
print(model_opt(a, b))

graphs = dynamo_compiler.importer(
    model, a, b
)

for g in graphs:
    g.lower_to_top_level_ir()
    print(g._imported_module)

# test bert
# dynamo_compiler = DynamoCompiler(
#     primary_registry=tosa.ops_registry,
#     aot_autograd_decomposition=inductor_decomp
# )
# model = BertModel.from_pretrained("bert-base-uncased")
# model.eval()
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_text = tokenizer(text, return_tensors="pt")
# print(model(**encoded_text))
# model_opt = torch.compile(model, backend=dynamo_compiler._compile_fx)
# print(model_opt(**encoded_text))
# print(model_opt(**encoded_text))

# graphs = dynamo_compiler.importer(
#     model, **encoded_text
# )

# for g in graphs:
#     g.lower_to_top_level_ir()
#     print(g._imported_module)
