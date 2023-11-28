import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertModel, BertTokenizer
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_text = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    module, params = dynamo_compiler.importer(model, **encoded_text)
    print(module)
    # print(params)
