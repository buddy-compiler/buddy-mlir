import os
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import BertTokenizer, BertForSequenceClassification
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from pathlib import Path
import numpy as np

model = BertForSequenceClassification.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
model.eval()
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

tokenizer = BertTokenizer.from_pretrained(
    "bhadresh-savani/bert-base-uncased-emotion"
)
inputs = {
    "input_ids": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
    "token_type_ids": torch.tensor([[0 for _ in range(5)]], dtype=torch.int64),
    "attention_mask": torch.tensor([[1 for _ in range(5)]], dtype=torch.int64),
}
with torch.no_grad():
    module, params = dynamo_compiler.importer(model, **inputs)

current_path = os.path.dirname(os.path.abspath(__file__))

with open(Path(current_path) / "bert.mlir", "w") as module_file:
    module_file.write(str(module))

float32_param = np.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params[:-1]]
)

float32_param.tofile(Path(current_path) / "arg0.data")

int64_param = params[-1].detach().numpy().reshape([-1])
int64_param.tofile(Path(current_path) / "arg1.data")
