# ===- import-llama2.py --------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the test of llama2 model.
#
# ===---------------------------------------------------------------------------

import os

import numpy
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._functorch.aot_autograd import aot_autograd_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


model_path = os.environ.get('LLAMA_MODEL_PATH')
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True)
prompt = "Hey, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.input_ids

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
    is_inference=True,
)

gm, params = dynamo_compiler.importer(
    model, torch.tensor([[1 for i in range(80)]], dtype=torch.int64)
)
with open(
    os.path.dirname(os.path.abspath(__file__)) + "/llama.mlir", "w"
) as module_file:
    print(gm, file=module_file)

all_param = numpy.concatenate(
    [param.detach().numpy().reshape([-1]) for param in params]
)
all_param.tofile(os.path.dirname(os.path.abspath(__file__)) + "/arg0.data")
