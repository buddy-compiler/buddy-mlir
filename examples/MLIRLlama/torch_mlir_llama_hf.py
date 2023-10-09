import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig
import torch._dynamo as dynamo
from buddy.LlamaCompiler import DynamoCompiler

tokenizer = LlamaTokenizer.from_pretrained('/llama-2-7B-hf')
model = LlamaForCausalLM.from_pretrained('/llama-2-7B-hf', torchscript=True)
prompt = "Hey, please say hello world to me!"
inputs = tokenizer(prompt, return_tensors="pt")
#print(inputs.input_ids)
generate_ids = model.generate(inputs.input_ids, max_length=30)
#print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# print(len(generate_ids))
# print(generate_ids)
# print("-----------------------")
inputs = inputs.input_ids
#print(inputs)

model.eval()
model_opt = dynamo.optimize(DynamoCompiler)(model)
# for (name, module) in model_opt.named_modules():
#     print(name, module)
# features_in_hook = []
# features_out_hook = []

# def hook(module, fea_in, fea_out):
#     features_in_hook.append(fea_in)
#     features_out_hook.append(fea_out)
#     return None

# layer_name = '_orig_mod.model.layers.0.self_attn.v_proj'
# for (name, module) in model_opt.named_modules():
#     if name == layer_name:
#         module.register_forward_hook(hook=hook)
result = model_opt(torch.tensor([[1 for i in range(80)]], dtype=torch.int64))
# print("------------------")
# print(features_in_hook)
# print(features_out_hook)
# print("------------------")
#print(result[0].shape)
#print(result[0])
